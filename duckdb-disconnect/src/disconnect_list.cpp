#include "disconnect_list.hpp"

#include "mini_json.hpp"
#include "url_host.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>

namespace duckdb {
namespace disconnect {

//! Many hosts are listed under several categories: the fingerprinting lists in
//! particular overlap heavily with Advertising and Analytics. When that
//! happens, tracker_category() reports the first category listed here, which
//! favours what the host *does* over how it does it. Categories missing from
//! this list (a newer list revision, say) sort after all known ones, in file
//! order. tracker_categories() always reports every category.
static const char *const CATEGORY_PRIORITY[] = {
    "Advertising", "Analytics",       "Social",     "Cryptomining",    "FingerprintingInvasive",
    "Content",     "EmailAggressive", "Email",      "Anti-fraud",      "ConsentManagers",
    "FingerprintingGeneral"};
static constexpr idx_t CATEGORY_PRIORITY_COUNT = sizeof(CATEGORY_PRIORITY) / sizeof(CATEGORY_PRIORITY[0]);

static inline char LowerAscii(char c) {
	return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

static inline uint64_t HashLower(const char *data, idx_t len) {
	uint64_t hash = 14695981039346656037ULL;
	for (idx_t i = 0; i < len; i++) {
		hash ^= static_cast<uint8_t>(LowerAscii(data[i]));
		hash *= 1099511628211ULL;
	}
	return hash;
}

static inline bool EqualsLower(const string &lowercase, const char *data, idx_t len) {
	if (lowercase.size() != len) {
		return false;
	}
	for (idx_t i = 0; i < len; i++) {
		if (lowercase[i] != LowerAscii(data[i])) {
			return false;
		}
	}
	return true;
}

static string ToLower(const string &input) {
	string result = input;
	for (auto &c : result) {
		c = LowerAscii(c);
	}
	return result;
}

//! Normalize a raw list pattern: lowercase, no surrounding whitespace, no
//! trailing dot on the host part.
static string NormalizePattern(const string &input) {
	string result;
	result.reserve(input.size());
	for (auto c : input) {
		if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
			continue;
		}
		result += LowerAscii(c);
	}
	const auto slash = result.find('/');
	auto host_end = slash == string::npos ? result.size() : slash;
	while (host_end > 0 && result[host_end - 1] == '.') {
		result.erase(host_end - 1, 1);
		host_end--;
	}
	return result;
}

void DisconnectList::AddEntry(const string &pattern, uint16_t category, uint32_t organization) {
	if (pattern.empty() || category >= categories.size()) {
		return;
	}
	const auto slash = pattern.find('/');
	const string host = pattern.substr(0, slash);
	if (host.empty()) {
		return;
	}
	const auto entry_index = UnsafeNumericCast<uint32_t>(triples.size());
	triples.push_back(ListTriple {pattern, category, organization});

	auto lookup = host_lookup.find(host);
	if (lookup == host_lookup.end()) {
		lookup = host_lookup.emplace(host, hosts.size()).first;
		hosts.emplace_back();
		hosts.back().host = host;
	}
	auto &entry = hosts[lookup->second];
	const uint64_t bit = uint64_t(1) << category;
	const uint16_t rank = category_rank[category];

	if (slash == string::npos) {
		entry.category_mask |= bit;
		if (!entry.has_host_rule || rank < category_rank[entry.category]) {
			entry.category = category;
			entry.organization = organization;
			entry.entry_index = entry_index;
		}
		entry.has_host_rule = true;
		return;
	}

	const string path = pattern.substr(slash);
	for (auto &rule : entry.path_rules) {
		if (rule.path == path) {
			rule.category_mask |= bit;
			if (rank < category_rank[rule.category]) {
				rule.category = category;
				rule.organization = organization;
				rule.entry_index = entry_index;
			}
			return;
		}
	}
	PathRule rule;
	rule.path = path;
	rule.pattern = pattern;
	rule.category_mask = bit;
	rule.category = category;
	rule.organization = organization;
	rule.entry_index = entry_index;
	entry.path_rules.push_back(std::move(rule));
}

void DisconnectList::BuildIndex() {
	host_lookup.clear();
	// Keep the load factor at 50% or below; open addressing degrades quickly above that.
	idx_t capacity = 16;
	while (capacity < hosts.size() * 2) {
		capacity *= 2;
	}
	buckets.assign(capacity, 0);
	bucket_mask = capacity - 1;
	for (idx_t i = 0; i < hosts.size(); i++) {
		auto &host = hosts[i].host;
		auto slot = HashLower(host.c_str(), host.size()) & bucket_mask;
		while (buckets[slot] != 0) {
			slot = (slot + 1) & bucket_mask;
		}
		buckets[slot] = UnsafeNumericCast<uint32_t>(i + 1);
	}
}

idx_t DisconnectList::FindHost(const char *host, idx_t len) const {
	auto slot = HashLower(host, len) & bucket_mask;
	while (buckets[slot] != 0) {
		const idx_t index = buckets[slot] - 1;
		if (EqualsLower(hosts[index].host, host, len)) {
			return index;
		}
		slot = (slot + 1) & bucket_mask;
	}
	return DConstants::INVALID_INDEX;
}

//! Does `path` start with the rule's prefix? Paths in the list are lowercase,
//! so the comparison ignores case.
static bool PathMatches(const UrlParts &parts, const string &prefix) {
	if (parts.path_len < prefix.size()) {
		return false;
	}
	for (idx_t i = 0; i < prefix.size(); i++) {
		if (LowerAscii(parts.path[i]) != prefix[i]) {
			return false;
		}
	}
	return true;
}

bool DisconnectList::Match(const char *url, idx_t len, uint64_t filter_mask, MatchResult &result) const {
	const auto parts = SplitUrl(url, len);
	const char *host = parts.host;
	idx_t host_len = parts.host_len;
	while (host_len > 0) {
		const auto index = FindHost(host, host_len);
		if (index != DConstants::INVALID_INDEX) {
			auto &entry = hosts[index];
			// A path rule is more specific than a host rule, so it wins.
			for (auto &rule : entry.path_rules) {
				if ((rule.category_mask & filter_mask) == 0) {
					continue;
				}
				if (PathMatches(parts, rule.path)) {
					result.pattern = &rule.pattern;
					result.category_mask = rule.category_mask;
					result.category = rule.category;
					result.organization = rule.organization;
					result.entry_index = rule.entry_index;
					return true;
				}
			}
			if (entry.has_host_rule && (entry.category_mask & filter_mask) != 0) {
				result.pattern = &entry.host;
				result.category_mask = entry.category_mask;
				result.category = entry.category;
				result.organization = entry.organization;
				result.entry_index = entry.entry_index;
				return true;
			}
		}
		// Walk up to the parent domain: "a.b.example.com" -> "b.example.com" -> ...
		const auto *dot = static_cast<const char *>(memchr(host, '.', host_len));
		if (!dot) {
			break;
		}
		const idx_t consumed = UnsafeNumericCast<idx_t>(dot - host) + 1;
		host += consumed;
		host_len -= consumed;
	}
	return false;
}

uint64_t DisconnectList::ParseCategoryFilter(const string &filter) const {
	uint64_t mask = 0;
	idx_t pos = 0;
	while (pos <= filter.size()) {
		auto comma = filter.find(',', pos);
		if (comma == string::npos) {
			comma = filter.size();
		}
		auto begin = pos;
		auto end = comma;
		while (begin < end && std::isspace(static_cast<unsigned char>(filter[begin]))) {
			begin++;
		}
		while (end > begin && std::isspace(static_cast<unsigned char>(filter[end - 1]))) {
			end--;
		}
		if (end > begin) {
			const auto name = ToLower(filter.substr(begin, end - begin));
			bool found = false;
			for (idx_t i = 0; i < categories.size(); i++) {
				if (ToLower(categories[i]) == name) {
					mask |= uint64_t(1) << i;
					found = true;
					break;
				}
			}
			if (!found) {
				throw InvalidInputException("Unknown Disconnect category \"%s\". Known categories: %s",
				                            filter.substr(begin, end - begin), StringUtil::Join(categories, ", "));
			}
		}
		pos = comma + 1;
	}
	if (mask == 0) {
		throw InvalidInputException("Category filter \"%s\" does not name any category", filter);
	}
	return mask;
}

void DisconnectList::SetCategories(vector<string> names) {
	if (names.size() > 64) {
		throw InvalidInputException("Disconnect list has %llu categories, at most 64 are supported",
		                            static_cast<uint64_t>(names.size()));
	}
	categories = std::move(names);
	default_mask = ALL_CATEGORIES;
	category_rank.resize(categories.size());
	for (idx_t i = 0; i < categories.size(); i++) {
		if (categories[i] == EXCLUDED_BY_DEFAULT) {
			default_mask &= ~(uint64_t(1) << i);
		}
		uint16_t rank = UnsafeNumericCast<uint16_t>(CATEGORY_PRIORITY_COUNT + i);
		for (idx_t p = 0; p < CATEGORY_PRIORITY_COUNT; p++) {
			if (categories[i] == CATEGORY_PRIORITY[p]) {
				rank = UnsafeNumericCast<uint16_t>(p);
				break;
			}
		}
		category_rank[i] = rank;
	}
}

shared_ptr<const DisconnectList> DisconnectList::BuildEmbedded() {
	auto list = shared_ptr<DisconnectList>(new DisconnectList());
	list->source = GENERATED_SOURCE_INFO;

	vector<string> category_names;
	for (idx_t i = 0; i < GENERATED_CATEGORY_COUNT; i++) {
		category_names.emplace_back(GENERATED_CATEGORIES[i]);
	}
	list->SetCategories(std::move(category_names));

	for (idx_t i = 0; i < GENERATED_ORG_COUNT; i++) {
		list->organizations.push_back(Organization {GENERATED_ORG_NAMES[i], GENERATED_ORG_URLS[i]});
	}
	for (idx_t i = 0; i < GENERATED_ENTRY_COUNT; i++) {
		auto &entry = GENERATED_ENTRIES[i];
		list->AddEntry(entry.pattern, entry.category, entry.organization);
	}
	list->BuildIndex();
	return list;
}

shared_ptr<const DisconnectList> DisconnectList::FromJson(const char *data, idx_t len, const string &source) {
	string error;
	auto document = ParseJson(data, len, error);
	if (!document) {
		throw InvalidInputException("Failed to parse \"%s\" as JSON: %s", source, error);
	}
	auto category_object = document->Member("categories");
	if (document->type != JsonValue::Type::OBJECT || !category_object ||
	    category_object->type != JsonValue::Type::OBJECT) {
		throw InvalidInputException("\"%s\" is not a Disconnect services list: no top-level \"categories\" object",
		                            source);
	}

	auto list = shared_ptr<DisconnectList>(new DisconnectList());
	list->source = source;

	vector<string> category_names;
	for (auto &category : category_object->members) {
		category_names.push_back(category.first);
	}
	list->SetCategories(std::move(category_names));

	std::unordered_map<string, uint32_t> org_lookup;
	for (idx_t category_index = 0; category_index < category_object->members.size(); category_index++) {
		auto &category_entries = *category_object->members[category_index].second;
		if (category_entries.type != JsonValue::Type::ARRAY) {
			continue;
		}
		for (auto &entry : category_entries.elements) {
			if (entry->type != JsonValue::Type::OBJECT) {
				continue;
			}
			for (auto &organization : entry->members) {
				if (organization.second->type != JsonValue::Type::OBJECT) {
					continue;
				}
				for (auto &site : organization.second->members) {
					// Property markers such as {"dnt": "eff"} carry a string
					// instead of an array of domains; they are not entries.
					if (site.second->type != JsonValue::Type::ARRAY) {
						continue;
					}
					const auto org_key = organization.first + "\x1f" + site.first;
					auto org_entry = org_lookup.find(org_key);
					if (org_entry == org_lookup.end()) {
						const auto org_index = UnsafeNumericCast<uint32_t>(list->organizations.size());
						list->organizations.push_back(Organization {organization.first, site.first});
						org_entry = org_lookup.emplace(org_key, org_index).first;
					}
					for (auto &domain : site.second->elements) {
						if (domain->type != JsonValue::Type::STRING) {
							continue;
						}
						list->AddEntry(NormalizePattern(domain->str), UnsafeNumericCast<uint16_t>(category_index),
						               org_entry->second);
					}
				}
			}
		}
	}
	if (list->hosts.empty()) {
		throw InvalidInputException("\"%s\" is not a Disconnect services list: it contains no domains", source);
	}
	list->BuildIndex();
	return list;
}

static std::mutex active_list_lock;
static shared_ptr<const DisconnectList> active_list;

shared_ptr<const DisconnectList> DisconnectList::Current() {
	std::lock_guard<std::mutex> guard(active_list_lock);
	if (!active_list) {
		active_list = BuildEmbedded();
	}
	return active_list;
}

void DisconnectList::Replace(shared_ptr<const DisconnectList> list) {
	std::lock_guard<std::mutex> guard(active_list_lock);
	active_list = std::move(list);
}

shared_ptr<const DisconnectList> DisconnectList::ResetToEmbedded() {
	auto list = BuildEmbedded();
	Replace(list);
	return list;
}

} // namespace disconnect
} // namespace duckdb
