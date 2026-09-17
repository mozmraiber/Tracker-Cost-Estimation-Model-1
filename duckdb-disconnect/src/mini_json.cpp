#include "mini_json.hpp"

namespace duckdb {
namespace disconnect {

const JsonValue *JsonValue::Member(const string &key) const {
	for (auto &member : members) {
		if (member.first == key) {
			return member.second.get();
		}
	}
	return nullptr;
}

namespace {

constexpr idx_t MAX_DEPTH = 64;

class Parser {
public:
	Parser(const char *data_p, idx_t len_p) : data(data_p), len(len_p) {
	}

	unique_ptr<JsonValue> ParseDocument(string &error) {
		auto value = ParseValue(0);
		if (!value) {
			error = failure;
			return nullptr;
		}
		SkipWhitespace();
		if (pos != len) {
			error = "trailing data at byte " + std::to_string(pos);
			return nullptr;
		}
		return value;
	}

private:
	const char *data;
	idx_t len;
	idx_t pos = 0;
	string failure;

	std::nullptr_t Fail(const string &message) {
		if (failure.empty()) {
			failure = message + " at byte " + std::to_string(pos);
		}
		return nullptr;
	}

	void SkipWhitespace() {
		while (pos < len) {
			const char c = data[pos];
			if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
				pos++;
			} else {
				break;
			}
		}
	}

	bool Consume(char expected) {
		if (pos < len && data[pos] == expected) {
			pos++;
			return true;
		}
		return false;
	}

	static void AppendUtf8(string &out, uint32_t codepoint) {
		if (codepoint < 0x80) {
			out += static_cast<char>(codepoint);
		} else if (codepoint < 0x800) {
			out += static_cast<char>(0xC0 | (codepoint >> 6));
			out += static_cast<char>(0x80 | (codepoint & 0x3F));
		} else if (codepoint < 0x10000) {
			out += static_cast<char>(0xE0 | (codepoint >> 12));
			out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
			out += static_cast<char>(0x80 | (codepoint & 0x3F));
		} else {
			out += static_cast<char>(0xF0 | (codepoint >> 18));
			out += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
			out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
			out += static_cast<char>(0x80 | (codepoint & 0x3F));
		}
	}

	bool ParseHex4(uint32_t &result) {
		if (pos + 4 > len) {
			return false;
		}
		result = 0;
		for (idx_t i = 0; i < 4; i++) {
			const char c = data[pos + i];
			uint32_t digit;
			if (c >= '0' && c <= '9') {
				digit = static_cast<uint32_t>(c - '0');
			} else if (c >= 'a' && c <= 'f') {
				digit = static_cast<uint32_t>(c - 'a' + 10);
			} else if (c >= 'A' && c <= 'F') {
				digit = static_cast<uint32_t>(c - 'A' + 10);
			} else {
				return false;
			}
			result = (result << 4) | digit;
		}
		pos += 4;
		return true;
	}

	bool ParseString(string &out) {
		if (!Consume('"')) {
			Fail("expected '\"'");
			return false;
		}
		while (pos < len) {
			const char c = data[pos++];
			if (c == '"') {
				return true;
			}
			if (c != '\\') {
				out += c;
				continue;
			}
			if (pos >= len) {
				break;
			}
			const char escape = data[pos++];
			switch (escape) {
			case '"':
			case '\\':
			case '/':
				out += escape;
				break;
			case 'b':
				out += '\b';
				break;
			case 'f':
				out += '\f';
				break;
			case 'n':
				out += '\n';
				break;
			case 'r':
				out += '\r';
				break;
			case 't':
				out += '\t';
				break;
			case 'u': {
				uint32_t codepoint;
				if (!ParseHex4(codepoint)) {
					Fail("invalid \\u escape");
					return false;
				}
				if (codepoint >= 0xD800 && codepoint <= 0xDBFF && pos + 1 < len && data[pos] == '\\' &&
				    data[pos + 1] == 'u') {
					const idx_t saved = pos;
					pos += 2;
					uint32_t low;
					if (ParseHex4(low) && low >= 0xDC00 && low <= 0xDFFF) {
						codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
					} else {
						pos = saved;
					}
				}
				AppendUtf8(out, codepoint);
				break;
			}
			default:
				Fail("invalid escape sequence");
				return false;
			}
		}
		Fail("unterminated string");
		return false;
	}

	//! Numbers, booleans and null are not modelled; skip over them.
	bool SkipScalar() {
		const idx_t start = pos;
		while (pos < len) {
			const char c = data[pos];
			if (c == ',' || c == '}' || c == ']' || c == ' ' || c == '\t' || c == '\n' || c == '\r') {
				break;
			}
			pos++;
		}
		if (pos == start) {
			Fail("unexpected character");
			return false;
		}
		return true;
	}

	unique_ptr<JsonValue> ParseValue(idx_t depth) {
		if (depth > MAX_DEPTH) {
			return Fail("maximum nesting depth exceeded");
		}
		SkipWhitespace();
		if (pos >= len) {
			return Fail("unexpected end of input");
		}
		auto value = make_uniq<JsonValue>();
		const char c = data[pos];
		if (c == '{') {
			pos++;
			value->type = JsonValue::Type::OBJECT;
			SkipWhitespace();
			if (Consume('}')) {
				return value;
			}
			while (true) {
				SkipWhitespace();
				string key;
				if (!ParseString(key)) {
					return nullptr;
				}
				SkipWhitespace();
				if (!Consume(':')) {
					return Fail("expected ':'");
				}
				auto member = ParseValue(depth + 1);
				if (!member) {
					return nullptr;
				}
				value->members.emplace_back(std::move(key), std::move(member));
				SkipWhitespace();
				if (Consume(',')) {
					continue;
				}
				if (Consume('}')) {
					return value;
				}
				return Fail("expected ',' or '}'");
			}
		}
		if (c == '[') {
			pos++;
			value->type = JsonValue::Type::ARRAY;
			SkipWhitespace();
			if (Consume(']')) {
				return value;
			}
			while (true) {
				auto element = ParseValue(depth + 1);
				if (!element) {
					return nullptr;
				}
				value->elements.push_back(std::move(element));
				SkipWhitespace();
				if (Consume(',')) {
					continue;
				}
				if (Consume(']')) {
					return value;
				}
				return Fail("expected ',' or ']'");
			}
		}
		if (c == '"') {
			value->type = JsonValue::Type::STRING;
			if (!ParseString(value->str)) {
				return nullptr;
			}
			return value;
		}
		value->type = JsonValue::Type::OTHER;
		if (!SkipScalar()) {
			return nullptr;
		}
		return value;
	}
};

} // namespace

unique_ptr<JsonValue> ParseJson(const char *data, idx_t len, string &error) {
	Parser parser(data, len);
	return parser.ParseDocument(error);
}

} // namespace disconnect
} // namespace duckdb
