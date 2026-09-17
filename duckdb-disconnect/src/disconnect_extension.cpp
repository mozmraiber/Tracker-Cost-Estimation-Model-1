#include "disconnect_list.hpp"
#include "url_host.hpp"

#include "duckdb/common/file_system.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {
namespace disconnect {

//===--------------------------------------------------------------------===//
// Scalar functions
//===--------------------------------------------------------------------===//

//! Holds the category filter of a two-argument call when it is a constant. The
//! filter is kept as text rather than as a bitmask because disconnect_load()
//! may install a list with different category indexes after binding.
struct CategoryFilterData : public FunctionData {
	explicit CategoryFilterData(string filter_p) : filter(std::move(filter_p)) {
	}

	string filter;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<CategoryFilterData>(filter);
	}
	bool Equals(const FunctionData &other) const override {
		return filter == other.Cast<CategoryFilterData>().filter;
	}
};

static unique_ptr<FunctionData> BindCategoryFilter(ClientContext &context, ScalarFunction &bound_function,
                                                   vector<unique_ptr<Expression>> &arguments) {
	if (!arguments[1]->IsFoldable()) {
		return nullptr;
	}
	const auto filter = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	if (filter.IsNull()) {
		return nullptr;
	}
	const auto filter_text = StringValue::Get(filter);
	// Reject unknown category names while the query is being planned.
	DisconnectList::Current()->ParseCategoryFilter(filter_text);
	return make_uniq<CategoryFilterData>(filter_text);
}

//! is_tracker(url): every category except the one DefaultCategoryMask() drops.
//! Pass a filter to the two-argument form to widen or narrow that.
static void IsTrackerFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto list = DisconnectList::Current();
	const auto mask = list->DefaultCategoryMask();
	UnaryExecutor::Execute<string_t, bool>(args.data[0], result, args.size(), [&](string_t url) {
		MatchResult match;
		return list->Match(url.GetData(), url.GetSize(), mask, match);
	});
}

static void IsTrackerFilteredFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto list = DisconnectList::Current();
	auto &bind_info = state.expr.Cast<BoundFunctionExpression>().bind_info;
	if (bind_info) {
		// Constant filter: resolve it once for the whole chunk.
		const auto mask = list->ParseCategoryFilter(bind_info->Cast<CategoryFilterData>().filter);
		UnaryExecutor::Execute<string_t, bool>(args.data[0], result, args.size(), [&](string_t url) {
			MatchResult match;
			return list->Match(url.GetData(), url.GetSize(), mask, match);
		});
		return;
	}
	BinaryExecutor::Execute<string_t, string_t, bool>(
	    args.data[0], args.data[1], result, args.size(), [&](string_t url, string_t filter) {
		    MatchResult match;
		    return list->Match(url.GetData(), url.GetSize(), list->ParseCategoryFilter(filter.GetString()), match);
	    });
}

//! Shared body of the scalar functions that describe a match; `extract` turns a
//! match into the string to return. Rows without a match become NULL.
template <class EXTRACT>
static void MatchStringFunction(DataChunk &args, Vector &result, EXTRACT extract) {
	auto list = DisconnectList::Current();
	UnaryExecutor::ExecuteWithNulls<string_t, string_t>(
	    args.data[0], result, args.size(), [&](string_t url, ValidityMask &mask, idx_t row) {
		    MatchResult match;
		    if (!list->Match(url.GetData(), url.GetSize(), ALL_CATEGORIES, match)) {
			    mask.SetInvalid(row);
			    return string_t();
		    }
		    return StringVector::AddString(result, extract(*list, match));
	    });
}

static void TrackerCategoryFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	MatchStringFunction(args, result,
	                    [](const DisconnectList &list, const MatchResult &match) -> const string & {
		                    return list.CategoryName(match.category);
	                    });
}

static void TrackerOwnerFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	MatchStringFunction(args, result, [](const DisconnectList &list, const MatchResult &match) -> const string & {
		return list.Org(match.organization).name;
	});
}

static void TrackerOwnerUrlFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	MatchStringFunction(args, result, [](const DisconnectList &list, const MatchResult &match) -> const string & {
		return list.Org(match.organization).url;
	});
}

static void TrackerPatternFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	MatchStringFunction(args, result,
	                    [](const DisconnectList &list, const MatchResult &match) -> const string & {
		                    return *match.pattern;
	                    });
}

//! Every category a URL is listed under, or NULL when it is not listed.
static void TrackerCategoriesFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto list = DisconnectList::Current();
	const auto count = args.size();

	UnifiedVectorFormat input;
	args.data[0].ToUnifiedFormat(count, input);
	const auto urls = UnifiedVectorFormat::GetData<string_t>(input);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto list_entries = FlatVector::GetData<list_entry_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	idx_t child_offset = 0;
	for (idx_t row = 0; row < count; row++) {
		const auto index = input.sel->get_index(row);
		list_entries[row] = list_entry_t(child_offset, 0);
		MatchResult match;
		if (!input.validity.RowIsValid(index) ||
		    !list->Match(urls[index].GetData(), urls[index].GetSize(), ALL_CATEGORIES, match)) {
			result_validity.SetInvalid(row);
			continue;
		}
		idx_t matched = 0;
		for (idx_t category = 0; category < list->Categories().size(); category++) {
			if ((match.category_mask & (uint64_t(1) << category)) == 0) {
				continue;
			}
			ListVector::Reserve(result, child_offset + matched + 1);
			// Reserve may reallocate, so the child data pointer is fetched again.
			auto child_data = FlatVector::GetData<string_t>(ListVector::GetEntry(result));
			child_data[child_offset + matched] =
			    StringVector::AddString(ListVector::GetEntry(result), list->CategoryName(UnsafeNumericCast<uint16_t>(category)));
			matched++;
		}
		list_entries[row] = list_entry_t(child_offset, matched);
		child_offset += matched;
	}
	ListVector::SetListSize(result, child_offset);
}

//! The host of a URL, using the same parsing the matcher uses.
static void UrlHostFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::ExecuteWithNulls<string_t, string_t>(
	    args.data[0], result, args.size(), [&](string_t url, ValidityMask &mask, idx_t row) {
		    const auto parts = SplitUrl(url.GetData(), url.GetSize());
		    if (parts.host_len == 0) {
			    mask.SetInvalid(row);
			    return string_t();
		    }
		    return StringVector::AddString(result, parts.host, parts.host_len);
	    });
}

//===--------------------------------------------------------------------===//
// Table functions
//===--------------------------------------------------------------------===//

struct ListSnapshotBindData : public TableFunctionData {
	shared_ptr<const DisconnectList> list;
	string path;
};

struct ListScanState : public GlobalTableFunctionState {
	idx_t offset = 0;

	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<GlobalTableFunctionState> InitListScan(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<ListScanState>();
}

static unique_ptr<FunctionData> BindEntries(ClientContext &context, TableFunctionBindInput &input,
                                            vector<LogicalType> &return_types, vector<string> &names) {
	names = {"pattern", "category", "organization", "organization_url"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR};
	auto result = make_uniq<ListSnapshotBindData>();
	result->list = DisconnectList::Current();
	return std::move(result);
}

static void ScanEntries(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<ListSnapshotBindData>();
	auto &state = data.global_state->Cast<ListScanState>();
	auto &list = *bind_data.list;
	auto &triples = list.Triples();

	const idx_t count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, triples.size() - state.offset);
	for (idx_t i = 0; i < count; i++) {
		auto &triple = triples[state.offset + i];
		auto &org = list.Org(triple.organization);
		output.SetValue(0, i, Value(triple.pattern));
		output.SetValue(1, i, Value(list.CategoryName(triple.category)));
		output.SetValue(2, i, Value(org.name));
		output.SetValue(3, i, Value(org.url));
	}
	state.offset += count;
	output.SetCardinality(count);
}

static void ListInfoColumns(vector<LogicalType> &return_types, vector<string> &names) {
	names = {"source", "hosts", "rules", "categories", "organizations"};
	return_types = {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
	                LogicalType::BIGINT};
}

static void EmitListInfo(const DisconnectList &list, DataChunk &output) {
	output.SetValue(0, 0, Value(list.Source()));
	output.SetValue(1, 0, Value::BIGINT(UnsafeNumericCast<int64_t>(list.HostCount())));
	output.SetValue(2, 0, Value::BIGINT(UnsafeNumericCast<int64_t>(list.Triples().size())));
	output.SetValue(3, 0, Value::BIGINT(UnsafeNumericCast<int64_t>(list.Categories().size())));
	output.SetValue(4, 0, Value::BIGINT(UnsafeNumericCast<int64_t>(list.OrganizationCount())));
	output.SetCardinality(1);
}

static unique_ptr<FunctionData> BindListInfo(ClientContext &context, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<string> &names) {
	ListInfoColumns(return_types, names);
	auto result = make_uniq<ListSnapshotBindData>();
	result->list = DisconnectList::Current();
	return std::move(result);
}

static void ScanListInfo(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<ListScanState>();
	if (state.offset > 0) {
		output.SetCardinality(0);
		return;
	}
	state.offset = 1;
	EmitListInfo(*data.bind_data->Cast<ListSnapshotBindData>().list, output);
}

static unique_ptr<FunctionData> BindLoad(ClientContext &context, TableFunctionBindInput &input,
                                         vector<LogicalType> &return_types, vector<string> &names) {
	ListInfoColumns(return_types, names);
	auto result = make_uniq<ListSnapshotBindData>();
	if (input.inputs[0].IsNull()) {
		throw InvalidInputException("disconnect_load requires a path to a Disconnect services JSON file");
	}
	result->path = StringValue::Get(input.inputs[0]);
	return std::move(result);
}

//! Read a whole file through DuckDB's file system, so that paths handled by
//! httpfs and friends work as well as local ones.
static string ReadFile(ClientContext &context, const string &path) {
	auto &fs = FileSystem::GetFileSystem(context);
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
	const auto file_size = handle->GetFileSize();
	static constexpr idx_t MAX_LIST_SIZE = 256 * 1024 * 1024;
	if (file_size > MAX_LIST_SIZE) {
		throw InvalidInputException("\"%s\" is %llu bytes, larger than the %llu byte limit for a services list", path,
		                            static_cast<uint64_t>(file_size), static_cast<uint64_t>(MAX_LIST_SIZE));
	}
	string contents;
	contents.resize(file_size);
	idx_t read_total = 0;
	while (read_total < file_size) {
		const auto bytes = handle->Read(&contents[read_total], file_size - read_total);
		if (bytes <= 0) {
			throw IOException("Failed to read \"%s\": unexpected end of file", path);
		}
		read_total += UnsafeNumericCast<idx_t>(bytes);
	}
	return contents;
}

static void ScanLoad(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<ListSnapshotBindData>();
	auto &state = data.global_state->Cast<ListScanState>();
	if (state.offset > 0) {
		output.SetCardinality(0);
		return;
	}
	state.offset = 1;
	const auto contents = ReadFile(context, bind_data.path);
	auto list = DisconnectList::FromJson(contents.c_str(), contents.size(), bind_data.path);
	DisconnectList::Replace(list);
	EmitListInfo(*list, output);
}

static unique_ptr<FunctionData> BindReset(ClientContext &context, TableFunctionBindInput &input,
                                          vector<LogicalType> &return_types, vector<string> &names) {
	ListInfoColumns(return_types, names);
	return make_uniq<ListSnapshotBindData>();
}

static void ScanReset(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<ListScanState>();
	if (state.offset > 0) {
		output.SetCardinality(0);
		return;
	}
	state.offset = 1;
	EmitListInfo(*DisconnectList::ResetToEmbedded(), output);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//

static void RegisterMatchString(ExtensionLoader &loader, const char *name, scalar_function_t function) {
	loader.RegisterFunction(ScalarFunction(name, {LogicalType::VARCHAR}, LogicalType::VARCHAR, function));
}

static void LoadInternal(ExtensionLoader &loader) {
	loader.SetDescription("Classifies URLs against the Disconnect services list");

	ScalarFunctionSet is_tracker("is_tracker");
	is_tracker.AddFunction(
	    ScalarFunction({LogicalType::VARCHAR}, LogicalType::BOOLEAN, IsTrackerFunction));
	is_tracker.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                                      IsTrackerFilteredFunction, BindCategoryFilter));
	loader.RegisterFunction(is_tracker);

	RegisterMatchString(loader, "tracker_category", TrackerCategoryFunction);
	RegisterMatchString(loader, "tracker_owner", TrackerOwnerFunction);
	RegisterMatchString(loader, "tracker_owner_url", TrackerOwnerUrlFunction);
	RegisterMatchString(loader, "tracker_pattern", TrackerPatternFunction);
	RegisterMatchString(loader, "disconnect_url_host", UrlHostFunction);

	loader.RegisterFunction(ScalarFunction("tracker_categories", {LogicalType::VARCHAR},
	                                       LogicalType::LIST(LogicalType::VARCHAR), TrackerCategoriesFunction));

	loader.RegisterFunction(TableFunction("disconnect_entries", {}, ScanEntries, BindEntries, InitListScan));
	loader.RegisterFunction(TableFunction("disconnect_list_info", {}, ScanListInfo, BindListInfo, InitListScan));
	loader.RegisterFunction(
	    TableFunction("disconnect_load", {LogicalType::VARCHAR}, ScanLoad, BindLoad, InitListScan));
	loader.RegisterFunction(TableFunction("disconnect_reset", {}, ScanReset, BindReset, InitListScan));
}

} // namespace disconnect
} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(disconnect, loader) {
	duckdb::disconnect::LoadInternal(loader);
}
}
