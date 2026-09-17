#pragma once

#include "disconnect_compat.hpp"

#include <string>
#include <utility>
#include <vector>

namespace duckdb {
namespace disconnect {

//! A deliberately small JSON reader, just large enough for the Disconnect
//! services list. Values that are neither objects, arrays nor strings are kept
//! as OTHER: the list format only ever uses those to mark properties such as
//! {"session-replay": "true"}, which the list builder skips.
struct JsonValue {
	enum class Type : uint8_t { OBJECT, ARRAY, STRING, OTHER };

	Type type = Type::OTHER;
	string str;                                                 // Type::STRING
	vector<std::pair<string, unique_ptr<JsonValue>>> members;   // Type::OBJECT
	vector<unique_ptr<JsonValue>> elements;                     // Type::ARRAY

	const JsonValue *Member(const string &key) const;
};

//! Parse a JSON document. Returns nullptr and fills `error` on malformed input.
unique_ptr<JsonValue> ParseJson(const char *data, idx_t len, string &error);

} // namespace disconnect
} // namespace duckdb
