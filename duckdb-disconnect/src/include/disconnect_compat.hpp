#pragma once

//! The list model, the URL splitter and the JSON reader are shared between two
//! builds: the DuckDB extension, and the standalone Python module in
//! src/python_module.cpp. They are written against a handful of names from
//! `duckdb.hpp` (`string`, `idx_t`, `InvalidInputException`, ...). When
//! DISCONNECT_NO_DUCKDB is defined those names are provided from the standard
//! library instead, so the core compiles with no DuckDB headers and links no
//! DuckDB library. Everything DuckDB-specific lives in disconnect_extension.cpp.

#ifndef DISCONNECT_NO_DUCKDB

#include "duckdb.hpp"

#else

#include <cctype>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace duckdb {

using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

using idx_t = uint64_t;

struct DConstants {
	static constexpr idx_t INVALID_INDEX = idx_t(-1);
};

template <class T, class... ARGS>
unique_ptr<T> make_uniq(ARGS &&...args) {
	return std::make_unique<T>(std::forward<ARGS>(args)...);
}

//! DuckDB's version checks the value survives the conversion; here the callers
//! have already established that, so this is a plain cast.
template <class DST, class SRC>
DST UnsafeNumericCast(SRC value) {
	return static_cast<DST>(value);
}

struct StringUtil {
	static string Join(const vector<string> &parts, const string &separator) {
		string result;
		for (idx_t i = 0; i < parts.size(); i++) {
			if (i > 0) {
				result += separator;
			}
			result += parts[i];
		}
		return result;
	}
};

inline string ToMessageString(const string &value) {
	return value;
}
inline string ToMessageString(const char *value) {
	return value ? string(value) : string("(null)");
}
inline string ToMessageString(uint64_t value) {
	return std::to_string(value);
}

//! Substitute each printf conversion in `format` with the next argument, which
//! has already been rendered to a string. Only the conversions the callers use
//! are recognized, and the length modifiers they carry are ignored.
inline string FormatMessage(const string &format, const vector<string> &args) {
	string result;
	idx_t next_arg = 0;
	for (size_t i = 0; i < format.size(); i++) {
		if (format[i] != '%') {
			result += format[i];
			continue;
		}
		if (i + 1 < format.size() && format[i + 1] == '%') {
			result += '%';
			i++;
			continue;
		}
		// Walk to the conversion character, past any flags, width and length modifiers.
		size_t spec = i + 1;
		while (spec < format.size() && !std::isalpha(static_cast<unsigned char>(format[spec]))) {
			spec++;
		}
		while (spec < format.size() && (format[spec] == 'l' || format[spec] == 'h' || format[spec] == 'z')) {
			spec++;
		}
		result += next_arg < args.size() ? args[next_arg++] : string();
		i = spec;
	}
	return result;
}

//! Stands in for DuckDB's exception of the same name: raised for a malformed
//! services list or an unknown category name, and mapped to ValueError by the
//! Python module.
class InvalidInputException : public std::runtime_error {
public:
	template <class... ARGS>
	explicit InvalidInputException(const string &format, ARGS &&...args)
	    : std::runtime_error(FormatMessage(format, {ToMessageString(std::forward<ARGS>(args))...})) {
	}
};

} // namespace duckdb

#endif
