#include "url_host.hpp"

namespace duckdb {
namespace disconnect {

static bool IsAllDigits(const char *data, idx_t len) {
	if (len == 0) {
		return false;
	}
	for (idx_t i = 0; i < len; i++) {
		if (data[i] < '0' || data[i] > '9') {
			return false;
		}
	}
	return true;
}

UrlParts SplitUrl(const char *data, idx_t len) {
	UrlParts result;
	if (!data) {
		return result;
	}
	// Trim surrounding whitespace.
	idx_t begin = 0;
	while (begin < len && (data[begin] == ' ' || data[begin] == '\t' || data[begin] == '\n' || data[begin] == '\r')) {
		begin++;
	}
	while (len > begin && (data[len - 1] == ' ' || data[len - 1] == '\t' || data[len - 1] == '\n' ||
	                       data[len - 1] == '\r')) {
		len--;
	}

	// Skip the scheme: either "scheme://" or a scheme-relative "//".
	idx_t pos = begin;
	for (idx_t i = begin; i + 2 < len; i++) {
		if (data[i] == ':' && data[i + 1] == '/' && data[i + 2] == '/') {
			pos = i + 3;
			break;
		}
		// A '/' or '?' before any "://" means there is no scheme at all.
		if (data[i] == '/' || data[i] == '?' || data[i] == '#') {
			break;
		}
	}
	if (pos == begin && len - begin >= 2 && data[begin] == '/' && data[begin + 1] == '/') {
		pos = begin + 2;
	}

	// The authority runs until the first '/', '?' or '#'.
	idx_t authority_begin = pos;
	idx_t authority_end = pos;
	while (authority_end < len && data[authority_end] != '/' && data[authority_end] != '?' &&
	       data[authority_end] != '#') {
		authority_end++;
	}

	// Strip userinfo ("user:pass@host").
	for (idx_t i = authority_end; i > authority_begin; i--) {
		if (data[i - 1] == '@') {
			authority_begin = i;
			break;
		}
	}

	const char *host = data + authority_begin;
	idx_t host_len = authority_end - authority_begin;

	if (host_len > 0 && host[0] == '[') {
		// Bracketed IPv6 literal: keep what is inside the brackets.
		idx_t close = 1;
		while (close < host_len && host[close] != ']') {
			close++;
		}
		host = host + 1;
		host_len = close - 1;
	} else {
		// Strip the port, but only when what follows the last ':' is numeric,
		// so that a bare IPv6 literal is left alone.
		for (idx_t i = host_len; i > 0; i--) {
			if (host[i - 1] == ':') {
				if (IsAllDigits(host + i, host_len - i)) {
					host_len = i - 1;
				}
				break;
			}
		}
	}
	// A fully qualified name may carry a trailing root dot.
	while (host_len > 0 && host[host_len - 1] == '.') {
		host_len--;
	}

	result.host = host;
	result.host_len = host_len;

	if (authority_end < len && data[authority_end] == '/') {
		idx_t path_end = authority_end;
		while (path_end < len && data[path_end] != '?' && data[path_end] != '#') {
			path_end++;
		}
		result.path = data + authority_end;
		result.path_len = path_end - authority_end;
	}
	return result;
}

} // namespace disconnect
} // namespace duckdb
