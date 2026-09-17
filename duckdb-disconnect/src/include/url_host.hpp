#pragma once

#include "disconnect_compat.hpp"

namespace duckdb {
namespace disconnect {

//! The host and path of a URL, as pointers into the original buffer (no copies).
//! Both may be empty; `host` is not lowercased (matching is case-insensitive).
struct UrlParts {
	const char *host = nullptr;
	idx_t host_len = 0;
	const char *path = nullptr;
	idx_t path_len = 0;
};

//! Split a URL into its host and path. Accepts full URLs ("https://a.b/c?d"),
//! scheme-relative URLs ("//a.b/c") and bare hostnames ("a.b"), and strips
//! userinfo, ports, trailing dots and IPv6 brackets. Never throws.
UrlParts SplitUrl(const char *data, idx_t len);

} // namespace disconnect
} // namespace duckdb
