//! The URL fields the feature vector is built from.
//!
//! Each accessor reproduces one expression of `sql/05_per_request_full.sql`,
//! which is where the columns `engineer_features` reads came from. Reproducing
//! the SQL rather than doing the obvious thing matters in a couple of places:
//! `num_query_params` is 1 and not 0 for a URL with no query, because
//! BigQuery's `SPLIT('', '&')` is one empty element; and `file_extension` is
//! the first dot-suffix in the *whole* URL that ends at `?`, `#` or the end of
//! the string, so `/gtag/js?id=1` has no extension at all while
//! `https://cdn.example.com?a=1` has `com`.

/// A request URL, split into the pieces the features key on.
pub struct Url<'a> {
    /// As given. `url_length` counts this whole string, and
    /// `file_extension`, `has_query_params` and `num_query_params` are all
    /// taken over it rather than over the path.
    full: &'a str,
    /// `NET.HOST(req.url)`, which is what both target encodings are keyed on.
    pub host: &'a str,
    /// `REGEXP_EXTRACT(req.url, r'https?://[^/]+(\/[^?#]*)')`.
    ///
    /// `/` when the URL has no path, which is what `COALESCE` gives
    /// `path_depth` in the SQL and what `fillna` gives the estimator's callers.
    /// The regex features and the embedding read this.
    pub path: &'a str,
}

impl<'a> Url<'a> {
    /// Split a URL without validating it.
    ///
    /// A bare `host/path` with no scheme parses too, as `llm-classifier`'s
    /// does. The SQL's `https?://` is required rather than optional, so it
    /// would have yielded NULL for one; the difference only shows up for a
    /// caller that omits the scheme, and answering from the path beats
    /// answering from nothing.
    pub fn parse(url: &'a str) -> Self {
        let after_scheme = match url.find("://") {
            Some(i) => &url[i + 3..],
            None => url,
        };
        let authority_end = after_scheme
            .find(['/', '?', '#'])
            .unwrap_or(after_scheme.len());
        let (authority, rest) = after_scheme.split_at(authority_end);

        let path_end = rest.find(['?', '#']).unwrap_or(rest.len());
        let path = &rest[..path_end];

        Self {
            full: url,
            host: host_of(authority),
            // No leading `/` means the SQL's path regex did not match, and
            // every reader of the path treats that as `/`.
            path: if path.starts_with('/') { path } else { "/" },
        }
    }

    /// `LENGTH(req.url)`, which counts characters rather than bytes.
    pub fn length(&self) -> usize {
        self.full.chars().count()
    }

    /// `ARRAY_LENGTH(SPLIT(path, '/')) - 1`, i.e. the number of `/` in the
    /// path. The leading one counts, so a path of `/` has depth 1.
    pub fn path_depth(&self) -> usize {
        self.path.matches('/').count()
    }

    /// `REGEXP_CONTAINS(req.url, r'\?')`.
    pub fn has_query(&self) -> bool {
        self.full.contains('?')
    }

    /// `ARRAY_LENGTH(SPLIT(COALESCE(REGEXP_EXTRACT(req.url, r'\?(.*)$'), ''), '&'))`.
    ///
    /// One more than the number of `&` after the first `?` -- and 1, not 0,
    /// when there is no query at all, because BigQuery splits the empty string
    /// into one empty element.
    pub fn num_query_params(&self) -> usize {
        match self.full.split_once('?') {
            Some((_, query)) => 1 + query.matches('&').count(),
            None => 1,
        }
    }

    /// `LOWER(REGEXP_EXTRACT(req.url, r'\.([a-zA-Z0-9]+)(?:\?|#|$)'))`.
    ///
    /// The leftmost dot-suffix of the whole URL that runs out at `?`, `#` or
    /// the end of the string; `None` when there is none, which the one-hots
    /// read the same way `engineer_features` reads a NULL column.
    pub fn file_extension(&self) -> Option<String> {
        let bytes = self.full.as_bytes();
        for start in 0..bytes.len() {
            if bytes[start] != b'.' {
                continue;
            }
            let mut end = start + 1;
            while end < bytes.len() && bytes[end].is_ascii_alphanumeric() {
                end += 1;
            }
            let terminated = end == bytes.len() || bytes[end] == b'?' || bytes[end] == b'#';
            if end > start + 1 && terminated {
                return Some(self.full[start + 1..end].to_ascii_lowercase());
            }
        }
        None
    }
}

/// The host of an authority, as `NET.HOST` reports it: no userinfo, no port.
fn host_of(authority: &str) -> &str {
    let after_userinfo = match authority.rfind('@') {
        Some(i) => &authority[i + 1..],
        None => authority,
    };
    // An IPv6 literal's colons are inside brackets, so only look past them.
    let port_search_from = after_userinfo.rfind(']').map_or(0, |i| i + 1);
    match after_userinfo[port_search_from..].find(':') {
        Some(i) => &after_userinfo[..port_search_from + i],
        None => after_userinfo,
    }
}
