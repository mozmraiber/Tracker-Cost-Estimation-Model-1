//! A CPython extension module exposing the same tracker classification the
//! `disconnect` DuckDB extension provides, without going through DuckDB. The
//! list model, matcher and URL parser are the very same sources; only the
//! calling convention differs (see src/disconnect_extension.cpp for the SQL
//! one). Built with DISCONNECT_NO_DUCKDB, so this links no DuckDB library.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "disconnect_list.hpp"
#include "url_host.hpp"

#include <fstream>
#include <new>
#include <vector>

#ifndef DISCONNECT_VERSION
#define DISCONNECT_VERSION "unknown"
#endif

namespace {

using duckdb::disconnect::ALL_CATEGORIES;
using duckdb::disconnect::DisconnectList;
using duckdb::disconnect::MatchResult;
using duckdb::disconnect::SplitUrl;
using duckdb::InvalidInputException;

//===--------------------------------------------------------------------===//
// Helpers
//===--------------------------------------------------------------------===//

//! Run `body` and turn any C++ exception into the matching Python one. An
//! unknown category or a malformed list is a ValueError, as in SQL it is an
//! InvalidInputException.
template <class FUN>
PyObject *Translated(FUN &&body) {
	try {
		return body();
	} catch (const InvalidInputException &e) {
		PyErr_SetString(PyExc_ValueError, e.what());
	} catch (const std::bad_alloc &) {
		PyErr_NoMemory();
	} catch (const std::exception &e) {
		PyErr_SetString(PyExc_RuntimeError, e.what());
	}
	return nullptr;
}

//! A borrowed UTF-8 view of a `str`. The buffer is cached on the object, so it
//! stays valid as long as the caller holds a reference to it.
struct UrlView {
	const char *data = nullptr;
	Py_ssize_t size = 0;
	bool is_null = false;
};

//! View `object` as a URL. `None` yields `is_null`, mirroring SQL NULL
//! propagation. Returns false with a Python error set on any other type.
bool ViewUrl(PyObject *object, UrlView &view) {
	if (object == Py_None) {
		view.is_null = true;
		return true;
	}
	if (!PyUnicode_Check(object)) {
		PyErr_Format(PyExc_TypeError, "expected str or None, not %s", Py_TYPE(object)->tp_name);
		return false;
	}
	view.data = PyUnicode_AsUTF8AndSize(object, &view.size);
	return view.data != nullptr;
}

//! Turn the `categories` argument into a filter mask. `None` means
//! `default_mask`, which is what the caller considers unfiltered; a str is the
//! comma-separated form SQL takes; any other sequence of str is joined with
//! commas first. Returns false with a Python error set.
bool ResolveMask(const DisconnectList &list, PyObject *categories, uint64_t default_mask, uint64_t &mask) {
	if (categories == nullptr || categories == Py_None) {
		mask = default_mask;
		return true;
	}
	duckdb::string filter;
	if (PyUnicode_Check(categories)) {
		Py_ssize_t size = 0;
		const auto *data = PyUnicode_AsUTF8AndSize(categories, &size);
		if (!data) {
			return false;
		}
		filter.assign(data, static_cast<size_t>(size));
	} else {
		PyObject *fast = PySequence_Fast(categories, "categories must be a str, a sequence of str, or None");
		if (!fast) {
			return false;
		}
		const auto count = PySequence_Fast_GET_SIZE(fast);
		for (Py_ssize_t i = 0; i < count; i++) {
			PyObject *item = PySequence_Fast_GET_ITEM(fast, i);
			Py_ssize_t size = 0;
			const auto *data = PyUnicode_Check(item) ? PyUnicode_AsUTF8AndSize(item, &size) : nullptr;
			if (!data) {
				if (!PyErr_Occurred()) {
					PyErr_Format(PyExc_TypeError, "category names must be str, not %s", Py_TYPE(item)->tp_name);
				}
				Py_DECREF(fast);
				return false;
			}
			if (i > 0) {
				filter += ',';
			}
			filter.append(data, static_cast<size_t>(size));
		}
		Py_DECREF(fast);
	}
	mask = list.ParseCategoryFilter(filter);
	return true;
}

PyObject *NewString(const duckdb::string &value) {
	return PyUnicode_FromStringAndSize(value.c_str(), static_cast<Py_ssize_t>(value.size()));
}

//! Every category the match is listed under, in list order.
PyObject *NewCategoryList(const DisconnectList &list, const MatchResult &match) {
	PyObject *result = PyList_New(0);
	if (!result) {
		return nullptr;
	}
	for (duckdb::idx_t category = 0; category < list.Categories().size(); category++) {
		if ((match.category_mask & (uint64_t(1) << category)) == 0) {
			continue;
		}
		PyObject *name = NewString(list.CategoryName(duckdb::UnsafeNumericCast<uint16_t>(category)));
		if (!name || PyList_Append(result, name) != 0) {
			Py_XDECREF(name);
			Py_DECREF(result);
			return nullptr;
		}
		Py_DECREF(name);
	}
	return result;
}

//! Shared body of the scalar functions that describe a match; `extract` turns a
//! match into the value to return. An unlisted URL and NULL both give None.
template <class EXTRACT>
PyObject *MatchAttribute(PyObject *args, PyObject *kwargs, const char *format, EXTRACT extract) {
	static const char *keywords[] = {"url", nullptr};
	PyObject *url_object = nullptr;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, const_cast<char **>(keywords), &url_object)) {
		return nullptr;
	}
	UrlView url;
	if (!ViewUrl(url_object, url)) {
		return nullptr;
	}
	if (url.is_null) {
		Py_RETURN_NONE;
	}
	return Translated([&]() -> PyObject * {
		auto list = DisconnectList::Current();
		MatchResult match;
		if (!list->Match(url.data, static_cast<duckdb::idx_t>(url.size), ALL_CATEGORIES, match)) {
			Py_RETURN_NONE;
		}
		return extract(*list, match);
	});
}

//===--------------------------------------------------------------------===//
// Scalar functions
//===--------------------------------------------------------------------===//

PyObject *PyIsTracker(PyObject *self, PyObject *args, PyObject *kwargs) {
	static const char *keywords[] = {"url", "categories", nullptr};
	PyObject *url_object = nullptr;
	PyObject *categories = nullptr;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O:is_tracker", const_cast<char **>(keywords), &url_object,
	                                 &categories)) {
		return nullptr;
	}
	UrlView url;
	if (!ViewUrl(url_object, url)) {
		return nullptr;
	}
	return Translated([&]() -> PyObject * {
		auto list = DisconnectList::Current();
		uint64_t mask = list->DefaultCategoryMask();
		if (!ResolveMask(*list, categories, list->DefaultCategoryMask(), mask)) {
			return nullptr;
		}
		// The category filter is validated even for a NULL url, so a typo is an
		// error rather than silently returning None.
		if (url.is_null) {
			Py_RETURN_NONE;
		}
		MatchResult match;
		if (list->Match(url.data, static_cast<duckdb::idx_t>(url.size), mask, match)) {
			Py_RETURN_TRUE;
		}
		Py_RETURN_FALSE;
	});
}

//! Classify a whole sequence in one call. Python call overhead dominates
//! per-URL classification, so this is what makes the module usable at crawl
//! scale; the matching itself runs with the GIL released.
PyObject *PyIsTrackerMany(PyObject *self, PyObject *args, PyObject *kwargs) {
	static const char *keywords[] = {"urls", "categories", nullptr};
	PyObject *urls_object = nullptr;
	PyObject *categories = nullptr;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O:is_tracker_many", const_cast<char **>(keywords), &urls_object,
	                                 &categories)) {
		return nullptr;
	}
	PyObject *urls = PySequence_Fast(urls_object, "urls must be an iterable of str");
	if (!urls) {
		return nullptr;
	}
	PyObject *result = Translated([&]() -> PyObject * {
		auto list = DisconnectList::Current();
		uint64_t mask = list->DefaultCategoryMask();
		if (!ResolveMask(*list, categories, list->DefaultCategoryMask(), mask)) {
			return nullptr;
		}
		const auto count = PySequence_Fast_GET_SIZE(urls);
		PyObject *output = PyList_New(count);
		if (!output) {
			return nullptr;
		}
		// Worked in chunks so the view buffers stay small whatever the input size.
		static constexpr Py_ssize_t CHUNK = 2048;
		std::vector<UrlView> views;
		std::vector<char> matched;
		views.reserve(CHUNK);
		matched.reserve(CHUNK);
		for (Py_ssize_t start = 0; start < count; start += CHUNK) {
			const auto chunk = count - start < CHUNK ? count - start : CHUNK;
			views.assign(static_cast<size_t>(chunk), UrlView());
			for (Py_ssize_t i = 0; i < chunk; i++) {
				if (!ViewUrl(PySequence_Fast_GET_ITEM(urls, start + i), views[static_cast<size_t>(i)])) {
					Py_DECREF(output);
					return nullptr;
				}
			}
			matched.assign(static_cast<size_t>(chunk), 0);
			// The views point into the input strings, which the sequence keeps
			// alive, so no Python object is touched in here.
			Py_BEGIN_ALLOW_THREADS;
			for (size_t i = 0; i < views.size(); i++) {
				auto &view = views[i];
				if (view.is_null) {
					matched[i] = 2;
					continue;
				}
				MatchResult match;
				matched[i] = list->Match(view.data, static_cast<duckdb::idx_t>(view.size), mask, match) ? 1 : 0;
			}
			Py_END_ALLOW_THREADS;
			for (Py_ssize_t i = 0; i < chunk; i++) {
				const auto flag = matched[static_cast<size_t>(i)];
				PyObject *value = flag == 2 ? Py_None : (flag == 1 ? Py_True : Py_False);
				Py_INCREF(value);
				PyList_SET_ITEM(output, start + i, value);
			}
		}
		return output;
	});
	Py_DECREF(urls);
	return result;
}

PyObject *PyTrackerCategory(PyObject *self, PyObject *args, PyObject *kwargs) {
	return MatchAttribute(args, kwargs, "O:tracker_category",
	                      [](const DisconnectList &list, const MatchResult &match) {
		                      return NewString(list.CategoryName(match.category));
	                      });
}

PyObject *PyTrackerCategories(PyObject *self, PyObject *args, PyObject *kwargs) {
	return MatchAttribute(args, kwargs, "O:tracker_categories",
	                      [](const DisconnectList &list, const MatchResult &match) {
		                      return NewCategoryList(list, match);
	                      });
}

PyObject *PyTrackerOwner(PyObject *self, PyObject *args, PyObject *kwargs) {
	return MatchAttribute(args, kwargs, "O:tracker_owner", [](const DisconnectList &list, const MatchResult &match) {
		return NewString(list.Org(match.organization).name);
	});
}

PyObject *PyTrackerOwnerUrl(PyObject *self, PyObject *args, PyObject *kwargs) {
	return MatchAttribute(args, kwargs, "O:tracker_owner_url", [](const DisconnectList &list, const MatchResult &match) {
		return NewString(list.Org(match.organization).url);
	});
}

PyObject *PyTrackerPattern(PyObject *self, PyObject *args, PyObject *kwargs) {
	return MatchAttribute(args, kwargs, "O:tracker_pattern", [](const DisconnectList &list, const MatchResult &match) {
		return NewString(*match.pattern);
	});
}

//! Where the matched entry sits in entries(), so a caller can join a
//! classification back onto the list without looking a pattern up again.
PyObject *PyTrackerIndex(PyObject *self, PyObject *args, PyObject *kwargs) {
	return MatchAttribute(args, kwargs, "O:tracker_index", [](const DisconnectList &list, const MatchResult &match) {
		return PyLong_FromUnsignedLong(static_cast<unsigned long>(match.entry_index));
	});
}

//! Everything known about one URL in a single lookup.
PyObject *PyMatch(PyObject *self, PyObject *args, PyObject *kwargs) {
	static const char *keywords[] = {"url", "categories", nullptr};
	PyObject *url_object = nullptr;
	PyObject *categories = nullptr;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O:match", const_cast<char **>(keywords), &url_object,
	                                 &categories)) {
		return nullptr;
	}
	UrlView url;
	if (!ViewUrl(url_object, url)) {
		return nullptr;
	}
	return Translated([&]() -> PyObject * {
		auto list = DisconnectList::Current();
		uint64_t mask = ALL_CATEGORIES;
		if (!ResolveMask(*list, categories, ALL_CATEGORIES, mask)) {
			return nullptr;
		}
		MatchResult match;
		if (url.is_null || !list->Match(url.data, static_cast<duckdb::idx_t>(url.size), mask, match)) {
			Py_RETURN_NONE;
		}
		auto &org = list->Org(match.organization);
		return Py_BuildValue("{s:N,s:N,s:N,s:N,s:N}", "pattern", NewString(*match.pattern), "category",
		                     NewString(list->CategoryName(match.category)), "categories",
		                     NewCategoryList(*list, match), "owner", NewString(org.name), "owner_url",
		                     NewString(org.url));
	});
}

//! The host of a URL, using the same parsing the matcher uses.
PyObject *PyUrlHost(PyObject *self, PyObject *args, PyObject *kwargs) {
	static const char *keywords[] = {"url", nullptr};
	PyObject *url_object = nullptr;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:url_host", const_cast<char **>(keywords), &url_object)) {
		return nullptr;
	}
	UrlView url;
	if (!ViewUrl(url_object, url)) {
		return nullptr;
	}
	if (url.is_null) {
		Py_RETURN_NONE;
	}
	const auto parts = SplitUrl(url.data, static_cast<duckdb::idx_t>(url.size));
	if (parts.host_len == 0) {
		Py_RETURN_NONE;
	}
	return PyUnicode_FromStringAndSize(parts.host, static_cast<Py_ssize_t>(parts.host_len));
}

//===--------------------------------------------------------------------===//
// List management
//===--------------------------------------------------------------------===//

PyObject *NewListInfo(const DisconnectList &list) {
	return Py_BuildValue("{s:N,s:K,s:K,s:K,s:K}", "source", NewString(list.Source()), "hosts",
	                     static_cast<unsigned long long>(list.HostCount()), "rules",
	                     static_cast<unsigned long long>(list.Triples().size()), "categories",
	                     static_cast<unsigned long long>(list.Categories().size()), "organizations",
	                     static_cast<unsigned long long>(list.OrganizationCount()));
}

PyObject *PyListInfo(PyObject *self, PyObject *) {
	return Translated([]() { return NewListInfo(*DisconnectList::Current()); });
}

PyObject *PyCategories(PyObject *self, PyObject *) {
	return Translated([]() -> PyObject * {
		auto list = DisconnectList::Current();
		PyObject *result = PyList_New(static_cast<Py_ssize_t>(list->Categories().size()));
		if (!result) {
			return nullptr;
		}
		for (duckdb::idx_t i = 0; i < list->Categories().size(); i++) {
			PyObject *name = NewString(list->Categories()[i]);
			if (!name) {
				Py_DECREF(result);
				return nullptr;
			}
			PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), name);
		}
		return result;
	});
}

//! The whole list, one dict per source line.
PyObject *PyEntries(PyObject *self, PyObject *) {
	return Translated([]() -> PyObject * {
		auto list = DisconnectList::Current();
		auto &triples = list->Triples();
		PyObject *result = PyList_New(static_cast<Py_ssize_t>(triples.size()));
		if (!result) {
			return nullptr;
		}
		for (duckdb::idx_t i = 0; i < triples.size(); i++) {
			auto &triple = triples[i];
			auto &org = list->Org(triple.organization);
			PyObject *entry =
			    Py_BuildValue("{s:N,s:N,s:N,s:N}", "pattern", NewString(triple.pattern), "category",
			                  NewString(list->CategoryName(triple.category)), "organization", NewString(org.name),
			                  "organization_url", NewString(org.url));
			if (!entry) {
				Py_DECREF(result);
				return nullptr;
			}
			PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), entry);
		}
		return result;
	});
}

//! Read a whole services JSON. Unlike disconnect_load() in SQL this goes
//! straight to the local file system: there is no DuckDB here to provide
//! httpfs and friends.
bool ReadFile(const char *path, duckdb::string &contents) {
	std::ifstream stream(path, std::ios::binary);
	if (!stream) {
		PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
		return false;
	}
	stream.seekg(0, std::ios::end);
	const auto file_size = static_cast<uint64_t>(stream.tellg());
	stream.seekg(0, std::ios::beg);
	static constexpr uint64_t MAX_LIST_SIZE = 256ULL * 1024 * 1024;
	if (file_size > MAX_LIST_SIZE) {
		PyErr_Format(PyExc_ValueError, "\"%s\" is %llu bytes, larger than the %llu byte limit for a services list",
		             path, static_cast<unsigned long long>(file_size), static_cast<unsigned long long>(MAX_LIST_SIZE));
		return false;
	}
	contents.resize(static_cast<size_t>(file_size));
	if (file_size > 0 && !stream.read(&contents[0], static_cast<std::streamsize>(file_size))) {
		PyErr_Format(PyExc_OSError, "Failed to read \"%s\": unexpected end of file", path);
		return false;
	}
	return true;
}

//! Install a list read from a services JSON, process-wide.
PyObject *PyLoad(PyObject *self, PyObject *args, PyObject *kwargs) {
	static const char *keywords[] = {"path", nullptr};
	PyObject *path_bytes = nullptr;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&:load", const_cast<char **>(keywords), PyUnicode_FSConverter,
	                                 &path_bytes)) {
		return nullptr;
	}
	PyObject *result = Translated([&]() -> PyObject * {
		const char *path = PyBytes_AS_STRING(path_bytes);
		duckdb::string contents;
		if (!ReadFile(path, contents)) {
			return nullptr;
		}
		auto list = DisconnectList::FromJson(contents.c_str(), contents.size(), path);
		DisconnectList::Replace(list);
		return NewListInfo(*list);
	});
	Py_DECREF(path_bytes);
	return result;
}

//! Go back to the list embedded at build time.
PyObject *PyReset(PyObject *self, PyObject *) {
	return Translated([]() { return NewListInfo(*DisconnectList::ResetToEmbedded()); });
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//

#define KWARG_METHOD(function) reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(function))

PyMethodDef METHODS[] = {
    {"is_tracker", KWARG_METHOD(PyIsTracker), METH_VARARGS | METH_KEYWORDS,
     "is_tracker(url, categories=None) -> bool | None\n\n"
     "Is the URL on the Disconnect list? `categories` restricts the match to a\n"
     "comma-separated string or a sequence of category names; the default of None\n"
     "matches any category except Content, which lists tracking companies' own\n"
     "first-party properties. Name Content explicitly to include it, or use\n"
     "tracker_category() to see it regardless. None in gives None out."},
    {"is_tracker_many", KWARG_METHOD(PyIsTrackerMany), METH_VARARGS | METH_KEYWORDS,
     "is_tracker_many(urls, categories=None) -> list[bool | None]\n\n"
     "is_tracker() over a whole sequence, without per-URL Python call overhead.\n"
     "The category filter is resolved once and the matching runs with the GIL\n"
     "released."},
    {"tracker_category", KWARG_METHOD(PyTrackerCategory), METH_VARARGS | METH_KEYWORDS,
     "tracker_category(url) -> str | None\n\n"
     "The primary category of the match, None if the URL is not listed. Hosts\n"
     "listed under several categories report the most telling one; use\n"
     "tracker_categories() for all of them."},
    {"tracker_categories", KWARG_METHOD(PyTrackerCategories), METH_VARARGS | METH_KEYWORDS,
     "tracker_categories(url) -> list[str] | None\n\n"
     "Every category the match is listed under, None if the URL is not listed."},
    {"tracker_owner", KWARG_METHOD(PyTrackerOwner), METH_VARARGS | METH_KEYWORDS,
     "tracker_owner(url) -> str | None\n\n"
     "The organization that owns the matched domain, None if not listed."},
    {"tracker_owner_url", KWARG_METHOD(PyTrackerOwnerUrl), METH_VARARGS | METH_KEYWORDS,
     "tracker_owner_url(url) -> str | None\n\n"
     "That organization's home page, None if the URL is not listed."},
    {"tracker_pattern", KWARG_METHOD(PyTrackerPattern), METH_VARARGS | METH_KEYWORDS,
     "tracker_pattern(url) -> str | None\n\n"
     "The list entry that matched, e.g. 'doubleclick.net', None if not listed."},
    {"tracker_index", KWARG_METHOD(PyTrackerIndex), METH_VARARGS | METH_KEYWORDS,
     "tracker_index(url) -> int | None\n\n"
     "The position in entries() of the list entry that matched, None if the URL\n"
     "is not listed. That entry is the one the other tracker_*() functions\n"
     "describe: a host listed under several categories has one entry per\n"
     "category, and this is the primary one."},
    {"match", KWARG_METHOD(PyMatch), METH_VARARGS | METH_KEYWORDS,
     "match(url, categories=None) -> dict | None\n\n"
     "Everything known about the URL in one lookup: pattern, category,\n"
     "categories, owner and owner_url. None if the URL is not listed."},
    {"url_host", KWARG_METHOD(PyUrlHost), METH_VARARGS | METH_KEYWORDS,
     "url_host(url) -> str | None\n\n"
     "The host of a URL, parsed exactly as the matcher parses it."},
    {"entries", PyEntries, METH_NOARGS,
     "entries() -> list[dict]\n\n"
     "The whole list: pattern, category, organization, organization_url."},
    {"categories", PyCategories, METH_NOARGS,
     "categories() -> list[str]\n\nThe category names of the loaded list, in list order."},
    {"list_info", PyListInfo, METH_NOARGS,
     "list_info() -> dict\n\nWhich list is loaded and how large it is."},
    {"load", KWARG_METHOD(PyLoad), METH_VARARGS | METH_KEYWORDS,
     "load(path) -> dict\n\n"
     "Replace the list from a Disconnect services JSON and return list_info().\n"
     "The loaded list is process-wide: it applies until reset()."},
    {"reset", PyReset, METH_NOARGS,
     "reset() -> dict\n\nGo back to the list embedded at build time, returning list_info()."},
    {nullptr, nullptr, 0, nullptr}};

PyDoc_STRVAR(MODULE_DOC, "Tracker classification against the Disconnect services list.\n"
                         "\n"
                         "The list Firefox uses for Enhanced Tracking Protection, compiled into this\n"
                         "module: nothing is downloaded or joined at query time.\n"
                         "\n"
                         "    >>> import disconnect\n"
                         "    >>> disconnect.is_tracker('https://www.google-analytics.com/collect?v=1')\n"
                         "    True\n"
                         "    >>> disconnect.tracker_owner('connect.facebook.net')\n"
                         "    'Meta'\n"
                         "\n"
                         "Input may be a full URL, a scheme-relative URL or a bare hostname, and parent\n"
                         "domains count: 'stats.g.doubleclick.net' matches the entry 'doubleclick.net'.\n"
                         "The same matching is available in SQL through the DuckDB extension built from\n"
                         "these sources.");

PyModuleDef MODULE = {PyModuleDef_HEAD_INIT, "disconnect", MODULE_DOC, -1, METHODS, nullptr, nullptr, nullptr, nullptr};

} // namespace

PyMODINIT_FUNC PyInit_disconnect(void) {
	PyObject *module = PyModule_Create(&MODULE);
	if (!module) {
		return nullptr;
	}
	if (PyModule_AddStringConstant(module, "__version__", DISCONNECT_VERSION) != 0) {
		Py_DECREF(module);
		return nullptr;
	}
	return module;
}
