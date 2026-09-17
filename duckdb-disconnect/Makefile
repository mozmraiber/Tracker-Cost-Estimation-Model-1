# Build the `disconnect` DuckDB extension against the Homebrew-installed DuckDB.
#
#   make            build build/disconnect.duckdb_extension
#   make module     build the standalone Python module into build/python
#   make test       build both, then run every test suite
#   make regen-list regenerate the embedded list from the source JSON
#   make clean      remove build artifacts

EXTENSION_NAME := disconnect
EXTENSION_VERSION := 0.1.0

# Override DUCKDB_PREFIX to build against a different DuckDB installation.
DUCKDB_PREFIX ?= $(shell brew --prefix duckdb 2>/dev/null)
DUCKDB := $(DUCKDB_PREFIX)/bin/duckdb
DUCKDB_INCLUDE := $(DUCKDB_PREFIX)/include

DUCKDB_VERSION := $(shell $(DUCKDB) -noheader -list -c "select library_version from pragma_version()")
DUCKDB_PLATFORM := $(shell $(DUCKDB) -noheader -list -c "select * from pragma_platform()")

PYTHON ?= python3
# The Python test suite needs the duckdb module; prefer the project virtualenv.
TEST_PYTHON ?= $(firstword $(wildcard ../.venv/bin/python) python3)
LIST_JSON ?= ../data/external/disconnect_services.json

BUILD_DIR := build
# The list model, matcher, URL parser and embedded data are shared between the
# DuckDB extension and the Python module; only the calling layer differs.
CORE_SOURCES := src/disconnect_list.cpp src/mini_json.cpp src/url_host.cpp src/generated/disconnect_data.cpp
SOURCES := src/disconnect_extension.cpp $(CORE_SOURCES)
OBJECTS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

CXXFLAGS ?= -O2
# -MMD -MP writes a .d per object so that a header change rebuilds every object
# that included it; without it a stale object keeps an outdated struct layout.
CXXFLAGS += -std=c++17 -fPIC -fvisibility=hidden -Wall -MMD -MP -I src/include
EXTENSION_CXXFLAGS := $(CXXFLAGS) -I $(DUCKDB_INCLUDE)

# The extension resolves DuckDB symbols from the process that loads it; it must
# not link its own copy of libduckdb.
ifeq ($(shell uname -s),Darwin)
	LDFLAGS += -dynamiclib -undefined dynamic_lookup
	MODULE_LDFLAGS := -bundle -undefined dynamic_lookup
else
	LDFLAGS += -shared
	MODULE_LDFLAGS := -shared
endif

RAW_LIBRARY := $(BUILD_DIR)/$(EXTENSION_NAME).lib
EXTENSION := $(BUILD_DIR)/$(EXTENSION_NAME).duckdb_extension

# The Python module is built against whichever interpreter TEST_PYTHON names, so
# that `make test` imports the module it just built.
MODULE_DIR := $(BUILD_DIR)/python
MODULE_SUFFIX := $(shell $(TEST_PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
MODULE_INCLUDE := $(shell $(TEST_PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
MODULE := $(MODULE_DIR)/$(EXTENSION_NAME)$(MODULE_SUFFIX)
MODULE_SOURCES := src/python_module.cpp $(CORE_SOURCES)
MODULE_OBJECTS := $(patsubst src/%.cpp,$(BUILD_DIR)/python-obj/%.o,$(MODULE_SOURCES))
# PyMODINIT_FUNC exports the init symbol itself, so -fvisibility=hidden holds.
MODULE_CXXFLAGS := $(CXXFLAGS) -DDISCONNECT_NO_DUCKDB -DDISCONNECT_VERSION='"$(EXTENSION_VERSION)"' \
                   -I $(MODULE_INCLUDE)

.PHONY: all clean test test-sql test-python test-module module regen-list check-duckdb check-python

all: $(EXTENSION) $(MODULE)

check-duckdb:
	@test -x "$(DUCKDB)" || { echo "DuckDB CLI not found at $(DUCKDB); install it with 'brew install duckdb' or set DUCKDB_PREFIX"; exit 1; }
	@test -d "$(DUCKDB_INCLUDE)/duckdb" || { echo "DuckDB headers not found in $(DUCKDB_INCLUDE)"; exit 1; }

check-python:
	@test -f "$(MODULE_INCLUDE)/Python.h" || { echo "Python headers not found in $(MODULE_INCLUDE); install the development headers for $(TEST_PYTHON)"; exit 1; }

-include $(OBJECTS:.o=.d) $(MODULE_OBJECTS:.o=.d)

$(BUILD_DIR)/%.o: src/%.cpp | check-duckdb
	@mkdir -p $(dir $@)
	$(CXX) $(EXTENSION_CXXFLAGS) -c $< -o $@

$(RAW_LIBRARY): $(OBJECTS)
	$(CXX) $(LDFLAGS) $(OBJECTS) -o $@

# DuckDB only loads binaries carrying a metadata footer that matches its own
# version, platform and ABI type.
$(EXTENSION): $(RAW_LIBRARY)
	$(PYTHON) scripts/append_extension_metadata.py \
		--library-file $(RAW_LIBRARY) \
		--extension-name $(EXTENSION_NAME) \
		--duckdb-version $(DUCKDB_VERSION) \
		--duckdb-platform $(DUCKDB_PLATFORM) \
		--extension-version $(EXTENSION_VERSION) \
		--abi-type CPP \
		--out-file $(EXTENSION)

#===-----------------------------------------------------------------------===#
# Python module
#===-----------------------------------------------------------------------===#

module: $(MODULE)

$(BUILD_DIR)/python-obj/%.o: src/%.cpp | check-python
	@mkdir -p $(dir $@)
	$(CXX) $(MODULE_CXXFLAGS) -c $< -o $@

# Python resolves its own symbols in the interpreter that imports the module, so
# this links neither libpython nor libduckdb.
$(MODULE): $(MODULE_OBJECTS)
	@mkdir -p $(MODULE_DIR)
	$(CXX) $(MODULE_LDFLAGS) $(MODULE_OBJECTS) -o $@

regen-list:
	$(PYTHON) scripts/generate_list_data.py $(LIST_JSON)

test: test-sql test-python test-module

test-sql: $(EXTENSION)
	$(DUCKDB) -unsigned -init /dev/null -bail -batch -c ".read test/disconnect.test.sql"

test-python: $(EXTENSION)
	$(TEST_PYTHON) test/test_disconnect.py $(EXTENSION)

test-module: $(MODULE)
	PYTHONPATH=$(MODULE_DIR) $(TEST_PYTHON) test/test_module.py

clean:
	rm -rf $(BUILD_DIR)
