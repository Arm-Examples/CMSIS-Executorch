# Host unit test of the firmware's pure C++ modules against the generated reference vectors.
#   make -C firmware/tests test
# Needs tests/generated/preprocess_vectors.h (host/make_vectors.py).

ifeq ($(origin CXX),default)
CXX := clang++
endif
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off
BUILD := build
SRC := ../src
SOURCES := preprocess_test.cpp $(SRC)/preprocess.cpp $(SRC)/uart_rx.cpp $(SRC)/motion_gate.cpp $(SRC)/json_out.cpp
HEADERS := $(wildcard $(SRC)/*.h) generated/preprocess_vectors.h

.PHONY: test clean FORCE

# Always rebuilt (a few seconds): the generated header can be rewritten within the same
# second as a previous build, which mtime-based make would miss.

test: $(BUILD)/preprocess_test
	./$(BUILD)/preprocess_test

$(BUILD)/preprocess_test: $(SOURCES) $(HEADERS) Makefile FORCE
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -I$(SRC) -Igenerated -o $@ $(SOURCES)

FORCE:

clean:
	rm -f $(BUILD)/preprocess_test
