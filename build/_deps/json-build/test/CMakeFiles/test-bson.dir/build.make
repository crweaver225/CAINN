# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/christopher/Desktop/CAINN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/christopher/Desktop/CAINN/build

# Include any dependencies generated for this target.
include _deps/json-build/test/CMakeFiles/test-bson.dir/depend.make

# Include the progress variables for this target.
include _deps/json-build/test/CMakeFiles/test-bson.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/json-build/test/CMakeFiles/test-bson.dir/flags.make

_deps/json-build/test/CMakeFiles/test-bson.dir/src/unit-bson.cpp.o: _deps/json-build/test/CMakeFiles/test-bson.dir/flags.make
_deps/json-build/test/CMakeFiles/test-bson.dir/src/unit-bson.cpp.o: _deps/json-src/test/src/unit-bson.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/christopher/Desktop/CAINN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/json-build/test/CMakeFiles/test-bson.dir/src/unit-bson.cpp.o"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test-bson.dir/src/unit-bson.cpp.o -c /home/christopher/Desktop/CAINN/build/_deps/json-src/test/src/unit-bson.cpp

_deps/json-build/test/CMakeFiles/test-bson.dir/src/unit-bson.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-bson.dir/src/unit-bson.cpp.i"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/christopher/Desktop/CAINN/build/_deps/json-src/test/src/unit-bson.cpp > CMakeFiles/test-bson.dir/src/unit-bson.cpp.i

_deps/json-build/test/CMakeFiles/test-bson.dir/src/unit-bson.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-bson.dir/src/unit-bson.cpp.s"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/christopher/Desktop/CAINN/build/_deps/json-src/test/src/unit-bson.cpp -o CMakeFiles/test-bson.dir/src/unit-bson.cpp.s

# Object files for target test-bson
test__bson_OBJECTS = \
"CMakeFiles/test-bson.dir/src/unit-bson.cpp.o"

# External object files for target test-bson
test__bson_EXTERNAL_OBJECTS = \
"/home/christopher/Desktop/CAINN/build/_deps/json-build/test/CMakeFiles/doctest_main.dir/src/unit.cpp.o"

_deps/json-build/test/test-bson: _deps/json-build/test/CMakeFiles/test-bson.dir/src/unit-bson.cpp.o
_deps/json-build/test/test-bson: _deps/json-build/test/CMakeFiles/doctest_main.dir/src/unit.cpp.o
_deps/json-build/test/test-bson: _deps/json-build/test/CMakeFiles/test-bson.dir/build.make
_deps/json-build/test/test-bson: _deps/json-build/test/CMakeFiles/test-bson.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/christopher/Desktop/CAINN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test-bson"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-bson.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/json-build/test/CMakeFiles/test-bson.dir/build: _deps/json-build/test/test-bson

.PHONY : _deps/json-build/test/CMakeFiles/test-bson.dir/build

_deps/json-build/test/CMakeFiles/test-bson.dir/clean:
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && $(CMAKE_COMMAND) -P CMakeFiles/test-bson.dir/cmake_clean.cmake
.PHONY : _deps/json-build/test/CMakeFiles/test-bson.dir/clean

_deps/json-build/test/CMakeFiles/test-bson.dir/depend:
	cd /home/christopher/Desktop/CAINN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/christopher/Desktop/CAINN /home/christopher/Desktop/CAINN/build/_deps/json-src/test /home/christopher/Desktop/CAINN/build /home/christopher/Desktop/CAINN/build/_deps/json-build/test /home/christopher/Desktop/CAINN/build/_deps/json-build/test/CMakeFiles/test-bson.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/json-build/test/CMakeFiles/test-bson.dir/depend

