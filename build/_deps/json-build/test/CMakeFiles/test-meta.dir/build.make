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
include _deps/json-build/test/CMakeFiles/test-meta.dir/depend.make

# Include the progress variables for this target.
include _deps/json-build/test/CMakeFiles/test-meta.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/json-build/test/CMakeFiles/test-meta.dir/flags.make

_deps/json-build/test/CMakeFiles/test-meta.dir/src/unit-meta.cpp.o: _deps/json-build/test/CMakeFiles/test-meta.dir/flags.make
_deps/json-build/test/CMakeFiles/test-meta.dir/src/unit-meta.cpp.o: _deps/json-src/test/src/unit-meta.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/christopher/Desktop/CAINN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/json-build/test/CMakeFiles/test-meta.dir/src/unit-meta.cpp.o"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test-meta.dir/src/unit-meta.cpp.o -c /home/christopher/Desktop/CAINN/build/_deps/json-src/test/src/unit-meta.cpp

_deps/json-build/test/CMakeFiles/test-meta.dir/src/unit-meta.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-meta.dir/src/unit-meta.cpp.i"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/christopher/Desktop/CAINN/build/_deps/json-src/test/src/unit-meta.cpp > CMakeFiles/test-meta.dir/src/unit-meta.cpp.i

_deps/json-build/test/CMakeFiles/test-meta.dir/src/unit-meta.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-meta.dir/src/unit-meta.cpp.s"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/christopher/Desktop/CAINN/build/_deps/json-src/test/src/unit-meta.cpp -o CMakeFiles/test-meta.dir/src/unit-meta.cpp.s

# Object files for target test-meta
test__meta_OBJECTS = \
"CMakeFiles/test-meta.dir/src/unit-meta.cpp.o"

# External object files for target test-meta
test__meta_EXTERNAL_OBJECTS = \
"/home/christopher/Desktop/CAINN/build/_deps/json-build/test/CMakeFiles/doctest_main.dir/src/unit.cpp.o"

_deps/json-build/test/test-meta: _deps/json-build/test/CMakeFiles/test-meta.dir/src/unit-meta.cpp.o
_deps/json-build/test/test-meta: _deps/json-build/test/CMakeFiles/doctest_main.dir/src/unit.cpp.o
_deps/json-build/test/test-meta: _deps/json-build/test/CMakeFiles/test-meta.dir/build.make
_deps/json-build/test/test-meta: _deps/json-build/test/CMakeFiles/test-meta.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/christopher/Desktop/CAINN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test-meta"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-meta.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/json-build/test/CMakeFiles/test-meta.dir/build: _deps/json-build/test/test-meta

.PHONY : _deps/json-build/test/CMakeFiles/test-meta.dir/build

_deps/json-build/test/CMakeFiles/test-meta.dir/clean:
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && $(CMAKE_COMMAND) -P CMakeFiles/test-meta.dir/cmake_clean.cmake
.PHONY : _deps/json-build/test/CMakeFiles/test-meta.dir/clean

_deps/json-build/test/CMakeFiles/test-meta.dir/depend:
	cd /home/christopher/Desktop/CAINN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/christopher/Desktop/CAINN /home/christopher/Desktop/CAINN/build/_deps/json-src/test /home/christopher/Desktop/CAINN/build /home/christopher/Desktop/CAINN/build/_deps/json-build/test /home/christopher/Desktop/CAINN/build/_deps/json-build/test/CMakeFiles/test-meta.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/json-build/test/CMakeFiles/test-meta.dir/depend

