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
include _deps/json-build/test/CMakeFiles/test-udt.dir/depend.make

# Include the progress variables for this target.
include _deps/json-build/test/CMakeFiles/test-udt.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/json-build/test/CMakeFiles/test-udt.dir/flags.make

_deps/json-build/test/CMakeFiles/test-udt.dir/src/unit-udt.cpp.o: _deps/json-build/test/CMakeFiles/test-udt.dir/flags.make
_deps/json-build/test/CMakeFiles/test-udt.dir/src/unit-udt.cpp.o: _deps/json-src/test/src/unit-udt.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/christopher/Desktop/CAINN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/json-build/test/CMakeFiles/test-udt.dir/src/unit-udt.cpp.o"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test-udt.dir/src/unit-udt.cpp.o -c /home/christopher/Desktop/CAINN/build/_deps/json-src/test/src/unit-udt.cpp

_deps/json-build/test/CMakeFiles/test-udt.dir/src/unit-udt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-udt.dir/src/unit-udt.cpp.i"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/christopher/Desktop/CAINN/build/_deps/json-src/test/src/unit-udt.cpp > CMakeFiles/test-udt.dir/src/unit-udt.cpp.i

_deps/json-build/test/CMakeFiles/test-udt.dir/src/unit-udt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-udt.dir/src/unit-udt.cpp.s"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/christopher/Desktop/CAINN/build/_deps/json-src/test/src/unit-udt.cpp -o CMakeFiles/test-udt.dir/src/unit-udt.cpp.s

# Object files for target test-udt
test__udt_OBJECTS = \
"CMakeFiles/test-udt.dir/src/unit-udt.cpp.o"

# External object files for target test-udt
test__udt_EXTERNAL_OBJECTS = \
"/home/christopher/Desktop/CAINN/build/_deps/json-build/test/CMakeFiles/doctest_main.dir/src/unit.cpp.o"

_deps/json-build/test/test-udt: _deps/json-build/test/CMakeFiles/test-udt.dir/src/unit-udt.cpp.o
_deps/json-build/test/test-udt: _deps/json-build/test/CMakeFiles/doctest_main.dir/src/unit.cpp.o
_deps/json-build/test/test-udt: _deps/json-build/test/CMakeFiles/test-udt.dir/build.make
_deps/json-build/test/test-udt: _deps/json-build/test/CMakeFiles/test-udt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/christopher/Desktop/CAINN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test-udt"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-udt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/json-build/test/CMakeFiles/test-udt.dir/build: _deps/json-build/test/test-udt

.PHONY : _deps/json-build/test/CMakeFiles/test-udt.dir/build

_deps/json-build/test/CMakeFiles/test-udt.dir/clean:
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && $(CMAKE_COMMAND) -P CMakeFiles/test-udt.dir/cmake_clean.cmake
.PHONY : _deps/json-build/test/CMakeFiles/test-udt.dir/clean

_deps/json-build/test/CMakeFiles/test-udt.dir/depend:
	cd /home/christopher/Desktop/CAINN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/christopher/Desktop/CAINN /home/christopher/Desktop/CAINN/build/_deps/json-src/test /home/christopher/Desktop/CAINN/build /home/christopher/Desktop/CAINN/build/_deps/json-build/test /home/christopher/Desktop/CAINN/build/_deps/json-build/test/CMakeFiles/test-udt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/json-build/test/CMakeFiles/test-udt.dir/depend

