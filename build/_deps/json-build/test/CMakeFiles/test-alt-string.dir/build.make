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
include _deps/json-build/test/CMakeFiles/test-alt-string.dir/depend.make

# Include the progress variables for this target.
include _deps/json-build/test/CMakeFiles/test-alt-string.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/json-build/test/CMakeFiles/test-alt-string.dir/flags.make

_deps/json-build/test/CMakeFiles/test-alt-string.dir/src/unit-alt-string.cpp.o: _deps/json-build/test/CMakeFiles/test-alt-string.dir/flags.make
_deps/json-build/test/CMakeFiles/test-alt-string.dir/src/unit-alt-string.cpp.o: _deps/json-src/test/src/unit-alt-string.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/christopher/Desktop/CAINN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/json-build/test/CMakeFiles/test-alt-string.dir/src/unit-alt-string.cpp.o"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test-alt-string.dir/src/unit-alt-string.cpp.o -c /home/christopher/Desktop/CAINN/build/_deps/json-src/test/src/unit-alt-string.cpp

_deps/json-build/test/CMakeFiles/test-alt-string.dir/src/unit-alt-string.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-alt-string.dir/src/unit-alt-string.cpp.i"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/christopher/Desktop/CAINN/build/_deps/json-src/test/src/unit-alt-string.cpp > CMakeFiles/test-alt-string.dir/src/unit-alt-string.cpp.i

_deps/json-build/test/CMakeFiles/test-alt-string.dir/src/unit-alt-string.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-alt-string.dir/src/unit-alt-string.cpp.s"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/christopher/Desktop/CAINN/build/_deps/json-src/test/src/unit-alt-string.cpp -o CMakeFiles/test-alt-string.dir/src/unit-alt-string.cpp.s

# Object files for target test-alt-string
test__alt__string_OBJECTS = \
"CMakeFiles/test-alt-string.dir/src/unit-alt-string.cpp.o"

# External object files for target test-alt-string
test__alt__string_EXTERNAL_OBJECTS = \
"/home/christopher/Desktop/CAINN/build/_deps/json-build/test/CMakeFiles/doctest_main.dir/src/unit.cpp.o"

_deps/json-build/test/test-alt-string: _deps/json-build/test/CMakeFiles/test-alt-string.dir/src/unit-alt-string.cpp.o
_deps/json-build/test/test-alt-string: _deps/json-build/test/CMakeFiles/doctest_main.dir/src/unit.cpp.o
_deps/json-build/test/test-alt-string: _deps/json-build/test/CMakeFiles/test-alt-string.dir/build.make
_deps/json-build/test/test-alt-string: _deps/json-build/test/CMakeFiles/test-alt-string.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/christopher/Desktop/CAINN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test-alt-string"
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-alt-string.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/json-build/test/CMakeFiles/test-alt-string.dir/build: _deps/json-build/test/test-alt-string

.PHONY : _deps/json-build/test/CMakeFiles/test-alt-string.dir/build

_deps/json-build/test/CMakeFiles/test-alt-string.dir/clean:
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build/test && $(CMAKE_COMMAND) -P CMakeFiles/test-alt-string.dir/cmake_clean.cmake
.PHONY : _deps/json-build/test/CMakeFiles/test-alt-string.dir/clean

_deps/json-build/test/CMakeFiles/test-alt-string.dir/depend:
	cd /home/christopher/Desktop/CAINN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/christopher/Desktop/CAINN /home/christopher/Desktop/CAINN/build/_deps/json-src/test /home/christopher/Desktop/CAINN/build /home/christopher/Desktop/CAINN/build/_deps/json-build/test /home/christopher/Desktop/CAINN/build/_deps/json-build/test/CMakeFiles/test-alt-string.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/json-build/test/CMakeFiles/test-alt-string.dir/depend
