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

# Utility rule file for NightlyBuild.

# Include the progress variables for this target.
include _deps/json-build/CMakeFiles/NightlyBuild.dir/progress.make

_deps/json-build/CMakeFiles/NightlyBuild:
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build && /usr/local/bin/ctest -D NightlyBuild

NightlyBuild: _deps/json-build/CMakeFiles/NightlyBuild
NightlyBuild: _deps/json-build/CMakeFiles/NightlyBuild.dir/build.make

.PHONY : NightlyBuild

# Rule to build all files generated by this target.
_deps/json-build/CMakeFiles/NightlyBuild.dir/build: NightlyBuild

.PHONY : _deps/json-build/CMakeFiles/NightlyBuild.dir/build

_deps/json-build/CMakeFiles/NightlyBuild.dir/clean:
	cd /home/christopher/Desktop/CAINN/build/_deps/json-build && $(CMAKE_COMMAND) -P CMakeFiles/NightlyBuild.dir/cmake_clean.cmake
.PHONY : _deps/json-build/CMakeFiles/NightlyBuild.dir/clean

_deps/json-build/CMakeFiles/NightlyBuild.dir/depend:
	cd /home/christopher/Desktop/CAINN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/christopher/Desktop/CAINN /home/christopher/Desktop/CAINN/build/_deps/json-src /home/christopher/Desktop/CAINN/build /home/christopher/Desktop/CAINN/build/_deps/json-build /home/christopher/Desktop/CAINN/build/_deps/json-build/CMakeFiles/NightlyBuild.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/json-build/CMakeFiles/NightlyBuild.dir/depend

