# CMake generated Testfile for 
# Source directory: /home/christopher/Desktop/CAINN/build/_deps/json-src/test/cmake_import_minver
# Build directory: /home/christopher/Desktop/CAINN/build/_deps/json-build/test/cmake_import_minver
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(cmake_import_minver_configure "/usr/local/bin/cmake" "-G" "Unix Makefiles" "-A" "" "-Dnlohmann_json_DIR=/home/christopher/Desktop/CAINN/build/_deps/json-build" "/home/christopher/Desktop/CAINN/build/_deps/json-src/test/cmake_import_minver/project")
set_tests_properties(cmake_import_minver_configure PROPERTIES  FIXTURES_SETUP "cmake_import_minver")
add_test(cmake_import_minver_build "/usr/local/bin/cmake" "--build" ".")
set_tests_properties(cmake_import_minver_build PROPERTIES  FIXTURES_REQUIRED "cmake_import_minver")
