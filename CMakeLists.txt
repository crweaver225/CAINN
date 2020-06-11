cmake_minimum_required(VERSION 2.6)
project(CAINN)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++17")


include_directories(include)
file(GLOB SOURCES "src/*cpp")
add_executable(CAINN ${SOURCES})

add_library(Neural_Network SHARED ${SOURCES})

