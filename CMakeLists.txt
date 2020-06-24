cmake_minimum_required(VERSION 2.6)
project(Neural_Network)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3")


include_directories(include)
file(GLOB SOURCES "src/*cpp")

add_library(Neural_Network SHARED ${SOURCES})







