cmake_minimum_required(VERSION 3.0)
project(Neural_Network)
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++17 -O3 -ffast-math -mavx -march=native")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++17 -O3 -mavx -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3 -mavx")


include_directories(include)
file(GLOB SOURCES "src/*cpp")

add_library(Neural_Network MODULE ${SOURCES})

include(FetchContent)

FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.7.3)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

target_link_libraries(Neural_Network PRIVATE nlohmann_json::nlohmann_json)







