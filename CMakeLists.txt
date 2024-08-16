cmake_minimum_required(VERSION 3.0)
project(Neural_Network)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++2a -O3")
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++2a -O3")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++2a -O3  -mavx -march=native")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3 -mavx")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")


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







