cmake_minimum_required(VERSION 3.6)
project(Neural_Network)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set(CMAKE_CXX_STANDARD 20)
	set(CMAKE_CXX_STANDARD_REQUIRED True)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3 -mavx -ffast-math -flto -funroll-loops -fno-exceptions")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_options(
        -O3
        -march=native
        -ffast-math
        -flto
        -funroll-loops
        -fno-exceptions
    )
    set(CMAKE_CXX_STANDARD 20)
endif()

include_directories(include)
file(GLOB SOURCES "src/*cpp")

add_library(Neural_Network MODULE ${SOURCES})

include(FetchContent)

FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

target_link_libraries(Neural_Network PRIVATE nlohmann_json::nlohmann_json)







