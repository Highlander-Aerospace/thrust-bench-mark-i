cmake_minimum_required(VERSION 3.17)
project(src)

set(CMAKE_CXX_STANDARD 14)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(include)

add_library(src core.cpp include/arduino-serial-lib.c)
target_link_libraries(src ${PYTHON_LIBRARIES})