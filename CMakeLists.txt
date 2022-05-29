cmake_minimum_required(VERSION 3.5)
project(allygator)

set(CMAKE_CXX_STANDARD 17)

find_package(pinocchio 2.6.3 REQUIRED)
find_package(example-robot-data 3.7.0 REQUIRED)

include_directories(include)

add_executable(${PROJECT_NAME} src/allygator.cpp src/ddp.cpp src/shooting.cpp)
target_link_libraries(${PROJECT_NAME} pinocchio::pinocchio example-robot-data::example-robot-data)
