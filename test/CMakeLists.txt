cmake_minimum_required(VERSION 3.14)
project(allygator_tests)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

FetchContent_MakeAvailable(googletest)

enable_testing()

include_directories("../include")

add_executable(test_numeric_diff test_numeric_diff.cpp ../src/numeric_diff.cpp)
target_link_libraries(test_numeric_diff gtest_main Eigen3::Eigen)

add_executable(test_problem test_problem.cpp ../src/numeric_diff.cpp)
target_link_libraries(test_problem gtest_main Eigen3::Eigen)

add_executable(test_solve test_solve.cpp ../src/ddp.cpp ../src/numeric_diff.cpp)
target_link_libraries(test_solve gtest_main Eigen3::Eigen)

include(GoogleTest)

gtest_discover_tests(test_numeric_diff)
