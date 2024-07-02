cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://mirror.ghproxy.com/https://github.com/jbeder/yaml-cpp.git
  GIT        yaml-cpp-0.7.0 
)

FetchContent_MakeAvailable(yaml-cpp)
