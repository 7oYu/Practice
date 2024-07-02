cmake_minimum_required(VERSION 3.4)

project(quick_demo)

aux_source_directory(./src sources)
find_package(OpenCV REQUIRED)	
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
