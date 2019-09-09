# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.14)

project (OpenCV_Tasks)

find_package(OpenCV          REQUIRED)
find_package(Cpp-Taskflow    REQUIRED)
find_package(GTest           REQUIRED)
find_package(fmt             REQUIRED)
find_package(spdlog          REQUIRED)
find_package(range-v3        REQUIRED)

add_executable(main)

target_sources(main PRIVATE main.cpp)

target_include_directories(main
                           PRIVATE
						   ${OpenCV_INLCUDE_DIRS})

target_link_libraries(main
                     PRIVATE
					 ${OpenCV_LIBS}
					 Cpp-Taskflow::Cpp-Taskflow
					 fmt::fmt
					 spdlog::spdlog
					 range-v3)

target_compile_features(main PRIVATE cxx_std_17)