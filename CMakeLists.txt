cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(wl-stream)
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")


# Dependencies
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} include)
add_subdirectory(external/CLI11)

# Find Boost
find_package(Boost REQUIRED COMPONENTS system)
list(APPEND CMAKE_PREFIX_PATH "libtorch")
find_package(Torch REQUIRED)


# Check if Boost.Asio is available
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    message(STATUS "Boost found: ${Boost_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "Boost not found")
endif ()

# Libraries
add_subdirectory(lib)


# Main
add_executable( wl-stream main.cpp )
target_link_libraries(wl-stream io_lib CLI11::CLI11)

# Install
install(TARGETS wl-stream
        RUNTIME DESTINATION bin)
