cmake_minimum_required(VERSION 3.11)
project(wl-stream)

# Dependencies
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} include)
add_subdirectory(external/CLI11)

# Find Boost
find_package(Boost REQUIRED COMPONENTS system)


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
