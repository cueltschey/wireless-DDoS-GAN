
file(GLOB IO_LIB_SRC *.cpp)
add_library(io_lib ${IO_LIB_SRC})
target_link_libraries(io_lib Boost::boost Boost::system ${OpenCV_LIBS} "${TORCH_LIBRARIES}")
