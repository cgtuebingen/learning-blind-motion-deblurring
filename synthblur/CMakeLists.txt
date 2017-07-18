cmake_minimum_required(VERSION 2.8)
project( convert )

if(NOT CMAKE_CXX_STANDARD) # don't override
    set(CMAKE_CXX_STANDARD 11)
endif()

find_package( OpenCV REQUIRED )
# find_package (glog)

include_directories(include)

add_executable( convert src/convert.cpp src/flow.cpp src/video.cpp src/meta.cpp src/blur.cpp)
target_link_libraries( convert ${OpenCV_LIBS} glog)
