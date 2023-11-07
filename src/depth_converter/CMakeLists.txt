cmake_minimum_required(VERSION 3.0.2)
project(depth_converter)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_ros
)

catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES PointCloudToDepthImage
#  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS} 
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
add_executable(transfer src/transfer.cpp)

target_link_libraries(transfer
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)
