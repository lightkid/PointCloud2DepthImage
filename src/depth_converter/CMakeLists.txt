cmake_minimum_required(VERSION 3.0.2)
project(depth_converter)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

# CUDA
set(ENABLE_CUDA true)
# set(ENABLE_CUDA false)
if(ENABLE_CUDA)
  find_package(CUDA REQUIRED)
  SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math)
  set(CUDA_NVCC_FLAGS 
  #       -gencode arch=compute_20,code=sm_20;
  #       -gencode arch=compute_20,code=sm_21;
  #       -gencode arch=compute_30,code=sm_30;
  #       -gencode arch=compute_35,code=sm_35;
  #       -gencode arch=compute_50,code=sm_50;
  #       -gencode arch=compute_52,code=sm_52; 
  #       -gencode arch=compute_60,code=sm_60;
      # -gencode arch=compute_61,code=sm_61
      -gencode arch=compute_86,code=sm_86
  ) 

  SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

  find_package(OpenCV REQUIRED)
  find_package(Eigen3 REQUIRED)
  find_package(PCL 1.7 REQUIRED)
  find_package(catkin REQUIRED COMPONENTS
    roscpp
    pcl_ros
    cv_bridge
  )

  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES depth_render_cuda
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
  CUDA_ADD_LIBRARY( depth_render_cuda
      src/depth_render.cu
  )
  add_executable(transfer src/transfer.cpp)
  target_link_libraries( transfer
    depth_render_cuda
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
  )
  # ==================================================================================================
else(ENABLE_CUDA)
  find_package(OpenCV REQUIRED)
  find_package(Eigen3 REQUIRED)
  find_package(PCL 1.7 REQUIRED)
  find_package(catkin REQUIRED COMPONENTS
    roscpp
    pcl_ros
    cv_bridge
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
endif(ENABLE_CUDA)