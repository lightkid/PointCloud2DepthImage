# PointCloud2DepthImage

平台：ROS noetic
GPU：RTX 3060 (8.6)
CUDA: 12.0

输入：点云sensor_msg::PointCloud2，话题名为/scan

转化为cv::Mat深度图，深度图的数据类型为CV_32FC1

输出：深度图sensor_msg::Image，话题名为/depth

run：

    rosrun depth_converter transfer

v1.0为cpu版本

v2.0为gpu版本

参考了FUEL中的仿真部分：https://github.com/HKUST-Aerial-Robotics/FUEL.git
