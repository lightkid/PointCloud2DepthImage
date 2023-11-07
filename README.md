# PointCloud2DepthImage

平台：ROS noetic
接收sensor_msg::PointCloud2类型message，话题名为/scan
转化为cv::Mat深度图，深度图的数据类型为CV_32FC1

run：

    rosrun depth_converter transfer

显示时候灰度比较浅