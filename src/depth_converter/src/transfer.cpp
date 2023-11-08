#include "depth_converter/transfer.h"
#include <pcl_ros/impl/transforms.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

ros::Publisher pub_depth;

void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg) {
  auto transfer = PC2DITransfer(360, 59, 180.0f, 7.0f); // h:[0,360] v:[-52,7]
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::fromROSMsg(*msg, cloud);
  // std::cout << "points num: " << cloud.points.size() << std::endl;

  // transform point cloud to depth image
  cv::Size size(360, 59);
  auto t1 = std::chrono::steady_clock::now();
  cv::Mat depth = transfer.getDepthImageCUDA(cloud, size);
  auto t2 = std::chrono::steady_clock::now();
  float dt = std::chrono::duration<float, std::milli>(t2 - t1).count();
  std::cout << "generate depth cost: " << dt << " ms" << std::endl;

  // depth image visualization
  // cv::imshow("depth", depth);
  // cv::waitKey(1);
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = ros::Time::now();
  out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  out_msg.image = depth.clone();
  pub_depth.publish(out_msg.toImageMsg());
}
int main(int argc, char **argv) {
  ros::init(argc, argv, "test");
  ros::NodeHandle nh;
  pub_depth = nh.advertise<sensor_msgs::Image>("depth", 1000);
  ros::Subscriber pcSub =
      nh.subscribe<sensor_msgs::PointCloud2>("/scan", 10, pointCloudCallback);
  ros::spin();
  return 0;
}