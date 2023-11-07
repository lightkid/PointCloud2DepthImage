#ifndef TRANSFER_H_
#define TRANSFER_H_

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class PC2DITransfer {
public:
  PC2DITransfer(float hfov, float vfov, float cxTheta, float cyPhi)
      : hfov_(hfov), vfov_(vfov), cxTheta_(cxTheta), cyPhi_(cyPhi) {
    // hfov_inv_ = 1.0f / hfov_;
    // vfov_inv_ = 1.0f / vfov_;
  }

  /**
   * @brief 由点云生成伪深度图
   *
   * @param points 局部坐标系下雷达点
   * @param depth  深度图像,对应为距离,单位为m
   * @param hfov   雷达水平FOV
   * @param vfov   雷达垂直FOV
   * @param size   图像尺寸
   */
  cv::Mat getDepthIamge(const pcl::PointCloud<pcl::PointXYZ> &pointCloud,
                        const cv::Size &size) {
    /* 右手系
              ^x
        y     |
        <-----|---
              |
        水平角度theta 垂直角度phi
    */
    cv::Mat res(size, CV_32FC1, cv::Scalar(0));
    const int width = size.width;
    const int height = size.height;
    const int cx = static_cast<int>(cxTheta_ * width / hfov_);
    const int cy = static_cast<int>(cyPhi_ * height / vfov_);
    std::cout << "cx: " << cx << std::endl;
    double max_phi = -10000.0f, min_phi = 10000.0f;
    int max_x = -1000, min_x = 1000;
    for (const auto &p : pointCloud.points) {
      Eigen::Vector3d v;
      v << p.x, p.y, p.z;
      double vnorm = v.norm();
      if (vnorm < 0.001) {
        //如果是因为架子挡住了，也添加进去，靠raycast过程剔除
        continue;
      }
      v = v / vnorm;
      Eigen::Vector2d vxy = v.head(2);
      double vxynorm = vxy.norm();
      if (vxynorm < 0.001) { //头顶的也不要
        continue;            // TODO：应该加上
      }
      vxy = vxy / vxynorm;
      double theta = sign(vxy(1)) * acos(vxy(0));
      double phi = asin(v(2));

      double px = width * theta * 180 * M_1_PI / hfov_;
      double py = height * phi * 180 * M_1_PI / vfov_;
      // if (abs(px) > max_phi) {
      //   max_phi = abs(px);
      // }
      // if (abs(px) < min_phi) {
      //   min_phi = abs(px);
      // }
      int x = (int)px, y = (int)py;
      // theta最小是0.5四舍五入的时候，深度图中间会出现一个黑线
      // x = (px > 0) ? (int)(px + 0.5) : (int)(px - 0.5);
      // y = (py > 0) ? (int)(py + 0.5) : (int)(py - 0.5);
      // if (abs(x) > max_x) {
      //   max_x = abs(x);
      // }
      // if (abs(x) < min_x) {
      //   min_x = abs(x);
      // }
      x = cx - x;
      y = cy - y;
      x = (width > x) ? (x < 0 ? 0 : x) : width - 1;
      y = (height > y) ? (y < 0 ? 0 : y) : height - 1;

      res.at<float>(y, x) = vnorm * 255 / 3; // TODO 不要用at
    }
    // std::cout << "max x: " << max_x << " min x: " << min_x << std::endl;
    // std::cout << "max phi: " << std::fixed << std::setprecision(3) << max_phi
    //           << " min phi: " << std::fixed << std::setprecision(3) <<
    //           min_phi
    //           << std::endl;
    return res;

    // // cv::Mat res(size, CV_32FC1, 0.0f);
    // double max_depth = 0;
    // const int width = size.width;
    // const int height = size.height;
    // uint8_t tmp[height][width];
    // // const int cx = static_cast<int>(cxTheta_ * width * hfov_inv_);
    // // const int cy = static_cast<int>(cyPhi_ * height * vfov_inv_);
    // for (const auto &p : pointCloud.points) {
    //   Eigen::Vector3d v;
    //   v << p.x, p.y, p.z;
    //   double vnorm = v.norm();
    //   if (vnorm < 0.05) { // 5cm以内认为是盲区
    //     continue;
    //   }
    //   v(2) = v(2) / vnorm;
    //   Eigen::Vector2d vxy = v.head(2);
    //   double vxynorm = vxy.norm();
    //   if (vxynorm < 0.05) { //头顶的也不要
    //     continue;           // TODO：应该加上
    //   }
    //   vxy = vxy / vxynorm;
    //   double theta = (-sign(vxy(1)) * acos(vxy(0))) * M_1_PI * 180.0f;
    //   double phi = asin(v(2)) * M_1_PI * 180.0f;
    //   double px = width * hfov_inv_ * (theta + cxTheta_); // theta=0 ->cx
    //   double py = height * vfov_inv_ * (phi + cyPhi_);    // phi=0 ->cy
    //   int x, y;
    //   x = (px > 0) ? (int)(px + 0.5) : (int)(px - 0.5);
    //   y = (py > 0) ? (int)(py + 0.5) : (int)(py - 0.5);
    //   x = (width > x) ? (x < 0 ? 0 : x) : width - 1;
    //   y = (height > y) ? (y < 0 ? 0 : y) : height - 1;
    //   // ROS_WARN("(%f,%f)->(%d,%d),depth:%f", px, py, x, y, vnorm);
    //   tmp[y][x] = vnorm * 255 / 3;
    //   if (vnorm > max_depth) {
    //     max_depth = vnorm;
    //   }
    //   res.at<float>(y, x) = vnorm;
    //   // std::cout << "ok" << std::endl;
    //   // if ((x > width - 1 || x < 0) || (y > height - 1 || y < 0)) {

    //   // }
    // }
    // // std::cout << max_depth << std::endl;
    // // cv::Mat res(size, CV_8UC1, tmp);
    // // return res;
  }

private:
  float hfov_; //雷达水平FOV
  float vfov_; //雷达垂直FOV
  float hfov_inv_;
  float vfov_inv_;
  float cxTheta_; //雷达中心水平角度
  float cyPhi_;   //雷达中心垂直角度

  // pcl::PointCloud<pcl::PointXYZ> pointCloud_;

  inline double sign(double x) {
    return x < 0.0 ? -1.0 : (x > 0.0 ? 1.0 : 0.0);
  }
};

#endif