#ifndef TRANSFER_H_
#define TRANSFER_H_

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

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
   * @return 深度图像,对应为距离,单位为m
   *
   * @param pointCloud 局部坐标系下雷达点
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
      int x = (int)px, y = (int)py;
      // theta最小是0.5四舍五入的时候，深度图中间会出现一个黑线
      // x = (px > 0) ? (int)(px + 0.5) : (int)(px - 0.5);
      // y = (py > 0) ? (int)(py + 0.5) : (int)(py - 0.5);
      x = cx - x;
      y = cy - y;
      x = (width > x) ? (x < 0 ? 0 : x) : width - 1;
      y = (height > y) ? (y < 0 ? 0 : y) : height - 1;
      res.at<float>(y, x) = vnorm; // TODO： 不要用at
      // TODO:可能多个光线投影在同一个像素上，如果该像素非0，则应该取二者最小值
    }
    return res;
  }

private:
  float hfov_; //雷达水平FOV
  float vfov_; //雷达垂直FOV
  // float hfov_inv_;
  // float vfov_inv_;
  float cxTheta_; //雷达中心水平角度
  float cyPhi_;   //雷达中心垂直角度
  inline double sign(double x) {
    return x < 0.0 ? -1.0 : (x > 0.0 ? 1.0 : 0.0);
  }
};

#endif