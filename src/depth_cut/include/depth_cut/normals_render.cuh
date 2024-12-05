#ifndef DEPTH_RENDER_CUH
#define DEPTH_RENDER_CUH

#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "helper_math.h"

// using namespace std;
// using namespace Eigen;

struct Parameter {
  int point_number, kernel_half_size;
  float fx, fy, cx, cy, k_depth_scaling_factor;
  int width, height;
  float r[9];
  float t[3];
};

class NormalsRender {
 public:
  NormalsRender(float _fx, float _fy, float _cx, float _cy, int _width,
                int _height, int _kernel_half_size, float _k_depth_scaling_factor);
  // void set_pose(float *r, float *t);
  ~NormalsRender();
  void set_data(cv::Mat& depth_img);
  void render_pose(float *r, float *t);
  bool get_cloud(std::vector<Eigen::Vector3d>& cloud){
    cloud.clear();
    cloud.resize(parameter_ptr_->point_number);
    for(int i = 0; i < parameter_ptr_->point_number;++i){
      cloud[i] = Eigen::Vector3d(cloud_ptr_[i].x, cloud_ptr_[i].y, cloud_ptr_[i].z);
    }
    return true;
  }
  bool get_normals(std::vector<Eigen::Vector3d>& normals){
    normals.clear();
    normals.resize(parameter_ptr_->point_number);
    for(int i = 0; i < parameter_ptr_->point_number;++i){
      normals[i] = Eigen::Vector3d(normals_ptr_[i].x, normals_ptr_[i].y, normals_ptr_[i].z);
    }
    return true;
  }
  bool get_valid(std::vector<int>& valids){
    valids.clear();
    valids.resize(parameter_ptr_->point_number);
    for(int i = 0; i < parameter_ptr_->point_number;++i){
      valids[i] = idx_valid_ptr_[i];
    }
    return true;
  }
  bool get_depth(std::vector<uint16_t>& depthimg){
    depthimg.clear();
    depthimg.resize(parameter_ptr_->point_number);
    for(int i = 0; i < parameter_ptr_->point_number;++i){
      depthimg[i] = depth_ptr_[i];
    }
    return true;
  }
  Parameter* get_param(){
    return parameter_ptr_;
  }

 private:
  // int cloud_size;

  // data
  // float3 *host_cloud_ptr;
  uint16_t *depth_ptr_;
  float3 *cloud_ptr_;
  float3 *normals_ptr_;
  int *idx_valid_ptr_;
  bool has_ptr_;

  // camera
  // Parameter parameter;
  Parameter *parameter_ptr_;
};

#endif