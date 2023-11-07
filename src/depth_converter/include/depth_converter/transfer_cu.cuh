#ifndef TRANSFER_CU_CUH_
#define TRANSFER_CU_CUH_

#include <cuda_runtime.h>

#include <vector>

struct Parameter {
  int point_number;
  float fx, fy, cx, cy;
  int width, height;
  float r[3][3];
  float t[3];
};

class DepthTransfer {
 public:
  DepthTransfer();
  ~DepthTransfer();
  void set_para(float _fx, float _fy, float _cx, float _cy, int _width,
                int _height);
  void set_data(std::vector<float> &cloud_data);
  void render_pose(double *transformation, int *host_ptr);

 private:
  int cloud_size;

  // data
  float3 *host_cloud_ptr;
  float3 *dev_cloud_ptr;
  bool has_devptr;

  // camera
  Parameter parameter;
  Parameter *parameter_devptr;
};

#endif