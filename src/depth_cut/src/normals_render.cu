#include "depth_cut/normals_render.cuh"

__global__ void result_init(Parameter *para_ptr, int *idx_valid_ptr, float3 *cloud_ptr, float3 *normals_ptr) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;
  int index = v * para_ptr->width + u;
  if (index >= para_ptr->point_number){
    // idx_valid_ptr[index] = 0;
    return;
  }else{
    idx_valid_ptr[index] = 0;
  }
  return;
}

__global__ void render_cloud(uint16_t *depth_ptr, Parameter *para_ptr, int *idx_valid_ptr, float3 *cloud_ptr) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;
  // if(u > para_ptr->width || v > para_ptr->height) return;
  int index = v * para_ptr->width + u;
  if (index >= para_ptr->point_number){
    // idx_valid_ptr[index] = 0;
    return;
  }
  uint16_t depth_u = depth_ptr[index];
  float depth = float(depth_u) / para_ptr->k_depth_scaling_factor;

  const int width = para_ptr->width;
  const int height = para_ptr->height;
  const float fx = para_ptr->fx;
  const float fy = para_ptr->fy;
  const float cx = para_ptr->cx;
  const float cy = para_ptr->cy;

  float3 point, transformed_point;

  point.x = (u - cx) * depth / fx;
  point.y = (v - cy) * depth / fy;
  point.z = depth;

  transformed_point.x = para_ptr->r[0] * point.x + para_ptr->r[1] * point.y + para_ptr->r[2] * point.z + para_ptr->t[0];
  transformed_point.y = para_ptr->r[3] * point.x + para_ptr->r[4] * point.y + para_ptr->r[5] * point.z + para_ptr->t[1];
  transformed_point.z = para_ptr->r[6] * point.x + para_ptr->r[7] * point.y + para_ptr->r[8] * point.z + para_ptr->t[2];

  cloud_ptr[index] = transformed_point;
  if(depth>0.1){
    idx_valid_ptr[index] = 1;
    
  }else{
    idx_valid_ptr[index] = 0;
  }
  // cloud_ptr[index].x = depth;
}

__global__ void render_normals(Parameter *para_ptr, int *idx_valid_ptr, float3 *cloud_ptr, float3 *normals_ptr) {

  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;
  
  int index = v * para_ptr->width + u;
  if (index >= para_ptr->point_number) return;

  auto vu2idx = [&](int _v, int _u){
        return _u + _v * para_ptr->width;
  };
  auto isvuValid = [&](int _v, int _u) -> bool{
    return idx_valid_ptr[_u + _v * para_ptr->width] == 1;
  };

  if(!(isvuValid(v,u) && 
       isvuValid(v, u - para_ptr->kernel_half_size) &&
       isvuValid(v, u + para_ptr->kernel_half_size) &&
       isvuValid(v - para_ptr->kernel_half_size, u) &&
       isvuValid(v + para_ptr->kernel_half_size, u))){
    // idx_valid_ptr[index] = 0;
    return;
  }
  float3 mid,left,right,up,down,dir_u,dir_v,normal;
  mid = cloud_ptr[vu2idx(v, u)];
  // u方向 l->r
  left = cloud_ptr[vu2idx(v, u - para_ptr->kernel_half_size)];
  if(length(left - mid) > 0.1){
    // idx_valid_ptr[index] = 0;
    return;
  }
  right = cloud_ptr[vu2idx(v, u + para_ptr->kernel_half_size)];
  if(length(right - mid) > 0.1){
    // idx_valid_ptr[index] = 0;
    return;
  }
  dir_u = right - left;
  // v方向 d->u
  up = cloud_ptr[vu2idx(v - para_ptr->kernel_half_size, u)];
  if(length(up - mid) > 0.1){
    // idx_valid_ptr[index] = 0;
    return;
  }
  down = cloud_ptr[vu2idx(v + para_ptr->kernel_half_size, u)];
  if(length(down - mid) > 0.1){
    // idx_valid_ptr[index] = 0;
    return;
  }
  dir_v = up - down;
  // normal = du x dv
  normal = cross(dir_u, dir_v);
  normal = normalize(normal);
  // normal长度代表该点在normal方向上的投影
  float length = dot(mid, normal);
  normal = normal * length;
  normals_ptr[index] = normal;
}

NormalsRender::NormalsRender(float _fx, float _fy, float _cx, float _cy, int _width,
                             int _height, int _kernel_half_size, float _k_depth_scaling_factor) : 
                                            depth_ptr_(NULL), cloud_ptr_(NULL), normals_ptr_(NULL), 
                                            parameter_ptr_(NULL), has_ptr_(false) {
  int pixel_num = _width * _height;
  cudaMallocManaged(&parameter_ptr_, sizeof(Parameter));
  cudaMallocManaged(&depth_ptr_, pixel_num * sizeof(uint16_t));
  cudaMallocManaged(&cloud_ptr_, pixel_num * sizeof(float3));
  cudaMallocManaged(&idx_valid_ptr_, pixel_num * sizeof(int));
  cudaMallocManaged(&normals_ptr_, pixel_num * sizeof(float3));
  
  parameter_ptr_->fx = _fx;
  parameter_ptr_->fy = _fy;
  parameter_ptr_->cx = _cx;
  parameter_ptr_->cy = _cy;
  parameter_ptr_->width = _width;
  parameter_ptr_->height = _height;
  parameter_ptr_->point_number = pixel_num;
  parameter_ptr_->kernel_half_size = _kernel_half_size;
  parameter_ptr_->k_depth_scaling_factor = _k_depth_scaling_factor;
  has_ptr_ = true;
}

NormalsRender::~NormalsRender() {
  cudaDeviceSynchronize();
  if (has_ptr_) {
    cudaFree(depth_ptr_);
    cudaFree(cloud_ptr_);
    cudaFree(normals_ptr_);
    cudaFree(idx_valid_ptr_);
    cudaFree(parameter_ptr_);
  }
}

void NormalsRender::set_data(cv::Mat& depth_img) {
  int rows = depth_img.rows;
  int cols = depth_img.cols;
  int img_size = rows * cols;
  if(img_size != parameter_ptr_->point_number){
    std::cout<<"size wrong"<<std::endl;
    return;
  }
  if (!depth_img.isContinuous()) {
      std::cerr << "Image is not continuous. Please use cv::Mat::reshape or cv::Mat::clone to make it continuous." << std::endl;
      return;
  }
  // depth_ptr_ = depth_img.ptr<uint16_t>(0);
  // cudaMemcpy(depth_ptr_, depth_img.data, img_size, cudaMemcpyHostToDevice);
  // cudaDeviceSynchronize();
  uint16_t* row_ptr = depth_img.ptr<uint16_t>(0);
  for(int i=0;i<img_size;++i){
    depth_ptr_[i] = *(row_ptr + i);
  }
  // uint16_t depth;
  // uint32_t proj_points_cnt = 0;
  // for (int v = 0; v < rows; v++) {
  //     row_ptr = depth_img.ptr<uint16_t>(v);
  //     for (int u = 0; u < cols; u++) {
  //         depth = (*row_ptr++);
          
  //         depth_ptr_[proj_points_cnt++] = depth;
  //         // if(v=160&&u==320)
  //         //   std::cout<<depth<<" "<<depth_ptr_[proj_points_cnt]<<std::endl;
          
  //     }
  // }
}
void NormalsRender::render_pose(float *r, float *t) {
  cudaMemcpy(parameter_ptr_->r, r, 9 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(parameter_ptr_->t, t, 3 * sizeof(float), cudaMemcpyHostToDevice);
  dim3 depth_block;
  dim3 depth_grid;
  depth_block.x = 32;
  depth_block.y = 32;
  depth_grid.x = (parameter_ptr_->width + depth_block.x - 1) / depth_block.x;
  depth_grid.y = (parameter_ptr_->height + depth_block.y - 1) / depth_block.y;
  // result_init<<<depth_grid, depth_block>>>(parameter_ptr_, idx_valid_ptr_, cloud_ptr_, normals_ptr_);
  render_cloud<<<depth_grid, depth_block>>>(depth_ptr_, parameter_ptr_, idx_valid_ptr_, cloud_ptr_);
  render_normals<<<depth_grid, depth_block>>>(parameter_ptr_, idx_valid_ptr_, cloud_ptr_, normals_ptr_);
  cudaDeviceSynchronize();
  // depth_output.getDevData(host_ptr);
}