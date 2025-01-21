#include <depth_cut/depth_cut_node.h>

Eigen::Vector3d camera_pos_;
Eigen::Quaterniond camera_q_;
cv_bridge::CvImagePtr cv_ptr;
cv::Mat depth_image;
std::vector<Eigen::Vector3d> proj_points;
std::vector<int> idx_valid;
std::vector<Eigen::Vector3d> local_normals;
cv::Mat var_image;
double k_depth_scaling_factor{1000.0};
int skip_pixel{2};
double cx = 321.04638671875;
double cy = 243.44969177246094;
double fx = 387.229248046875;
double fy = 387.229248046875;
int kernel_half_size{3}; // 5/2
ros::Publisher var_img_pub;
ros::Publisher obj_img_pub;
ros::Publisher marker_pub;
ros::Publisher normals_pub;
ros::Publisher cloud_pub;
sensor_msgs::Image depth_img;
// NormalsRender nr;

void ProjDepthImage(){
    // 像素每个点计算一个世界位置
    // int cols = depth_image.cols;
    // int rows = depth_image.rows;
    // proj_points.clear();
    // proj_points.resize(rows * cols / skip_pixel/skip_pixel, Eigen::Vector3d::Zero());
    // idx_valid.clear();
    // idx_valid.resize(rows*cols / skip_pixel/skip_pixel, 0);

    camera_pos_ = Eigen::Vector3d::Zero();
    camera_q_ = Eigen::Quaterniond(-0.5, 0.5, -0.5, 0.5);

    Eigen::Matrix3d camera_r = camera_q_.toRotationMatrix();

    // uint16_t* row_ptr;
    double depth;
    uint32_t proj_points_cnt = 0;
    Eigen::Vector3d proj_pt;
    // for (int v = 0; v < rows; v+=skip_pixel) {
    //     row_ptr = depth_image.ptr<uint16_t>(v);
    //     for (int u = 0; u < cols; u+=skip_pixel) {
    //         // Eigen::Vector3d proj_pt;
    //         depth = (*row_ptr) / k_depth_scaling_factor;
    //         // if(v==240&&u==320){
    //             // std::cout<<"depth:"<<(*row_ptr)<<std::endl;
    //         // }
    //         if(depth < 0.2) {
    //             idx_valid[proj_points_cnt++] = 0;
    //             // proj_points_cnt++;
    //             continue;
    //         }
    //         proj_pt(0) = (u - cx) * depth / fx;
    //         proj_pt(1) = (v - cy) * depth / fy;
    //         proj_pt(2) = depth;

    //         proj_pt = camera_r * proj_pt + camera_pos_;

    //         // if (u == 320 && v == 240) std::cout << "depth: " << depth << std::endl;
    //         proj_points[proj_points_cnt] = proj_pt;
    //         idx_valid[proj_points_cnt++] = 1;
    //         row_ptr = row_ptr + skip_pixel;
    //     }
    // }
    if (depth_img.encoding == sensor_msgs::image_encodings::TYPE_16UC1){
        // 获取图像的宽度和高度
        int width = depth_img.width;
        int height = depth_img.height;
        proj_points.clear();
        proj_points.resize(height * width / skip_pixel/skip_pixel, Eigen::Vector3d::Zero());
        idx_valid.clear();
        idx_valid.resize(height*width / skip_pixel/skip_pixel, 0);

        // 获取图像数据指针
        const uint16_t* data = reinterpret_cast<const uint16_t*>(&depth_img.data[0]);

        // 遍历图像中的每个像素
        for (int row = 0; row < height; row += skip_pixel)
        {
            for (int col = 0; col < width; col += skip_pixel)
            {
                // 计算像素索引
                int index = row * width + col;

                // 获取像素值
                depth = data[index] / k_depth_scaling_factor;
                // uint16_t pixel_value = data[index];
                // std::cout<<pixel_value<<" ";
                // 这里你可以对像素值进行操作，例如打印
                // ROS_INFO("Pixel value at (%d, %d): %u", col, row, pixel_value);
                if(depth < 0.2) {
                    idx_valid[proj_points_cnt++] = 0;
                    // proj_points_cnt++;
                    continue;
                }
                proj_pt(0) = (col - cx) * depth / fx;
                proj_pt(1) = (row - cy) * depth / fy;
                proj_pt(2) = depth;

                proj_pt = camera_r * proj_pt + camera_pos_;

                // if (u == 320 && v == 240) std::cout << "depth: " << depth << std::endl;
                proj_points[proj_points_cnt] = proj_pt;
                idx_valid[proj_points_cnt++] = 1;
                // row_ptr = row_ptr + skip_pixel;
            }
            // std::cout<<std::endl;
        }
    }
}

void PublishClouds(){
    pcl::PointXYZ pt;
    pcl::PointCloud<pcl::PointXYZ> cloud;

    for(auto & p : proj_points){
        pt.x = p(0);
        pt.y = p(1);
        pt.z = p(2);
        cloud.push_back(pt);
    }
    ROS_WARN("cloud pt:%ld",cloud.size());

    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;
    cloud.header.frame_id = "map";
    sensor_msgs::PointCloud2 cloud_msg;

    pcl::toROSMsg(cloud, cloud_msg);
    cloud_pub.publish(cloud_msg);
}

void CalNormalVectors(){
    // 为每个像素计算一个局部法向量
    // int cols = depth_image.cols/skip_pixel;
    // int rows = depth_image.rows/skip_pixel;
    int cols = depth_img.width/skip_pixel;
    int rows = depth_img.height/skip_pixel;
    auto vu2idx = [&](int _v, int _u){
        return _u + _v * cols;
    };
    local_normals.clear();
    local_normals.resize(rows*cols,Eigen::Vector3d::Zero());
    Eigen::Vector3d mid,left,right,up,down,dir_u,dir_v,normal;
    for (int v = kernel_half_size; v < rows - kernel_half_size; v++) {
        for (int u = kernel_half_size; u < cols-kernel_half_size; u++) {
            if(idx_valid[vu2idx(v,u)] != 1){
                continue;
            }
            mid = proj_points[vu2idx(v, u)];
            // u方向 l->r
            left = proj_points[vu2idx(v, u - kernel_half_size)];
            if((left - mid).norm() > 0.1) continue;
            right = proj_points[vu2idx(v, u + kernel_half_size)];
            if((right - mid).norm() > 0.1) continue;
            // 判断是否在邻域
            dir_u = right - left;
            // v方向 d->u
            up = proj_points[vu2idx(v - kernel_half_size, u)];
            if((up - mid).norm() > 0.1) continue;
            down = proj_points[vu2idx(v + kernel_half_size, u)];
            if((down - mid).norm() > 0.1) continue;
            dir_v = up - down;
            // normal = du x dv
            normal = dir_u.cross(dir_v);
            normal.normalize();
            // normal长度代表该点在normal方向上的投影
            double length = proj_points[vu2idx(v, u)].dot(normal);
            normal = normal * length;
            local_normals[vu2idx(v, u)] = normal;
        }
    }
}

void PublishWorldPointsNormals(){
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker marker;

    // 设置 marker 的 frame_id 和时间戳
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();

    // 设置 marker 的命名空间和 ID
    marker.ns = "arrow_markers";
    marker.action = visualization_msgs::Marker::MODIFY;

    // 设置 marker 的类型为 ARROW，并设置其大小
    marker.type = visualization_msgs::Marker::ARROW;
    marker.scale.x = 0.01;
    marker.scale.y = 0.02;
    marker.scale.z = 0.02;

    // 设置 marker 的生命周期
    marker.lifetime = ros::Duration();

    int points_num = proj_points.size();

    for (int i = 0; i <points_num; i+=10) {
        auto &pos = proj_points[i];
        auto nrl = local_normals[i];
        // Eigen::Vector3d e3 = Eigen::Vector3d(0,0,1);
        nrl.normalize();
        if(nrl.norm() < 0.1){
            continue;
        }
        // (nrl.x()<0.1&&nrl.y()<0.1)
        // if(nrl.z()>-0.1 || nrl.z() < -0.8) {
        //     continue;
        // }
        // auto dir = e3.cross(nrl);
        // dir.normalize();
        // double cosa = nrl(2);
        // 设置 marker 的颜色
        if(std::abs(nrl(2)) > cos(25.0 / 180.0 * M_PI)){
            // ground
            marker.color.r = 0;
            marker.color.g = 0;
            marker.color.b = 1;
            marker.color.a = 1.0;
        }else{
            marker.color.r = 1;
            marker.color.g = 0;
            marker.color.b = 0;
            marker.color.a = 1.0;
        }
        
        // 为每个箭头设置起点和终点
        geometry_msgs::Point start, end;
        start.x = pos(0);
        start.y = pos(1);
        start.z = pos(2);
        end.x = pos(0) + nrl(0) * 0.1;
        end.y = pos(1) + nrl(1) * 0.1;
        end.z = pos(2) + nrl(2) * 0.1;

        // 设置 marker 的 pose，箭头方向由起点和终点决定
        marker.points.clear();
        marker.points.push_back(start);
        marker.points.push_back(end);

        // 设置唯一的 ID
        marker.id = i;

        // 将 marker 添加到 marker_array 中
        marker_array.markers.push_back(marker);
    }
    // 发布 marker_array
    marker_pub.publish(marker_array);
}

void PublishNormals(){
    // 将法向量和投影长度单独显示，用于可视化不同平面簇
    pcl::PointXYZ pt;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    int points_num = local_normals.size();
    for(int i = 0; i < points_num; i+=100){
        auto nrl = local_normals[i];
        // nrl.normalize();
        if(nrl.norm() < 0.1){
            continue;
        }
        // if(nrl.z()>-0.1 || nrl.z() < -0.8) {
        //     continue;
        // }
        pt.x = nrl(0);
        pt.y = nrl(1);
        pt.z = nrl(2);
        cloud.push_back(pt);
    }
    ROS_WARN("nrls pt:%ld",cloud.size());

    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;
    cloud.header.frame_id = "map";
    sensor_msgs::PointCloud2 cloud_msg;

    pcl::toROSMsg(cloud, cloud_msg);
    normals_pub.publish(cloud_msg);
}

void CalDepthVariance(){
    // 计算在相机系下这些点的深度方差
    uint16_t* row_ptr;
    int cols = depth_image.cols;
    int rows = depth_image.rows;
    var_image = cv::Mat::zeros(depth_image.size(), CV_32FC1);
    double depth, depth_sum{0.0};
    int kernel_num = (kernel_half_size * 2 + 1) * (kernel_half_size * 2 + 1);
    for (int v = kernel_half_size; v < rows - kernel_half_size; v++) {
        for (int u = kernel_half_size; u < cols-kernel_half_size; u++) {
            // sum
            depth_sum = 0.0;
            for(int vv = v - kernel_half_size; vv <= v + kernel_half_size; ++vv){
                row_ptr = depth_image.ptr<uint16_t>(vv) + u - kernel_half_size;
                for(int uu = u - kernel_half_size; uu <= u + kernel_half_size; ++uu){
                    depth = (*row_ptr++) / k_depth_scaling_factor;
                    depth_sum += depth;
                }
            }
            double avg = depth_sum / kernel_num;
            double var = 0.0;
            // variance
            for(int vv = v - kernel_half_size; vv <= v + kernel_half_size; ++vv){
                row_ptr = depth_image.ptr<uint16_t>(vv) + u - kernel_half_size;
                for(int uu = u - kernel_half_size; uu <= u + kernel_half_size; ++uu){
                    depth = (*row_ptr++) / k_depth_scaling_factor;
                    var += (depth - avg) * (depth - avg);
                }
            }
            var = var / kernel_num;
            var_image.at<float>(v, u) = var > 0.1 ? 0 : 255;
        }
    }
    cv_bridge::CvImage out_msg;
    out_msg.header.stamp = ros::Time::now();
    out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    out_msg.image = var_image.clone();
    // var_img_pub.publish(out_msg.toImageMsg());
}

void DepthCallback(const sensor_msgs::ImageConstPtr &img){
    // get depth image
    auto start_time = ros::Time::now();
    depth_img = *img;
    // cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
    // std::cout<<"type:"<<img->encoding<<std::endl;
    // if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    //     std::cout<<"ok"<<std::endl;
    //     (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, k_depth_scaling_factor);
    // }
    // cv_ptr->image.copyTo(depth_image);
    // ROS_WARN("get img");
    // if (img->encoding == sensor_msgs::image_encodings::TYPE_16UC1){
    //     // 获取图像的宽度和高度
    //     int width = img->width;
    //     int height = img->height;

    //     // 获取图像数据指针
    //     const uint16_t* data = reinterpret_cast<const uint16_t*>(&img->data[0]);

    //     // 遍历图像中的每个像素
    //     for (int row = 0; row < height; row+=40)
    //     {
    //         for (int col = 0; col < width; col+=40)
    //         {
    //             // 计算像素索引
    //             int index = row * width + col;

    //             // 获取像素值
    //             uint16_t pixel_value = data[index];
    //             std::cout<<pixel_value<<" ";
    //             // 这里你可以对像素值进行操作，例如打印
    //             // ROS_INFO("Pixel value at (%d, %d): %u", col, row, pixel_value);
    //         }
    //         std::cout<<std::endl;
    //     }
    // }
    
    ProjDepthImage();
    PublishClouds();
    CalNormalVectors();
    auto end_time = ros::Time::now();
    
    // ROS_INFO("get1,%ld",idx_valid.size());
    PublishWorldPointsNormals();
    PublishNormals();
    // TODO:把深度图的原图和计算结果完整的保存下来，存成txt文件，用来检查特殊方向的产生原因
    
    ROS_WARN("used time:%f",(end_time - start_time).toSec()*1000);
}

void DepthCallbackCUDA(const sensor_msgs::ImageConstPtr &img){
    // get depth image
    auto start_time = ros::Time::now();
    cv_ptr = cv_bridge::toCvCopy(img, img->encoding);

    if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, k_depth_scaling_factor);
    }
    cv_ptr->image.copyTo(depth_image);

    int cols = depth_image.cols;
    int rows = depth_image.rows;

    NormalsRender nr(fx,fy,cx,cy,cols,rows,kernel_half_size, k_depth_scaling_factor);
    
    nr.set_data(depth_image);
    // std::vector<uint16_t> de;
    // nr.get_depth(de);

    camera_pos_ = Eigen::Vector3d::Zero();
    camera_q_ = Eigen::Quaterniond(-0.5, 0.5, -0.5, 0.5);

    Eigen::Matrix3d camera_r = camera_q_.toRotationMatrix();
    float tt[3] = {camera_pos_(0),camera_pos_(1),camera_pos_(2)};
    float rr[9];
    for(int i=0;i<3;++i){
        for(int j=0;j<3;++j){
            rr[j+3*i] = camera_r(i,j);
        }
    }
    
    nr.render_pose(rr,tt);

    // auto para = nr.get_param();
    // nr.get_valid(idx_valid);
    // nr.get_cloud(proj_points);
    // int cnt=0,cnt1=0,cnt2=0,cnt3=0;
    // std::cout<<"num"<<para->point_number<<std::endl;
    // for(int i=0;i<para->point_number;++i){
    //     if(idx_valid[i]!=1){
    //         // std::cout<<i<<std::endl;
    //         cnt++;
    //     }
    //     if(!(proj_points[i](0) > 0.1)){
    //         cnt1++;
    //     }
    //     if(!(de[i]>100)){
    //         cnt2++;
    //     }
    //     if(!(*(depth_image.ptr<uint16_t>(0) + i) > 100)){
    //         cnt3++;
    //     }
    // }
    // std::cout<<cnt<<" "<<cnt1<<" "<<cnt2<<" "<<cnt3<<std::endl;
    // ROS_WARN("get_cloud");
    auto end_time = ros::Time::now();
    if(nr.get_cloud(proj_points) && nr.get_normals(local_normals)){
        // PublishClouds();
        PublishWorldPointsNormals();
        PublishNormals();
    }
    
    ROS_WARN("used time:%f",(end_time - start_time).toSec()*1000);
}

int main(int argc, char** argv){
    ros::init(argc, argv, "depth_cut_node");
    ros::NodeHandle nh;
    // var_img_pub = nh.advertise<sensor_msgs::Image>("/var_img", 2); //发布每个点的方差
    // obj_img_pub = nh.advertise<sensor_msgs::Image>("/obj_img", 2); //发布最后不同类别的
    marker_pub = nh.advertise<visualization_msgs::MarkerArray>("/objects", 1);
    normals_pub = nh.advertise<sensor_msgs::PointCloud2>("/normals", 2);
    cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud", 2);
    ros::Subscriber depth_sub = nh.subscribe<sensor_msgs::Image>("/camera/depth/image_rect_raw", 2, DepthCallback);
    // Eigen::Quaterniond camera_q1_ = Eigen::Quaterniond(0.707, 0.0, 0.707, 0.0);
    // Eigen::Quaterniond camera_q2_ = Eigen::Quaterniond(0.707, 0.0, 0.0, -0.707);
    // auto q = camera_q1_ * camera_q2_;
    // ROS_INFO("%f,%f,%f,%f",q.x(),q.y(),q.z(),q.w());
    // ros::Rate loop_rate(40);
    // while(ros::ok()){
        // ros::spinOnce();
    // }
    camera_pos_ = Eigen::Vector3d::Zero();
    camera_q_ = Eigen::Quaterniond(-0.5, 0.5, -0.5, 0.5);

    Eigen::Matrix3d camera_r = camera_q_.toRotationMatrix();
    ros::spin();
    return 0;
}