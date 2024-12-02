#include <depth_cut/depth_cut_node.h>


cv_bridge::CvImagePtr cv_ptr;
cv::Mat depth_image;
std::vector<Eigen::Vector3d> proj_points;
std::vector<int> idx_valid;
std::vector<Eigen::Vector3d> local_normals;
cv::Mat var_image;
double k_depth_scaling_factor{1000.0};
double cx = 321.04638671875;
double cy = 243.44969177246094;
double fx = 387.229248046875;
double fy = 387.229248046875;
int kernel_half_size{1}; // 5/2
ros::Publisher var_img_pub;
ros::Publisher obj_img_pub;
ros::Publisher marker_pub;
ros::Publisher normals_pub;

void ProjDepthImage(){
    // 像素每个点计算一个世界位置
    int cols = depth_image.cols;
    int rows = depth_image.rows;
    proj_points.clear();
    proj_points.resize(rows * cols, Eigen::Vector3d::Zero());
    idx_valid.clear();
    idx_valid.resize(rows*cols, 0);

    uint16_t* row_ptr;
    double depth;
    uint32_t proj_points_cnt = 0;
    for (int v = 0; v < rows; v++) {
        row_ptr = depth_image.ptr<uint16_t>(v);
        for (int u = 0; u < cols; u++) {
            Eigen::Vector3d proj_pt;
            depth = (*row_ptr++) / k_depth_scaling_factor;
            if(depth < 0.5) continue;
            proj_pt(0) = (u - cx) * depth / fx;
            proj_pt(1) = (v - cy) * depth / fy;
            proj_pt(2) = depth;

            // proj_pt = proj_pt + md_.camera_pos_;

            // if (u == 320 && v == 240) std::cout << "depth: " << depth << std::endl;
            proj_points[proj_points_cnt] = proj_pt;
            idx_valid[proj_points_cnt++] = 1;
        }
    }
}

void CalNormalVectors(){
    // 为每个像素计算一个局部法向量
    int cols = depth_image.cols;
    int rows = depth_image.rows;
    auto vu2idx = [&](int _v, int _u){
        return _u + _v * cols;
    };
    local_normals.clear();
    local_normals.resize(rows*cols,Eigen::Vector3d::Zero());
    Eigen::Vector3d left,right,up,down,dir_u,dir_v,normal;
    for (int v = kernel_half_size; v < rows - kernel_half_size; v++) {
        for (int u = kernel_half_size; u < cols-kernel_half_size; u++) {
            if(idx_valid[vu2idx(v,u)] != 1){
                continue;
            }
            // u方向 l->r
            left = proj_points[vu2idx(v, u - kernel_half_size)];
            right = proj_points[vu2idx(v, u + kernel_half_size)];
            dir_u = right - left;
            // v方向 d->u
            up = proj_points[vu2idx(v - kernel_half_size, u)];
            down = proj_points[vu2idx(v + kernel_half_size, u)];
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
    marker.header.frame_id = "/map";
    marker.header.stamp = ros::Time::now();

    // 设置 marker 的命名空间和 ID
    marker.ns = "arrow_markers";
    marker.action = visualization_msgs::Marker::ADD;

    // 设置 marker 的类型为 ARROW，并设置其大小
    marker.type = visualization_msgs::Marker::ARROW;
    marker.scale.x = 0.1;
    marker.scale.y = 0.2;
    marker.scale.z = 0.2;

    // 设置 marker 的生命周期
    marker.lifetime = ros::Duration();

    int points_num = proj_points.size();

    for (int i = 0; i <points_num; ++i) {
        auto &pos = proj_points[i];
        auto nrl = local_normals[i];
        Eigen::Vector3d e3 = Eigen::Vector3d(0,0,1);
        nrl.normalize();
        // auto dir = e3.cross(nrl);
        // dir.normalize();
        // double cosa = nrl(2);
        // 设置 marker 的颜色
        marker.color.r = nrl(0);
        marker.color.g = nrl(1);
        marker.color.b = nrl(2);
        marker.color.a = 1.0;
        // 为每个箭头设置起点和终点
        geometry_msgs::Point start, end;
        start.x = pos(0);
        start.y = pos(1);
        start.z = pos(2);
        end.x = pos(0) + nrl(0) * 0.2;
        end.y = pos(1) + nrl(1) * 0.2;
        end.z = pos(2) + nrl(2) * 0.2;

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

    for(auto & nrl : local_normals){
        pt.x = nrl(0);
        pt.y = nrl(1);
        pt.z = nrl(2);
        cloud.push_back(pt);
    }

    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;
    cloud.header.frame_id = "normals";
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
    var_img_pub.publish(out_msg.toImageMsg());
}

void DepthCallback(const sensor_msgs::ImageConstPtr &img){
    // get depth image
    cv_ptr = cv_bridge::toCvCopy(img, img->encoding);

    if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, k_depth_scaling_factor);
    }
    cv_ptr->image.copyTo(depth_image);
    ProjDepthImage();
    CalNormalVectors();
    PublishWorldPointsNormals();
    PublishNormals();
}

int main(int argc, char** argv){
    ros::init(argc, argv, "depth_cut_node");
    ros::NodeHandle nh;
    var_img_pub = nh.advertise<sensor_msgs::Image>("/var_img", 2); //发布每个点的方差
    obj_img_pub = nh.advertise<sensor_msgs::Image>("/obj_img", 2); //发布最后不同类别的
    marker_pub = nh.advertise<visualization_msgs::MarkerArray>("/objects", 1);
    normals_pub = nh.advertise<sensor_msgs::PointCloud2>("/normals", 2);
    ros::Subscriber depth_sub = nh.subscribe<sensor_msgs::Image>("/pcl_render_node/depth", 2, DepthCallback);
    // ros::Rate loop_rate(40);
    // while(ros::ok()){
        // ros::spinOnce();
    // }
    ros::spin();
    return 0;
}