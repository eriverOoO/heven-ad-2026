// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <atomic>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <filesystem>
#include <limits>
#include <optional>
#include <csignal>
#include <chrono>
#include <unistd.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <fast_lio/localization_health.hpp>
#include <fast_lio/longitudinal_position_guard.hpp>
#include <fast_lio/mapping_planar_position_guard.hpp>
#include <fast_lio/mode_policy.hpp>
#include <fast_lio/point_validation.hpp>
#include <fast_lio/pose_utils.hpp>
#include <fast_lio/wheel_velocity_buffer.hpp>
#include <fast_lio/wheel_position_increment.hpp>
#include <fast_lio/wheel_velocity_update.hpp>
#include "preprocess.hpp"
#include <ikd-Tree/ikd_Tree.h>

using fast_lio::Mode;
using fast_lio::ScanTimingMode;

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

std::vector<float> res_last;
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string map_file_path, lid_topic, imu_topic;
string initial_pose_topic, odom_topic, map_topic, registered_topic, path_topic, status_topic;
string save_map_service;
string odom_frame, base_frame, imu_frame, lidar_frame;
Mode mode = Mode::kMapping;
ScanTimingMode scan_timing_mode = ScanTimingMode::kRolling;
bool initial_pose_ready = false;
M3D base_R_imu(Eye3d);
V3D base_T_imu(Zero3d);

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
std::vector<std::uint8_t> point_selected_surf;
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool    is_first_lidar = true;
double lidar_mean_scantime = 0.0;
int scan_num = 0;
std::atomic<double> last_lidar_receipt_steady_sec{0.0};
std::atomic<double> last_imu_receipt_steady_sec{0.0};

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI());
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI());
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI());
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<fast_lio::Preprocess> p_pre(new fast_lio::Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    std::cout << "catch sig %d" << sig << std::endl;
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) 
{
    if (!initial_pose_ready) return;
    if (!fast_lio::matches_lidar_frame(msg->header.frame_id, lidar_frame)) {
        static rclcpp::Clock wrong_frame_log_clock(RCL_STEADY_TIME);
        RCLCPP_ERROR_THROTTLE(
            rclcpp::get_logger("fastlio_node"), wrong_frame_log_clock, 2000,
            "discarded PointCloud2 in frame '%s'; expected '%s'",
            msg->header.frame_id.c_str(), lidar_frame.c_str());
        return;
    }
    constexpr std::size_t kMaximumInputPoints = 200000;
    if (!fast_lio::valid_morai_pointcloud_layout(*msg) ||
        !fast_lio::valid_pointcloud_shape(*msg, kMaximumInputPoints)) {
        RCLCPP_ERROR(rclcpp::get_logger("fastlio_node"),
                     "discarded malformed or oversized MORAI PointCloud2");
        return;
    }
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    double cur_time = get_time_sec(msg->header.stamp);
    p_pre->process(msg, ptr);
    last_lidar_receipt_steady_sec.store(
        omp_get_wtime(), std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(mtx_buffer);
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
        time_buffer.clear();
        lidar_pushed = false;
        lidar_mean_scantime = 0.0;
        scan_num = 0;
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    last_timestamp_lidar = cur_time;
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr msg_in)
{
    if (!initial_pose_ready) return;
    last_imu_receipt_steady_sec.store(
        omp_get_wtime(), std::memory_order_relaxed);
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
    

    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (scan_timing_mode == ScanTimingMode::kInstantaneous)
        {
            lidar_end_time = meas.lidar_beg_time;
        }
        else if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            std::cerr << "Too few input point cloud!\n";
        }
        else
        {
            const auto maximum_time_point = std::max_element(
                meas.lidar->points.begin(), meas.lidar->points.end(),
                [](const PointType & lhs, const PointType & rhs) {
                    return lhs.curvature < rhs.curvature;
                });
            const double scan_duration = maximum_time_point->curvature / double(1000);
            if (scan_duration < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            }
            else
            {
                scan_num ++;
                lidar_end_time = meas.lidar_beg_time + scan_duration;
                lidar_mean_scantime += (scan_duration - lidar_mean_scantime) / scan_num;
            }
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = odom_frame;
        pubLaserCloudFull->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    /*
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
    */
}

void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = imu_frame;
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
    laserCloudFullRes3.header.frame_id = odom_frame;
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap)
{
    // The canonical map is the ikd-tree itself.  Do not accumulate current
    // scans here: localization must publish its loaded fixed map unchanged.
    if (ikdtree.Root_Node == nullptr) return;
    const PointVector points = ikdtree.Snapshot();
    PointCloudXYZI map_cloud;
    map_cloud.points.assign(points.begin(), points.end());
    map_cloud.width = static_cast<std::uint32_t>(map_cloud.size());
    map_cloud.height = 1;
    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(map_cloud, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = odom_frame;
    pubLaserCloudMap->publish(laserCloudmsg);
}

void save_to_pcd()
{
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(map_file_path, *pcl_wait_pub);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(
    const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped,
    std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br,
    bool publish_tf)
{
    // FastLIO's state is world_T_imu.  The AD graph consumes world_T_base.
    const M3D world_R_base = state_point.rot.toRotationMatrix() * base_R_imu.transpose();
    const V3D world_T_base = state_point.pos - world_R_base * base_T_imu;
    Eigen::Quaterniond base_q(world_R_base);
    odomAftMapped.header.frame_id = odom_frame;
    odomAftMapped.child_frame_id = base_frame;
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    odomAftMapped.pose.pose.position.x = world_T_base.x();
    odomAftMapped.pose.pose.position.y = world_T_base.y();
    odomAftMapped.pose.pose.position.z = world_T_base.z();
    odomAftMapped.pose.pose.orientation.x = base_q.x();
    odomAftMapped.pose.pose.orientation.y = base_q.y();
    odomAftMapped.pose.pose.orientation.z = base_q.z();
    odomAftMapped.pose.pose.orientation.w = base_q.w();
    const V3D base_velocity =
        fast_lio::world_velocity_to_body(world_R_base, state_point.vel);
    odomAftMapped.twist.twist.linear.x = base_velocity.x();
    odomAftMapped.twist.twist.linear.y = base_velocity.y();
    odomAftMapped.twist.twist.linear.z = base_velocity.z();
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }
    const M3D base_velocity_covariance =
        world_R_base.transpose() * P.block<3, 3>(12, 12) * world_R_base;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            odomAftMapped.twist.covariance[row * 6 + column] =
                base_velocity_covariance(row, column);
        }
    }
    pubOdomAftMapped->publish(odomAftMapped);

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = odom_frame;
    trans.header.stamp = odomAftMapped.header.stamp;
    trans.child_frame_id = base_frame;
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    if (publish_tf && tf_br) {
        tf_br->sendTransform(trans);
    }
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    const M3D world_R_base = state_point.rot.toRotationMatrix() * base_R_imu.transpose();
    const V3D world_T_base = state_point.pos - world_R_base * base_T_imu;
    Eigen::Quaterniond base_q(world_R_base);
    msg_body_pose.pose.position.x = world_T_base.x();
    msg_body_pose.pose.position.y = world_T_base.y();
    msg_body_pose.pose.position.z = world_T_base.z();
    msg_body_pose.pose.orientation.x = base_q.x();
    msg_body_pose.pose.orientation.y = base_q.y();
    msg_body_pose.pose.orientation.z = base_q.z();
    msg_body_pose.pose.orientation.w = base_q.w();
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = odom_frame;

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("fastlio_node", options)
    {
        this->declare_parameter<string>("mode", "mapping");
        this->declare_parameter<string>("map_path", "");
        this->declare_parameter<string>("base_frame", "base_link");
        this->declare_parameter<string>("odom_frame", "odom");
        // The integration profile and launch overrides use these original-style
        // top-level names.  Keep common.* only for upstream lidar/IMU inputs.
        this->declare_parameter<string>("initial_pose_topic", "/ad/localization/input/initial_pose");
        this->declare_parameter<string>("wheel_velocity.topic", "/ad/localization/input/wheel_speed");
        this->declare_parameter<bool>("wheel_velocity.enabled", true);
        this->declare_parameter<bool>(
            "wheel_velocity.header_stamp_is_device_time", false);
        this->declare_parameter<bool>("wheel_velocity.use_nonholonomic_constraints", true);
        this->declare_parameter<double>("wheel_velocity.maximum_age_sec", 0.25);
        this->declare_parameter<double>("wheel_velocity.maximum_future_sec", 0.03);
        this->declare_parameter<double>("wheel_velocity.maximum_abs_speed_mps", 20.0);
        this->declare_parameter<double>("wheel_velocity.lateral_variance", 0.04);
        this->declare_parameter<double>("wheel_velocity.vertical_variance", 0.04);
        this->declare_parameter<bool>(
            "wheel_velocity.preserve_predicted_longitudinal_position", false);
        this->declare_parameter<bool>(
            "wheel_velocity.preserve_predicted_lateral_position_in_mapping", false);
        this->declare_parameter<double>(
            "wheel_velocity.position_integration_period_sec", 0.0);
        this->declare_parameter<double>(
            "wheel_velocity.longitudinal_position_variance_floor", 0.04);
        this->declare_parameter<string>("odometry_topic", "/ad/localization/odometry");
        this->declare_parameter<string>("map_topic", "/ad/localization/fastlio/map");
        this->declare_parameter<string>("registered_points_topic", "/ad/localization/fastlio/registered_points");
        this->declare_parameter<string>("path_topic", "/ad/localization/fastlio/path");
        this->declare_parameter<string>("status_topic", "/ad/localization/fastlio/status");
        this->declare_parameter<string>("save_map_service", "/ad/localization/fastlio/save_map");
        this->declare_parameter<string>("world_frame", "odom");
        this->declare_parameter<string>("body_frame", "base_link");
        this->declare_parameter<vector<double>>("extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("extrinsic_R", vector<double>());
        this->declare_parameter<string>("imu_frame", "imu_link");
        this->declare_parameter<string>("lidar_frame", "lidar_link");
        this->declare_parameter<vector<double>>("base_to_imu_T", {0.0, 0.0, 1.5685});
        this->declare_parameter<vector<double>>("base_to_imu_R", {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.effect_map_en", false);
        this->declare_parameter<bool>("publish.map_en", true);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<bool>("publish_tf", true);
        this->declare_parameter<double>("publish.map_period_sec", 5.0);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<bool>("common.time_sync_en", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", 2);  // VELO16 only
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", fast_lio::SEC);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<string>(
            "preprocess.scan_timing_mode", "rolling");
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("feature_extract_enable", false);
        this->declare_parameter<bool>("runtime_pos_log_enable", false);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
        this->declare_parameter<int>("pcd_save.interval", -1);
        this->declare_parameter<int>("localization.minimum_effective_points", 20);
        this->declare_parameter<double>("localization.maximum_mean_residual_m", 0.5);
        this->declare_parameter<int>("localization.healthy_scan_count", 3);
        this->declare_parameter<int>("localization.lost_scan_count", 3);
        this->declare_parameter<double>("localization.sensor_timeout_sec", 0.5);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());

        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en, false);
        this->get_parameter_or<bool>("publish.map_en", map_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<bool>("publish_tf", publish_tf_, true);
        this->get_parameter_or<double>("publish.map_period_sec", map_publish_period_sec_, 5.0);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        string mode_string;
        this->get_parameter_or<string>("mode", mode_string, "mapping");
        mode = fast_lio::parse_mode(mode_string);
        this->get_parameter_or<string>("map_path", map_file_path, "");
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/ad/sensors/lidar/points");
        this->get_parameter_or<string>("common.imu_topic", imu_topic,"/ad/sensors/imu/data");
        this->get_parameter_or<string>("initial_pose_topic", initial_pose_topic, "/ad/localization/input/initial_pose");
        this->get_parameter_or<string>("wheel_velocity.topic", wheel_velocity_topic_, "/ad/localization/input/wheel_speed");
        this->get_parameter_or<bool>("wheel_velocity.enabled", wheel_velocity_enabled_, true);
        this->get_parameter_or<bool>(
            "wheel_velocity.header_stamp_is_device_time",
            wheel_velocity_header_stamp_is_device_time_, false);
        this->get_parameter_or<bool>("wheel_velocity.use_nonholonomic_constraints", wheel_velocity_use_nhc_, true);
        this->get_parameter_or<double>("wheel_velocity.maximum_age_sec", wheel_velocity_maximum_age_sec_, 0.25);
        this->get_parameter_or<double>("wheel_velocity.maximum_future_sec", wheel_velocity_maximum_future_sec_, 0.03);
        this->get_parameter_or<double>("wheel_velocity.maximum_abs_speed_mps", wheel_velocity_maximum_abs_speed_mps_, 20.0);
        this->get_parameter_or<double>("wheel_velocity.lateral_variance", wheel_velocity_lateral_variance_, 0.04);
        this->get_parameter_or<double>("wheel_velocity.vertical_variance", wheel_velocity_vertical_variance_, 0.04);
        this->get_parameter_or<bool>(
            "wheel_velocity.preserve_predicted_longitudinal_position",
            wheel_velocity_preserve_predicted_longitudinal_position_, false);
        this->get_parameter_or<bool>(
            "wheel_velocity.preserve_predicted_lateral_position_in_mapping",
            wheel_velocity_preserve_predicted_lateral_position_in_mapping_, false);
        this->get_parameter_or<double>(
            "wheel_velocity.position_integration_period_sec",
            wheel_velocity_position_integration_period_sec_, 0.0);
        this->get_parameter_or<double>(
            "wheel_velocity.longitudinal_position_variance_floor",
            wheel_velocity_longitudinal_position_variance_floor_, 0.04);
        this->get_parameter_or<string>("odometry_topic", odom_topic, "/ad/localization/odometry");
        this->get_parameter_or<string>("map_topic", map_topic, "/ad/localization/fastlio/map");
        this->get_parameter_or<string>("registered_points_topic", registered_topic, "/ad/localization/fastlio/registered_points");
        this->get_parameter_or<string>("path_topic", path_topic, "/ad/localization/fastlio/path");
        this->get_parameter_or<string>("status_topic", status_topic, "/ad/localization/fastlio/status");
        this->get_parameter_or<string>("save_map_service", save_map_service, "/ad/localization/fastlio/save_map");
        this->get_parameter_or<string>("base_frame", base_frame, "base_link");
        this->get_parameter_or<string>("odom_frame", odom_frame, "odom");
        this->get_parameter_or<string>("imu_frame", imu_frame, "imu_link");
        this->get_parameter_or<string>("lidar_frame", lidar_frame, "lidar_link");
        if (!fast_lio::valid_relative_frame(lidar_frame)) {
            throw std::runtime_error("lidar_frame must be a valid relative TF frame");
        }
        this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<double>("filter_size_corner",filter_size_corner_min,0.5);
        this->get_parameter_or<double>("filter_size_surf",filter_size_surf_min,0.5);
        this->get_parameter_or<double>("filter_size_map",filter_size_map_min,0.5);
        this->get_parameter_or<double>("cube_side_length",cube_len,200.f);
        this->get_parameter_or<float>("mapping.det_range",DET_RANGE,300.f);
        this->get_parameter_or<double>("mapping.fov_degree",fov_deg,180.f);
        this->get_parameter_or<double>("mapping.gyr_cov",gyr_cov,0.1);
        this->get_parameter_or<double>("mapping.acc_cov",acc_cov,0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov",b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov",b_acc_cov,0.0001);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, fast_lio::SEC);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        string scan_timing_mode_string;
        this->get_parameter_or<string>(
            "preprocess.scan_timing_mode", scan_timing_mode_string, "rolling");
        scan_timing_mode = fast_lio::parse_scan_timing_mode(
            scan_timing_mode_string);
        p_imu->set_point_undistortion_enabled(
            fast_lio::ScanTimingPolicy::applies_point_undistortion(
                scan_timing_mode));
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<int>("pcd_save.interval", pcd_save_interval, -1);
        int minimum_effective_points = 0;
        int healthy_scan_count = 0;
        int lost_scan_count = 0;
        double maximum_mean_residual_m = 0.0;
        this->get_parameter("localization.minimum_effective_points", minimum_effective_points);
        this->get_parameter("localization.maximum_mean_residual_m", maximum_mean_residual_m);
        this->get_parameter("localization.healthy_scan_count", healthy_scan_count);
        this->get_parameter("localization.lost_scan_count", lost_scan_count);
        this->get_parameter(
            "localization.sensor_timeout_sec",
            localization_sensor_timeout_sec_);
        if (minimum_effective_points <= 0 || healthy_scan_count <= 0 ||
            lost_scan_count <= 0 ||
            !std::isfinite(localization_sensor_timeout_sec_) ||
            localization_sensor_timeout_sec_ <= 0.0) {
            throw std::runtime_error("localization health scan counts must be positive");
        }
        if (wheel_velocity_maximum_age_sec_ <= 0.0 ||
            wheel_velocity_maximum_future_sec_ < 0.0 ||
            wheel_velocity_maximum_abs_speed_mps_ <= 0.0 ||
            wheel_velocity_lateral_variance_ <= 0.0 ||
            wheel_velocity_vertical_variance_ <= 0.0 ||
            wheel_velocity_position_integration_period_sec_ < 0.0 ||
            wheel_velocity_longitudinal_position_variance_floor_ <= 0.0) {
            throw std::runtime_error("wheel velocity limits and variances must be positive");
        }
        localization_health_ = std::make_unique<fast_lio::LocalizationHealth>(
            fast_lio::LocalizationHealthConfig{
                static_cast<std::size_t>(minimum_effective_points),
                maximum_mean_residual_m,
                static_cast<std::size_t>(healthy_scan_count),
                static_cast<std::size_t>(lost_scan_count)});
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());
        vector<double> launch_extrin_t, launch_extrin_r;
        this->get_parameter("extrinsic_T", launch_extrin_t);
        this->get_parameter("extrinsic_R", launch_extrin_r);
        if (!launch_extrin_t.empty()) extrinT = launch_extrin_t;
        if (!launch_extrin_r.empty()) extrinR = launch_extrin_r;

        if (p_pre->time_unit != fast_lio::SEC || p_pre->N_SCANS != 16 || p_pre->SCAN_RATE != 10) {
            throw std::runtime_error("MORAI PointCloud2 profile requires 16 rings, 10 Hz, and seconds point time");
        }
        if (p_pre->point_filter_num <= 0) {
            throw std::runtime_error("point_filter_num must be positive");
        }
        p_pre->set(p_pre->feature_enabled, p_pre->N_SCANS, p_pre->blind, p_pre->point_filter_num);

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id = odom_frame;

        // /*** variables definition ***/
        // int effect_feat_num = 0, frame_num = 0;
        // double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
        // bool flg_EKF_converged, EKF_stop_flg = 0;

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

        if (extrinT.size() != 3 || !fast_lio::finite_vector(extrinT) ||
            !fast_lio::valid_rotation_matrix(extrinR)) {
            throw std::runtime_error(
                "mapping.extrinsic_T/R must be a finite translation and proper rotation");
        }
        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        vector<double> base_to_imu_t, base_to_imu_r;
        this->get_parameter("base_to_imu_T", base_to_imu_t);
        this->get_parameter("base_to_imu_R", base_to_imu_r);
        if (base_to_imu_t.size() != 3 || !fast_lio::finite_vector(base_to_imu_t) ||
            !fast_lio::valid_rotation_matrix(base_to_imu_r)) {
            throw std::runtime_error(
                "base_to_imu_T/R must be a finite translation and proper rotation");
        }
        base_T_imu << VEC_FROM_ARRAY(base_to_imu_t);
        base_R_imu << MAT_FROM_ARRAY(base_to_imu_r);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        fill(epsi, epsi+23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        /*** ROS graph: PointCloud2 only, with an explicit base-frame initial pose gate. ***/
        sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
        // MORAI bridge publishes sensor data BEST_EFFORT; a reliable request
        // would be incompatible and silently receive no IMU samples.
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, rclcpp::SensorDataQoS().keep_last(100), imu_cbk);
        initial_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(initial_pose_topic, 1,
            std::bind(&LaserMappingNode::initial_pose_cbk, this, std::placeholders::_1));
        if (wheel_velocity_enabled_) {
            wheel_velocity_sub_ = this->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
                wheel_velocity_topic_, rclcpp::QoS(rclcpp::KeepLast(20)).reliable(),
                std::bind(&LaserMappingNode::wheel_velocity_cbk, this, std::placeholders::_1));
        }
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(registered_topic, 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
        pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            map_topic, rclcpp::QoS(1).reliable().transient_local());
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>(odom_topic, 20);
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>(path_topic, 20);
        status_publisher_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
            status_topic, 10);
        if (publish_tf_) {
            tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        }

        //------------------------------------------------------------------------------------------------------
        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        timer_ = rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this));

        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>(save_map_service, std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

        if (mode == Mode::kLocalization) {
            if (map_file_path.empty() || pcl::io::loadPCDFile(map_file_path, *featsFromMap) != 0 || featsFromMap->empty()) {
                throw std::runtime_error("localization mode requires a readable non-empty map_path PCD");
            }
            const auto invalid_map_point = std::find_if(
                featsFromMap->points.begin(), featsFromMap->points.end(),
                [](const PointType & point) {
                    return !fast_lio::finite_xyz(point.x, point.y, point.z);
                });
            if (invalid_map_point != featsFromMap->points.end()) {
                throw std::runtime_error(
                    "localization map_path contains non-finite XYZ points");
            }
            ikdtree.set_downsample_param(filter_size_map_min);
            ikdtree.Build(featsFromMap->points);
            if (map_pub_en) publish_map(pubLaserCloudMap_);  // transient-local fixed map, exactly once
        } else if (map_pub_en) {
            const auto period = std::chrono::milliseconds(
                static_cast<int64_t>(std::max(1.0, map_publish_period_sec_) * 1000.0));
            map_pub_timer_ = rclcpp::create_timer(this, this->get_clock(), period,
                std::bind(&LaserMappingNode::map_publish_callback, this));
        }

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

    ~LaserMappingNode() = default;

private:
    void initial_pose_cbk(const geometry_msgs::msg::PoseStamped::ConstSharedPtr message)
    {
        // In localization mode the handoff node owns this seed stream.  It
        // keeps the pose current while GNSS is authoritative, then stops the
        // stream as soon as FastLIO is selected.
        if (!fast_lio::ModePolicy::accepts_initial_pose_update(
                mode, initial_pose_ready)) return;
        // The adapter publishes a base_link pose expressed in odom, so the
        // header identifies the reference frame, not the body frame.
        if (!message->header.frame_id.empty() && message->header.frame_id != odom_frame) {
            RCLCPP_WARN(this->get_logger(), "initial pose must be expressed in %s", odom_frame.c_str());
            return;
        }
        if (!fast_lio::valid_pose(message->pose)) {
            RCLCPP_WARN(this->get_logger(), "initial pose must be finite with a valid quaternion");
            return;
        }
        const auto &q = message->pose.orientation;
        Eigen::Quaterniond base_q(q.w, q.x, q.y, q.z);
        if (base_q.norm() < 1e-6) return;
        base_q.normalize();
        const M3D world_R_base = base_q.toRotationMatrix();
        state_point.pos << message->pose.position.x, message->pose.position.y, message->pose.position.z;
        state_point.pos += world_R_base * base_T_imu;
        state_point.rot = world_R_base * base_R_imu;
        state_point.offset_T_L_I = Lidar_T_wrt_IMU;
        state_point.offset_R_L_I = Lidar_R_wrt_IMU;
        kf.change_x(state_point);
        wheel_position_anchor_.reset();
        wheel_position_anchor_time_.reset();
        wheel_position_anchor_lateral_speed_mps_ = 0.0;
        wheel_position_anchor_cumulative_forward_distance_m_.reset();
        wheel_position_anchor_cumulative_lateral_distance_m_.reset();
        latest_applied_wheel_velocity_.reset();
        const double now_steady = omp_get_wtime();
        last_lidar_receipt_steady_sec.store(
            now_steady, std::memory_order_relaxed);
        last_imu_receipt_steady_sec.store(
            now_steady, std::memory_order_relaxed);
        const bool first_pose = !initial_pose_ready;
        initial_pose_ready = true;
        if (first_pose) {
            RCLCPP_INFO(this->get_logger(), "initial pose accepted; processing enabled");
        }
    }

    void wheel_velocity_cbk(
        const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr message)
    {
        const double header_stamp = get_time_sec(message->header.stamp);
        const double arrival_stamp = this->get_clock()->now().seconds();
        const double stamp = wheel_velocity_header_stamp_is_device_time_ ?
            arrival_stamp : header_stamp;
        const double speed = message->twist.twist.linear.x;
        const double variance = message->twist.covariance[0];
        const double lateral_speed = message->twist.twist.linear.y;
        const double lateral_variance = message->twist.covariance[7];
        const bool has_lateral_measurement =
            std::isfinite(lateral_variance) && lateral_variance > 0.0;
        if (message->header.frame_id != base_frame || header_stamp <= 0.0 ||
            stamp <= 0.0 || !std::isfinite(header_stamp) ||
            !std::isfinite(stamp) || !std::isfinite(speed) ||
            !std::isfinite(variance) || variance <= 0.0 ||
            std::abs(speed) > wheel_velocity_maximum_abs_speed_mps_ ||
            (has_lateral_measurement &&
             (!std::isfinite(lateral_speed) ||
              std::abs(lateral_speed) > wheel_velocity_maximum_abs_speed_mps_))) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "discarded invalid wheel velocity measurement");
            return;
        }
        fast_lio::TimedWheelVelocity sample{stamp, speed, variance};
        if (wheel_velocity_header_stamp_is_device_time_) {
            sample.distance_stamp_sec = header_stamp;
        }
        if (has_lateral_measurement) {
            sample.lateral_speed_mps = lateral_speed;
            sample.lateral_variance = lateral_variance;
        }
        if (!wheel_velocity_buffer_.push(sample)) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "discarded duplicate or out-of-order wheel velocity");
        }
    }

    bool apply_latest_wheel_velocity(double scan_end_time)
    {
        if (!wheel_velocity_enabled_) return false;
        const auto measurement = wheel_velocity_buffer_.take_for_scan(
            scan_end_time,
            wheel_velocity_maximum_age_sec_,
            wheel_velocity_maximum_future_sec_);
        if (!measurement) return false;

        state_ikfom updated_state = kf.get_x();
        auto updated_covariance = kf.get_P();
        Eigen::Vector3d world_velocity = updated_state.vel;
        const auto update = fast_lio::WheelVelocityUpdate::apply(
            updated_state.rot.toRotationMatrix() * base_R_imu.transpose(),
            measurement->forward_speed_mps,
            measurement->forward_variance,
            measurement->lateral_speed_mps,
            measurement->lateral_speed_mps ?
                measurement->lateral_variance : wheel_velocity_lateral_variance_,
            wheel_velocity_use_nhc_,
            wheel_velocity_vertical_variance_,
            world_velocity,
            updated_covariance);
        if (!update.accepted) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "wheel velocity IEKF auxiliary update was rejected");
            return false;
        }
        updated_state.vel = world_velocity;
        kf.change_x(updated_state);
        kf.change_P(updated_covariance);
        latest_applied_wheel_velocity_ = *measurement;
        const Eigen::Vector3d fused_body_velocity =
            (updated_state.rot.toRotationMatrix() * base_R_imu.transpose()).transpose() *
            world_velocity;
        RCLCPP_INFO_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "wheel velocity fused: measured (%.3f, %.3f) m/s, state (%.3f, %.3f) m/s",
            measurement->forward_speed_mps,
            measurement->lateral_speed_mps.value_or(
                std::numeric_limits<double>::quiet_NaN()),
            fused_body_velocity.x(), fused_body_velocity.y());
        return true;
    }

    bool preserve_predicted_longitudinal_position(
        const state_ikfom &predicted_state,
        const fast_lio::LongitudinalPositionGuard::Covariance &predicted_covariance,
        double scan_end_time)
    {
        if (!wheel_velocity_preserve_predicted_longitudinal_position_ ||
            !latest_applied_wheel_velocity_) return false;

        state_ikfom corrected_state = kf.get_x();
        auto corrected_covariance = kf.get_P();
        Eigen::Vector3d corrected_position = corrected_state.pos;
        const Eigen::Matrix3d world_R_body =
            predicted_state.rot.toRotationMatrix() * base_R_imu.transpose();
        Eigen::Vector3d wheel_predicted_position = predicted_state.pos;
        if (wheel_position_anchor_ && wheel_position_anchor_time_) {
            fast_lio::WheelPositionIncrement::Result increment;
            if (wheel_position_anchor_cumulative_forward_distance_m_ &&
                wheel_position_anchor_cumulative_lateral_distance_m_ &&
                latest_applied_wheel_velocity_->cumulative_forward_distance_m &&
                latest_applied_wheel_velocity_->cumulative_lateral_distance_m) {
                if (!fast_lio::WheelPositionIncrement::cumulative_distance_advanced(
                        *wheel_position_anchor_cumulative_forward_distance_m_,
                        *wheel_position_anchor_cumulative_lateral_distance_m_,
                        *latest_applied_wheel_velocity_->
                            cumulative_forward_distance_m,
                        *latest_applied_wheel_velocity_->
                            cumulative_lateral_distance_m)) {
                    // A fresh wheel sample may be reused for several LiDAR
                    // scans.  Reapplying its unchanged cumulative distance
                    // would pin the position to the previous scan.
                    return false;
                }
                increment = fast_lio::WheelPositionIncrement::apply_displacement(
                    world_R_body,
                    *wheel_position_anchor_,
                    *latest_applied_wheel_velocity_->cumulative_forward_distance_m -
                        *wheel_position_anchor_cumulative_forward_distance_m_,
                    *latest_applied_wheel_velocity_->cumulative_lateral_distance_m -
                        *wheel_position_anchor_cumulative_lateral_distance_m_,
                    2.0 * wheel_velocity_maximum_abs_speed_mps_ *
                        wheel_velocity_maximum_age_sec_);
            } else {
                const double integration_interval =
                    fast_lio::WheelPositionIncrement::select_interval(
                        scan_end_time - *wheel_position_anchor_time_,
                        wheel_velocity_position_integration_period_sec_,
                        2.0 * wheel_velocity_maximum_age_sec_);
                increment = fast_lio::WheelPositionIncrement::integrate_planar(
                    world_R_body,
                    *wheel_position_anchor_,
                    wheel_position_anchor_speed_mps_,
                    wheel_position_anchor_lateral_speed_mps_,
                    latest_applied_wheel_velocity_->forward_speed_mps,
                    latest_applied_wheel_velocity_->lateral_speed_mps.value_or(0.0),
                    integration_interval,
                    2.0 * wheel_velocity_maximum_age_sec_);
            }
            if (increment.accepted) {
                wheel_predicted_position = increment.position;
            } else {
                wheel_position_anchor_.reset();
                wheel_position_anchor_time_.reset();
                wheel_position_anchor_lateral_speed_mps_ = 0.0;
                wheel_position_anchor_cumulative_forward_distance_m_.reset();
                wheel_position_anchor_cumulative_lateral_distance_m_.reset();
            }
        }
        double removed_forward_correction_m = 0.0;
        double removed_lateral_correction_m = 0.0;
        bool guard_accepted = false;
        if (mode == Mode::kMapping &&
            wheel_velocity_preserve_predicted_lateral_position_in_mapping_) {
            const auto guard = fast_lio::MappingPlanarPositionGuard::apply(
                world_R_body, wheel_predicted_position, predicted_covariance,
                wheel_velocity_longitudinal_position_variance_floor_,
                corrected_position, corrected_covariance);
            guard_accepted = guard.accepted;
            removed_forward_correction_m = guard.removed_forward_correction_m;
            removed_lateral_correction_m = guard.removed_lateral_correction_m;
        } else {
            const auto guard = fast_lio::LongitudinalPositionGuard::apply(
                world_R_body, wheel_predicted_position, predicted_covariance,
                wheel_velocity_longitudinal_position_variance_floor_,
                corrected_position, corrected_covariance);
            guard_accepted = guard.accepted;
            removed_forward_correction_m = guard.removed_correction_m;
        }
        if (!guard_accepted) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "LiDAR position guard was rejected");
            return false;
        }

        corrected_state.pos = corrected_position;
        kf.change_x(corrected_state);
        kf.change_P(corrected_covariance);
        wheel_position_anchor_ = corrected_position;
        wheel_position_anchor_time_ = scan_end_time;
        wheel_position_anchor_speed_mps_ =
            latest_applied_wheel_velocity_->forward_speed_mps;
        wheel_position_anchor_lateral_speed_mps_ =
            latest_applied_wheel_velocity_->lateral_speed_mps.value_or(0.0);
        wheel_position_anchor_cumulative_forward_distance_m_ =
            latest_applied_wheel_velocity_->cumulative_forward_distance_m;
        wheel_position_anchor_cumulative_lateral_distance_m_ =
            latest_applied_wheel_velocity_->cumulative_lateral_distance_m;
        RCLCPP_INFO_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "LiDAR correction removed: forward %.3f m, lateral %.3f m",
            removed_forward_correction_m, removed_lateral_correction_m);
        return true;
    }

    void timer_callback()
    {
        update_sensor_watchdog();
        if(sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                return;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                record_localization_match(0, INFINITY);
                return;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** The fixed localization map is immutable and must not be pruned. ***/
            if (mode == Mode::kMapping) lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                return;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                record_localization_match(0, INFINITY);
                return;
            }
            
            normvec->resize(feats_down_size);
            laserCloudOri->resize(feats_down_size);
            corr_normvect->resize(feats_down_size);
            point_selected_surf.assign(feats_down_size, true);
            res_last.assign(feats_down_size, -1000.0F);
            feats_down_world->resize(feats_down_size);

            if(0) // If you need to see map point, change to "if(1)"
            {
                ikdtree.PCL_Storage = ikdtree.Snapshot();
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            const state_ikfom imu_predicted_state = kf.get_x();
            const auto imu_predicted_covariance = kf.get_P();
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            // Keep the velocity constraint as the final update for this scan.
            // Otherwise scan-to-map cross covariance can immediately erase
            // it in a longitudinally degenerate tunnel.  The fused velocity
            // propagates position from the next IMU interval onward.
            const bool wheel_velocity_applied =
                apply_latest_wheel_velocity(Measures.lidar_end_time);
            if (wheel_velocity_applied) {
                preserve_predicted_longitudinal_position(
                    imu_predicted_state, imu_predicted_covariance,
                    Measures.lidar_end_time);
            }
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish control-facing odometry only while fixed-map matching is healthy. *******/
            const bool localization_output_ready =
                record_localization_match(effct_feat_num, res_mean_last);
            if (mode == Mode::kMapping || localization_output_ready) {
                publish_odometry(pubOdomAftMapped_, tf_broadcaster_, publish_tf_);
            }

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            if (mode == Mode::kMapping) map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath_);
            if (scan_pub_en)      publish_frame_world(pubLaserCloudFull_);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body_);
            if (effect_pub_en) publish_effect_world(pubLaserCloudEffect_);
            // if (map_pub_en) publish_map(pubLaserCloudMap_);

        }
    }

    void map_publish_callback()
    {
        if (mode == Mode::kMapping && map_pub_en) publish_map(pubLaserCloudMap_);
    }

    bool record_localization_match(std::size_t effective_points, double mean_residual_m)
    {
        if (mode == Mode::kMapping) return true;
        bool output_ready = false;
        if (sensor_stream_stale_) {
            localization_health_->mark_stale();
        } else {
            output_ready = localization_health_->observe(
                effective_points, mean_residual_m);
        }
        const auto state = localization_health_->state();

        diagnostic_msgs::msg::DiagnosticArray array;
        array.header.stamp = this->now();
        diagnostic_msgs::msg::DiagnosticStatus status;
        status.name = "fast_lio/fixed_map_match";
        status.hardware_id = map_file_path;
        if (state == fast_lio::LocalizationHealthState::kHealthy) {
            status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
            status.message = "fixed-map scan matching is healthy";
        } else if (state == fast_lio::LocalizationHealthState::kLost) {
            status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
            status.message = "fixed-map scan matching is lost; odometry suppressed";
        } else {
            status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
            status.message = "waiting for consecutive healthy fixed-map matches";
        }
        diagnostic_msgs::msg::KeyValue state_value;
        state_value.key = "state";
        state_value.value = fast_lio::to_string(state);
        diagnostic_msgs::msg::KeyValue points_value;
        points_value.key = "effective_points";
        points_value.value = std::to_string(effective_points);
        diagnostic_msgs::msg::KeyValue residual_value;
        residual_value.key = "mean_residual_m";
        residual_value.value = std::isfinite(mean_residual_m) ?
            std::to_string(mean_residual_m) : "unavailable";
        status.values = {state_value, points_value, residual_value};
        array.status.push_back(std::move(status));
        status_publisher_->publish(array);

        if (!output_ready) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "fixed-map localization %s: %zu effective points, residual %.3f m; odometry suppressed",
                fast_lio::to_string(state), effective_points, mean_residual_m);
        }
        return output_ready;
    }

    void update_sensor_watchdog()
    {
        if (mode != Mode::kLocalization || !initial_pose_ready) {
            return;
        }
        const double now_steady = omp_get_wtime();
        const double lidar_age = now_steady -
            last_lidar_receipt_steady_sec.load(std::memory_order_relaxed);
        const double imu_age = now_steady -
            last_imu_receipt_steady_sec.load(std::memory_order_relaxed);
        sensor_stream_stale_ =
            !std::isfinite(lidar_age) || !std::isfinite(imu_age) ||
            lidar_age > localization_sensor_timeout_sec_ ||
            imu_age > localization_sensor_timeout_sec_;
        if (!sensor_stream_stale_) {
            return;
        }

        localization_health_->mark_stale();
        if (now_steady - last_watchdog_report_steady_sec_ < 0.5) {
            return;
        }
        last_watchdog_report_steady_sec_ = now_steady;

        std::size_t lidar_queue_depth = 0;
        std::size_t time_queue_depth = 0;
        std::size_t imu_queue_depth = 0;
        {
            std::lock_guard<std::mutex> lock(mtx_buffer);
            lidar_queue_depth = lidar_buffer.size();
            time_queue_depth = time_buffer.size();
            imu_queue_depth = imu_buffer.size();
        }

        const auto value = [](const std::string & key, const std::string & text) {
            diagnostic_msgs::msg::KeyValue result;
            result.key = key;
            result.value = text;
            return result;
        };
        diagnostic_msgs::msg::DiagnosticStatus status;
        status.name = "fast_lio/fixed_map_match";
        status.hardware_id = map_file_path;
        status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
        status.message = "LiDAR or IMU input is stale; odometry suppressed";
        status.values = {
            value("state", "lost"),
            value("lidar_age_sec", std::to_string(lidar_age)),
            value("imu_age_sec", std::to_string(imu_age)),
            value("lidar_queue_depth", std::to_string(lidar_queue_depth)),
            value("time_queue_depth", std::to_string(time_queue_depth)),
            value("imu_queue_depth", std::to_string(imu_queue_depth)),
        };
        diagnostic_msgs::msg::DiagnosticArray array;
        array.header.stamp = this->now();
        array.status.push_back(std::move(status));
        status_publisher_->publish(array);
    }

    void map_save_callback(std_srvs::srv::Trigger::Request::ConstSharedPtr req, std_srvs::srv::Trigger::Response::SharedPtr res)
    {
        if (mode != Mode::kMapping || map_file_path.empty()) {
            res->success = false;
            res->message = "map snapshots are mapping-only and require map_path";
            return;
        }
        const PointVector snapshot = ikdtree.Snapshot();
        PointCloudXYZI snapshot_cloud;
        snapshot_cloud.points.assign(snapshot.begin(), snapshot.end());
        snapshot_cloud.width = static_cast<std::uint32_t>(snapshot_cloud.size());
        snapshot_cloud.height = 1;
        const std::filesystem::path target(map_file_path);
        std::error_code filesystem_error;
        if (!target.parent_path().empty()) std::filesystem::create_directories(target.parent_path(), filesystem_error);
        if (filesystem_error) {
            res->success = false;
            res->message = "cannot create map snapshot parent directory: " + filesystem_error.message();
            return;
        }
        const string temporary = map_file_path + ".tmp";
        if (snapshot.empty() || pcl::io::savePCDFileBinary(temporary, snapshot_cloud) != 0 ||
            std::rename(temporary.c_str(), map_file_path.c_str()) != 0) {
            std::remove(temporary.c_str());
            res->success = false;
            res->message = "failed to save ikd-tree snapshot atomically";
            return;
        }
        res->success = true;
        res->message = "ikd-tree snapshot saved";
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr status_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr initial_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr
        wheel_velocity_sub_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr map_pub_timer_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;
    std::unique_ptr<fast_lio::LocalizationHealth> localization_health_;
    double localization_sensor_timeout_sec_{0.5};
    bool sensor_stream_stale_{false};
    double last_watchdog_report_steady_sec_{0.0};

    bool effect_pub_en = false, map_pub_en = false;
    bool publish_tf_{true};
    double map_publish_period_sec_{5.0};
    std::string wheel_velocity_topic_;
    bool wheel_velocity_enabled_{true};
    bool wheel_velocity_header_stamp_is_device_time_{false};
    bool wheel_velocity_use_nhc_{true};
    double wheel_velocity_maximum_age_sec_{0.25};
    double wheel_velocity_maximum_future_sec_{0.03};
    double wheel_velocity_maximum_abs_speed_mps_{20.0};
    double wheel_velocity_lateral_variance_{0.04};
    double wheel_velocity_vertical_variance_{0.04};
    bool wheel_velocity_preserve_predicted_longitudinal_position_{false};
    bool wheel_velocity_preserve_predicted_lateral_position_in_mapping_{false};
    double wheel_velocity_position_integration_period_sec_{0.0};
    double wheel_velocity_longitudinal_position_variance_floor_{0.04};
    fast_lio::WheelVelocityBuffer wheel_velocity_buffer_{200};
    std::optional<fast_lio::TimedWheelVelocity> latest_applied_wheel_velocity_;
    std::optional<Eigen::Vector3d> wheel_position_anchor_;
    std::optional<double> wheel_position_anchor_time_;
    double wheel_position_anchor_speed_mps_{0.0};
    double wheel_position_anchor_lateral_speed_mps_{0.0};
    std::optional<double> wheel_position_anchor_cumulative_forward_distance_m_;
    std::optional<double> wheel_position_anchor_cumulative_lateral_distance_m_;
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    double epsi[23] = {0.001};

    FILE *fp{nullptr};
    ofstream fout_pre, fout_out, fout_dbg;
};

namespace fast_lio {
rclcpp::Node::SharedPtr make_fastlio_node(const rclcpp::NodeOptions &options)
{
    return std::make_shared<LaserMappingNode>(options);
}
}  // namespace fast_lio
