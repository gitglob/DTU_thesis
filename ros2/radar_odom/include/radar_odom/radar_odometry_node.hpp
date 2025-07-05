#ifndef RADAR_ODOMETRY_HPP
#define RADAR_ODOMETRY_HPP

// must
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"

// handle pointclouds
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp> 

// custom msg
#include <custom_msgs/msg/radar_odometry_stamped.hpp>

// transform between coordinates
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.h"

// dbscan header only implementation
#include <vector>

// to save xyz data for clustering experimentation
#include <fstream>

// to execute linear regression as a solutoin to Least Squares problem
#include "radar_odom/radar_helpers.h"
#include "eigen/Dense"

// to generate random integers
#include <stdlib.h>
#include <time.h>

// classic libraries
#include <cmath>
#include <iostream>
#include <string>

class RadarOdom : public rclcpp::Node{
public:
    RadarOdom();

private:
    // callback functions
    void radar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void radar_cartesian_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // subscribers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_radar_;

    // publishers
    rclcpp::Publisher<custom_msgs::msg::RadarOdometryStamped>::SharedPtr publisher_;

    // other functions
    std::tuple<bool, std::vector<int>, float> Ransac(std::vector<float> theta_list, 
                                                    std::vector<float> psi_list, 
                                                    std::vector<float>  vr_list, 
                                                    float error_threshold, int max_iter, float stop_error, float stop_inl, 
                                                    int num_points_thresh,
                                                    bool ransac_flag, 
                                                    std::string criteria);
    std::tuple<bool, float, float, float> inliers_LSQ(std::vector<int> best_inliers, 
                                                std::vector<float> theta_list, 
                                                std::vector<float> psi_list, 
                                                std::vector<float> vr_list);
    std::tuple<float, float> ego_motion(float v_s, float alpha, float b_, float l_, float beta_);

    // global variables
    // define the radar c.f. offset from the vehicle c.f. -> hardcoded based on the extrinsic calibration from "rd25.urdf.xacro"
//   <joint name="radar_front_right_joint" type="fixed">
//     <origin xyz="${radar_front_dual_x} ${-radar_front_dual_y} ${radar_front_height-chassis_offset}" rpy="0 -0.09 -0.61" />
//     <parent link="chassis"/>
//     <child link="radar_front_right" />
//   </joint>
    float l_ = 1.02; // x_s
    float b_ = -0.36; // y_s
    float beta_ = -0.61; // a_s
};


#endif