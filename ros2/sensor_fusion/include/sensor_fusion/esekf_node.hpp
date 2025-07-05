#ifndef ESEKF_HPP
#define ESEKF_HPP

// must
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"

// other pacakges
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>

// custom msg
#include <custom_msgs/msg/radar_odometry_stamped.hpp>
#include <custom_msgs/msg/esekf_state_stamped.hpp>
#include <custom_msgs/msg/position_stamped.hpp>

// classic libraries
#include <iostream>

// linear algebra
#include <eigen/Dense>
#include <eigen/Geometry>

// to add gaussian noise
#include <random>

// ESEKF state vector 
struct state_vector {
    Eigen::Vector3f p; // position
    Eigen::Vector3f v; // velocity
    Eigen::Quaternionf q; // orientation
    Eigen::Vector3f ab; // linear acceleration bias
    Eigen::Vector3f wb; // angular velocity bias
    Eigen::Vector3f g; // gravity
} ;

// ESEKF error-sate vector
struct error_state_vector {
    Eigen::Vector3f p; // position
    Eigen::Vector3f v; // velocity
    Eigen::Vector3f theta; // velocity
    Eigen::Vector3f ab; // linear acceleration bias
    Eigen::Vector3f wb; // angular velocity bias
    Eigen::Vector3f g; // gravity
} ;

class ESEKF : public rclcpp::Node{
public:
    ESEKF();

private:
    // callback functions
    // void gps_imu_callback()
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg);
    void radar_odom_callback(const custom_msgs::msg::RadarOdometryStamped::SharedPtr msg);

    // subscribers
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_gps_;
    rclcpp::Subscription<custom_msgs::msg::RadarOdometryStamped>::SharedPtr sub_radar_odom_;

    // publishers
    rclcpp::Publisher<custom_msgs::msg::EsekfStateStamped>::SharedPtr publisher_;
    rclcpp::Publisher<custom_msgs::msg::PositionStamped>::SharedPtr position_pub_;

    // Sensor Fusion functions
    void nominal_state(Eigen::Vector3f& a_imu, Eigen::Vector3f& w_imu);
    void predict(Eigen::Vector3f a_imu, Eigen::Vector3f omega_imu);
    Eigen::MatrixXf F_x(Eigen::Vector3f a_imu, Eigen::Vector3f w_imu);
    Eigen::MatrixXf Fw();
    Eigen::MatrixXf Qw();
    void true_state();
    void correct(Eigen::Vector3f y_mu, int flag);
    Eigen::Vector3f get_innov(Eigen::Vector3f mu, Eigen::Vector3f h);
    Eigen::Vector3f get_h(int flag);
    Eigen::MatrixXf get_H(int flag);
    void reset();
    Eigen::MatrixXf Jac1(int flag);
    Eigen::MatrixXf Jac2();
    Eigen::MatrixXf Jac3(Eigen::Quaternionf q);
    float glr(float m0, float m1, float r, float u, float S);
    void publish(custom_msgs::msg::EsekfStateStamped);

    // global variables
    
    // States
    state_vector x_;
	error_state_vector dx_; 
    state_vector rollback_state_; // rollback state of the ESEKF in case a fault is detected

    // Covariances
    Eigen::MatrixXf Pg_;
    Eigen::MatrixXf Pr_;
	Eigen::Vector3f innov_;

    // Measurement noise matrices
    Eigen::Matrix3f V_gps_;
    Eigen::Matrix3f V_radar_;

    // Base 2 IMU rotation matrix
    Eigen::Matrix3f R_b2i_;

    // Matrices Fw and Qw
    Eigen::MatrixXf Fw_;
    Eigen::MatrixXf Qw_;

    // Sensor offsets
    Eigen::Vector3f gps_offset_;
    Eigen::Vector3f radar_offset_;

    /*
    variance of position (GPS), acceleration & angular velocity (IMU) - from ros topic
    variance of radar - tuning
    These are used inside the Measurement Noise matrix
    */
    float var_p__ = 0.019599999999999996 ;
    float var_a__ = 0.01;
    float var_w__ = 0.0006;
    float var_r__ = 0.01;

    /*
    Variances that are used inside the covariance matrix P and matrix Qw.
    */
    float var_gps_ = var_p__;
    float var_radar_ = var_r__;
    float var_v_ = var_a__*1000;
    float var_a_ = var_a__*100;
    float var_theta_ = var_w__*0.01;
    float var_w_ = var_w__*0.001;

    // IMU prediction
    int imu_counter_ = 0;
    double t_imu_ = 0;
    double dt_imu_ = 0;

    // GPS correction
    double utm_x_ = 0; // UTM coordinate offset
    double utm_y_ = 0;
    int gps_counter_ = 0; // GPS callback counter
    double t_gps_ = 0;
    double dt_gps_ = 0;
    float v_gps_heading_; // stores the heading velocity of the vehicle based on consecutive gps measurements
    Eigen::Vector3f gps_prev_;

    // radar correction 
    int radar_counter_ = 0;

    // fault detection
    bool gps_radar_fault_ = false; // flag that indicates if there is sth wrong with GPS/Radar
    std::vector<float> r_; // vector that contains the velocity residuals for fault detection
    int window_size_ = 40; // window size for fault detection
    // fault threshold for fault detection
    float rising_thresh_ = 3.037;
    float falling_thresh_ = rising_thresh_/10;
    float fault_thresh_ = rising_thresh_; 
    int fault_counter_ = 0; // fault counter
    int no_fault_counter_ = 0;

    // results from residual analysis on the radar odometry algorithm on the non-fault dataset
    int df_ = 6.652259949237751;
    float m0_ = 0.008440058269720194;
    float S_ = 1;

    // type of fusion
    int fusion_type_;

    // ESEKF time (basically GPS time)
    double t_esekf_ = 0;
};


#endif