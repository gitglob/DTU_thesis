#ifndef ESEKF_FUNCTIONS_HPP
#define ESEKF_FUNCTIONS_HPP

// to return multiple variables
#include <tuple>
// mathematical operations
#include <math.h>
// quaternion operations
#include "sensor_fusion/quaternion_functions.hpp"

/* 
Functions fot the PREDICTION step of the Kalman filter, which utilizes the IMU
*/

/*
Performs update of the nominal state
*/
void ESEKF::nominal_state(Eigen::Vector3f& a_imu, 
                          Eigen::Vector3f& w_imu) {
    // find the current robot rotation matrix
    Eigen::Matrix3f R_robot = x_.q.toRotationMatrix();
    
    // extract the current Rotation (IMU frame rotation + robot rotation)
    Eigen::Matrix3f R = R_b2i_.transpose()*R_robot;
    
    // angular velocity
    Eigen::Vector3f w_theta = R*(w_imu - x_.wb)*dt_imu_;
    float w_norm = w_theta.norm();
    float q_w = cos(w_norm/2);
    Eigen::Vector3f q_xyz = (w_theta/w_norm)*sin(w_norm/2);
    Eigen::Quaternionf q_omega(q_w, q_xyz(0), q_xyz(1), q_xyz(2));

    // apply dynamics to position, velocity and orientation
    x_.p = x_.p + x_.v*dt_imu_ + 0.5*(R*(a_imu-x_.ab) + x_.g)*(dt_imu_*dt_imu_);
    x_.v = x_.v + (R * (a_imu - x_.ab) + x_.g)*dt_imu_;
    
    x_.q = quat_mult(x_.q, q_omega); // quaternion multiplication    
    x_.q.normalize();
}

// predict the next state
void ESEKF::predict(Eigen::Vector3f a_imu, Eigen::Vector3f omega_imu) {
    
    Eigen::MatrixXf Fx = F_x(a_imu, omega_imu);

    Eigen::MatrixXf pg1 = (Fx * Pg_) * (Fx.transpose());
    Eigen::MatrixXf pr1 = (Fx * Pr_) * (Fx.transpose());
    Eigen::MatrixXf p2 = (Fw_ * Qw_*(dt_imu_*dt_imu_)) * (Fw_.transpose());

    // covariance matrix update
    Pg_ = pg1 + p2;
    Pr_ = pr1 + p2;
}

// calculate state transition matrix (including drift - last 6 elements)
Eigen::MatrixXf ESEKF::F_x(Eigen::Vector3f a_imu, 
                        Eigen::Vector3f w_imu) {
    // find the current robot rotation matrix
    Eigen::MatrixXf R_robot = x_.q.toRotationMatrix();
    
    // extract the current Rotation (IMU frame rotation + robot rotation)
    Eigen::MatrixXf R = R_b2i_.transpose() * R_robot;

    // linear acceleration
    Eigen::Vector3f real_a = a_imu - x_.ab;
    // angular velocity
    Eigen::Vector3f w_theta = R*(w_imu - x_.wb)*dt_imu_;
    
    // extract dq from angular velocity
    float w_norm = w_theta.norm();
    float q_w = cos(w_norm/2);
    Eigen::Vector3f q_xyz = (w_theta/w_norm)*sin(w_norm/2);
    Eigen::Quaternionf q_omega(q_w, q_xyz(0), q_xyz(1), q_xyz(2));  
    
    // convert dq to rotation matrix
    Eigen::Matrix3f Rw = q_omega.toRotationMatrix();

    // shortcuts
    // A = skew(real_a);
    Eigen::Matrix3f A;
    A << 0, -real_a(2), real_a(1),
        real_a(2), 0, -real_a(0),
        -real_a(1), real_a(0), 0;
    Eigen::MatrixXf RA = R * A;
    
    // initialize matrix F
    Eigen::MatrixXf F(18,18);
    F.setZero(18,18);

    // fill the matrix with the kinematic equations
    // matrix.block(i,j,p,q) - Block of size (p,q), starting at (i,j)	

    // F[0:3, 0:3] = I3
    F.block(0,0, 3,3) = Eigen::MatrixXf::Identity(3, 3);
    // F[0:3, 3:6] = I3*dt
    F.block(0,3, 3,3) = Eigen::MatrixXf::Identity(3, 3)*dt_imu_;

    // F[3:6, 3:6] = I3
    F.block(3,3, 3,3) = Eigen::MatrixXf::Identity(3, 3);
    // F[3:6, 6:9] = -RA*dt
    F.block(3,6, 3,3) = -RA*dt_imu_;
    // F[3:6, 9:12] = -R*dt
    F.block(3,9, 3,3) = -R*dt_imu_;
    // F[3:6, 15:18] = I3*dt
    F.block(3,15, 3,3) = Eigen::MatrixXf::Identity(3, 3)*dt_imu_;

    // F[6:9, 6:9] = Rw.transpose()
    F.block(6,6, 3,3) = Rw.transpose(); 
    // F[6:9, 12:15] = -I3*dt
    F.block(6,12, 3,3) = Eigen::MatrixXf::Identity(3, 3)*(-dt_imu_);

    // F[9:18, 9:18] = np.eye(9)
    F.block(9,9, 9,9) = Eigen::MatrixXf::Identity(9, 9);
        
    return (F);
}


// calculate noise state transition matrix
Eigen::MatrixXf ESEKF::Fw() {    
    Eigen::MatrixXf Fw(18,12);
    Fw.setZero(18,12);

    // change the rest of the matrix elements
    Fw.block(3,0, 3,3) = Eigen::MatrixXf::Identity(3, 3);
    Fw.block(6,3, 3,3) = Eigen::MatrixXf::Identity(3, 3);
    Fw.block(9,6, 3,3) = Eigen::MatrixXf::Identity(3, 3);
    Fw.block(12,9, 3,3) = Eigen::MatrixXf::Identity(3, 3);
    
    return (Fw);
}

// calculate noise covariance
Eigen::MatrixXf ESEKF::Qw() {
    Eigen::MatrixXf Qw(12,12);
    Qw.setZero(12,12);
    
    Qw.block(0,0, 3,3) = Eigen::MatrixXf::Identity(3, 3)*var_v_;
    Qw.block(3,3, 3,3) = Eigen::MatrixXf::Identity(3, 3)*var_theta_;
    Qw.block(6,6, 3,3) = Eigen::MatrixXf::Identity(3, 3)*var_a_;
    Qw.block(9,9, 3,3) = Eigen::MatrixXf::Identity(3, 3)*var_w_;

    return (Qw);
}


/* 
Functions fot the CORRECTION step of the Kalman filter, which utilizes the Radar and the GPS
*/

// calculate the true state based on the nominal state and the error state
void ESEKF::true_state() {     
    Eigen::Quaternionf dq = Eigen::AngleAxisf(dx_.theta(0), Eigen::Vector3f::UnitX())
                            * Eigen::AngleAxisf(dx_.theta(1), Eigen::Vector3f::UnitY())
                            * Eigen::AngleAxisf(dx_.theta(2), Eigen::Vector3f::UnitZ());
    
    x_.p = x_.p + dx_.p;
    x_.v = x_.v + dx_.v;
    x_.q = quat_mult(x_.q, dq); // quaternion multiplication
    x_.q.normalize();
    x_.ab = x_.ab + dx_.ab;
    x_.wb = x_.wb + dx_.wb;
    x_.g = x_.g + dx_.g;
}

// flag: 0 -> position sensor, 2 -> velocity from radar odom
void ESEKF::correct(Eigen::Vector3f y_mu, int flag) {   
    
    // Kalman gain
    Eigen::MatrixXf H = get_H(flag); // jacobian
    Eigen::MatrixXf K;
    if (flag==0) {
        K = (Pg_*H.transpose()) * ((H*Pg_*H.transpose()) + V_gps_).inverse(); // Kalman gain
    } else {
        K = (Pr_*H.transpose()) * ((H*Pr_*H.transpose()) + V_radar_).inverse(); // Kalman gain
    }
    
    // error
    Eigen::Vector3f h = get_h(flag);
    innov_ = get_innov(y_mu, h);
    Eigen::VectorXf dx_vec = K * innov_;
    dx_.p = dx_vec.head(3);
    dx_.v = dx_vec.segment(3, 3);
    dx_.theta = dx_vec.segment(6, 3);
    dx_.ab = dx_vec.segment(9, 3);
    dx_.wb = dx_vec.segment(12, 3);
    dx_.g = dx_vec.segment(15, 3);
    
    // covariance update
    if (flag==0) {
        Pg_ = (Eigen::MatrixXf::Identity(18,18) - (K*H)) * Pg_;
    } else {
        Pr_ = (Eigen::MatrixXf::Identity(18,18) - (K*H)) * Pr_;
    }
}

// flag: 0 -> position sensor, 1 -> orientation sensor, 2 -> velocity sensor
Eigen::Vector3f ESEKF::get_innov(Eigen::Vector3f mu, Eigen::Vector3f h) {
    Eigen::Vector3f inno = mu - h;
    
    return(inno);
}

// flag: 0 -> position sensor, 2 -> velocity sensor
Eigen::Vector3f ESEKF::get_h(int flag) {  
    if ( flag == 0 ) {
        // Eigen::Matrix3f R_qi = x_.qg_i.toRotationMatrix();
        // return (R_qi*x_.p + x_.pg_i);
        return (x_.p);
    } else {        
        return (x_.v);
    }
}

// get the full H matrix
Eigen::MatrixXf ESEKF::get_H(int flag) {
    Eigen::MatrixXf J1 = Jac1(flag);
    Eigen::MatrixXf J2 = Jac2();
    Eigen::MatrixXf J = J1 * J2;

    return (J);
}

// ESEKF reset
void ESEKF::reset() {   
    // reset dx
    dx_.p.setZero();
    dx_.v.setZero();
    dx_.theta.setZero();
    dx_.ab.setZero();
    dx_.wb.setZero();
    dx_.g.setZero();
}

// Jacobians
// Jacobian 1
Eigen::MatrixXf ESEKF::Jac1(int flag) {
    Eigen::MatrixXf J1(3,19);
    J1.setZero(3,19);
    if (flag == 0) { // position measurement
        J1.block(0,0, 3,3) = Eigen::MatrixXf::Identity(3, 3);
    } else if (flag == 2) { // velocity measurement
        J1.block(0,3, 3,3) = Eigen::MatrixXf::Identity(3, 3);
    }
        
    return(J1);
}

// Jacobian 2
Eigen::MatrixXf ESEKF::Jac2() {
    // initialize zero matrix
    Eigen::MatrixXf J2(19,18);
    J2.setZero(19,18);
    
    J2.block(0,0, 6,6) = Eigen::MatrixXf::Identity(6, 6);
    J2.block(6,6, 4,3) = Jac3(x_.q);
    J2.block(10,9, 9,9) = Eigen::MatrixXf::Identity(9, 9);

    return (J2);
}

// Jacobian 3
Eigen::MatrixXf ESEKF::Jac3(Eigen::Quaternionf q) {
    // std::cout << "[" << q.w() << " , " << q.x() << " , " << q.y() << " , " << q.z() << "]" << std::endl;
    Eigen::MatrixXf M(4,3);
    M << -q.x(), -q.y(), -q.z(),
        q.w(), -q.z(), q.y(),
        q.z(), q.w(), -q.x(),
        -q.y(), q.x(), q.w();

    return (M*0.5);
}

// Fault detection

/*
m0 : mean of non-fault residuals
m1 : mean of window sample subsample
r : residual value of every subsample item
u : degrees of freedom of non-fault residuals
S : 1, because of univariate
*/
float ESEKF::glr(float m0, float m1, float r, float u, float S=1) {
    float l1 = log(1 + (1/u)*(r-m1)*(1/S)*(r-m1));
    float l0 = log(1 + (1/u)*(r-m0)*(1/S)*(r-m0));
    
    return (-l1 + l0);
}


#endif