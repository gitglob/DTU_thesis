#ifndef __QUAT_FUNCTIONS_H__
#define __QUAT_FUNCTIONS_H__

#include <eigen/Dense>
#include <eigen/Geometry>
#include <cmath>

/*
Quaternion functions
*/

// page 25, Sola's
Eigen::Matrix3f quat2matrix(Eigen::Quaternionf q) {
    float qw = q.w();
    float qx = q.x();
    float qy = q.y();
    float qz = q.z();
    Eigen::Matrix3f R;

    R << pow(qw,2) + pow(qx,2) - pow(qy,2) - pow(qz,2), 2*(qx*qy - qw*qz),                              2*(qx*qz + qw*qy),
        2*(qx*qy + qw*qz),                              pow(qw,2) - pow(qx,2) + pow(qy,2) - pow(qz,2),  2*(qy*qz - qw*qx),
        2*(qx*qz - qw*qy),                              2*(qy*qz + qw*qx),                              pow(qw,2) - pow(qx,2) - pow(qy,2) + pow(qz,2);
    
    return (R);
}

// as defined in Sola, page 6
// The product of two rotation quaternions will be equivalent to the rotation a2 + b2i + c2j + d2k (q1)
// followed by the rotation a1 + b1i + c1j + d1k (q0).
Eigen::Quaternionf quat_mult(Eigen::Quaternionf p, Eigen::Quaternionf q) {
    float pw = p.w();
    float px = p.x();
    float py = p.y();
    float pz = p.z();

    float qw = q.w();
    float qx = q.x();
    float qy = q.y();
    float qz = q.z();
    
    float q_w = pw*qw - px*qx - py*qy - pz*qz;
    float q_x = pw*qx + px*qw + py*qz - pz*qy;
    float q_y = pw*qy - px*qz + py*qw + pz*qx;
    float q_z = pw*qz + px*qy - py*qx + pz*qw;
    
    Eigen::Quaternionf retq;
    retq.w() = q_w;
    retq.x() = q_x;
    retq.y() = q_y;
    retq.z() = q_z;

    return (retq);
}    

// convert Rotation matrix to euler angles
bool closeEnough(const float& a, const float& b, const float& epsilon = std::numeric_limits<float>::epsilon()) {
    return (epsilon > std::abs(a - b));
}
Eigen::Vector3f matrix2euler(Eigen::Matrix3f R) {
    float PI = 3.14159265358979323846264f;
    //check for gimbal lock
    if (closeEnough(R(2,0), -1.0f)) {
        float x = 0; //gimbal lock, value of x doesn't matter
        float y = PI / 2;
        float z = x + atan2(R(0,1), R(0,2));
        return { x, y, z };
    } else if (closeEnough(R(2,0), 1.0f)) {
        float x = 0;
        float y = -PI / 2;
        float z = -x + atan2(-R(0,1), -R(0,2));
        return { x, y, z };
    } else { //two solutions exist
        float x1 = -asin(R(2,0));
        float x2 = PI - x1;

        float y1 = atan2(R(2,1) / cos(x1), R(2,2) / cos(x1));
        float y2 = atan2(R(2,1) / cos(x2), R(2,2) / cos(x2));

        float z1 = atan2(R(1,0) / cos(x1), R(0,0) / cos(x1));
        float z2 = atan2(R(1,0) / cos(x2), R(0,0) / cos(x2));

        //choose one solution to return
        Eigen::Vector3f r;
        // for example the "shortest" rotation
        if ((std::abs(x1) + std::abs(y1) + std::abs(z1)) <= (std::abs(x2) + std::abs(y2) + std::abs(z2))) {
            r << x1, y1, z1;
        } else {
            r << x2, y2, z2;
        }
        return r;
    }
}

#endif