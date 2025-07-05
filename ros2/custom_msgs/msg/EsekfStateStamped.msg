# this represents the state vector of the ESEKF for the IMU-GPS-Radar sensor fusion
# inside the header, the "frame" entry denotes which sensor returned the state vector, not the coordinate frame
# the coordinate frame is always "base_link"

std_msgs/Header header
geometry_msgs/Point p
geometry_msgs/Point v
geometry_msgs/Quaternion q
geometry_msgs/Point ab
geometry_msgs/Point wb
geometry_msgs/Point g
# geometry_msgs/Point pg_i
# geometry_msgs/Quaternion qg_i
# geometry_msgs/Point pr_i
# geometry_msgs/Quaternion qr_i

