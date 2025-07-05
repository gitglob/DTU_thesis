# this represents the return values of the radar_odometry algorithm
# it also includes the timestamp of when the radar_points were obtained

std_msgs/Header header
int64 num_points
float64 heading_velocity
float64 yaw_rate
float64 ransac_ssr
float64 lsq_ssr