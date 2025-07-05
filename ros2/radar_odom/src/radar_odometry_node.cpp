#include "radar_odom/radar_odometry_node.hpp"
#include "radar_odom/radar_odometry_functions.hpp"

float deg2rad(float x){
	return (x * M_PI / 180);
}

float rad2deg(float x){
	return (x * 180 / M_PI);
}
	
// contructor
RadarOdom::RadarOdom() : Node("radar_odometry"){
	std::cout << "Initiating radar odometry..." << std::endl;

	sub_radar_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
	"/radar/front_right/points", 10, std::bind(&RadarOdom::radar_callback, this, std::placeholders::_1));

	publisher_ = this->create_publisher<custom_msgs::msg::RadarOdometryStamped>("/radar/radar_odom", 10);
}

// callback for Radar messages
void RadarOdom::radar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){

	// extract the timestamp of the message
	int sec = msg->header.stamp.sec;
	int nsec = msg->header.stamp.nanosec;

	// // extract the number of radar data points
	int num_radar_points = msg->width;

	// perform outlier rejection to remove false radar readings based on range readings
	// access all radar fields[] with iterators
	sensor_msgs::PointCloud2Iterator<float> 	iter_range(*msg, "range"),
												iter_azimuth(*msg, "azimuth_angle"),
												iter_elevation(*msg, "elevation_angle"),
												iter_speed_radial(*msg, "radial_speed");

	// arrays to save readings
	std::vector<float> theta_list;
	std::vector<float> psi_list;
	std::vector<float> vr_list;
	std::vector<float> r_list;

	// iterate over every instance of every field that we are interested in, save the radial speed and construct the A matrix
	int count_zero = 0;
	int i = 0;
	while (iter_range != iter_range.end()) 
	{
		// extract radar range reading
		r_list.push_back(*iter_range);
		theta_list.push_back(deg2rad(*iter_azimuth)); // CAREFUL! ros angle is in degrees, but the algorithm needs rads
		psi_list.push_back(deg2rad(*iter_elevation)); // CAREFUL! ros angle is in degrees, but the algorithm needs rads
		vr_list.push_back(*iter_speed_radial);
		if (*iter_speed_radial == 0) {
			++count_zero;
		}

		// increment the iterators
		++iter_range; ++iter_speed_radial; ++iter_azimuth;
		// increment the counter
		++i;
	}

    // tuning
    // Ransac
	// minimum number of radar returns
	int num_points_thresh = 5;
    // use sub-samples flag
    bool ransac_flag = true;
    // stoppage criteria
    std::string crit = "num_inliers";
    int max_iter = 50; // maximum iterations
    float stop_error = 0.05; // total error %
    float stop_inl = 0.9; // % of inliers
    // percentage of error allowed to be considered inlier
    float error_threshold = 0.1;

    // Filtering
    bool radial_speed_filtering = false;
    float radial_threshold = 0; // removing all 0 radial speed returns
    bool range_filtering = false;
    float range_threshold = 1; // meters
    bool velocity_filtering = false;
    float velocity_thresholod = 0.5; // percentage of velocity deviation that is allowed between iterations

    // LSQ criteria
    float lsq_inlier_pct_thresh = 0.5; // % of Ransac inliers required to accept LSQ result
    int lsq_inlier_num_thresh = 5; // number of Ransac inliers required to accept LSQ result

    // filter based on radial_speeds returns. There should almost never be 0 radial speeds. If there are, we are probably stationary or sth is wrong
    if (radial_speed_filtering) {
    	remove_radial(vr_list, theta_list, psi_list, r_list, radial_threshold);
        std::cout << "# of radar readings after radial speed filtering: " << theta_list.size() << ". Removed " << num_radar_points - vr_list.size() << " points!" << std::endl;
	}

    // filter based on range, to avoid objects that are too close to the radar sensor and block the view
    if (range_filtering) {
        remove_low_range(vr_list, theta_list, psi_list, r_list, range_threshold);
        std::cout << "# of radar readings after range filtering: " << theta_list.size() << ". Removed " << num_radar_points - vr_list.size() << " points!" << std::endl;
	}

    // step 1 - ransac
    // if false, ransac is applied with 2 random pooints, otherwise, it uses a subsample
    // of the total radar_points
    // criteria can either be "error" or "num_inliers"
    // std::cout << "Step 1: Ransac on the radar pointcloud to find inliers..." << std::endl;
	bool ret_Ransac; 
	std::vector<int> inliers; 
	float ransac_SSR = -1;
	float lsq_SSR = -1;
	std::string criteria;
    std::tie(ret_Ransac, inliers, ransac_SSR) = Ransac(theta_list, psi_list, vr_list, 
												error_threshold, max_iter, stop_error, stop_inl, num_points_thresh, 
												ransac_flag=true, criteria="num_inliers");

	int retflag;
	float v, w;
    // check Ransac gave enough radial speed returns
    if (!ret_Ransac) {
        // std::cout << "Not enough non-zero radial_speed returns!" << std::endl;
        retflag = 1;
	} else {
		// check if Ransac gave an adequate amount of inliers
        if (inliers.size() < (lsq_inlier_pct_thresh*theta_list.size()) || (inliers.size() < lsq_inlier_num_thresh)) {
            // std::cout << "Warning! Not enough inliers!!" << std::endl;
			retflag = 2;
		} else {
            retflag = 0;
            // step 2 - LSQ
    		// std::cout << "Step 2: LSQ on the inlier set..." << std::endl;
			bool LSQ_retval;
			float v_s, alpha;
            std::tie(LSQ_retval, v_s, alpha, lsq_SSR) = inliers_LSQ(inliers, theta_list, psi_list, vr_list);

			// check if LSQ succeedeed
			if (!LSQ_retval) {
		        std::invalid_argument("Sensor linear velocity (v_x) == 0, this should never happen!!");
			} else {
				// step 3 - ego-motion
    			// std::cout << "Step 3: Ego-motion estimation to find velocity profile..." << std::endl;
				float v, w;
				std::tie(v, w) = ego_motion(v_s, alpha, b_, l_, beta_);

				// reject fits that indicate the problem of the largest group being moving targets (too big velocity deviation from previous reading)
				// if (velocity_filtering) { // THIS GOES TO THE SENSOR FUSION MODE
				// 	v_diff = abs(v_prev - v);
				// 	if (v_diff > abs(velocity_thresholod*v_prev) and v_prev != 0) {
				// 		std::cout << "Danger!! Large velocity deviation: "<< v_prev - v << "m/sec! Skipping this iteration..." << std::endl;
				// 		retflag = 3;
				// 	}
				// }

				// publish the returned velocity profile
				auto message = custom_msgs::msg::RadarOdometryStamped();
				message.header = msg->header;
				message.header.frame_id = "base_link";
				message.num_points = num_radar_points;
				message.heading_velocity = v;
				message.yaw_rate = w;
				message.ransac_ssr = ransac_SSR;
				message.lsq_ssr = lsq_SSR;
				publisher_->publish(message);
			}
		}
	}
}


int main(int argc, char *argv[]) {
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<RadarOdom>());
	rclcpp::shutdown();
	return 0;
}
