// custom packages
#include "sensor_fusion/esekf_node.hpp"
#include "sensor_fusion/esekf_functions.hpp"

// GeographicLib for handling GPS messages
#include <GeographicLib/GeoCoords.hpp>
#include <GeographicLib/UTMUPS.hpp>

// to throw exception error
#include <stdexcept>

#define LOW 0.00000000000001 
#define GRAV 9.80665
#define PI 3.14159265

ESEKF::ESEKF() : Node("gps_imu_radar_fusion"){
	std::cout << "Initiating GPS-IMU-Radar fusion..." << std::endl;
	// DEBUG
	// Eigen::Vector3f v(2, 2, 2);
	// std::cout << v.norm() << std::endl;
	// std::cout << v.squaredNorm() << std::endl;
	
	// initialize gps and radar sensor translational offsets
	gps_offset_ << -0.01, 0, 2.13;
    radar_offset_ << 0.915, 0, 0.895;

	// initialize Covariance matrices
	Pg_.setZero(18,18);
	Pg_.diagonal() << var_gps_, var_gps_, var_gps_,
						var_v_, var_v_, var_v_,
						var_theta_, var_theta_, var_theta_,
						var_a_, var_a_, var_a_,
						var_w_, var_w_, var_w_,
						LOW, LOW, LOW;
	Pr_.setZero(18,18);
	Pr_.diagonal() << var_gps_, var_gps_, var_gps_,
						var_radar_, var_radar_, var_radar_,
						var_theta_, var_theta_, var_theta_,
						var_a_, var_a_, var_a_,
						var_w_, var_w_, var_w_,
						LOW, LOW, LOW;

	// initialize state
	x_.p.setZero();
	x_.v.setZero();
	// this is the ones used in Jupyter, BUT they respond to a completely different coordinate frame!!!
	// x_.q.w() = 0.99680171;
	// x_.q.vec() << 0, 0, 0.07991469;
	double roll = 0, pitch = 0, yaw = 0.16; // We need an orientation measurement device to set it correctly!
	Eigen::Quaternionf q;
	q = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX())
		* Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())
		* Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
	x_.q = q;
	std::cout << "Initial orientation: \nQuaternion: " << std::endl << x_.q.coeffs() << std::endl;
	std::cout << "Euler: " << x_.q.toRotationMatrix().eulerAngles(0, 1, 2) << std::endl;
	x_.ab << 0.10895214, -0.80556905, -0.09362127;
	x_.wb << -0.01907755,  0.01299578, -0.00510084;
	x_.g << 0, 0, -GRAV;

	// initialize rollback state
	rollback_state_ = x_;

	// initialize error state
	dx_.p.setZero();
	dx_.v.setZero();
	dx_.theta.setZero();
	dx_.ab.setZero();
	dx_.wb.setZero();
	dx_.g.setZero();

	// initialize Measurement Noise
	V_gps_ << var_p__, 0, 0,
			0, var_p__, 0,
			0, 0, var_p__;
	V_radar_ << var_r__, 0, 0,
			0, var_r__, 0,
			0, 0, var_r__;

	// initialize Rotation Matrices
	R_b2i_ = Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX())
			* Eigen::AngleAxisf(0,  Eigen::Vector3f::UnitY())
			* Eigen::AngleAxisf(1.5708, Eigen::Vector3f::UnitZ());

	// initialize Fw and Qw
	Fw_ = Fw();
	Qw_ = Qw();

	// initialize first GPS measurement
	gps_prev_ << 0, 0, 0;

	// initialize innovation
	innov_.setZero();

	// initialize state publisher
	publisher_ = this->create_publisher<custom_msgs::msg::EsekfStateStamped>("/sensor_fusion/state", 10);

	// publish the initial state
	auto message = custom_msgs::msg::EsekfStateStamped();
	publish(message);

	// gps position publisher
	position_pub_ = this->create_publisher<custom_msgs::msg::PositionStamped>("/gps/position", 10);

    // publish the initial gps position in map frame
    auto message1 = custom_msgs::msg::PositionStamped();
    message1.header.frame_id = "INIT";
    message1.x = 0;
    message1.y = 0;
    position_pub_->publish(message1);

	/* define type of fusion: 
	0 -> gps + radar + imu
	1 -> gps + imu
	2 -> radar + imu 
	*/
	fusion_type_ = 0;
	// initialize callback functions
	sub_gps_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
		"/gps/fix", 10, std::bind(&ESEKF::gps_callback, this, std::placeholders::_1));
}

/*
IMU callback function which perform the prediction step of the ESEKF.
*/
void ESEKF::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg){
	// std::cout << "\t\tReceived IMU data..." << std::endl;

	// increase IMU callback counter
	imu_counter_ += 1;
    
	// extract IMU data
	int sec = msg->header.stamp.sec;
	double nanosec = msg->header.stamp.nanosec;
	double t_imu = sec+(nanosec/1000000000);
	// initialize IMU timing if that's the first reading
	if (imu_counter_ == 1) {
		t_imu_ = t_imu;
	}	
	// calculate dt
	dt_imu_ = t_imu - t_imu_;

	// std::cout << "Sec: " << sec << " Nanosec: " << nanosec << std::endl;
	// std::cout << "\ntime: " << t_imu_ << std::endl;
	// std::cout << "IMU dt: " << dt_imu_ << std::endl;
	Eigen::Vector3f a(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
	Eigen::Vector3f w(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

	// check for timing error and perform prediction step
	if (dt_imu_ > 0 && imu_counter_>1 && dt_imu_ < 0.1) {
		nominal_state(a, w);
		predict(a, w);

		// publish the state
		auto message = custom_msgs::msg::EsekfStateStamped();
		message.header = msg->header;
		message.header.frame_id = "IMU";
		publish(message);
	}

	// if there was a timing error, this IMU measurement doesn't count
	if (dt_imu_ > 0.1) {
		imu_counter_ -= 1;
	} else {
		// keep the current timestamp
		t_imu_ = t_imu;
	}
}

void ESEKF::gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg){
	// std::cout << "\t\tReceived GPS data..." << std::endl;

	// increase GPS callback counter
	gps_counter_ += 1;

	// if we have a radar+imu sensor fusion, we only need the GPS once
	if (!((fusion_type_ == 2) && (gps_counter_>1))) {
		// extract GPS data
		int sec = msg->header.stamp.sec;  // - 1635257600;
		double nanosec = msg->header.stamp.nanosec;
		double t_gps = sec+(nanosec/1000000000);

		// initialize GPS timing if that's the first reading
		if (gps_counter_ == 1) {
			t_gps_ = t_gps;
		}	
		dt_gps_ = t_gps - t_gps_;
		t_esekf_ += dt_gps_;
		std::cout << "TIME: " << t_esekf_ << std::endl;

		double lat = msg->latitude;
		double lon = msg->longitude;
		// double alt = msg->altitude;

		// convert GPS [lat,long,alt] to [x,y,z]
		// https://geographiclib.sourceforge.io/C++/doc/classGeographicLib_1_1UTMUPS.html#ac333eaff84cc487fee67440de3383bf7
		int zone;
		bool northp;
		double x, y, gamma, k;
		GeographicLib::UTMUPS::Forward(lat, lon, zone, northp, x, y, gamma, k);
		// I didn't find a way to convert altitude to z, investigate if need be
		// std::cout << "[x, y]" << x << " , " << y << std::endl;

		// set the UTM coordinate offset
		if (gps_counter_ == 1) {
			utm_x_ = x;
			utm_y_ = y;

			// only initialize the other subscribers after we 've set the map frame
			sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
			"/ouster/imu", 10, std::bind(&ESEKF::imu_callback, this, std::placeholders::_1));
			if ((fusion_type_ == 0) || (fusion_type_ == 2)) {
				sub_radar_odom_ = this->create_subscription<custom_msgs::msg::RadarOdometryStamped>(
				"/radar/radar_odom", 10, std::bind(&ESEKF::radar_odom_callback, this, std::placeholders::_1));
			}
		}

		// remove the base UTM offset // this includes the gps offset
		x = x - utm_x_;
		y = y - utm_y_;

		// remove the GPS offset
		// x = x - gps_offset_(0);
		// y = y - gps_offset_(1);

		// add gps "small ramp" fault between 20 and 40 seconds
		if (t_esekf_ >= 20 && t_esekf_ <= 40) {
			// Define random generator with Gaussian distribution
			const double mean = 0.5;
			const double stddev = 0.1;
			std::mt19937 generator(std::random_device{}());
			std::normal_distribution<double> dist(mean, stddev);

			// add noise to the signal
			x = x + dist(generator);

			std::cout << "Adding GPS noise... x = x + " << dist(generator) << std::endl;
		}

		// get the gps velocity measurement based on the current and previous gps position measurement
		Eigen::Vector3f y_mu(x, y, 0);
		// std::cout << "GPS measurement: " << y_mu << std::endl;
		if (dt_gps_>0 && gps_counter_>1) { // the second condition is to ensure that we have had at least one GPS measurement before that
			Eigen::Vector3f v_gps_mu = (y_mu - gps_prev_)/dt_gps_;
			// convert to heading velocity
			v_gps_heading_ = sqrt(v_gps_mu(0)*v_gps_mu(0) + v_gps_mu(1)*v_gps_mu(1));
		} else {
			v_gps_heading_ = 0;
		}

		// publish gps position in map frame
		auto message1 = custom_msgs::msg::PositionStamped();
		message1.header = msg->header;
		message1.header.frame_id = "map";
		message1.x = y_mu(0);
		message1.y = y_mu(1);
		position_pub_->publish(message1);

		// if there is nothing wrong with the GPS, perform correction
		// if ((int(gps_counter_/100) % 2) == 0) {
		if (true) {
			if (!gps_radar_fault_) {
				std::cout << "GPS correction..." << std::endl;
				// perform correction step
				correct(y_mu, 0);
				true_state();
				reset();
			}
		}

		// update the "previous" gps reading as the current one
		gps_prev_ = y_mu;

		// publish the state
		auto message = custom_msgs::msg::EsekfStateStamped();
		// don't add timing, as GPS and IMU have different timestamps smh
		message.header = msg->header; 
		message.header.frame_id = "GPS";
		publish(message);

		// keep this timestamp to calculate gps velocities
		t_gps_ = t_gps;
	}
}

void ESEKF::radar_odom_callback(const custom_msgs::msg::RadarOdometryStamped::SharedPtr msg){
	// std::cout << "\t\tReceived radar odometry data..." << std::endl;
    
	// increase Radar callback counter
	radar_counter_ += 1;

	// extract Radar Odometry data
	float heading_vel = msg->heading_velocity;

	// perform fault detection
	float vel_resid = heading_vel - v_gps_heading_;
	// check if we reached the end of the window
	if (r_.size() < window_size_) {
		r_.push_back(vel_resid);
	} else {
		// iterate over every sample of the window and find the max g value
		std::vector<float> g_list;
		for (int i=0; i<window_size_; ++i) {
			float subsum = 0;
			for (int j=i; j<window_size_; ++j) {
				// extract the residual mean of the current subsample
				float s1 = 0; // sum
				for (int k=j; k<window_size_; ++k) {
					s1 += r_[k];
				}
				float m1 = s1/(window_size_-j); // mean
				// find the glr term and add it to the sum of the current subsample
				float temp = glr(m0_, m1, r_[j], df_, S_);
				subsum += temp;
			}
			g_list.push_back(((1+df_)/2)*subsum);
		}

		// keep the max value of g
		float g = g_list[0];
		for(auto it = std::begin(g_list); it != std::end(g_list); ++it) {
			if (*it > g) {
				g = *it;
			}
		}
				
		// compare to threshold
		// std::cout << "g vs h: " << g << " | " << fault_thresh_ << std::endl;
		if (g > fault_thresh_) {
			// fault detected
			gps_radar_fault_ = true;
			std::cout << "GPS or Radar just got fucked! Investigate 8-|" << std::endl;
			std::cout << "g vs h: " << g << " | " << fault_thresh_ << std::endl;
			// throw std::invalid_argument( "FAULT DETECTED" );

			// increase fault counter
			fault_counter_ += 1;
			// reset no fault counter
			no_fault_counter_ = 0;

			// if this is the first fault, rollback the state
			if (fault_counter_ == 1) {
				x_ = rollback_state_;
				// enable falling threshold
				fault_thresh_ = falling_thresh_;
			}
		} else {
			// increase no fault counter
			no_fault_counter_ += 1;

			// set rollback state as the current one
			rollback_state_ = x_;
			
			// if 5 consecutive samples presented no fault, then the vehicle is fault-free
			if (no_fault_counter_ == 5) {
				// no fault detected
				gps_radar_fault_ = false;

				// reset fault counter
				fault_counter_ = 0;

				// enable rising threshold
				fault_thresh_ = rising_thresh_;
			}
		}

		// pop the first residual of the residual vector
		r_.erase(r_.begin());
	}

	// convert polar velocity to cartesian using the current orientation
	// auto e = x_.q.toRotationMatrix().eulerAngles(0, 1, 2); // convert quaternion to euler
	Eigen::Vector3f e = matrix2euler(quat2matrix(x_.q));
	// std::cout << "Orientation:\n" << e << std::endl;
	float v_z = x_.v(2); // we have no v_z information from radar_odometry, as it operates in 2d
	float v_x = cos(e(2))*heading_vel;
	float v_y = sin(e(2))*heading_vel;
	Eigen::Vector3f v_mu(v_x, v_y, 0);
	// std::cout << "Orientation:" << e*(180.0/PI) << " (degrees) \nv_mu: " << heading_vel << "\nv: \n" << v_mu << std::endl;

	// Perform correction (here, if we had a third velocity estimation method, we would check if there is a fault in the radar as well)
	correct(v_mu, 2);
	true_state();
	reset();

	// publish the state
	auto message = custom_msgs::msg::EsekfStateStamped();
	message.header = msg->header;
	message.header.frame_id = "Radar";
	publish(message);
}

// publishes the ESEKF state vector to /sensor_fusion/state
void ESEKF::publish(custom_msgs::msg::EsekfStateStamped message) {
	message.p.x = x_.p(0);
	message.p.y = x_.p(1);
	message.p.z = x_.p(2);
	message.v.x = x_.v(0);
	message.v.y = x_.v(1);
	message.v.z = x_.v(2);
	message.q.w = x_.q.w();
	message.q.x = x_.q.x();
	message.q.y = x_.q.y();
	message.q.z = x_.q.z();
	message.ab.x = x_.ab(0);
	message.ab.y = x_.ab(1);
	message.ab.z = x_.ab(2);
	message.wb.x = x_.wb(0);
	message.wb.y = x_.wb(1);
	message.wb.z = x_.wb(2);
	message.g.x = x_.g(0);
	message.g.y = x_.g(1);
	message.g.z = x_.g(2);
	publisher_->publish(message);
}

int main(int argc, char *argv[]) {
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<ESEKF>());
	rclcpp::shutdown();
	return 0;
}


