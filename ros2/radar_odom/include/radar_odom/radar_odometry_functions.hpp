#ifndef RADAR_ODOMETRY_FUNCTIONS_HPP
#define RADAR_ODOMETRY_FUNCTIONS_HPP

#include <math.h>
#include <tuple> // to return multiple arguments

/*
function to remove all the radar readings with 0 radial speed or radial speed < val
*/
void remove_radial(std::vector<float> vr_list, std::vector<float> theta_list, std::vector<float> psi_list, std::vector<float> r_list, int val=0){
    std::vector<int> remove_idxs;
    
    // identify the indexes of the items that need to be removed
    for (int i=0; i<vr_list.size(); ++i) {
        if (abs(vr_list[i]) <= val) {
            remove_idxs.push_back(i);
        }
    }
            
    // remove the items in reverse order
    for (std::vector<int>::reverse_iterator i = remove_idxs.rbegin(); i != remove_idxs.rend(); ++i ) { 
        vr_list.erase(vr_list.begin() + *i);
        theta_list.erase(theta_list.begin() + *i);
        psi_list.erase(psi_list.begin() + *i);
        r_list.erase(r_list.begin() + *i);
    } 
}

/*
function to remove all the radar readings that have very low range
*/
void remove_low_range(std::vector<float> vr_list, std::vector<float> theta_list, std::vector<float> psi_list, std::vector<float> r_list, int range_thresh){
    std::vector<int> remove_idxs;
    
    // identify the indexes of the items that need to be removed
    for (int i=0; i<r_list.size(); ++i) {
        if (r_list[i] <= range_thresh) {
            remove_idxs.push_back(i);
        }
    }
            
    // remove the items in reverse order
    for (std::vector<int>::reverse_iterator i = remove_idxs.rbegin(); i != remove_idxs.rend(); ++i ) { 
        vr_list.erase(vr_list.begin() + *i);
        theta_list.erase(theta_list.begin() + *i);
        psi_list.erase(psi_list.begin() + *i);
        r_list.erase(r_list.begin() + *i);
    } 
}

/*
Function that performs Ransac on the Radar pointcloud for radar odometry.
Input:
- theta_list : list of azimuth angles
- psi_list : list of elevation angles
- vr_list : list of radial speed values
- error_threshold : velocity residual error for radar return to be considered inlier
- max_iter : maximum RANSAC iterations
- stop_error : total % error for RANSAC to stop
- stop_inl : total % of inliers for RANSAC to stop
- num_pts_thresh : minimum number of radar returns for radar odometry to be possible
- ransac_flag : whether RANSAC will use a subsample of radar returns and LSQ or just 2 points to extract the sensor velocity profile
- criteria : "error" | "num_inliers" criteria stop RANSAC iterations

Returns:
- Ransac flag (bool) : indicates if ransac succeeded
- num_inliers : number of inliers that Ransac found, if it succeeded
- SSR : sum of squared residuals as a metric of how well LSQ fitted the data
*/
std::tuple<bool, std::vector<int>, float> RadarOdom::Ransac(std::vector<float> theta_list, std::vector<float> psi_list, std::vector<float>  vr_list, 
                float error_threshold, int max_iter, float stop_error, float stop_inl, 
                int num_pts_thresh, bool ransac_flag=true, std::string criteria="num_inliers") {

    // extract the number of radar points inside the pointcloud
    int num_radar_points = theta_list.size();

    // vectors that keep the ids of the best inliers and outliers of the Radar points
    std::vector<int> best_inliers, best_outliers;

    // check if there are enough radar returns to perform radar odometry
    if (num_radar_points < num_pts_thresh){ // the threshold number chosen arbitrarily
        return std::make_tuple(false, best_inliers, 100);
    }

    // initialize some necessary variables
    int i = 0;
    int max_inliers = 0;
    float min_e = 9999;
    bool cond = false;
    float v_x, v_y, v_s, alpha;
    float R = 0;
    float SSR = 0;
    float best_R, best_SSR;

    // if the criteria is error %, we need the sum of all the absolute radial speeds
    int sum_vr = 0;
    for (int j=0; j<num_radar_points; ++j){
        sum_vr += abs(vr_list[j]);
    }

    // Ransac loop
    while (true) {
        // exit criteria
        if (cond || i>=max_iter) {
            break;
        }

        // check if we will use 2 points or a subsample during ransac
        if (!ransac_flag) {
            // generate 2 random numbers between 0 and *# of radar points*-1
            int p1 = rand() % num_radar_points;
            int p2 = rand() % num_radar_points;
            while (p2==p1) {
                p2 = rand() % num_radar_points;
            }
            
            // Analytical approach (solving the 2x2 system)
            // with elevation
            float num = (cos(psi_list[p1])*cos(theta_list[p1])*vr_list[p2]) - (cos(psi_list[p2])*cos(theta_list[p2])*vr_list[p1]);
            float denom = (cos(psi_list[p1])*cos(theta_list[p1])*cos(psi_list[p2])*sin(theta_list[p2])) - (cos(psi_list[p1])*sin(theta_list[p1])*cos(psi_list[p2])*cos(theta_list[p2]));
            v_y = num / denom;
            v_x = ( vr_list[p2] - (cos(psi_list[p2])*sin(theta_list[p2])*v_y) ) / (cos(psi_list[p2])*cos(theta_list[p2]));
        } else {
            // generate random subsample of semi-random size m = num_radar_points/n (m>=2)
			std::vector<int> subsample;
            int n_ = (rand() % 4) + 1; // n in [1, 5]
            int m = round(num_radar_points/n_); // 1/n_ of total points is arbitrarily chosen
            // make sure m >= 2
            while (m < 2){
                n_ -= 1;
                m = round(num_radar_points/n_);
            }
            // generate random indeces in [0, num_radar_points)
            while (subsample.size() < m) {
                int p = rand() % num_radar_points;
                // no duplicates - if p already was in the subsample, skip it
                bool dupl_flag = false;
                for (uint k=0; k<subsample.size(); ++k) {
                    if (subsample[k] == p) {
                        dupl_flag = true;
                        break;
                    }
                }
                // if p was not in the subsample, add it
                if (!(dupl_flag)) {
                    subsample.push_back(p);
                }
            }

            // initialize the A matrix and b vector for the least squares prloblem
            Eigen::MatrixXf A(subsample.size(), 2);
            Eigen::VectorXf b(subsample.size());

            // iterate over the entire subsample
            for (uint j=0; j<subsample.size(); ++j){
                // fill the A matrix
                // using just the azimuth angle
                // A(j,0) = cos(theta_list[subsample[j]]);
                // A(j,1) = sin(theta_list[subsample[j]]);
                // using azimuth + elevation
                A(j,0) = cos(psi_list[subsample[j]])*cos(theta_list[subsample[j]]);
                A(j,1) = cos(psi_list[subsample[j]])*sin(theta_list[subsample[j]]);
                b(j) =  vr_list[subsample[j]];
            }

            // solve the LSQ problem
            Eigen::VectorXf x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
            // get an estimate of how good the fit was
            std::tie(R, SSR) = r_squared(A, b, x);

            // the solution is going to be used to extract the sensor velocity and heading direction
            // v_x and v_y are the radar sensor linear velocities in the x,y directions
            v_x = x(0);
            v_y = x(1);
        }

        // now find the sensor velocity and yaw rate
        if (v_x != 0){ // avoid division by 0 (non-moving vehicle)
            alpha = atan2(v_y, v_x);
            v_s = -v_x/cos(alpha);

            // calculate the error of the current fit and the number of outliers/inliers
            int sum_e = 0;
            std::vector<int> current_inliers; // index of all the inlier radar readings
            std::vector<int> current_outliers; // index of all the outlier radar readings
            for (int j=0; j<num_radar_points; ++j) {	
                // measure the error between the radial speed and the sensor velocity
                float e = std::abs(vr_list[j] - v_s);

                // if the current radar reading has a small error, compared to the velocity profile of the randomly chosen subsample/2 points, it is an inlier
                if (e<error_threshold) {
                    current_inliers.push_back(j);
                } else {
                    current_outliers.push_back(j);
                }

                // count the accumulated error to determine best fit
                sum_e += e;
            }

            // keep the best fit in terms of minimum error or maximum inliers (this is the same most of the times)
            if ( criteria.compare("num_inliers") == 0 && current_inliers.size() > max_inliers) {
                // keep ransac berformance
                best_R = R;
                best_SSR = SSR;
                // save the maximum number of inliers
                max_inliers = current_inliers.size();
                min_e = sum_e;
                // make sure that the best_outliers and best_inliers are empty vectors
                while (!best_inliers.empty()) {
                    best_inliers.pop_back();
                }
                while (!best_outliers.empty()) {
                    best_outliers.pop_back();
                }

                // now append the new best values
                for (uint j=0; j<current_inliers.size(); ++j) {
                    best_inliers.push_back(current_inliers[j]);
                }
                for (uint j=0; j<current_outliers.size(); ++j) {
                    best_outliers.push_back(current_outliers[j]);
                }
            }
            if ( criteria.compare("error") == 0 && sum_e < min_e) {
                // keep ransac berformance
                best_R = R;
                best_SSR = SSR;
                // save the maximum number of inliers
                max_inliers = current_inliers.size();
                min_e = sum_e;
                // make sure that the best_outliers and best_inliers are empty vectors
                while (!best_inliers.empty()) {
                    best_inliers.pop_back();
                }
                while (!best_outliers.empty()) {
                    best_outliers.pop_back();
                }

                // now append the new best values
                for (uint j=0; j<current_inliers.size(); ++j) {
                    best_inliers.push_back(current_inliers[j]);
                }
                for (uint j=0; j<current_outliers.size(); ++j) {
                    best_outliers.push_back(current_outliers[j]);
                }
            }
        }
        
        // increase the iteration counter
        ++i;

        // determine which is the first condition to break the loop, based on the criteria given
        if (criteria.compare("error") == 0){
            cond = min_e < stop_error*sum_vr; // the minimum error being less than (stop_error)% of the sum of radial velocities
        } else {
            cond = max_inliers > round(stop_inl*num_radar_points); // the max inliers being more than (stop_in)% of the points 
        }
    } // end Ransac loop

    return std::make_tuple(true, best_inliers, best_SSR);
}

/*
Function that performs LSQ to the subsample of radar points (inliers) given by Ransac
*/
std::tuple<bool, float, float, float> RadarOdom::inliers_LSQ(std::vector<int> best_inliers, 
                                                    std::vector<float> theta_list, 
                                                    std::vector<float> psi_list, 
                                                    std::vector<float> vr_list) {
    // initialize the A matrix and b vector for the least squares prloblem
    Eigen::MatrixXf A(best_inliers.size(), 2);
    Eigen::VectorXf b(best_inliers.size());

    // iterate over all the inliers
    for (uint i=0; i<best_inliers.size(); ++i){
        // fill the A matrix
        // using just the azimuth angle
        // A(i,0) = cos(theta_list[best_inliers[i]]);
        // A(i,1) = sin(theta_list[best_inliers[i]]);
        // using azimuth + elevation
        A(i,0) = cos(psi_list[best_inliers[i]])*cos(theta_list[best_inliers[i]]);
        A(i,1) = cos(psi_list[best_inliers[i]])*sin(theta_list[best_inliers[i]]);
        b(i) =  vr_list[best_inliers[i]];
    }

    // solve the LSQ problem
    Eigen::VectorXf x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

    // get an estimate of how good the fit was
    float R, SSR;
    std::tie(R, SSR) = r_squared(A, b, x);
    
    // the solution is going to be used to extract the sensor velocity and heading direction
    // v_x and v_y are the radar sensor linear velocities in the x,y directions w.r.t. the local radar sensor coordinate frame
    float v_x = x(0);
    float v_y = x(1);

    float a;
    float v_s;
    bool retval;
    if (v_x != 0) { 
        // calculate sensor velocity and direction
        a = atan2(v_y, v_x);
        v_s = -v_x/cos(a);
        retval = true;
    } else {
        retval = false;
    }

    return std::make_tuple(retval, v_s, a, SSR);
}

/*
Function that extracts the vehicle radial velocity, using the sensor velocity and direction.
On top of that, it calculates the vehicly yaw rate, using the Ackerman condition.
*/
std::tuple<float, float> RadarOdom::ego_motion(float v_s, float alpha, float b_, float l_, float beta_) {
    // if the vehicle is moving, proceed to find the sensor and vehicle velocity and yaw rates
    float v, omega;    
    if (v_s != 0) {
        // now calculate the vehicle velocity and yaw rate based on the Ackerman condition
        v = (cos(alpha + beta_) - ((b_/l_)*sin(alpha+beta_)))*v_s;
        omega = sin(alpha+beta_)/l_ * v_s;

    } else {
        v = 0;
        omega = 0;
    }

    return std::make_tuple(v, omega);
}

#endif