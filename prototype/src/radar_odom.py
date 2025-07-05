import numpy as np
from random import randrange


# function to remove all the radar readings with 0 radial speed or radial speed < val
def remove_zero_radial(vr, the, ra, val=0):
    remove_idxs = []
    keep_idxs = []
    
    for j in range(len(vr)):
        if abs(vr[j]) <= val:
            remove_idxs.append(j)
        else:
            keep_idxs.append(j)
            
    vr = [vr[j] for j in keep_idxs]
    the = [the[j] for j in keep_idxs]
    ra = [ra[j] for j in keep_idxs]

    return vr, the, ra, remove_idxs

# function to remove all the radar readings that have very low range
def remove_low_range(vr, the, ra, range_thresh):
    remove_idxs = []
    keep_idxs = []
    
    for j in range(len(vr)):
        if ra[j] < range_thresh:
            remove_idxs.append(j)
        else:
            keep_idxs.append(j)
            
    vr = [vr[j] for j in keep_idxs]
    the = [the[j] for j in keep_idxs]
    ra = [ra[j] for j in keep_idxs]

    return vr, the, ra, remove_idxs

# part 1 - Ransac function
def Ransac(theta, v_r, error_threshold, max_iter, stop_e, stop_in, ransac_flag=True, criteria="num_inliers"):    
    num_radar_points = len(theta)
    # check if we do not have enough non-zero radial_speed returns
    # c1_ = np.count_nonzero(v_r) < round(num_radar_points/8)
    c2_ = np.count_nonzero(v_r) < 5 # 5 is chosen arbitrarily
    if c2_:
        return False, [], [], 0, 100, 100, None
        
    i = 0
    max_inliers = 0
    min_e = 9999
    best_inliers = []
    best_outliers = []
    v_s = None
    alpha = None
    
    # if the criteria is error %, we need the sum of all the absolute radial speeds
    sum_vr = 0
    for j in range(num_radar_points):
        sum_vr += abs(v_r[j])
    
    # Ransac loop
    while (True):
        # exit criteria
        if i>0:
            if cond or i>=max_iter:
                break
        
        # check if we will use 2 points or a subsample during ransac
        if (not ransac_flag):
            # generate random number between 0 and *# of radar points*-1
            p1 = randrange(num_radar_points)
            p2 = randrange(num_radar_points)
            while (p2==p1):
                p2 = randrange(num_radar_points)

            # Analytical approach (solving the 2x2 system)
            num = (np.cos(theta[p1])*v_r[p2]) - (np.cos(theta[p2])*v_r[p1])
            denom = (np.cos(theta[p1])*np.sin(theta[p2])) - (np.sin(theta[p1])*np.cos(theta[p2]))
            v_y = num/denom
            v_x = ( v_r[p2] - (np.sin(theta[p2])*v_y) ) / np.cos(theta[p2])
        else:
            # generate random subsample of semi-random size m >= 2
            subsample = []
            n_ = 6
            m = round(num_radar_points/n_)
            while m < 2:
                n_ -= 1
                m = round(num_radar_points/n_)
            for j in range(m):
                # generate random indeces in [0, num_radar_points)
                p = randrange(0, num_radar_points)
                # no duplicates!!
                while (p in subsample):
                    p = randrange(0, num_radar_points)
                subsample.append(p)
    
            # initialize matrix A and B for the LSQ problem
            A = []
            b = []

            # iterate over the subsample
            for j in subsample:
                # fill the A and b matrix
                A1 = np.cos(theta[j])
                A2 = np.sin(theta[j])
                A.append([A1, A2])
                b.append(v_r[j])

            # solve LSQ    
            sol = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
            # x
            v_x = sol[0][0]
            v_y = sol[0][1]
            # sum of squared residuals
            SSR = sol[1]

        # now find the sensor velocity and yaw rate
        if v_x != 0: # avoid division by 0 (non-moving vehicle)
            alpha = np.arctan2(v_y,v_x)
            v_s = -v_x/np.cos(alpha)

            # calculate the error of the current fit and the number of outliers/inliers
            sum_e = 0
            current_inliers = [] # index of all the inlier radar readings
            current_outliers = [] # index of all the outlier radar readings
            for j in range(num_radar_points):
                # measure the error between the radial speed and the sensor velocity
                e = abs(v_r[j] - (v_x*np.cos(theta[j]) + v_y*np.sin(theta[j]))) 

                # if the current radar reading has a small error, compared to the velocity profile of 
                # the randomly chosen subsample/2 points, it is an inlier
                if ( e < (error_threshold*abs(v_r[j])) ):
                    current_inliers.append(j)
                else:
                    current_outliers.append(j)
                # count the accumulated error to determine best fit
                sum_e += e

            # keep the best fit in terms of minimum error and maximum inliers (this is the same most of the times)
            if (len(current_inliers) > max_inliers):
                max_inliers = len(current_inliers)
                min_e = sum_e
                # calculate the % of inliers and
                pct_e = 100*min_e/sum_vr
                pct_inl = 100*max_inliers/num_radar_points
                if criteria == "num_inliers":
                    # keep the inliers and outliers
                    best_inliers = current_inliers
                    best_outliers = current_outliers
            if (sum_e < min_e):
                min_e = sum_e
                max_inliers = len(current_inliers)
                # calculate the % of error
                pct_e = 100*min_e/sum_vr
                pct_inl = 100*max_inliers/num_radar_points
                if criteria=="error":
                    best_inliers = current_inliers
                    best_outliers = current_outliers

        i+=1
        
        # determine which is the first condition to break the loop, based on the criteria given
        if criteria=="error":
            cond = min_e < stop_e*sum_vr # the minimum error being less than (stop_e)% of the sum of radial velocities
        else:
            cond = max_inliers > round(stop_in*num_radar_points) # the max inliers being more than (stop_in)% of the points 
            
        # give proper warning if ransac did not converge at max_iter
        if (i == max_iter):
            print(f"Warning! Ransac did not converge but stopped at {max_iter} iterations.")
            
            # depending on criteria, give diagnostic message
            if criteria == "error":
                print(f"Minimum error %: {pct_e}.")
            else:     
                print(f"% of inliers: {pct_inl}.")

    return True, best_inliers, best_outliers, i, pct_e, pct_inl, SSR

# part 2 - LSQ
# perform LSQ to get the sensor velocity and direction based on the inliers from Ransac,  
# the azimuth and the radial_speed radar readings
def inliers_LSQ(best_inliers, theta, v_r, error_threshold):
    # print()"Step 2: LSQ on the inlier set.")
    # initialize the A matrix and b vector for the least squares prloblem
    A = []
    b = []

    # iterate over all the inliers
    for j in best_inliers:
        # fill the A and b matrix
        A1 = np.cos(theta[j])
        A2 = np.sin(theta[j])
        A.append([A1, A2])
        b.append(v_r[j])

    # solve the LSQ problem 
    sol = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    # the solution is going to be used to extract the sensor velocity and heading direction
    # v_x and v_y are the radar sensor linear velocities in the x,y directions
    v_x = sol[0][0]
    v_y = sol[0][1]
    # sum of squared residuals
    SSR = sol[1]

    # calculate sensor velocity and direction
    alpha = np.arctan2(v_y,v_x)
    v_s = -v_x/np.cos(alpha)

    # get new, improved inliers
    final_inliers = []
    final_outliers = []
    sum_e = 0
    for j in range(len(v_r)):
        # measure the error between the radial speed and the sensor velocity
        e = abs(v_r[j] - (v_x*np.cos(theta[j]) + v_y*np.sin(theta[j]))) 
        sum_e += e

        # if the current radar reading has a small error, it is an inlier
        if (e<error_threshold):
            final_inliers.append(j)
        else:
            final_outliers.append(j)
            
    # determine which is the first condition to break the loop, based on the criteria given
    pct_e = 100*sum_e/sum([abs(_) for _ in v_r])
    pct_inl = 100*len(final_inliers)/len(v_r)
    
    return v_x, v_y, v_s, alpha, final_inliers, final_outliers, pct_e, pct_inl, SSR

# part 3 - ego-motion estimation
def ego_motion(v_s, alpha, b_, l_, beta_):
    # print()"Step 3: Ego-motion estimation.")
    # if the vehicle is moving, proceed to find the sensor and vehicle velocity and yaw rates
    if (v_s != 0):
        # now calculate the vehicle velocity and yaw rate based on the Ackerman condition
        v = ( np.cos(alpha + beta_) - ((b_/l_)*np.sin(alpha+beta_)) ) * v_s
        omega = (np.sin(alpha+beta_)/l_) * v_s
    else:
        print("\t\t\t WTF!!Non-moving vehicle!!")
        v = 0
        omega = 0
    
    return v, omega

# apply radar odometry on the the current radar scan
'''
Returns:

0. retflag : 0 = Not enough non-zero radial_speed returns, 1 = Ransac didn't return enough inliers, 2 = 
1. v : vehicle radial velocity
2. w : vehicle yaw rate
'''
def radar_odometry(t, r_list, theta_list, v_r_list, v_prev):
    # tuning
    # Ransac
    # use sub-samples flag
    ransac_flag = True
    # stoppage criteria
    crit = "num_inliers"
    max_iter = 50 # maximum iterations
    stop_error = 0.05 # total error %
    stop_inl = 0.9 # % of inliers
    # percentage of error allowed to be considered inlier
    error_threshold = 0.1

    # Filtering
    radial_speed_filtering = False
    radial_threshold = 0 # removing all 0 radial speed returns
    range_filtering = False
    range_threshold = 1 # meters
    velocity_filtering = False
    velocity_thresholod = 0.5 # percentage of velocity deviation that is allowed between iterations

    # LSQ criteria
    lsq_inlier_pct_thresh = 0.5 # % of Ransac inliers required to accept LSQ result
    lsq_inlier_num_thresh = 5 # number of Ransac inliers required to accept LSQ result

    # LSQ performance
    ssr_ransac, ssr_lsq = None, None

    # print time and size of radar scan
    print(f"At time: {t} seconds.")
    print(f"# of radar readings: {len(theta_list)}")

    # filter based on radial_speeds returns. There should almost never be 0 radial speeds. If there are, we are probably stationary or sth is wrong
    if radial_speed_filtering:
        v_r_list, theta_list, r_list, remove_idxs = remove_zero_radial(v_r_list, theta_list, r_list, radial_threshold)
        print(f"# of radar readings after radial speed filtering: {len(theta_list)}. Removed {len(remove_idxs)} points!")

    # filter based on range, to avoid objects that are too close to the radar sensor and block the view
    if range_filtering:
        v_r_list, theta_list, r_list, remove_idxs = remove_low_range(v_r_list, theta_list, r_list, range_threshold)
        print(f"# of radar readings after range filtering: {len(theta_list)}. Removed {len(remove_idxs)} points!")
    
    # step 1 - ransac
    # if false, ransac is applied with 2 random pooints, otherwise, it uses a subsample
    # of the total radar_points
    # criteria can either be "error" or "num_inliers"
    print("Step 1. Ransac...")
    retval, inliers, outliers, it, pct_e_ransac, pct_inl_ransac, ssr_ransac = Ransac(theta_list, v_r_list, 
                                                                        error_threshold, max_iter,
                                                                        stop_error, stop_inl,
                                                                        ransac_flag=True, criteria=crit)
    print(f"\t# of inliers: {len(inliers)}")
    print(f"\t# of iterations: {it}")
    print(f"\tSSR: {ssr_ransac}")

    # check if we had enough radial speed returns
    if not retval:
        print(f"Not enough non-zero radial_speed returns ({np.count_nonzero(v_r_list)}) !")
        retflag = 1
        v, omega = 0, 0
    else:
        # check if Ransac gave an adequate amount of inliers
        if len(inliers) < lsq_inlier_pct_thresh*len(theta_list) or len(inliers) < lsq_inlier_num_thresh:
            print("Warning! Not enough inliers!!")
            retflag = 2
            v, omega = None, None
        else:
            retflag = 0
            # step 2 - LSQ
            print("Step 2. LSQ...")
            v_x, v_y, v_s, alpha, final_inliers, final_outliers, pct_e_lsq, pct_inl_lsq, ssr_lsq = inliers_LSQ(inliers, 
                                                                                                            theta_list, 
                                                                                                            v_r_list, 
                                                                                                            error_threshold)
            print(f"\tLSQ SSR: {ssr_lsq}")
            print(f"\tLSQ # of optimized inliers: {len(final_inliers)}")
            print(f"\tLSQ Error: {pct_e_lsq} %")
            print(f"\tSensor velocity: {v_s} (m/sec)\n\tSensor direction: {alpha} (rad)")

            # step 3 - ego-motion
            l_ = 0.915; # x_s
            b_ = 0; # y_s
            beta_ = 0; # a_s
            print("Step 3. Ego-motion...")
            v, omega = ego_motion(v_s, alpha, b_, l_, beta_)
            print(f"\tVehicle velocity: {v} (m/s)\n\tVehicle yaw rate: {omega} (rad/sec)")

            # reject fits that indicate the problem of the largest group being moving targets (too big velocity deviation from previous reading)
            if velocity_filtering:
                v_diff = abs(v_prev - v)
                if v_diff > abs(velocity_thresholod*v_prev) and v_prev != 0:
                    print(f"Danger!! Large velocity deviation: {v_prev - v} m/sec! Skipping this iteration...")
                    retflag = 3
                    v, omega, ssr_lsq = None, None, None

    return t, v, omega, ssr_ransac, ssr_lsq, retflag

# apply radar odometry on the entire dataset
def radar_odometry_on_full_dataset(t_radar, r_list, theta_list, v_r_list):
    # iterate over the entire video
    v_r_used = []
    theta_used = []
    v_list = []
    omega_list = []
    inlier_list = []
    outlier_list = []
    v_x_list = []
    v_y_list = []
    alpha_list = []

    # keep the performance of ransac
    e_pct_ransac = []
    e_pct_lsq = []
    inl_pct_ransac = []
    inl_pct_lsq = []

    # tuning
    # Ransac
    # use sub-samples flag
    ransac_flag = False
    # stoppage criteria
    crit = "num_inliers"
    max_iter = 50 # maximum iterations
    stop_error = 0.05 # total error %
    stop_inl = 0.9 # % of inliers
    # percentage of error allowed to be considered inlier
    error_threshold = 0.1

    # Filtering
    radial_speed_filtering = False
    radial_threshold = 0 # removing all 0 radial speed returns
    range_filtering = False
    range_threshold = 1 # meters
    velocity_filtering = False
    velocity_thresholod = 0.5 # percentage of velocity deviation that is allowed between iterations

    # LSQ criteria
    lsq_inlier_pct_thresh = 0.5 # % of Ransac inliers required to accept LSQ result
    lsq_inlier_num_thresh = 5 # number of Ransac inliers required to accept LSQ result

    # multi-scans aggregation
    num_scans = 2
    #r_list, theta_list, v_r_list, t_radar = multi_scans(r_list, theta_list, v_r_list, t_radar, num_scans)

    # loop over every radar point of the current radar pointcloud
    for i in range(len(theta_list)):
        print(f"\n\tvideo frame: {i}")
        print(f"At time: {t_radar[i] - t_radar[0]} seconds.")
        # extract the azimuth angles and radial speeds for one specific radar reading
        theta = theta_list[i]
        v_r = v_r_list[i]
        r = r_list[i]
        print(f"# of radar readings: {len(theta)}")

        # filter based on radial_speeds returns. There should almost never be 0 radial speeds. If there are, we are probably stationary or sth is wrong
        if radial_speed_filtering:
            v_r, theta, r, remove_idxs = remove_zero_radial(v_r, theta, r, radial_threshold)
            print(f"# of radar readings after radial speed filtering: {len(theta)}. Removed {len(remove_idxs)} points!")

        # filter based on range, to avoid objects that are too close to the radar sensor and block the view
        if range_filtering:
            v_r, theta, r, remove_idxs = remove_low_range(v_r, theta, r, range_threshold)
            print(f"# of radar readings after range filtering: {len(theta)}. Removed {len(remove_idxs)} points!")
        
        # step 1 - ransac
        # if false, ransac is applied with 2 random pooints, otherwise, it uses a subsample
        # of the total radar_points
        # criteria can either be "error" or "num_inliers"
        print("Step 1. Ransac...")
        retval, inliers, outliers, it, pct_e_ransac, pct_inl_ransac = Ransac(theta, v_r, 
                                                                            error_threshold, max_iter,
                                                                            stop_error, stop_inl,
                                                                            ransac_flag=True, criteria=crit)
        print(f"# of inliers: {len(inliers)}")
        print(f"# of iterations: {it}")

        # if we had enough radial speed returns
        if retval:
            # if ransac succedeed, go to the next steps
            if len(inliers) < lsq_inlier_pct_thresh*len(theta) or len(inliers) < lsq_inlier_num_thresh:
                print("Warning! Not enough inliers!!")
                final_inliers = []
                final_outliers = []
                if v_list:
                    v = v_list[-1]
                    omega = omega_list[-1]
                    v_x = v_x_list[-1]
                    v_y = v_y_list[-1]
                    alpha = alpha_list[-1]
                    pct_e_ransac = e_pct_ransac[-1]
                    pct_e_lsq = e_pct_lsq[-1]
                    pct_inl_ransac = inl_pct_ransac[-1]
                    pct_inl_lsq = inl_pct_lsq[-1]
                else:
                    v, omega, v_x, v_y,  = 0, 0, 0, 0
                    pct_e_ransac, pct_e_lsq, pct_inl_ransac, pct_inl_lsq = 100, 100, 100, 100
            else:
                # step 2 - LSQ
                print("Step 2. LSQ...")
                v_x, v_y, v_s, alpha, final_inliers, final_outliers, pct_e_lsq, pct_inl_lsq = inliers_LSQ(inliers, 
                                                                                        theta, 
                                                                                        v_r, 
                                                                                        error_threshold)
                print(f"LSQ # of optimized inliers: {len(final_inliers)}")
                print(f"LSQ Error: {pct_e_lsq} %")
                print(f"Sensor velocity: {v_s} (m/sec)\nSensor direction: {alpha} (rad)")

                # step 3 - ego-motion
                l_ = 0.915; # x_s
                b_ = 0; # y_s
                beta_ = 0; # a_s
                print("Step 3. Ego-motion...")
                v, omega = ego_motion(v_s, alpha, b_, l_, beta_)
                print(f"Vehicle velocity: {v} (m/s)\nVehicle yaw rate: {omega} (rad/sec)")

                # reject fits that indicate the problem of the largest group being moving targets
                if velocity_filtering:
                    if v_list:
                        v_diff = abs(v_list[-1] - v)
                        if v_diff > abs(velocity_thresholod*v_list[-1]) and v_list[-1] != 0:
                            print(f"Danger!! Large velocity deviation: {v_list[-1] - v} m/sec! Skipping this iteration...")
                            v = v_list[-1]
                            omega = omega_list[-1]
                            v_x = v_x_list[-1]
                            v_y = v_y_list[-1]
                
            # keep all the data
            v_r_used.append(v_r) # radial speeds of current frame
            theta_used.append(theta) # azimuth angles of current frame
            
            inlier_list.append(final_inliers) # final inliers
            outlier_list.append(final_outliers) # final outliers
            
            e_pct_ransac.append(pct_e_ransac) # ransac fit error
            e_pct_lsq.append(pct_e_lsq) # lsq fit error
            inl_pct_ransac.append(pct_inl_ransac) # ransac fit inlier pct
            inl_pct_lsq.append(pct_inl_lsq) # lsq fit inlier pct
            
            v_list.append(v) # vehicle linear vel
            omega_list.append(omega) # vehicle angular vel
            v_x_list.append(v_x) # sensor linear velocity on its x-axis
            v_y_list.append(v_y) # sensor linear velocity on its y-axis
            alpha_list.append(alpha) # sensor orientation
        else:
            # if not enough radial speed returns
            # keep all the data
            v_r_used.append(v_r) # radial speeds of current frame
            theta_used.append(theta) # azimuth angles of current frame
            v_list.append(0) # vehicle linear vel
            omega_list.append(0) # vehicle angular vel
            inlier_list.append([]) # final inliers
            outlier_list.append([]) # final outliers
            e_pct_ransac.append(100) # ransac fit error
            e_pct_lsq.append(100) # lsq fit error
            inl_pct_ransac.append(100) # ransac fit inlier pct
            inl_pct_lsq.append(100) # lsq fit inlier pct
            v_x_list.append(0) # sensor linear velocity on its x-axis
            v_y_list.append(0) # sensor linear velocity on its y-axis
            if alpha_list:
                alpha_list.append(alpha_list[-1]) # sensor orientation
            else:
                alpha_list.append(0)