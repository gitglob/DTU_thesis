import numpy as np

import globals
from helpers import *
from radar_odom import radar_odometry

# update of the nominal state (ignores noise & inaccuracies) based on system dynamics
def nominal_state(dt, x_prev, a_imu, omega_imu):
    p_prev = x_prev[0:3]
    v_prev = x_prev[3:6]
    q_prev = x_prev[6:10]
    a_b = x_prev[10:13]
    omega_b = x_prev[13:16]
    g = x_prev[16:19]
    pg_i = x_prev[19:22]
    qg_i = x_prev[22:26]
    pr_i = x_prev[26:29]
    qr_i = x_prev[29:33]
    # find the current robot rotation matrix
    R_robot = quat2matrix(q_prev)
    
    # extract the current Rotation (IMU frame rotation + robot rotation)
    R = globals.R_b2i.T@R_robot
    
    # angular velocity
    w_theta = R@(omega_imu - omega_b)*dt
    w_norm = np.linalg.norm(w_theta)
    q_w = np.cos(w_norm/2)
    q_xyz = (w_theta/w_norm)*np.sin(w_norm/2)
    q_omega = np.hstack((q_w, q_xyz.flatten())).reshape((4,1))   

    # apply dynamics to position, velocity and orientation
    p = p_prev + v_prev*dt + 0.5*(R @ (a_imu-a_b) + g)*(dt**2)
    v = v_prev + (R @ (a_imu - a_b) + g)*dt
    
    q = quat_mult(q_prev, q_omega)    
    
    # the nominal state
    x_out = np.vstack((p, v, q, a_b, omega_b, g, pg_i, qg_i, pr_i, qr_i))
    
    return x_out

# predict the next
def predict(dt, x, dx_prev, P_prev, 
            a_imu, omega_imu,
            var_v, var_theta, var_alpha, var_omega):
    
    Fx = F_x(dt, x, a_imu, omega_imu)
    Fw = F_w()
    Qw = Q_w(dt, var_v, var_theta, var_alpha, var_omega)

#     dx = Fx @ dx_prev #+ Fw @ w # this is always zero!

    t1 = (Fx @ P_prev) @ Fx.T
    t2 = (Fw @ Qw) @ Fw.T

    P = t1 + t2 # this is the only term we are interested in
    
    return P

# calculate state transition matrix (including drift - last 6 elements)
def F_x(dt, x, a_imu, omega_imu):
    # extract linear acceleration and angular velocity biases
    a_b = x[10:13]
    omega_b = x[13:16]
    # find the current robot rotation matrix
    q_ = x[6:10]
    R_robot = quat2matrix(q_)
    
    # extract the current Rotation (IMU frame rotation + robot rotation)
    R = globals.R_b2i.T @ R_robot

    # linear acceleration
    real_a = a_imu - a_b
    # angular velocity
    w_theta = R@(omega_imu - omega_b)*dt
    
    # extract dq from angular velocity
    w_norm = np.linalg.norm(w_theta)
    q_w = np.cos(w_norm/2)
    q_xyz = (w_theta/w_norm)*np.sin(w_norm/2)
    q_omega = np.hstack((q_w, q_xyz.flatten())).reshape((4,1))    
    
    # convert dq to rotation matrix
    Rw = quat2matrix(q_omega.flatten())

    # shortcuts
    A = skew(real_a)
    RA = R @ A
    
    F = np.zeros((24, 24))
    F[0:3, 0:3] = globals.I3
    F[0:3, 3:6] = globals.I3*dt
    
    F[3:6, 3:6] = globals.I3
    F[3:6, 6:9] = -RA*dt
    F[3:6, 9:12] = -R*dt
    F[3:6, 15:18] = globals.I3*dt
    
    F[6:9, 6:9] = Rw.T
    F[6:9, 12:15] = -globals.I3*dt
    
    F[9:24, 9:24] = np.eye(15)
        
    return F

# calculate noise state transition matrix
def F_w():    
    M = np.zeros((24,12))
    M[3:6, 0:3] = globals.I3
    M[6:9, 3:6] = globals.I3
    M[9:12, 6:9] = globals.I3
    M[12:15, 9:12] = globals.I3

    return M

# calculate noise covariance
def Q_w(dt, var_v, var_theta, var_alpha, var_omega):
    
    Q = np.zeros((12,12))
    Q[0:3, 0:3] = var_v * globals.I3 * dt**2
    Q[3:6, 3:6] = var_theta * globals.I3 * dt**2
    Q[6:9, 6:9] = var_alpha * globals.I3 * dt**2
    Q[9:12, 9:12] = var_omega * globals.I3 * dt**2
    
    return Q

# page 17 - Solas
# calculate skew symmetric matrix of vector in se(3)
def skew(x):
    x = x.flatten()
    
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]]).reshape((3,3))

# calculate the true state based on the nominal state and the error state
def true_state(x_in, dx, flag):  
    p = x_in[0:3]
    v = x_in[3:6]
    q = x_in[6:10]
    a = x_in[10:13]
    w = x_in[13:16]
    g = x_in[16:19]
    pg_i = x_in[19:22]
    qg_i = x_in[22:26]
    pr_i = x_in[26:29]
    qr_i = x_in[29:33]
    
    dp = dx[0:3]
    dv = dx[3:6]
    dtheta = dx[6:9]
    dq = euler2quat(dtheta, rot_order)
    da = dx[9:12]
    dw = dx[12:15]
    dg = dx[15:18]
    dp_i = dx[18:21]
    dtheta_i = dx[21:24]
    dq_i = euler2quat(dtheta_i, rot_order)
    
    p_t = p + dp
    v_t = v + dv
    q_t = quat_mult(q, dq)
    a_t = a + da
    w_t = w + dw
    g_t = g + dg
    if flag==0:
        pg_i_t = pg_i + dp_i
        qg_i_t = quat_mult(qg_i, dq_i)
        pr_i_t = pr_i
        qr_i_t = qr_i
    if flag==2:
        pr_i_t = pr_i + dp_i
        qr_i_t = quat_mult(qr_i, dq_i)    
        pg_i_t = pg_i
        qg_i_t = qg_i    
    
    x_out = np.vstack((p_t, v_t, q_t, a_t, w_t, g_t, pg_i_t, qg_i_t, pr_i_t, qr_i_t))

    return x_out

# flag: 0 -> position sensor, 1 -> orientation sensor, 2 -> otherwise
def correct(x, P, y_mu, V, flag):    
    # Kalman gain
    H = get_H(x, y_mu, flag) # jacobian
    K = (P @ H.T) @ np.linalg.inv(((H@P)@H.T) + V) # Kalman gain
#     print("K: ", K[3:6, :])
    
    # error
    h = get_h(x, flag)
    innov, v = get_innov(x, y_mu, h, flag)
    dx = K @ innov
    
    # covariance update
    P = (np.eye(24) - (K@H)) @ P
    
    return innov, dx, P, v

# flag: 0 -> position sensor, 1 -> orientation sensor, 2 -> velocity sensor
def get_innov(x, mu, h, flag):
    if flag==0:
        inno = mu - h
        
        return inno, None
    elif flag==1: # not working
        e = quat2euler(h)
        R = quat2matrix(h)

        q_mu = euler2quat(mu)
        R_mu = quat2matrix(q_mu)
                
        # page 19, Solas
        R_inno = R.T@R_mu
        phi = np.arccos( (np.trace(R_inno) - 1)/2 )
        u = v_func(R_inno - R_inno.T)/(2 * np.sin(phi))
        
        inno = u*phi
#         inno = matrix2euler(R_inno) - quat2euler(h)

        return inno, None
    elif flag==2:
        # convert polar velocity to cartesian
        q = x[6:10]
        e = quat2euler(q)
        v_x = np.cos(e[2])*mu
        v_y = np.sin(e[2])*mu
        v_z = x[5] # we have no v_z information from radar_odometry, as it operates in 2d
        v_mu = np.array([v_x, v_y, v_z]).reshape(3,1)
        
        inno = v_mu - h
        
        return inno, v_mu
    else:
        raise

# page 19, solas
# this function is the reverse of the "skew" function
def v_func(R):
    return np.array([R[2,1], R[0,2], R[1,0]]).reshape((3,1))

# page 19, solas
# this function
def log_func(R):
    phi = np.arccos((np.trace(R) - 1)/2)
    u = v_func(R - R.T)/(2 * np.sin(phi))
    retval = skew(u*phi)
    
    return retval

# flag: 0 -> position sensor, 1 -> orientation sensor, 2 -> velocity sensor
def get_h(x, flag):        
    if flag == 0:
        p_t = x[0:3]
        p_i = x[19:22]
        q_i = x[22:26]
        R_qi = quat2matrix(q_i)
        
        ret = np.matmul(R_qi, p_t) + p_i
        
        return ret
    elif flag == 1:
        q_t = x[6:10]
        q_i = x[22:26]
        
        ret = quat_mult(q_i, q_t)

        return ret
    elif flag==2:
        v_t = x[3:6]
        p_i = x[19:22]
        q_i = x[22:26]
        R_qi = quat2matrix(q_i)
        
        ret = np.matmul(R_qi, v_t)
        
        return ret

# get the full H matrix
def get_H(x, mu, flag):    
    J1 = Jac1(x, mu, flag)
    J2 = Jac2(x)
    J = J1 @ J2 

    return J

# ESEKF reset
def reset(dx, P):
    dtheta = dx[6:9]
    
    G = np.zeros((24,24))
    G[0:6, 0:6] = np.eye(6)
    G[6:9, 6:9] = globals.I3 - skew(0.5*dtheta)
    G[9:24, 9:24] = np.eye(15)
    
#     P = (G @ P) @ G.T # for better accuracy
    P = P
    dx = np.zeros((24,1))
    
    return dx, P

# Jacobians
# Jacobian 1
def Jac1(x, mu, flag):
    J1 = np.zeros((3,26))
    
    if flag==0: # position measurement
        q_i = x[22:26]
        J1[0:3, 0:3] = quat2matrix(q_i)
        J1[0:3, 19:22] = globals.I3
    elif flag==1: # orientation measurement
        J1[0:3, 6:9] = globals.I3
        J1[0:3, 22:25] = quat2matrix(mu).T
    elif flag==2: # velocity measurement
        J1[0:3, 3:6] = globals.I3
        
    return J1

# Jacobian 2
def Jac2(x):
    q = x[6:10]
    q_i = x[22:26]
    
    J2 = np.zeros((26,24))
    
    J2[0:3, 0:3] = globals.I3
    J2[3:6, 3:6] = globals.I3
    J2[6:10, 6:9] = Jac3(q)
    J2[10:22, 9:21] = np.eye(12)
    J2[22:26, 21:24] = Jac3(q_i)

    return J2

# Jacobian 3
def Jac3(q):
    q_w, q_x, q_y, q_z = q
    
    return 0.5 * np.array([[-q_x, -q_y, -q_z],
                         [q_w, -q_z, q_y],
                         [q_z, q_w, -q_x],
                         [-q_y, q_x, q_w]]).reshape((4,3))

# Sample gps to test how many seconds we can go without it
def sample_gps(t_gps, p_gps, sample_rate=10):
    t_gpss = []
    p_gpss = []
    sample_rate = sample_rate
    for i in range(len(t_gps)):
        if i%sample_rate == 0:
            t_gpss.append(t_gps[i])
            p_gpss.append(p_gps[i])
            
    t_gpss = np.array(t_gpss)
    p_gpss = np.array(p_gpss)

    return t_gpss, p_gpss

def sensor_fusion(t_gps, p_gps, 
                t_imu, a_imu, w_imu, theta0_z, lin_acc_imu_bias, ang_vel_imu_bias,
                t_radar, r_radar, theta_radar, vr_radar,
                gps_sample_rate):
    # ESEKF parameters
    # load velocity, orientation, acceleration and ang. velocity bias variances
    var_v = np.array([globals.var_a, globals.var_a, globals.var_a])*1000 # hand-tuning
    var_alpha = np.array([globals.var_a, globals.var_a, globals.var_a])*100 # IMU topic

    var_theta = np.array([globals.var_w, globals.var_w, globals.var_w])*0.01 # hand-tuning
    var_omega = np.array([globals.var_w, globals.var_w, globals.var_w])*0.001 # IMU topic

    # model system noise (12x1)
    w = np.hstack((var_v, var_theta, var_alpha, var_omega)).reshape((12,1))

    # measurement covariance vector
    var_gps = np.array([globals.var_p, globals.var_p, globals.var_p])
    var_mag = np.array([0, 0, globals.var_w])
    var_radar = np.array([globals.var_r, globals.var_r, globals.var_r])

    # measurement noise covariance matrix 
    V_gps = var_gps*np.identity(3)
    V_mag = var_mag*np.identity(3)
    V_radar = var_radar*np.identity(3)

    # step 1. initialize state and covariance for the very first iteration
    # nominal and error state
    p0 = np.array([0, 0, 0]).reshape(3,1)
    v0 = np.array([0, 0, 0]).reshape(3,1)
    q0 = euler2quat(np.array([0, 0, theta0_z]))
    a_b0 = lin_acc_imu_bias.reshape(3,1)
    omega_b0 = ang_vel_imu_bias.reshape(3,1)
    g0 = np.array([0, 0, -globals.GRAV]).reshape(3,1)
    pgi0 = globals.gps_off
    qgi0 = np.array([1, 0, 0, 0]).reshape(4,1)
    pri0 = globals.radar_off
    qri0 = np.array([1, 0, 0, 0]).reshape(4,1)
    x = np.vstack([p0, v0, q0, a_b0, omega_b0, g0, pgi0, qgi0, pri0, qri0])
    dx = np.zeros((24,1))

    # initialize error-state covariance matrix based on how good our initial estimate is
    k0 = 1
    k1 = 1
    k2 = 1
    k3 = 1
    k4 = 1
    Pg_vec = np.array([var_gps[0]*k0, var_gps[1]*k0, var_gps[2]*k0, # position
                    var_v[0]*k1, var_v[1]*k1, var_v[2]*k1, # velocity
                    var_theta[0]*k2, var_theta[1]*k2, var_theta[2]*k2, # orientation
                    var_alpha[0]*k3, var_alpha[1]*k3, var_alpha[2]*k3, # lin. acc. bias
                    var_omega[0]*k4, var_omega[1]*k4, var_omega[2]*k4, # ang. velocity bias
                    1e-14, 1e-14, 1e-14, # gravity
                    1e-14, 1e-14, 1e-14,  # extrinsics - translation
                    1e-14, 1e-14, 1e-14,]) # extrinsics - rotation
    Pg = Pg_vec*np.identity(24)
    Pg_prev = Pg

    k00 = 1
    k11 = 1
    k22 = 1
    k33 = 1
    k44 = 1
    Pr_vec = np.array([var_gps[0]*k00, var_gps[1]*k00, var_gps[2]*k00, # position
                    var_radar[0]*k11, var_radar[1]*k11, var_radar[2]*k11, # velocity
                    var_theta[0]*k22, var_theta[1]*k22, var_theta[2]*k22, # orientation
                    var_alpha[0]*k33, var_alpha[1]*k33, var_alpha[2]*k33, # lin. acc. bias
                    var_omega[0]*k44, var_omega[1]*k44, var_omega[2]*k44, # ang. velocity bias
                    1e-14, 1e-14, 1e-14, # gravity
                    1e-14, 1e-14, 1e-14,  # extrinsics - translation
                    1e-14, 1e-14, 1e-14,]) # extrinsics - rotation
    Pr = Pr_vec*np.identity(24)
    Pr_prev = Pr

    # initialize time
    t_prev = 0
    t = 0
    dt = 0

    # initialize flags as false - if True, it means that we are reading from this signal
    IMU_flag = False
    GPS_flag = False 
    RO_flag = False

    # counter for IMU/radar sample
    i_imu = 1
    i_gps = 1
    i_radar = 1
    i = -1

    # to keep the performance of LSQ in the radar odometry
    # keep the performance of Least Squares
    ssr_ransac_list = []
    ssr_lsq_list = []

    # keep the lsq timestamps
    t_lsq = []

    # initialize empty lists to keep history of state variables
    dx_list = np.empty((0,24), float)
    x_list = np.empty((0,33), float)
    t_list = np.empty((0,1), float)
    t_ro_list = np.empty((0,1), float)
    innovation_list = np.empty((0,3), float)
    v_ro_list = np.empty((0,3), float)

    # sample GPS to see how long we can go without it
    t_gps, p_gps = sample_gps(t_gps, p_gps, sample_rate=gps_sample_rate)

    # main loop (predictions : IMU, measurements : Radar)
    while True:
        print(f"\t\tIteration #{i}.")
        # if we have an imu signal, we are at the prediction phase
        if IMU_flag:
            print("IMU signal...")
            # set time
            t = t_imu[i_imu]
            dt = t - t_prev
                
            # get imu readings (3x1)
            alpha_imu = np.array(a_imu[i_imu]).reshape((3,1))
            omega_imu = np.array(w_imu[i_imu]).reshape((3,1))
            
            # step 2. nominal state kinematics
            x = nominal_state(dt, x, alpha_imu, omega_imu)   
                
            # step 3. error state kinematics - prediction
            Pg = predict(dt, x, dx, Pg_prev, alpha_imu, omega_imu, var_v, var_theta, var_alpha, var_omega) # gps
            Pr = predict(dt, x, dx, Pr_prev, alpha_imu, omega_imu, var_v, var_theta, var_alpha, var_omega) # radar
            
            # move to the next imu signal
            i_imu+=1
            IMU_flag = False
            t_prev = t
            
        # if we have a gps signal, we are in the measurement phase
        if GPS_flag:
            print("\t\t\t\t\t\t\t\tGPS signal...")
            # set time
            t = t_gps[i_gps]
            dt = t - t_prev

            # get measurement - gps coordinates
            y_mu = p_gps[i_gps, :].reshape(3,1)

            # step 4. correction
            inno, dx, Pg, _ = correct(x, Pg, y_mu, V_gps, flag=0)
            dx_list = np.append(dx_list, dx.T, axis=0)
            innovation_list = np.append(innovation_list, inno.T, axis=0)
            
            # step 5. true state update
            x_t = true_state(x, dx, flag=0)
            
            # step 6. injection of error to nominal state
            x = x_t
            
            # step 7. ESKF reset
            dx, Pg = reset(dx, Pg)
            
            # move to the next gps signal
            i_gps+=1
            GPS_flag = False
            
        # if we have a radar signal, we are in the measurement phase
        if RO_flag:
            print("\t\t\t\tRadar Odom signal...")
            # get current polar velocity of vehicle (needed for radar odometry velocity filtering)
            polar_vel = (x[3]**2 + x[4]**2)**0.5 # convert cartesian velocity to polar

            # extract time and velocity from radar odometry
            t_odom, v_odom, w_odom, ssr_ransac, ssr_lsq, retflag = radar_odometry(t_radar[i_radar], r_radar[i_radar], theta_radar[i_radar], vr_radar[i_radar], polar_vel)
            ssr_ransac_list.append(ssr_ransac)
            ssr_lsq_list.append(ssr_lsq)

            # check if radar odometry succeded
            if retflag == 0:
                # set time
                t = t_odom
                t_ro_list = np.append(t_ro_list, t)
                dt = t - t_prev

                # get measurement - radar odometry velocities
                v_mu = v_odom

                # step 4. correction
                inno, dx, Pr, v_ro = correct(x, Pr, v_mu, V_radar, flag=2)
                dx_list = np.append(dx_list, dx.T, axis=0)
                innovation_list = np.append(innovation_list, inno.T, axis=0)
                v_ro_list = np.append(v_ro_list, v_ro.T, axis=0)
                
                # step 5. true state update
                x_t = true_state(x, dx, flag=2)
                
                # step 6. injection of error to nominal state
                x = x_t
                
                # step 7. ESKF reset
                dx, Pr = reset(dx, Pr)
            else:
                print(f"Radar Odometry failed. Flag: #{retflag}")
                
            # move to the next radar signal
            i_radar+=1
            RO_flag = False
            
        # save state values
        x_list = np.append(x_list, x.T, axis=0)
        
        # save timestamps
        t_list = np.append(t_list, t)
                
        # update states
        Pg_prev = Pg
        Pr_prev = Pr
        
        # determine which signal is next if none is over
        # check if the signals are over
        imu_t = t_imu[i_imu] if i_imu < len(t_imu) else 999999
        gps_t = t_gps[i_gps] if i_gps < len(t_gps) else 999999
        odom_t = t_radar[i_radar] if i_radar < len(t_radar) else 999999
        # find the minimum out of the next signal times
        next_times = [imu_t, gps_t, odom_t]
        minval = min(next_times)
        # get the flag that corresponds to the minimum next timing
        if minval == imu_t:
            IMU_flag = True
        if minval == gps_t:
            GPS_flag = True
        if minval == odom_t:
            RO_flag = True
            
        i+=1

        # # early stop for debug
        # if i == 2000:
        #     break
            
        # stop when we have finished all our input readings
        if i_imu >= len(t_imu) and i_gps >= len(t_gps) and i_radar >= len(t_radar):
            break

    print("Finito!")

    return t_list, x_list, t_ro_list, v_ro_list, ssr_ransac_list, ssr_lsq_list
