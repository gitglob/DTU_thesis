import numpy as np
from pathlib import Path
from utils import project_root

from src.helpers import *
from src.preprocess import *
from src.sensor_fusion import *
from src.visualize import *

   
# main function
def main():
    # LOAD DATA
    ## GPS
    print("Loading data...")
    t_gps, p_gps, v_gps, a_gps, gps_bias = LoadGPS()
    ## IMU
    t_imu, a_imu, w_imu, q_imu, theta0_z, ab_imu, wb_imu = LoadIMU()
    ## Radar
    t_radar, r_radar, theta_radar, vr_radar = LoadRadar()

    ## make sure that all the sensor data have the same length
    print("GPS, IMU, Radar shapes: ")
    print(f"GPS: {np.shape(t_gps)} IMU: {np.shape(t_imu)} Radar: {np.shape(t_radar)}")

    # SENSOR FUSION
    print("Performing sensor fusion...")
    [t_list, x_list, t_ro_list, v_ro_list, ssr_ransac_list, ssr_lsq_list] = sensor_fusion(t_gps, p_gps, 
                                                                                t_imu, a_imu, w_imu, theta0_z, ab_imu, wb_imu, 
                                                                                t_radar, r_radar, theta_radar, vr_radar, 50)

    # VISUALIZE
    print("Visualizing...")
    vis_LSQ(t_radar, ssr_ransac_list, ssr_lsq_list)
    vis_v(t_list, x_list, t_gps, v_gps, t_radar, t_ro_list, v_ro_list)
    vis_traj_2d(x_list, p_gps)
    vis_traj_3d(x_list, p_gps)



if __name__ == "__main__":
    main()

