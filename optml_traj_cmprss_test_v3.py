# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:41:59 2019

@author: Bo-Hsun Chen
"""

import sys
import numpy as np
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
from math import pi
from keras.models import load_model
import keras.backend as Keras
from TrajectoryPlanning import CubicTrajetoryPlanning_v2, cbc_traj_extrm_pts
import timeit

np.set_printoptions( formatter={'float': '{: 0.3f}'.format} )

MODEL_NUMBER = 6
seeds = [ 5461 , 123 , 445 , 2500, 1111, 111, 33, 1 ]

Read_ini_cond = 1
#path_idx = str(4)
#path_idx = "1_exp" # for experiment
#path_idx = "2_obs"
#path_idx = "1_obs"
#path_idx = "temp"
#path_idx = "worst"
#path_idx = "test"
path_idx = "worst_5"

#ini_cond_path = "./GA_data/ini_cond(" + str(MODEL_NUMBER) + ").csv"
#ini_cond_path = "./ini_cond_hard.csv" # read hard cond. for Syntec report section 5.2 to 5.3
ini_cond_path = "./ini_cond_hard_" + path_idx + ".csv"
#ini_cond_path = "train_data(33)_extreme.csv" # for paper 4-3-2 part

directory = "simulation_result/4_6/data/"

Use_EKF = False
Set_ini_theta_B_zero = True # non-zero for Syntec report Section 5.2
Earlystopping_iLQR = False
Valid_by_decmprss = 0

Gen_theta_by_DNN = 0
Gen_x_mB_by_DNN = 0

Create_train_set = 0
Create_feat_and_decmprss = 0

PID_switch = 0
x_track_switch = 0

## for realistic robot arm
Add_frctn = 0
Add_delay = 0

Valid_opt = 0
# X : via_x*4, via_y*4, v_avg*3, theta_B_0, x_mB_0
grdnt_mask = np.array( [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ], dtype=float )

if( "path_idx" in globals() ):
    if( path_idx in path_idx == "2_obs" ):
        nrmlz_obs_set = np.array( [ [ 0.18, 0.34, 0.28 ], [ 0.82, 0.66, 0.28 ] ], dtype=float ) # ( nrmlz_x, nrmlz_y, nrmlz_r )
    elif( path_idx == "1_obs" ):
        nrmlz_obs_set = np.array( [ [ 0.5, 0.5, 0.3 ] ], dtype=float ) # ( nrmlz_x, nrmlz_y, nrmlz_r )
    else:
        nrmlz_obs_set = np.array([])

else:
    nrmlz_obs_set = np.array([])


Save_result = 1
Save_result_2 = 0
Save_result_3 = 0
Save_result_4 = 0

Print_SSE = 1
Print_task_time = 0
Print_nrmlz_X = 1

Plot_figures = 1
Plot_figures2 = 0
Calc_elps_time = 0

ep_num = 15
train_data_num = 1
DNN_x_mB_o_dim = 0
DNN_theta_o_dim = 0
if( "path_idx" in globals() ):
    if( path_idx == "worst" ):
        Worst_case = 1

#tf.set_random_seed( seeds[MODEL_NUMBER-1] )
np.random.seed( seeds[MODEL_NUMBER-1] )
#np.random.seed( int( time.time() ) )

model_idx_str = str( seeds[ ( MODEL_NUMBER - 2 + len(seeds) ) % len(seeds) ] )
#model_theta_path = "ML_data/model/ISOSC_theta(" + model_idx_str + ")_extreme.h5"
#model_x_mB_path = "ML_data/model/ISOSC_x_mB(" + model_idx_str + ")_extreme.h5"
train_data_path = "./ML_data/train_data(" + str( seeds[MODEL_NUMBER-1] ) + ")_extreme.csv"
#label_data_theta_path = "./ML_data/label_data_theta(" + str( seeds[MODEL_NUMBER-1] ) + ")_extreme.csv"
#label_data_x_mB_path = "./ML_data/label_data_x_mB(" + str( seeds[MODEL_NUMBER-1] ) + ")_extreme.csv"

#train_data_path = "./ML_data/train_data(" + str( seeds[MODEL_NUMBER-1] ) + ")_exp.csv"
#model_theta_path = "ML_data/model/ISOSC_theta(" + model_idx_str + ")_exp.h5"
#model_x_mB_path = "ML_data/model/ISOSC_x_mB(" + model_idx_str + ")_exp.h5"
#label_data_theta_path = "./ML_data/label_data_theta(" + str( seeds[MODEL_NUMBER-1] ) + ")_exp.csv"
#label_data_x_mB_path = "./ML_data/label_data_x_mB(" + str( seeds[MODEL_NUMBER-1] ) + ")_exp.csv"


if( Gen_theta_by_DNN == True and 'DNN_model_theta' not in globals() ):
    DNN_model_theta = load_model( model_theta_path ) # load DNN model
    print("load DNN model of theta : ", model_idx_str )
    
if( 'DNN_model_theta' in globals() ):
    DNN_theta_o_dim = DNN_model_theta.output.shape.as_list()[1]

if( Gen_x_mB_by_DNN == True and 'DNN_model_x_mB' not in globals() ):
    DNN_model_x_mB = load_model( model_x_mB_path ) # load DNN model
    print("load DNN model of x_mB : ", model_idx_str )
    
    
if( 'DNN_model_x_mB' in globals() ):
    DNN_x_mB_o_dim = DNN_model_x_mB.output.shape.as_list()[1]

if( Gen_x_mB_by_DNN == True ):
    
    jacobian_t_x_to_IC = []
    for i in range( DNN_x_mB_o_dim ):
        jacobian_t_x_to_IC.append( Keras.gradients( DNN_model_x_mB.output[:,i], DNN_model_x_mB.input )[0] )
    jacobian_t_x_to_IC = Keras.function( [ DNN_model_x_mB.input, Keras.learning_phase() ], jacobian_t_x_to_IC )

# jacobian = np.vstack( jacobian_t_x_to_IC( [ [1-D_array] , 0 ] ) ).reshape( DNN_x_mB_o_dim , -1 )

## Global condition
dt = 0.012 # [sec]
grvty = 9810 # [mm/sec^2]
SPACE_UP_LIMIT = 650 # [mm]
SPACE_DOWN_LIMIT = 0 # [mm]
SPACE_RIGHT_LIMIT = 650 # [mm]
SPACE_LEFT_LIMIT = 0 # [mm]
BOX_HALF_LENGTH = 125 # [mm]
LEARNING_PHASE = 0
obs_set = np.copy( nrmlz_obs_set )
for i in range(len(nrmlz_obs_set)):
    
    obs_set[i,0] = obs_set[i,0] * ( SPACE_RIGHT_LIMIT - SPACE_LEFT_LIMIT ) + SPACE_LEFT_LIMIT
    obs_set[i,1] = obs_set[i,1] * ( SPACE_UP_LIMIT - SPACE_DOWN_LIMIT ) + SPACE_DOWN_LIMIT
    obs_set[i,2] = obs_set[i,2] * ( SPACE_UP_LIMIT - SPACE_DOWN_LIMIT )
    
## physical conditions
m = 1 # [kg]
std_tau = 120 # [N-mm] of torque sensor

if( train_data_path[-7:-4] == "exp" ):
    v_avg_min = 50 # [mm/sec]
    v_avg_max = 100 # [mm/sec]
else:
    v_avg_min = 100 # [mm/sec]
    v_avg_max = 150 # [mm/sec]
    
theta_B_0_range = 30 * pi/180 # [rad]
omega_B_0_range = theta_B_0_range / dt # [rad/sec]
via_pt_num = 4
intrpltn_num = 1
P = (3e-8) * np.array([[ (dt**3)/6, 0,  0 ], # initial covariance matrix for EKF
                       [ 0, (dt**2)/2,  0 ],
                       [ 0,         0, dt ]], dtype=float )
param_num = intrpltn_num * ( via_pt_num - 1 ) # parameters: theta1, theta2, ...

## parameters for iLQR
dim_x = 4
dim_u = 1
w_x = 5e-8
if( train_data_path[-7:-4] == "exp" ):
    w_omega = 1e-2
else:
    w_omega = 1e-4


## parameters for realistic robot arm environment
mu_s = np.tan( 3 * pi / 180.0 )
delay_buffer = 15 * [0.0]

####################
### FUNCTION SET ###
####################

from Libraries import cnstrct_trans_set, PID_controller, PD_ctrl, imped_ctrl, set_random_ini_cond, denrmlz_values
from Libraries import read_ini_cond, extrct_feat_pts, bndry_check, creat_train_data, normalize_X, get_theta_feat_by_DNN
from Libraries import get_x_mB_feat_by_DNN, prtl_J2_prtl_t_and_x, prtl_J22_prtl_nrmlz_x_y_v_avg, calc_t_total, adam_opt
from Libraries import permut, optmz_J2_wrt_IC, EKF, calc_t_stamp, smooth_step, calc_frctn_f
from class_libraries import ILQR, prtl_f_prtl_x_and_u
######################
#### FUNCTION END ####
######################


train_data_set = []
label_data_theta_set = []
label_data_theta_len = 0
label_data_x_mB_set = []
label_data_x_mB_len = 0
nrmlz_X_opt = []
if( Save_result_2 == True ):
    theta_B_diff_set = [] # for paper 4-4-3
    x_mB_diff_set = [] # for paper 4-4-3
dist_SSE_set = [] # for paper 4-4-2
bfr_opt_SSE_set, aftr_opt_SSE_set = np.zeros( ( 2, train_data_num ), dtype=float ) # for paper 4-5-1-2
best_ini_cond_set = [] # for paper 4-5-1-2

if( Calc_elps_time == True ): start_time = timeit.default_timer()
for train_idx in range( train_data_num * ( 1 + 2 * Valid_opt ) ):
    
    ### initialize the environment ###    
    ## initial kinematic  conditions ##
    if( Valid_opt == True and ( train_idx % 3 == 1 ) ):
        print("\nopt by total squared x:")
#        print("normalized x: \n", nrmlz_X_opt_by_x )
        via_x, via_y, v_avg, theta_B_0, x_mB_0 = denrmlz_values( nrmlz_X_opt_by_x[0:via_pt_num],
                                                                 nrmlz_X_opt_by_x[via_pt_num:2*via_pt_num],
                                                                 nrmlz_X_opt_by_x[2*via_pt_num:3*via_pt_num-1],
                                                                 nrmlz_X_opt_by_x[-2],
                                                                 nrmlz_X_opt_by_x[-1] ) # recover values, [mm,rad,sec]
    
    elif( Valid_opt == True and ( train_idx % 3 == 2 ) ):
        print("\nopt by total time")
#        print("normalized x: \n", nrmlz_X_opt_by_T )
        via_x, via_y, v_avg, theta_B_0, x_mB_0 = denrmlz_values( nrmlz_X_opt_by_T[0:via_pt_num],
                                                                 nrmlz_X_opt_by_T[via_pt_num:2*via_pt_num],
                                                                 nrmlz_X_opt_by_T[2*via_pt_num:3*via_pt_num-1],
                                                                 nrmlz_X_opt_by_T[-2],
                                                                 nrmlz_X_opt_by_T[-1] ) # recover values, [mm,rad,sec]
        
    elif( Read_ini_cond == False ):
        via_x, via_y, v_avg, theta_B_0, x_mB_0 = set_random_ini_cond() # Normalized values!!! [1/1]
        via_x, via_y, v_avg, theta_B_0, x_mB_0 = denrmlz_values( via_x, via_y, v_avg, theta_B_0, x_mB_0 ) # recover values, [mm,rad,sec]
        
    else:
        via_x, via_y, v_avg, theta_B_0, x_mB_0 = read_ini_cond( ini_cond_path, train_idx )
                   
    if( path_idx == "test_" ):
        theta_B, x_Bx, x_By = np.zeros( ( 3, int(10.0/dt) ), dtype=float )
        theta_B[0] = theta_B_0
        t_stamp = np.zeros(1)
        
    elif( Set_ini_theta_B_zero == True ):       
        via_theta = np.zeros( param_num, dtype=float )
        t_stamp = calc_t_stamp( via_x, via_y, v_avg )
        x_Bx = cbc_traj_extrm_pts( t_stamp, via_x, 0, dt, np.nan )
        x_By = cbc_traj_extrm_pts( t_stamp, via_y, 0, dt, len(x_Bx) )
#        x_Bx, x_By, _, t_duration = CubicTrajetoryPlanning_v2( via_x, via_y, np.concatenate( ( [theta_B_0], via_theta ) ) , v_avg , dt )
        theta_B = np.zeros( len(x_Bx), dtype=float )
        theta_B[0] = theta_B_0

    else:
        via_theta = np.array( [ -pi/6, pi/6, 0 ], dtype=float ) # for hard cond. and Syntec Report
        t_stamp = calc_t_stamp( via_x, via_y, v_avg )
        x_Bx = cbc_traj_extrm_pts( t_stamp, via_x, 0, dt, np.nan )
        x_By = cbc_traj_extrm_pts( t_stamp, via_y, 0, dt, len(x_Bx) )
        theta_B = cbc_traj_extrm_pts( t_stamp, np.concatenate( ( [theta_B_0], via_theta ) ), 0, dt, len(x_Bx) )
        
    itr_num = len( x_Bx )
    t_axis = np.arange( 0, itr_num*dt - 1e-10, dt ) # [sec]
    
    nrmlz_X_test = creat_train_data( via_pt_num, via_x, via_y, v_avg, theta_B_0, x_mB_0 ) # [mm,sec,rad]
    normalize_X( nrmlz_X_test ) # [1/1]
    
    if( Gen_theta_by_DNN == True ):
        
        omega_0, theta_feat_pts = get_theta_feat_by_DNN( nrmlz_X_test, DNN_model_theta, t_axis[-1], theta_B_0 ) # [rad,sec]
        theta_B_hat = cbc_traj_extrm_pts( theta_feat_pts[:,0] , theta_feat_pts[:,1] , omega_0, dt, itr_num ) #[rad]
    
    if( Gen_x_mB_by_DNN == True ):
        
        x_mB_hat_feat_pts = get_x_mB_feat_by_DNN( nrmlz_X_test, DNN_model_x_mB, t_axis[-1], x_mB_0 ) # [mm,sec]
        x_mB_hat = cbc_traj_extrm_pts( x_mB_hat_feat_pts[:,0], x_mB_hat_feat_pts[:,1], 0, dt, itr_num ) # [mm]
    
    x_B, v_B, a_B = cnstrct_trans_set( x_Bx, x_By, np.append( [0], np.diff(t_stamp) ) ) # generate ( x_B, v_B, a_B ) of the box
    
    ## variable space initialization ##
    x_mB, v_mB, a_mB =  np.zeros( ( 3, itr_num ), dtype=float ) # [mm], [mm/sec], [mm/sec^2]
    #a_mB_cntr, a_mB_box, a_mB_grvty = np.zeros( ( 3, itr_num ), dtype=float ) # for Syntec report
    theta_B_iLQR, theta_B_PID, omega_B, alpha_B = np.zeros( ( 4, itr_num ), dtype=float ) # [rad], [rad/sec], [rad/sec^2]
    theta_B_iLQR[0] = theta_B_0
    nrml_f, frctn_f = np.zeros( ( 2, itr_num ), dtype=float ) # [N]
    msr_tau, tau = np.zeros( ( 2, itr_num ), dtype=float ) # [N-mm]

    ## kinematics and force initial conditions
    x_mB[0] = x_mB_0
    v_mB[0] = ( 0 - v_B[0,0] ) * np.cos( theta_B[0] ) + ( 0 - v_B[0,1] ) * np.sin( theta_B[0] )
    nrml_f[0] = m * alpha_B[0] * x_mB[0] + 2 * m * omega_B[0] * v_mB[0] + m * ( a_B[0,1] * np.cos(theta_B[0]) + grvty * np.cos(theta_B[0]) - a_B[0,0] * np.sin(theta_B[0]) )
    nrml_f[0] = 0.001 * nrml_f[0] #[N]
    
    ## initialize variable space for iLQR
    iLQR = ILQR( itr_num, dim_x, dim_u, w_x, w_omega, grvty, dt )

    ## evaluation index
    dist_SSE = []
    energy_set = []
    if( train_idx % 3 == 0 ):
        best_SSE = np.inf # for paper 4-5-1-1
    
    #theta_B_set = []
    #x_mB_set = []

    for ep_idx in range( ep_num+2 ):
        
        PID_controller( 0 , True ) # reset PID controller
        PD_ctrl( 0 , True ) # reset PD controller
        # imped_ctrl( 0 , True )
        est_state = np.array( [ [ x_mB[0] ], [ v_mB[0] ], [ a_mB[0] ] ] )
        est_x_mB = np.copy( x_mB )
        
        ###############################
        ### Dynamic simulation part ###
        ###############################
        for idx in range( itr_num - 1 ):
            
            if( ep_idx > 0 and ep_idx < ep_num + 1 ):
                
                ## state feedback control for iLQR ##
                state = np.array([ [x_mB[idx]], [v_mB[idx]], [a_mB[idx]], [theta_B[idx]] ])
                omega_B[idx] = K_set[idx].dot( state ) + k_set[idx]        
    #            omega_B[idx] = np.clip( omega_B[idx], -0.15, 0.15 )
                theta_B[idx+1] = theta_B[idx] + omega_B[idx] * dt
                theta_B_iLQR[idx+1] = theta_B[idx+1]
    #            theta_B[idx+1] = np.clip( theta_B[idx+1], -pi/3, pi/3 )
                
            ## update the dynamic state of the sliding box ##
            a_mB[idx+1] = omega_B[idx]**2 * x_mB[idx] - ( a_B[idx,0] * np.cos(theta_B[idx]) + a_B[idx,1] * np.sin(theta_B[idx]) + grvty * np.sin(theta_B[idx]) )
            
            nrml_f[idx+1] = m * alpha_B[idx] * x_mB[idx] + 2 * m * omega_B[idx] * v_mB[idx] + m * ( a_B[idx,1] * np.cos(theta_B[idx]) + grvty * np.cos(theta_B[idx]) - a_B[idx,0] * np.sin(theta_B[idx]) )
            nrml_f[idx+1] = 0.001 * nrml_f[idx+1] # unit translation to [N]
            
            frctn_f[idx+1] = Add_frctn * calc_frctn_f( nrml_f[idx+1], a_mB[idx+1], mu_s, v_mB[idx], m ) # [N]
            a_mB[idx+1] = a_mB[idx+1] + frctn_f[idx+1] / m * 1000
                
            v_mB[idx+1] = v_mB[idx] + a_mB[idx+1] * dt
            x_mB[idx+1] = x_mB[idx] + v_mB[idx+1] * dt + 0.5 * a_mB[idx+1] * dt**2
            x_mB[idx+1], v_mB[idx+1] = bndry_check( x_mB[idx+1], v_mB[idx+1] ) # Boundary check
            
            tau[idx+1] = x_mB[idx+1] * nrml_f[idx+1] # [N-mm]=[kg-m/sec^2 - mm]
            
            
            ## extended Kalman filter part ##
            if( Use_EKF == True ):
            
        #        msr_tau[idx+1] = tau[idx+1] * np.random.normal( 1., 0.03 )
                msr_tau[idx+1] = tau[idx+1] + np.random.normal( 0., 120 )        
                
                if( idx > 100 ):                   
                    est_state, P = EKF( est_state, 1000*np.array([[msr_tau[idx+1]]]), P,  m, theta_B[idx], omega_B[idx], alpha_B[idx], a_B[idx,:] )                
                    est_state[0,0], est_state[1,0] = bndry_check( est_state[0,0], est_state[1,0] ) # boundary check for EKF
                   
                else:
                    est_state = np.array( [ [ x_mB[idx+1] ], [ v_mB[idx+1] ], [ a_mB[idx+1] ] ] )
            
                est_x_mB[idx+1] = est_state[0,0]
                
            else:
                est_x_mB[idx+1] = x_mB[idx+1]
            
            ## PID controller part ##
            if( ep_idx == ep_num + 1 and PID_switch == True ):
                
                theta_B_PID[idx+1] = PID_controller( est_x_mB[idx+1] , False ) * pi/180 # [rad]
                # theta_B_PID[idx+1] = imped_ctrl( est_x_mB[idx+1], False, dt ) * pi/180
                if( Add_delay == True ):
                    delay_buffer.append( np.clip( theta_B_PID[idx+1], -pi/6, pi/6 ) )
                    theta_B[idx+1] = theta_B[idx+1] + delay_buffer.pop(0)
                
                else:
                    theta_B[idx+1] = theta_B[idx+1] + theta_B_PID[idx+1]
                    theta_B[idx+1] = np.clip( theta_B[idx+1], -pi/3, pi/3 )               
                
                omega_B[idx+1] = ( theta_B[idx+1] - theta_B[idx] ) / dt
            
            if( ep_idx == ep_num + 1 and x_track_switch == True ):
                
                theta_B_PID[idx+1] = PID_controller( est_x_mB[idx+1] - x_mB_hat[idx+1] * smooth_step( (idx+1) / itr_num ), False ) * pi/180 # [rad]
#                theta_B_PID[idx+1] = imped_ctrl( est_x_mB[idx+1], False ) * pi/180
                theta_B[idx+1] = theta_B_hat[idx+1] + theta_B_PID[idx+1]
#                theta_B[idx+1] = theta_B[idx+1] + theta_B_PID[idx+1]
                theta_B[idx+1] = np.clip( theta_B[idx+1], -pi/3, pi/3 )               
                omega_B[idx+1] = ( theta_B[idx+1] - theta_B[idx] ) / dt
            
            if( Gen_theta_by_DNN == True and ( PID_switch and x_track_switch ) == False ):
                omega_B[idx+1] = ( theta_B[idx+1] - theta_B[idx] ) / dt
                
            alpha_B[idx+1] = ( omega_B[idx+1] - omega_B[idx] ) / dt
            if( len(energy_set) < ep_idx + 1 ):
                energy_set.append(0.)
            energy_set[ep_idx] = energy_set[ep_idx] + omega_B[idx] * tau[idx] * dt
            
        ### iLQR part ###
        K_set, k_set = iLQR.implmnt( theta_B, omega_B, a_B, x_mB, v_mB, a_mB )
            
#        dist_SSE.append( np.sqrt( np.mean( x_mB**2 ) ) )
        dist_SSE.append( np.sum( x_mB**2 ) * dt )
        
        ## Early stopping for iLQR
        if( Earlystopping_iLQR == True and ep_num > 0 ):
            if( len(dist_SSE) > 1 and abs( dist_SSE[-1] - dist_SSE[-2] ) < 0.01 ):
                x_mB_iLQR = np.copy( x_mB )
                break
        
        if( Create_feat_and_decmprss == True ):
            
            theta_feat_pts, omega_0 = extrct_feat_pts( t_axis, theta_B, 0.0001 * (pi/180) ) # [sec], [rad]
            x_mB_feat_pts, _ = extrct_feat_pts( t_axis, x_mB, 0.001 ) # [mm,sec]
            decmprss_traj_theta = cbc_traj_extrm_pts( theta_feat_pts[:,0] , theta_feat_pts[:,1] , omega_0, dt, itr_num )
            decmprss_traj_x_mB = cbc_traj_extrm_pts( x_mB_feat_pts[:,0] , x_mB_feat_pts[:,1] , 0, dt, itr_num )
            
        if( ep_idx == ep_num + 1 and Valid_by_decmprss == True ):
            
            plt.plot( x_Bx, x_By )
            plt.show()
            
            plt.plot( t_axis, theta_B * 180/pi )
            plt.plot( t_axis, decmprss_traj_theta * 180/pi )
            plt.show()
            
            plt.plot( t_axis, x_mB )
            plt.plot( t_axis, decmprss_traj_x_mB )
            plt.show()
            
            ## check whether DNN generates ideal feature theta points and omega_0 ( ep_num > 0 )
#            print( "label: ", omega_0 )
#            omega_0, theta_feat_pts = get_theta_feat_by_DNN( nrmlz_X_test, DNN_model, t_axis[-1], theta_B_0 )            
#            print( "predict: ", omega_0 )
#            plt.scatter(theta_feat_pts[:,0],theta_feat_pts[:,1])
#            plt.show()
            
            # theta_B = decmprss_traj
            # if( len(theta_B) > itr_num ): theta_B = theta_B[:itr_num]
            # elif( len(theta_B) < itr_num ): theta_B = np.append( theta_B, np.array( ( itr_num - len(theta_B) ) * [theta_B[-1]] ) )
        
        if( ep_idx == ep_num ):
            x_mB_iLQR = np.copy( x_mB )
    
    if( Plot_figures == True ):
        
        if( len(nrmlz_obs_set) > 0 ):
            fig, ax = plt.subplots()
            for obst_idx in range(len(nrmlz_obs_set)):       
                ax.add_artist(plt.Circle(obs_set[obst_idx,0:2], obs_set[obst_idx,2], fill=False))        
        
        plt.plot( x_Bx, x_By )
        plt.plot( [ x_Bx[0] + BOX_HALF_LENGTH * np.cos( theta_B[0] ) , x_Bx[0] - BOX_HALF_LENGTH * np.cos( theta_B[0] ) ],
                  [ x_By[0] + BOX_HALF_LENGTH * np.sin( theta_B[0] ) , x_By[0] - BOX_HALF_LENGTH * np.sin( theta_B[0] ) ], 'r-' )
        plt.scatter( x_Bx[0] + x_mB[0] * np.cos( theta_B[0] ),
                     x_By[0] + x_mB[0] * np.sin( theta_B[0] ), c="k" )
        plt.axis('equal')
        plt.show()        
        
        plt.plot( t_axis, theta_B * 180/pi )
        if( ep_num > 0 ):
            plt.plot( t_axis, theta_B_iLQR * 180/pi )
        if( Gen_theta_by_DNN == True ):
            plt.plot( t_axis, theta_B_hat * 180/pi )
        plt.show()        
        
        plt.plot( t_axis, x_mB )
        if( ep_num > 0 ):
            plt.plot( t_axis, x_mB_iLQR )
        if( Gen_x_mB_by_DNN == True ):
            plt.plot( t_axis, x_mB_hat )
        plt.show()
    
    if( Save_result_2 == True ):
        theta_B_diff_set.append(  theta_B_hat - theta_B_iLQR ) # only for paper 4-3
        x_mB_diff_set.append( x_mB_hat - x_mB_iLQR ) # only for paper 4-3
    
    # Compare opt by sum-of-squared-x with total time under running dynamic system and select the best
    if( Valid_opt == 1 ):
        if( ( -2.0 * Worst_case + 1.0 ) * dist_SSE[-1] < best_SSE ):
            
            if( train_idx % 3 == 2 ):
                print("best is opt by total time")
                best_idx = best_IC_permut_idx[1]
            
            best_SSE = ( -2.0 * Worst_case + 1.0 ) * dist_SSE[-1]
            best_ini_cond = creat_train_data( via_pt_num, via_x, via_y, v_avg, theta_B_0, x_mB_0 ) # [mm,sec,rad]
            if( 'best_idx' in globals() ):
                best_ini_cond = np.append( best_ini_cond, best_idx )
            best_theta_B = np.copy( theta_B )
            best_x_mB = np.copy( x_mB )
            best_theta_B_hat = np.copy( theta_B_hat )
            best_x_mB_hat = np.copy( x_mB_hat )
        
        elif( train_idx % 3 == 2 ):
            print("best is opt by sum of squared x")
            best_idx = best_IC_permut_idx[0]
            best_ini_cond = np.append( best_ini_cond, best_idx )
        
        aftr_opt_SSE_set[int(train_idx/3)] = best_SSE
        
        if( train_idx % 3 == 2 ):
            best_ini_cond_set.append( best_ini_cond )
    
    if( Valid_opt == True and train_idx % 3 == 0 ):
        print("\n======================")
    
    if(Print_SSE == True and train_idx % 100 == 0 ): print( "Data", train_idx, "x : %.4f" %dist_SSE[-1], ", E: %.4f" %energy_set[-1] )
    if(Print_task_time == True ): print( "task time: %.4f" %t_axis[-1] )
    if(Print_nrmlz_X == True ): print( nrmlz_X_test )
    
    dist_SSE_set.append( dist_SSE[-1] ) # for paper 4-4-2
    
    if( Valid_opt == True and ( train_idx % 3 == 0 ) ):
        
        bfr_opt_SSE_set[int(train_idx/3)] = dist_SSE[-1]
        nrmlz_X_set, IC_num = permut( nrmlz_X_test, grdnt_mask )
#        nrmlz_X_set = [np.array([1., 1/3, 2/3, 0., 1., 1., 0., 0., 1., 1., 1., 0., 1.])] # for debugging
#        IC_num = 1 # for debugging
        

        J2_best = 1e8
        t_total_best = 1e8
        
        GD_process_set = []
        J2_process_set = []
        process_len_set = []
        best_IC_permut_idx = np.zeros( 2, dtype=int )
        
        for permut_idx in range(IC_num):
            
            nrmlz_X, t_total, J2, J2_process, GD_process, process_len = optmz_J2_wrt_IC( nrmlz_X_set[permut_idx], DNN_model_x_mB, jacobian_t_x_to_IC, grdnt_mask, nrmlz_obs_set, Worst_case  )
            if( Worst_case == True ):
                print("[", permut_idx, "]: optimized J2 : %.4f" %J2 )
            J2 = ( -2.0 * Worst_case + 1.0 ) * J2
            t_total = ( -2.0 * Worst_case + 1.0 ) *  t_total
#            print("optimized t_total : %.4f" %t_total )
            GD_process_set.append( GD_process )
            J2_process_set.append( J2_process )
            process_len_set.append( process_len )
            
            if( J2 < J2_best ):
                nrmlz_X_opt_by_x = nrmlz_X
                J2_best = J2
                best_IC_permut_idx[0] = permut_idx
            
            if( t_total < t_total_best and J2 < 8e7 ):
                nrmlz_X_opt_by_T = nrmlz_X
                t_total_best = t_total
                best_IC_permut_idx[1] = permut_idx
        
#        GD_process_set = np.array([np.array(xi) for xi in GD_process_set] )
        GD_process_set = np.vstack( GD_process_set )
        J2_process_set = np.vstack( J2_process_set )
        process_len_set = np.array( process_len_set )
        if( Plot_figures2 == True ):
#            for permut_idx in range(IC_num):
#                plt.plot( GD_process_set[permut_idx] )
#            plt.show()
            
            for permut_idx in range(IC_num):
                plt.plot( J2_process_set[ permut_idx, : process_len_set[permut_idx] ] )
            plt.show()
        
#        print("\nbest optimized J2: %.4f" %J2_best, "\n" )
            
    if( Create_train_set == True ):
        
#        theta_feat_pts, omega_0 = extrct_feat_pts( t_axis, theta_B, 0.0001 * (pi/180) ) # [sec], [rad]
#        x_mB_feat_pts, _ = extrct_feat_pts( t_axis, x_mB, 0.0001 ) # [mm]
#        decmprss_traj_theta = cbc_traj_extrm_pts( theta_feat_pts[:,0] , theta_feat_pts[:,1] , omega_0, dt, itr_num )
#        decmprss_traj_x_mB = cbc_traj_extrm_pts( x_mB_feat_pts[:,0] , x_mB_feat_pts[:,1] , 0., dt, itr_num )
        
#        plt.plot( theta_B )
#        plt.plot( decmprss_traj )
#        plt.show()

        # create a row of training data        
        train_data = creat_train_data( via_pt_num, via_x, via_y, v_avg, theta_B_0, x_mB_0 ) # [mm,sec,rad]
        train_data_set.append( train_data )
        
        label_data_theta = np.append( [ omega_0 ], theta_feat_pts.reshape(-1)[2:] ) # [rad,sec]
        label_data_theta = np.append( [ len(label_data_theta) ], label_data_theta )
        label_data_theta_set.append( label_data_theta )
        
        if( len( label_data_theta ) > label_data_theta_len ):
            label_data_theta_len = len( label_data_theta )
        
        label_data_x_mB = x_mB_feat_pts.reshape(-1)[2:] # [rad,sec]
        label_data_x_mB = np.append( [ len(label_data_x_mB) ], label_data_x_mB )
        label_data_x_mB_set.append( label_data_x_mB )

        if( len( label_data_x_mB ) > label_data_x_mB_len ):
            label_data_x_mB_len = len( label_data_x_mB )

if( Calc_elps_time == True ): print( 'Elapsed Time: ', timeit.default_timer() - start_time )
    
if( Create_train_set == True ):    
    
    train_data_set = np.vstack( train_data_set )
    np.savetxt( train_data_path, train_data_set, delimiter="," )

    ## align labeled data
    for i in range( train_data_num ):
        
        for j in range( label_data_theta_len - len( label_data_theta_set[i] ) ):
            label_data_theta_set[i] = np.append( label_data_theta_set[i], [ np.inf ] )
        
        for j in range( label_data_x_mB_len - len( label_data_x_mB_set[i] ) ):
            label_data_x_mB_set[i] = np.append( label_data_x_mB_set[i], [ np.inf ] )
            

    label_data_theta_set = np.vstack( label_data_theta_set )
    np.savetxt( label_data_theta_path, label_data_theta_set, delimiter="," )
    label_data_x_mB_set = np.vstack( label_data_x_mB_set )
    np.savetxt( label_data_x_mB_path, label_data_x_mB_set, delimiter="," )

    
### record simulation results
if( Save_result == True ):
    
    window_half = 2
    for i in range( window_half, len(msr_tau)-window_half ):
        msr_tau[i] = np.median( msr_tau[ (i - window_half) : (i + window_half) ] )
            
    np.savetxt( "./" + directory + "x_Bx_" + path_idx + ".csv", x_Bx )
    np.savetxt( "./" + directory + "x_By_" + path_idx + ".csv", x_By )
    np.savetxt( "./" + directory + "t_axis_" + path_idx + ".csv", t_axis )
    
    if( PID_switch == True ):
        np.savetxt( "./" + directory + "theta_B_PID_" + path_idx + ".csv", theta_B )
        np.savetxt( "./" + directory + "x_mB_PID_" + path_idx + ".csv", x_mB )
    
    if( ep_num > 0 ):
        np.savetxt( "./" + directory + "theta_B_iLQR_" + path_idx + ".csv", theta_B_iLQR )
        np.savetxt( "./" + directory + "x_mB_iLQR_" + path_idx + ".csv", x_mB_iLQR )
    
    if( Gen_theta_by_DNN == True ):
        np.savetxt( "./" + directory + "theta_B_hat(" + model_idx_str + ")_" + path_idx + ".csv", theta_B_hat )
        
    if( Gen_x_mB_by_DNN == True ):
        np.savetxt( "./" + directory + "x_mB_hat(" + model_idx_str + ")_" + path_idx + ".csv", x_mB_hat )
    
    if( x_track_switch == 1 ):
        np.savetxt( "./" + directory + "theta_B_NN(" + model_idx_str + ")_" + path_idx + ".csv", theta_B )
        np.savetxt( "./" + directory + "x_mB_NN(" + model_idx_str + ")_" + path_idx + ".csv", x_mB )
        
if( Save_result_2 == True and len(theta_B_diff_set) > 0 and len(x_mB_diff_set) > 0 ): # for paper 4-3-2
    
    file_name = "./" + directory + "x_mB_diff(" + model_idx_str + ").csv"
    with open( file_name, "w" ) as f:
        f.write( "\n".join(",".join( map( str, x ) ) for x in x_mB_diff_set ) )
    
    file_name = "./" + directory + "theta_B_diff(" + model_idx_str + ").csv"
    with open( file_name, "w" ) as f:
        f.write( "\n".join(",".join( map( str, x ) ) for x in theta_B_diff_set ) )
   
if( Save_result_3 == True ): # for paper 4-4-2
    
    dist_SSE_set = np.array( dist_SSE_set )
    if( ep_num > 0 and Gen_theta_by_DNN == 0 and Gen_x_mB_by_DNN == 0 ):
        np.savetxt( "./" + directory + "x_mB_iLQR_RMSE.csv", dist_SSE_set )
    
    if( ep_num == 0 and Gen_theta_by_DNN == True and Gen_x_mB_by_DNN == True ):
        np.savetxt( "./" + directory + "x_mB_NN_RMSE(" + model_idx_str + ").csv", dist_SSE_set )
    

if( Valid_opt == True and Save_result_4 == True ): # for paper 4-5-1-1
    
    best_ini_cond_set = np.vstack( best_ini_cond_set )
    
    ## for paper 4-5-1-1
    if(train_data_num == 1 ):
        np.savetxt( "./" + directory + "t_axis_" + path_idx + ".csv", t_axis )
        np.savetxt( "./" + directory + "theta_B_hat(" + model_idx_str + ")_" + path_idx + ".csv", best_theta_B_hat )
        np.savetxt( "./" + directory + "x_mB_hat(" + model_idx_str + ")_" + path_idx + ".csv", best_x_mB_hat )
        np.savetxt( "./" + directory + "theta_B(" + model_idx_str + ")_" + path_idx + ".csv", best_theta_B )
        np.savetxt( "./" + directory + "x_mB(" + model_idx_str + ")_" + path_idx + ".csv", best_x_mB )
        np.savetxt( "./" + directory + "best_ini_cond(" + model_idx_str + ")_" + path_idx + ".csv", best_ini_cond, delimiter="," )
        np.savetxt( "./" + directory + "GD_process(" + model_idx_str + ")_" + path_idx + ".csv", GD_process_set, delimiter="," )
        np.savetxt( "./" + directory + "J2_process(" + model_idx_str + ")_" + path_idx + ".csv", J2_process_set, delimiter="," )
        np.savetxt( "./" + directory + "process_len(" + model_idx_str + ")_" + path_idx + ".csv", process_len_set, delimiter="," )
    
    ## for paper 4-5-1-2
    else:
        np.savetxt( "./" + directory + "best_ini_cond_set(" + model_idx_str + ").csv", best_ini_cond_set, delimiter="," )
        np.savetxt( "./" + directory + "bfr_opt_SSE_set(" + model_idx_str + ").csv", bfr_opt_SSE_set, delimiter="," )
        np.savetxt( "./" + directory + "aftr_opt_SSE_set(" + model_idx_str + ").csv", aftr_opt_SSE_set, delimiter="," )
    
#np.savetxt( "./SyntecReport/est_x_mB.csv", est_x_mB)
#np.savetxt( "./SyntecReport/msr_tau.csv", msr_tau)


