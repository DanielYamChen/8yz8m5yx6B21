# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:41:59 2019

@author: Bo-Hsun Chen
"""

import sys
import time
import numpy as np
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
from math import pi
from keras.models import load_model
from TrajectoryPlanning import CubicTrajetoryPlanning_v2, cbc_traj_extrm_pts

np.set_printoptions( formatter={'float': '{: 0.3f}'.format} )

MODEL_NUMBER = 1
seeds = [ 5460 , 12 , 445 , 2500, 1111, 111, 11, 1 ]
Read_ini_cond = 1
#ini_cond_path = "./GA_data/ini_cond(" + str(MODEL_NUMBER) + ").csv"
#ini_cond_path = "./ini_cond_hard.csv" # read hard cond. for Syntec report section 5.2 to 5.3
#ini_cond_path = "./ini_cond/ini_cond_hard_2.csv"
ini_cond_path = "./ini_cond/5460_9.csv"
iLQR_switch = 0
Use_EKF = 0
Gen_theta_by_DNN = 1
Create_train_set = 0
PID_switch = 1
Valid_orgnl_decmprss = 0
ep_num = 0
train_data_num = 1

#tf.set_random_seed( seeds[MODEL_NUMBER-1] )
np.random.seed( seeds[MODEL_NUMBER+1] )
#np.random.seed( int( time.time() ) )

model_path = "ML_data/ISOSC(" + str(seeds[MODEL_NUMBER-1]) + ").h5"

## Global condition
dt = 0.012 # [sec]
grvty = 9810 # [mm/sec^2]
SPACE_UP_LIMIT = 650 # [mm]
SPACE_DOWN_LIMIT = 0 # [mm]
SPACE_RIGHT_LIMIT = 650 # [mm]
SPACE_LEFT_LIMIT = 0 # [mm]
BOX_HALF_LENGTH = 125 # [mm]

## physical conditions
m = 1 # [kg]
std_tau = 120 # [N-mm] of torque sensor
v_avg_min = 100 # [mm/sec]
v_avg_max = 150 # [mm/sec]
theta_B_0_range = 30 * pi/180 # [rad]
omega_B_0_range = theta_B_0_range / dt
via_pt_num = 4
intrpltn_num = 1
P = (3e-8) * np.array([[ (dt**3)/6, 0,  0 ], # covariance matrix for Kalman filter
                       [ 0, (dt**2)/2,  0 ],
                       [ 0,         0, dt ]], dtype=float)
param_num = intrpltn_num * ( via_pt_num - 1 ) # parameters: theta1, theta2, ...
X_BOUND = np.array([ -pi/6 * np.ones( param_num, dtype=float ), pi/6 * np.ones( param_num, dtype=float ) ]) # x upper and lower bounds


## parameters for iLQR
dim_x = 4
dim_u = 1
w_x = 1e-9
w_omega = 1e-4


####################
### FUNCTION SET ###
####################

## construct translation trajectory (x,v,a) of the box
def cnstrct_trans_set( x_Bx, x_By ):
      
    x_B = np.vstack((x_Bx, x_By)).T # [mm]
    v_B = np.vstack( ( [0,0], np.diff( x_B, axis=0 ) ) ) / dt # [mm/sec]

    # blending the velocity plot
    t_idx = 0
    for i in t_duration[0:-1]:
        t_idx = t_idx + int(i/dt)
        v_B[t_idx] = ( v_B[t_idx-1] + v_B[t_idx+1] ) /2

    a_B = np.vstack( ( [0,0], np.diff( v_B, axis=0 ) ) ) / dt # [mm/sec^2]
    
    return x_B, v_B, a_B
        
def PID_controller( error , reset ):

    if not hasattr( PID_controller, "error_prvs" ):
        PID_controller.error_prvs = error
    
    if not hasattr( PID_controller, "integral_term" ):
        PID_controller.integral_term = 0
    
    if( reset == True ):
        PID_controller.error_prvs = error
        PID_controller.integral_term = 0
    
    ## for Syntec report Part I
#    P_coefficient = 0.0 #[deg/mm]
#    I_coefficient = 0.0
#    D_coefficient = 0.00
    
    ## for Syntec report Part II
#    P_coefficient = 0.7 #[deg/mm]
#    I_coefficient = 0.0
#    D_coefficient = 0.00
    
    ## perfect ##
    P_coefficient = 0.7 #[deg/mm]
    I_coefficient = 0.2
    D_coefficient = 0.034
    
    ## perfect with using iLQR and EKF ##
#    P_coefficient = 0.7 #[deg/mm]
#    I_coefficient = 1.0
#    D_coefficient = 0.03
    
    ## perfect with using iLQR ##
#    P_coefficient = 0.3 #[deg/mm]
#    I_coefficient = 0.03
#    D_coefficient = 0.0
    
    ## for DNN_model ##
    P_coefficient = 0.5 #[deg/mm]
    I_coefficient = 0.0
    D_coefficient = 0.03
    
    PID_controller.integral_term = PID_controller.integral_term + error * dt
    delta_x_command = P_coefficient * ( error + I_coefficient * PID_controller.integral_term + D_coefficient * ( error - PID_controller.error_prvs ) / dt )
    PID_controller.error_prvs = error
    
#    if( delta_x_command > 90.0 ):
#        return 90.0
#    elif( delta_x_command < -90.0 ):
#        return -90.0
#    else:
   
    return delta_x_command

def imped_ctrl( err , reset ):
    
    if not hasattr( imped_ctrl, "err_prvs" ):
        imped_ctrl.err_prvs = 0
    
    if not hasattr( imped_ctrl, "err_prprvs" ):
        imped_ctrl.err_prprvs = 0
    
    if( reset == True ):
        imped_ctrl.err_prvs = 0
        imped_ctrl.err_prprvs = 0
        
    k = 0.6 #[deg/mm]
    m = 0.0002
#    k = 2 #[deg/mm]
#    m = 1000   
    
    c = 2 * np.sqrt( k * m )
    
    delta_x_command = ( k + c / dt + m / dt**2 ) * err - ( c / dt + 2 * m / dt**2 ) * imped_ctrl.err_prvs + m / dt**2 * imped_ctrl.err_prprvs
    imped_ctrl.err_prprvs = imped_ctrl.err_prvs
    imped_ctrl.err_prvs = err
    
#    delta_x_command = ( err * dt**2 + ( c * dt + 2 * m ) * imped_ctrl.err_prvs - m * imped_ctrl.err_prprvs ) / ( k* dt**2 + c * dt + m )
#    imped_ctrl.err_prprvs = imped_ctrl.err_prvs
#    imped_ctrl.err_prvs = delta_x_command
    
    return delta_x_command

## randomly choose 4 via_points, average linear speed set, initial angle of box, initial relative position
def set_random_ini_cond():
    
    # randomly choose 4 point in each side
    temp_x = [ 0, np.random.uniform( 0, 1 ), 
               1, np.random.uniform( 0, 1 ) ]
    
    temp_y = [ np.random.uniform( 0, 1 ), 1,
               np.random.uniform( 0, 1 ), 0 ]
    
    # randomly decide the connection order    
    order = np.random.permutation(4)        
    
    # decide via points
    via_x = []
    via_y = []
    for i in range( via_pt_num ):
        via_x.append( temp_x[order[i]] )
        via_y.append( temp_y[order[i]] )
    
    via_x = np.array( via_x )
    via_y = np.array( via_y )
    
    return via_x, via_y, np.random.uniform( 0, 1, size=via_pt_num-1 ), np.random.uniform( 0, 1 ), np.random.uniform( 0, 1 )

## recover normalized values to original range
def denrmlz_values( via_x, via_y, v_avg, theta_B_0, x_mB_0 ):
    
    via_x = via_x * ( SPACE_RIGHT_LIMIT - SPACE_LEFT_LIMIT ) + SPACE_LEFT_LIMIT # [mm]
    via_y = via_y * ( SPACE_UP_LIMIT - SPACE_DOWN_LIMIT ) + SPACE_DOWN_LIMIT # [mm]
    v_avg = v_avg * ( v_avg_max - v_avg_min ) + v_avg_min #[mm/sec]
    theta_B_0 = ( theta_B_0 * 2 - 1 ) * theta_B_0_range #[rad]
    x_mB_0 = ( x_mB_0 * 2 - 1 ) * BOX_HALF_LENGTH #[mm]
    
    return via_x, via_y, v_avg, theta_B_0, x_mB_0

## read recored generated initial kinematic condition
def read_ini_cond( path ):
    
    ini_cond = np.genfromtxt( path )
    ini_cond = ini_cond.tolist()
    via_x = []
    via_y = []
    v_avg = []
    for i in range( via_pt_num ):
        via_x.append( ini_cond.pop(0) )
    via_x = np.array( via_x )
    
    for i in range( via_pt_num ):
        via_y.append( ini_cond.pop(0) )    
    via_y = np.array( via_y )
    
    for i in range( via_pt_num - 1 ):
        v_avg.append( ini_cond.pop(0) )    
    v_avg = np.array( v_avg )
    
    theta_B_0 = ini_cond.pop(0)
    x_mB_0 = ini_cond.pop(0)
    
    return via_x, via_y, v_avg, theta_B_0, x_mB_0

## get gradient of f to state x and input u
def prtl_f_prtl_x_and_u( theta_B, omega_B, a_B, x_mB ):
    temp = a_B[0] * np.sin(theta_B) - a_B[1] * np.cos(theta_B) - grvty * np.cos(theta_B)
    
    return np.array([[            1, dt, dt**2/2,    0,              0 ],
                     [            0,  1,      dt,    0,              0 ],
                     [   omega_B**2,  0,       0, temp, 2*x_mB*omega_B ],
                     [            0,  0,       0,    1,             dt ]], dtype=float )

## extended Kalman filter
def EKF( state_prvs, msr_state, P_prvs, m, theta_B, omega_B, alpha_B, a_B ):
    
    std_tau = 120 # [N-mm]
        
    # state = [ x_mB v_mB a_mB ]^T 
    # msr_state = [ torque ]

    A = np.array([[            1, dt,  0.5*(dt**2) ],
                  [            0,  1,           dt ],
                  [ (omega_B**2),  0,            0 ]], dtype=float )
    
    u = np.array( [ [0], [0], [ - a_B[0] * np.cos(theta_B) - a_B[1] * np.sin(theta_B) - grvty * np.sin(theta_B) ] ], dtype=float )

    C = np.array( [[ 2*m*alpha_B*state_prvs[0,0] + 2*m*omega_B*state_prvs[1,0] + m*a_B[1]*np.cos(theta_B) + m*grvty*np.cos(theta_B) - m*a_B[0]*np.sin(theta_B),
                    2*m*omega_B*state_prvs[0,0],
                    0 ]], dtype=float )
                        
    # measurement covariance matrix 
    R = np.array([[ std_tau ]])
    
    # systematic covariance matrix       
    Q = 3e-6 * np.array([[ (dt**3)/6,         0,  0 ],
                         [         0, (dt**2)/2,  0 ],
                         [         0,         0, dt ]], dtype=float )
#    Q = 3e-10 * np.array([[ (dt**3)/6,         0,  0 ],
#                         [         0, (dt**2)/2,  0 ],
#                         [         0,         0, dt ]], dtype=float )
               
    I = np.identity( 3, dtype=float )
    
    state_bar = A.dot( state_prvs ) + u
#    print(state_bar)
    P_bar = A.dot( P_prvs.dot( A.T ) ) + Q
    #             3*3       3*1                      1*3      3*3       3*1       1
    KalmanGain = P_bar.dot( C.T ).dot( np.linalg.inv( C.dot( P_bar.dot( C.T ) ) + R ) )
#    print(KalmanGain)
    est_state = state_bar + KalmanGain.dot(  msr_state - C.dot( state_bar ) )
#    print(msr_state-C.dot( state_bar ))
    P = ( I - KalmanGain.dot( C ) ).dot( P_bar )
    
    return est_state, P

## extract extreme points of the trajectory
def extrct_feat_pts( t_axis, traj ):
    
    wndw_size = 7
    tol = 0.0001 * (pi/180) # [rad]
    
    feat_pts = []
    feat_pts.append( [ 0, traj[0] ] )
    ## find extreme point
    
    
    for i in range( wndw_size, len(traj)-wndw_size ):
        
        head = i - wndw_size
        tail = i + wndw_size
        segment = traj[head:tail]
        
        if( np.max( segment ) - np.min( segment ) >= tol ): # avoid flate segment
            if( ( head + np.argmax( segment ) ) == i or ( head + np.argmin(segment) ) == i ):
                
                feat_pts.append( [ t_axis[i], traj[i] ] )
    
    feat_pts.append( [ t_axis[-1], traj[-1] ] )    
    feat_pts = np.vstack( feat_pts )
    
    v_0 = ( traj[1] - traj[0] ) / ( t_axis[1] - t_axis[0] )
    
    return feat_pts, v_0

## boundary constraints for the internal sliding object
def bndry_check( x, v ):
    
    if( x >= BOX_HALF_LENGTH or x <= - BOX_HALF_LENGTH ):
                
        if( np.sign( x ) > 0 ): # at right boudary 
            
            return BOX_HALF_LENGTH, np.clip( v, -np.inf, 0. )
                
        else: # at left boundary
            
            return - BOX_HALF_LENGTH, np.clip( v, 0., np.inf )
    
    else:
        
        return x, v

## create a stroke of training data
def creat_train_data( via_pt_num, via_x, via_y, v_avg, theta_B_0, x_mB_0 ):
    
    ini_cond = []
    for i in range( via_pt_num ):
        ini_cond.append( via_x[i] )
    
    for i in range( via_pt_num ):
        ini_cond.append( via_y[i] )
        
    for i in range( via_pt_num-1 ):
        ini_cond.append( v_avg[i] )
    
    ini_cond.append( theta_B_0 )
    ini_cond.append( x_mB_0 )
    
    return np.array( ini_cond )
    
## normalized x_test to 0 to 1
def normalize_x( nrmlz_x ):
    
    # raw_x : via_x*4 [mm], via_y*4 [mm], v_avg*3 [mm/sec], theta_B_0 [rad], x_mB_0 [mm]
    nrmlz_x[:4] = ( nrmlz_x[:4] - SPACE_LEFT_LIMIT ) / ( SPACE_RIGHT_LIMIT - SPACE_LEFT_LIMIT ) # via_x*4
    nrmlz_x[4:8] = ( nrmlz_x[4:8] - SPACE_DOWN_LIMIT ) / ( SPACE_UP_LIMIT - SPACE_DOWN_LIMIT ) # via_y*4
    nrmlz_x[8:11] = ( nrmlz_x[8:11] - v_avg_min ) / ( v_avg_max - v_avg_min ) # v_avg*3
    nrmlz_x[11] = ( nrmlz_x[11] / theta_B_0_range + 1 ) / 2
    nrmlz_x[12] = ( nrmlz_x[12] / BOX_HALF_LENGTH + 1 ) / 2

    return

def get_theta_feat_by_DNN( nrmlz_x_test, DNN_model, t_duration, theta_B_0 ):

    nrmlz_y = DNN_model.predict( nrmlz_x_test.reshape(1,-1) ).reshape(-1)
    
    nrmlz_y[0] = nrmlz_y[0] * ( 2 * omega_B_0_range ) - omega_B_0_range # denormalize omega_B_0
    
    # transform the relative form of time stamps to absolute form
    for i in range( 3, len(nrmlz_y), 2 ):
        nrmlz_y[i] = nrmlz_y[i] + nrmlz_y[i-2]
       
    for i in range( 1, len(nrmlz_y), 2 ):
    
        nrmlz_y[i] = nrmlz_y[i] * t_duration # time_stamp
        nrmlz_y[i+1] = nrmlz_y[i+1] * ( 2 * theta_B_0_range ) - theta_B_0_range # theta_B_feat
    
    omega_0 = nrmlz_y[0]
    nrmlz_y = np.delete( nrmlz_y, [0] )
    nrmlz_y = nrmlz_y.reshape(-1,2)
    
    t_offset = nrmlz_y[0,0]
    for i in range( len(nrmlz_y) ): # stretch the time stamp of theta_B to full scale
        nrmlz_y[i,0] = ( nrmlz_y[i,0] - t_offset ) / ( nrmlz_y[-1,0] - t_offset ) * t_duration
    
    nrmlz_y[0,1] = theta_B_0 # [rad]
    
    return omega_0, nrmlz_y

    
######################
#### FUNCTION END ####
######################
if( Gen_theta_by_DNN==True ): DNN_model = load_model( model_path ) # load DNN model
    
train_data_set = []
label_data_set = []
label_data_len = 0
for train_idx in range( train_data_num ):
    
    ### initialize the environment ###    
    ## initial kinematic  conditions ##
    if( Read_ini_cond == False ):
        via_x, via_y, v_avg, theta_B_0, x_mB_0 = set_random_ini_cond() # !!!Normalized values
        via_x, via_y, v_avg, theta_B_0, x_mB_0 = denrmlz_values( via_x, via_y, v_avg, theta_B_0, x_mB_0 ) # recover values        
    
    else:
        via_x, via_y, v_avg, theta_B_0, x_mB_0 = read_ini_cond( ini_cond_path )
       
    via_theta = np.zeros( param_num, dtype=float )
    x_Bx, x_By, theta_B, t_duration = CubicTrajetoryPlanning_v2( via_x, via_y, np.concatenate( ( [theta_B_0], via_theta ) ) , v_avg , dt )
    theta_B[1:] = 0.    
    
    itr_num = len( x_Bx )
    t_axis = np.arange( 0, itr_num*dt - 1e-10, dt ) # [sec]
    
    if( Gen_theta_by_DNN == True ):
        
        nrmlz_x_test = creat_train_data( via_pt_num, via_x, via_y, v_avg, theta_B_0, x_mB_0 )
        normalize_x( nrmlz_x_test )
        omega_0, theta_feat_pts = get_theta_feat_by_DNN( nrmlz_x_test, DNN_model, t_axis[-1], theta_B_0 )
        theta_B = cbc_traj_extrm_pts( theta_feat_pts[:,0] , theta_feat_pts[:,1] , omega_0, dt, itr_num )
        
       
    x_B, v_B, a_B = cnstrct_trans_set( x_Bx, x_By ) # generate ( x_B, v_B, a_B ) of the box
    
    ## variable space initialization ##
    x_mB, v_mB, a_mB =  np.zeros( ( 3, itr_num ), dtype=float ) # [mm], [mm/sec], [mm/sec^2]
    theta_B_iLQR, theta_B_PID, omega_B, alpha_B = np.zeros( ( 4, itr_num ), dtype=float ) # [rad], [rad/sec], [rad/sec^2]
    nrml_f = np.zeros( itr_num, dtype=float ) # [N]
    msr_tau, tau = np.zeros( ( 2, itr_num ), dtype=float ) # [N-mm]

    ## kinematics and force initial conditions
    x_mB[0] = x_mB_0
    v_mB[0] = ( 0 - v_B[0,0] ) * np.cos( theta_B[0] ) + ( 0 - v_B[0,1] ) * np.sin( theta_B[0] )
    nrml_f[0] = m * alpha_B[0] * x_mB[0] + 2 * m * omega_B[0] * v_mB[0] + m * ( a_B[0,1] * np.cos(theta_B[0]) + grvty * np.cos(theta_B[0]) - a_B[0,0] * np.sin(theta_B[0]) )
    nrml_f[0] = 0.001 * nrml_f[0]

    ## initialize variable space for iLQR
    if( iLQR_switch == True ):
        
        F_set = np.zeros( ( itr_num, dim_x, dim_x + dim_u ), dtype=float ) # dim(F): n * (n+m)
        f_set = np.zeros( ( itr_num, dim_x, 1 ), dtype=float ) # dim(f): n * 1 
        K_set = np.zeros( ( itr_num, dim_u, dim_x ), dtype=float ) # dim(K): m * n
        k_set = np.zeros( ( itr_num, dim_u, 1 ), dtype=float ) # dim(k): m * 1
        V_set = np.zeros( ( itr_num, dim_x, dim_x ), dtype=float ) # dim(V): n * n
        V_set[-1,0,0] = w_x
        v_set = np.zeros( ( itr_num, dim_x, 1 ), dtype=float ) # dim(v): n * 1
        Q_set = np.zeros( ( itr_num, dim_x + dim_u, dim_x + dim_u ), dtype=float ) # dim(Q): (n+m) * (n+m)
        q_set = np.zeros( ( itr_num, dim_x + dim_u, 1 ), dtype=float ) # dim(q): (n+m) * 1

    ## evaluation index
    dist_RMSE = np.zeros( ep_num+2, dtype=float )
    energy_set = np.zeros( ep_num+2, dtype=float )
    #theta_B_set = []
    #x_mB_set = []

    for ep_idx in range( ep_num + 2 ):
        
        PID_controller( 0 , True ) # reset PID controller
        imped_ctrl( 0 , True )
        est_state = np.array( [ [ x_mB[0] ], [ v_mB[0] ], [ a_mB[0] ] ] )
        est_x_mB = np.copy( x_mB )
        
        ###############################
        ### Dynamic simulation part ###
        ###############################
        for idx in range( itr_num - 1 ):
            
            if( ep_idx > 0 and ep_idx < ep_num + 1 and iLQR_switch == True ):
                
                ## state feedback control for iLQR ##
                state = np.array([ [x_mB[idx]], [v_mB[idx]], [a_mB[idx]], [theta_B[idx]] ])
                omega_B[idx] = K_set[idx].dot( state ) + k_set[idx]        
                # omega_B[idx] = np.clip( omega_B[idx], -0.15, 0.15 )
                theta_B[idx+1] = theta_B[idx] + omega_B[idx] * dt               
                # theta_B[idx+1] = np.clip( theta_B[idx+1], -pi/3, pi/3 )
            
            theta_B_iLQR[idx+1] = theta_B[idx+1]    
            ## update the dynamic state of the sliding box ##
            a_mB[idx+1] = omega_B[idx]**2 * x_mB[idx] - ( a_B[idx,0] * np.cos(theta_B[idx]) + a_B[idx,1] * np.sin(theta_B[idx]) + grvty * np.sin(theta_B[idx]) )
            
            nrml_f[idx+1] = m * alpha_B[idx] * x_mB[idx] + 2 * m * omega_B[idx] * v_mB[idx] + m * ( a_B[idx,1] * np.cos(theta_B[idx]) + grvty * np.cos(theta_B[idx]) - a_B[idx,0] * np.sin(theta_B[idx]) )
            nrml_f[idx+1] = 0.001 * nrml_f[idx+1] # unit translation to [N]
            
            v_mB[idx+1] = v_mB[idx] + a_mB[idx+1] * dt
            x_mB[idx+1] = x_mB[idx] + v_mB[idx+1] * dt + 0.5 * a_mB[idx+1] * dt**2
            x_mB[idx+1], v_mB[idx+1] = bndry_check( x_mB[idx+1], v_mB[idx+1] ) # Boundary check
            
            tau[idx+1] = x_mB[idx+1] * nrml_f[idx+1] # [N-mm]=[kg-m/sec^2 - mm]
            
            
            ## extended Kalman filter part ##
            if( Use_EKF == True ):
            
                # msr_tau[idx+1] = tau[idx+1] * np.random.normal( 1., 0.03 )
                msr_tau[idx+1] = tau[idx+1] + np.random.normal( 0., 120 )        
                
                if( idx>100 ):
                    
                    est_state, P = EKF( est_state, 1000*np.array([[msr_tau[idx+1]]]), P,  m, theta_B[idx], omega_B[idx], alpha_B[idx], a_B[idx,:] )                
                    est_state[0,0], est_state[1,0] = bndry_check( est_state[0,0], est_state[1,0] ) # boundary check for EKF
                   
                else:
                    est_state = np.array( [ [ x_mB[idx+1] ], [ v_mB[idx+1] ], [ a_mB[idx+1] ] ] )
            
                est_x_mB[idx+1] = est_state[0,0]
                
            else:
                est_x_mB[idx+1] = x_mB[idx+1]
            
            ## PID controller part ##
            if( ep_idx == ep_num+1 and PID_switch == True ):
                
                theta_B_PID[idx+1] = PID_controller( est_x_mB[idx+1] , False ) * pi/180
                # theta_B_PID[idx+1] = imped_ctrl( est_x_mB[idx+1] , False ) * pi/180
                theta_B[idx+1] = theta_B[idx+1] + theta_B_PID[idx+1]
                theta_B[idx+1] = np.clip( theta_B[idx+1], -pi/3, pi/3 )               
                omega_B[idx+1] = ( theta_B[idx+1] - theta_B[idx] ) / dt
            
            if( Gen_theta_by_DNN == True and PID_switch == False ):
                omega_B[idx+1] = ( theta_B[idx+1] - theta_B[idx] ) / dt
                
            alpha_B[idx+1] = ( omega_B[idx+1] - omega_B[idx] ) / dt
            energy_set[ep_idx] = energy_set[ep_idx] + omega_B[idx] * tau[idx] * dt
            
        #################
        ### iLQR part ###
        #################
        if( iLQR_switch == True ):
            for i in range( itr_num-2, -1, -1 ):
                
                F_set[i] = prtl_f_prtl_x_and_u( theta_B[i], omega_B[i], a_B[i], x_mB[i] ) 
                f_set[i] = np.array([ [x_mB[i+1]], [v_mB[i+1]], [a_mB[i+1]], [theta_B[i+1]]]) - F_set[i].dot( np.array([ [x_mB[i]], [v_mB[i]], [a_mB[i]], [theta_B[i]], [omega_B[i]] ]) )
                
                Q_set[i] = F_set[i].T.dot( V_set[i+1].dot( F_set[i] ) )
                Q_set[i,0,0] = Q_set[i,0,0] + w_x # specialized for this case
                Q_set[i,dim_x,dim_x] = Q_set[i,dim_x,dim_x] + w_omega # specialized for this case
                q_set[i] = F_set[i].T.dot( V_set[i+1].dot( f_set[i] ) ) + F_set[i].T.dot( v_set[i+1] )
                
                K_set[i] = - Q_set[ i, dim_x:, :dim_x ] / ( Q_set[ i, dim_x:, dim_x: ] + 1e-9 )
                k_set[i] = - q_set[ i, dim_x:, : ] / ( Q_set[ i, dim_x:, dim_x: ] + 1e-9 )            
                
                V_set[i] = ( Q_set[ i,:dim_x, :dim_x ]
                           + Q_set[ i, :dim_x, dim_x: ].dot( K_set[i] )
                           + K_set[i].T.dot( Q_set[ i, dim_x: , :dim_x ] )
                           + K_set[i].T.dot( Q_set[ i, dim_x: , dim_x: ].dot( K_set[i] ) ) 
                           )
                
                v_set[i] = ( q_set[ i, :dim_x, :]
                           + Q_set[ i, :dim_x, dim_x: ].dot( k_set[i] )
                           + K_set[i].T.dot( q_set[ i, dim_x: , : ] )
                           + K_set[i].T.dot( Q_set[ i, dim_x: , dim_x: ].dot( k_set[i] ) ) 
                           )
                
        dist_RMSE[ep_idx] = np.sqrt( np.average( x_mB**2 ) )
        
        theta_feat_pts, omega_0 = extrct_feat_pts( t_axis, theta_B )
        decmprss_traj = cbc_traj_extrm_pts( theta_feat_pts[:,0] , theta_feat_pts[:,1] , omega_0, dt, itr_num )
               
        if( ep_idx == ep_num and Valid_orgnl_decmprss == True ):
            
            plt.plot( t_axis, theta_B )
            plt.plot( t_axis, decmprss_traj )
            plt.scatter(theta_feat_pts[:,0],theta_feat_pts[:,1])
            plt.show()
            theta_B__ = theta_B
            x_mB__ = x_mB
            theta_B = decmprss_traj
    
    print( "Data", train_idx, "x : %.4f" %dist_RMSE[-1], ", E: %.4f" %energy_set[-1] )
    
    if( Create_train_set == True ):
        
        theta_feat_pts, omega_0 = extrct_feat_pts( t_axis, theta_B )
        decmprss_traj = cbc_traj_extrm_pts( theta_feat_pts[:,0] , theta_feat_pts[:,1] , omega_0, dt, itr_num )
        
#        plt.plot( theta_B )
#        plt.plot( decmprss_traj )
#        plt.show()

        # create a row of training data        
        train_data = creat_train_data( via_pt_num, via_x, via_y, v_avg, theta_B_0, x_mB_0 )
        train_data_set.append( train_data )
    
        label_data = np.append( np.array([omega_0]), theta_feat_pts.reshape(-1) )
        label_data_set.append( label_data )

        if( len(label_data) > label_data_len ):
            label_data_len = len(label_data)
    
if( Create_train_set == True ):    
    
    train_data_set = np.vstack( train_data_set )
#    np.savetxt( "./ML_data/train_data(" + str( seeds[MODEL_NUMBER-1] ) + ").csv", train_data_set, delimiter="," )

    ## align labeled data
    for i in range( train_data_num ):
        for j in range( label_data_len - len( label_data_set[i] ) ):
            label_data_set[i] = np.append( label_data_set[i], [ np.inf ] )

    label_data_set = np.vstack( label_data_set )
#    np.savetxt( "./ML_data/label_data(" + str( seeds[MODEL_NUMBER-1] ) + ").csv", label_data_set, delimiter="," )

    
### record experiment results
#now = datetime.now()
'''
if( Read_ini_cond == False ):
    ini_cond = []
    for i in range(via_pt_num):
        ini_cond.append( via_x[i] )
    
    for i in range(via_pt_num):
        ini_cond.append( via_y[i] )    
    
    ini_cond.append( theta_B_0 )
    ini_cond.append( x_mB_0 )
    ini_cond.append( v_avg )
    
    np.savetxt( "./GA_data/ini_cond(" + str(MODEL_NUMBER) + ").csv", np.array( ini_cond ), delimiter=",")
    np.savetxt( "./GA_data/dist_RMSE_set(" + str(MODEL_NUMBER) + ").csv", dist_RMSE_set, delimiter=",")
    np.savetxt( "./GA_data/best points_rad(" + str(MODEL_NUMBER) + ").csv", best_points, delimiter=",")
    
    with open( "./GA_data/best_points(" + str(MODEL_NUMBER) + ").txt", 'w') as out_file:
        
        out_file.write( "best points (rad):\n")
        for i in range(len(best_points)):
            out_file.write( str( best_points[i] ) )
            out_file.write( "\n")
        out_file.write( "\nbest dist_RMSE (mm):\n")
        out_file.write( str( -1*best_dist_RMSE_minus ) )
        out_file.write( "\n\n")
    #    out_file.write( "elapsed time (ms):\n")
    #    out_file.write( str( int( 1000 * elapsed_time ) ) )


    
# via_theta_set = np.vstack( via_theta_set )
# for i in range(len(via_theta_set)):
    # plt.plot( via_theta_set[i,:] * 180/pi )
# plt.show()

#traj_theta_set = np.vstack( traj_theta_set )
#for i in range(len(traj_theta_set)):
#    plt.plot(traj_theta_set[i,:])
#plt.show()

#x_mB_set = np.vstack( x_mB_set )
#for i in range(len(x_mB_set)):
#    plt.plot(x_mB_set[i,:])
#plt.show()
'''
## testing

    
### plot simulation result ###
#plt.plot( t_axis, est_x_mB )
#plt.plot( t_axis, x_mB )
#plt.show()
#plt.plot(est_x_mB-x_mB)
#plt.show()
#print( np.sqrt( np.average( (est_x_mB-x_mB)**2 ) ) )
#plt.plot( t_axis, theta_B*180/pi )
#plt.show()
#plt.plot( t_axis[:250], theta_B[:250]*180/pi )    
#plt.show()
#plt.plot( t_axis[4:], alpha_B[4:]*180/pi )
    

## Syntec Report ##
window_half = 2
for i in range( window_half, len(msr_tau)-window_half ):
    msr_tau[i] = np.median( msr_tau[ (i - window_half) : (i + window_half) ] )
        
#np.savetxt( "./SyntecReport/x_Bx.csv", x_Bx, delimiter=',' )
#np.savetxt( "./SyntecReport/x_By.csv", x_By, delimiter=',' )

#np.savetxt( "./SyntecReport/theta_B_PID.csv", theta_B )
#np.savetxt( "./SyntecReport/theta_B_iLQR.csv", theta_B )
#np.savetxt( "./SyntecReport/theta_B_hybrid.csv", theta_B )

#np.savetxt( "./SyntecReport/t_axis.csv", t_axis, delimiter=',' )

#np.savetxt( "./SyntecReport/x_mB_PID.csv", x_mB)
#np.savetxt( "./SyntecReport/x_mB_iLQR.csv", x_mB)
#np.savetxt( "./SyntecReport/x_mB_hybrid.csv", x_mB)

#theta_B_set = np.array( theta_B_set )
#np.savetxt( "./SyntecReport/theta_B_set.csv", theta_B_set, delimiter=',' )
#x_mB_set = np.array( x_mB_set )
#np.savetxt( "./SyntecReport/x_mB_set.csv", x_mB_set, delimiter=',' )

#np.savetxt( "./SyntecReport/est_x_mB.csv", est_x_mB)
#np.savetxt( "./SyntecReport/msr_tau.csv", msr_tau)

#np.savetxt( "./SyntecReport/a_mB_cntr.csv", a_mB_cntr )
#np.savetxt( "./SyntecReport/a_mB_box.csv", a_mB_box)
#np.savetxt( "./SyntecReport/a_mB_grvty.csv",a_mB_grvty )
