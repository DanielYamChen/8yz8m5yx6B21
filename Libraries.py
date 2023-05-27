# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:48:22 2017

@author: Biorola
"""
from math import pi
import numpy as np
from numpy.linalg import norm
import itertools
import matplotlib.pyplot as plt
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

## Global condition
dt = 0.012 # [sec]
grvty = 9810 # [mm/sec^2]
SPACE_UP_LIMIT = 650 # [mm]
SPACE_DOWN_LIMIT = 0 # [mm]
SPACE_RIGHT_LIMIT = 650 # [mm]
SPACE_LEFT_LIMIT = 0 # [mm]
BOX_HALF_LENGTH = 125 # [mm]
LEARNING_PHASE = 0


## physical conditions
m = 1 # [kg]
std_tau = 120 # [N-mm] of torque sensor
# v_avg_min = 100 # [mm/sec]
# v_avg_max = 150 # [mm/sec]
v_avg_min = 50 # [mm/sec]
v_avg_max = 100 # [mm/sec]
theta_B_0_range = 30 * pi/180 # [rad]
omega_B_0_range = theta_B_0_range / dt # [rad/sec]
via_pt_num = 4
intrpltn_num = 1


## construct translation trajectory (x,v,a) of the box
def cnstrct_trans_set( x_Bx, x_By, t_duration ):
      
    x_B = np.vstack( ( x_Bx, x_By ) ).T # [mm]
    v_B = np.vstack( ( [ 0, 0 ], np.diff( x_B, axis=0 ) ) ) / dt # [mm/sec]

    # blending the velocity plot
    t_idx = 0
    for i in t_duration[0:-1]:
        t_idx = t_idx + int(i/dt)
        try:
            v_B[t_idx] = ( v_B[t_idx-1] + v_B[t_idx+1] ) / 2
        except:
            continue
        
    a_B = np.vstack( ( [ 0, 0 ], np.diff( v_B, axis=0 ) ) ) / dt # [mm/sec^2]
    
    return x_B, v_B, a_B
    
    
def PID_controller( error, reset ):

    if not hasattr( PID_controller, "error_prvs" ):
        PID_controller.error_prvs = error
    
    if not hasattr( PID_controller, "integral_term" ):
        PID_controller.integral_term = 0
    
    if( reset == True ):
        PID_controller.error_prvs = error
        PID_controller.integral_term = 0
    
    ## for PID_switch, for paper 4-1 ##
    # P_coefficient = 0.7 #[deg/mm]
    # I_coefficient = 0.2
    # D_coefficient = 0.034
    
    ## perfect with using iLQR and EKF ##
#    P_coefficient = 0.7 #[deg/mm]
#    I_coefficient = 1.0
#    D_coefficient = 0.03
    
    ## for x_track, for paper all ##
    # P_coefficient = 0.4 # [deg/mm]
    # I_coefficient = 0.2
    # D_coefficient = 0.08
    
    ## for PID_switch, for delay added in realistic ##
    P_coefficient = 0.3 #[deg/mm]
    I_coefficient = 0.2
    D_coefficient = 2.0
    
    
    PID_controller.integral_term = PID_controller.integral_term + error * dt
    delta_x_command = P_coefficient * ( error + I_coefficient * PID_controller.integral_term + D_coefficient * ( error - PID_controller.error_prvs ) / dt )
    PID_controller.error_prvs = error
    
#    if( delta_x_command > 90.0 ):
#        return 90.0
#    elif( delta_x_command < -90.0 ):
#        return -90.0
#    else:
   
    return delta_x_command


def PD_ctrl( error, reset ):
    
    if not hasattr( PID_controller, "error_prvs" ):
        PID_controller.error_prvs = error
        
    if( reset == True ):
        PID_controller.error_prvs = error
    
    ## perfect ##
    P_coefficient = 0.7 # [deg/mm]
    D_coefficient = P_coefficient * 0.
        
    delta_x_command = P_coefficient * error + D_coefficient * ( error - PID_controller.error_prvs ) / dt
    PID_controller.error_prvs = error
    
#    if( delta_x_command > 90.0 ):
#        return 90.0
#    elif( delta_x_command < -90.0 ):
#        return -90.0
#    else:
   
    return delta_x_command


def imped_ctrl( err, reset ):
    
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
    
    segment = 4
    line_choice = np.random.randint( segment, size=4 )
    
    # randomly choose 4 point in each side
    temp_x = [ 1.0 / ( 2 * segment ) * line_choice[0], np.random.uniform( 0, 1 ), 
               1.0 - 1.0 / ( 2 * segment ) * line_choice[2], np.random.uniform( 0, 1 ) ]
    
    temp_y = [ np.random.uniform( 0, 1 ), 1.0 - 1.0 / ( 2 * segment ) * line_choice[1],
               np.random.uniform( 0, 1 ), 1.0 / ( 2 * segment ) * line_choice[3] ]
    
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
def read_ini_cond( path, idx ):
    
    ini_cond = np.loadtxt( path, delimiter=',' )
    if( ini_cond.ndim > 1 ):
        ini_cond = ini_cond[idx]

    via_x = np.copy( ini_cond[ : via_pt_num ] )
    via_y = np.copy( ini_cond[ via_pt_num : 2*via_pt_num ] )
    v_avg = np.copy( ini_cond[ 2*via_pt_num : 3*via_pt_num-1 ] )
    theta_B_0 = ini_cond[-2]
    x_mB_0 = ini_cond[-1]
    
    return via_x, via_y, v_avg, theta_B_0, x_mB_0


## extract extreme points of the trajectory
def extrct_feat_pts( t_axis, traj, tol ):
    
    wndw_size = 7
    # tol = 0.0001 * (pi/180) # [rad]
    # tol = 0.001 # [mm]
    
    feat_pts = []
    feat_pts.append( [ 0., traj[0] ] )
    
    ## find extreme point
    for i in range( wndw_size, len(traj) - wndw_size ):
        
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
def normalize_X( nrmlz_X ):
    
    # raw_x : via_x*4 [mm], via_y*4 [mm], v_avg*3 [mm/sec], theta_B_0 [rad], x_mB_0 [mm]
    nrmlz_X[:via_pt_num] = ( nrmlz_X[:via_pt_num] - SPACE_LEFT_LIMIT ) / ( SPACE_RIGHT_LIMIT - SPACE_LEFT_LIMIT ) # via_x*4, [mm] -> [1/1]
    nrmlz_X[via_pt_num:2*via_pt_num] = ( nrmlz_X[via_pt_num:2*via_pt_num] - SPACE_DOWN_LIMIT ) / ( SPACE_UP_LIMIT - SPACE_DOWN_LIMIT ) # via_y*4, [mm] -> [1/1]
    nrmlz_X[2*via_pt_num:3*via_pt_num-1] = ( nrmlz_X[2*via_pt_num:3*via_pt_num-1] - v_avg_min ) / ( v_avg_max - v_avg_min ) # v_avg*3, [mm/sec] -> [1/1]
    nrmlz_X[-2] = ( nrmlz_X[-2] / theta_B_0_range + 1 ) / 2 # theta_B_0, [rad] -> [1/1]
    nrmlz_X[-1] = ( nrmlz_X[-1] / BOX_HALF_LENGTH + 1 ) / 2 # x_mB_0, [mm] -> [1/1]

    return


## get feature points of theta and omega_0 given nrmlz I.C. from DNN and dnrmlz and append them to theta_B_0
def get_theta_feat_by_DNN( nrmlz_X_test, DNN_model, t_total, theta_B_0 ):

    y = DNN_model.predict( nrmlz_X_test.reshape(1,-1) ).reshape(-1)
    
    omega_0 = y[0] * ( 2 * omega_B_0_range ) - omega_B_0_range # denormalize omega_B_0, [rad/sec]
    y = np.delete( y, [0] )
    
    # transform the increment form of time stamps to absolute form
    for i in range( 2, len(y), 2 ):
        y[i] = y[i] + y[i-2]
       
    for i in range( 0, len(y), 2 ):
    
        y[i] = y[i] / y[-2] * t_total # time_stamp, [1/1] -> [sec]
        y[i+1] = y[i+1] * ( 2 * theta_B_0_range ) - theta_B_0_range # theta_B_feat, [1/1] -> [rad]
    
    y = np.append( [ 0., theta_B_0 ], y )
    y = y.reshape( -1, 2 )
    
    return omega_0, y


## get feature points of x_mB given nrmlz I.C. from DNN and dnrmlz and append them to x_mB_0
def get_x_mB_feat_by_DNN( nrmlz_X_test, DNN_model, t_total, x_mB_0 ):

    y = DNN_model.predict( nrmlz_X_test.reshape(1,-1) ).reshape(-1)
      
    # transform the increment form of time stamps to absolute form
    for i in range( 2, len(y), 2 ):
        y[i] = y[i] + y[i-2]
       
    for i in range( 0, len(y), 2 ):
    
        y[i] = y[i] / y[-2] * t_total # time_stamp, [1/1] -> [sec]
        y[i+1] = y[i+1] * ( 2 * BOX_HALF_LENGTH ) - BOX_HALF_LENGTH # x_mB_hat_feat, [1/1] -> [mm]
    
    y = np.append( [ 0., x_mB_0 ], y )
    y = y.reshape( -1, 2 )
    
    return y


## calculate gradient value of Cost function 21 to vector [ (delta_T_mB_prime,i , x_mB_hat,i)_i=1~N ]
def prtl_J2_prtl_t_and_x( nrmlz_dT_mB_, x_hat_, x_mB_0, t_tot, DNN_x_mB_o_dim ):
    
    dT_mB = t_tot * np.append( nrmlz_dT_mB_, [0.] ) # 1 ~ N+1
    x_hat = np.append( [x_mB_0], x_hat_ ) # 0 ~ N
    
    weight = 0.
    p_J2_p_t_and_x = np.zeros( ( 1, DNN_x_mB_o_dim ), dtype=float )
    for i in range( 0, int( DNN_x_mB_o_dim / 2 ) ): # i = 1 ~ N
        
        p_J2_p_t_and_x[0,2*i] = ( 13 * x_hat[i]**2
                          + 9 * x_hat[i] * x_hat[i+1]
                          + 13 * x_hat[i+1]**2 ) / 35 # partial J2 to partial delta_T_prime_mB,i
        
        weight = weight + p_J2_p_t_and_x[0,2*i] * dT_mB[i] / t_tot
        
        if( i == int( DNN_x_mB_o_dim / 2 ) - 1 ):
            p_J2_p_t_and_x[0,2*i+1] = ( 9 * dT_mB[i] * x_hat[i]
                                  + 26 * dT_mB[i] * x_hat[i+1] ) / 35 # partial J2 to partial x_mB_hat,i
           
        else:
            p_J2_p_t_and_x[0,2*i+1] = ( 9 * dT_mB[i] * x_hat[i]
                                  + 26 * ( dT_mB[i] + dT_mB[i+1] ) * x_hat[i+1]
                                  + 9 * dT_mB[i+1] * x_hat[i+2] ) / 35 # partial J2 to partial x_mB_hat,i
                      
    return p_J2_p_t_and_x, weight


## calculate gradient value of Cost function 2 to vector [ nrmlz_x_B,i=0~3, nrmlz_y_B,i=0~3, nrmlz_v_avg,i=1~3 ]
def prtl_J22_prtl_nrmlz_x_y_v_avg( nrmlz_X ):
    
    via_x, via_y, v_avg, _, __ = denrmlz_values( nrmlz_X[0:via_pt_num],
                                                 nrmlz_X[via_pt_num:2*via_pt_num],
                                                 nrmlz_X[2*via_pt_num:3*via_pt_num-1],
                                                 nrmlz_X[-2],
                                                 nrmlz_X[-1] ) # recover values, [mm,rad,sec]
    
    p_J22_p_nrmlz_IC_1 = np.zeros( 3*via_pt_num-1, dtype=float )
    for i in range( 0, via_pt_num ): # i = 0 ~ 3
        
        try:
            if( i == 0 ):
                p_J22_p_nrmlz_IC_1[i] = - ( 2 * via_x[i+1] - 2 * via_x[i] ) / ( 2 * v_avg[i] * np.sqrt( ( via_x[i+1] - via_x[i] )**2 + ( via_y[i+1] - via_y[i] )**2 ) + 1e-10 )
                p_J22_p_nrmlz_IC_1[via_pt_num+i] = - ( 2 * via_y[i+1] - 2 * via_y[i] ) / ( 2 * v_avg[i] * np.sqrt( ( via_x[i+1] - via_x[i] )**2 + ( via_y[i+1] - via_y[i] )**2 ) + 1e-10  )
            
            elif( i == via_pt_num - 1 ):
                p_J22_p_nrmlz_IC_1[i] = ( 2 * via_x[i] - 2 * via_x[i-1] ) / ( 2 * v_avg[i-1] * np.sqrt( ( via_x[i] - via_x[i-1] )**2 + ( via_y[i] - via_y[i-1] )**2 ) + 1e-10  )
                p_J22_p_nrmlz_IC_1[via_pt_num+i] = ( 2 * via_y[i] - 2 * via_y[i-1] ) / ( 2 * v_avg[i-1] * np.sqrt( ( via_x[i] - via_x[i-1] )**2 + ( via_y[i] - via_y[i-1] )**2 ) + 1e-10  )
                
            else:
                p_J22_p_nrmlz_IC_1[i] = ( ( 2 * via_x[i] - 2 * via_x[i-1] )
                                        / ( 2 * v_avg[i-1] * np.sqrt( ( via_x[i] - via_x[i-1] )**2 + ( via_y[i] - via_y[i-1] )**2 ) + 1e-10  )
                                        - ( 2 * via_x[i+1] - 2 * via_x[i] )
                                        / ( 2 * v_avg[i] * np.sqrt( ( via_x[i+1] - via_x[i] )**2 + ( via_y[i+1] - via_y[i] )**2 ) + 1e-10  )
                                        )
                p_J22_p_nrmlz_IC_1[via_pt_num+i] = ( ( 2 * via_y[i] - 2 * via_y[i-1] )
                                                   / ( 2 * v_avg[i-1] * np.sqrt( ( via_x[i] - via_x[i-1] )**2 + ( via_y[i] - via_y[i-1] )**2 ) + 1e-10  )
                                                   - ( 2 * via_y[i+1] - 2 * via_y[i] )
                                                   / ( 2 * v_avg[i] * np.sqrt( ( via_x[i+1] - via_x[i] )**2 + ( via_y[i+1] - via_y[i] )**2 ) + 1e-10  )
                                                   )
        
        except:
            continue
        
        if( i > 0 ):
            p_J22_p_nrmlz_IC_1[2*via_pt_num+i-1] = - np.sqrt( ( via_x[i] - via_x[i-1] )**2 + ( via_y[i] - via_y[i-1] )**2 ) / ( v_avg[i-1]**2 + 1e-10  )
    
    p_J22_p_nrmlz_IC_1[ 0 : via_pt_num ] = p_J22_p_nrmlz_IC_1[ 0 : via_pt_num ] * ( SPACE_RIGHT_LIMIT - SPACE_LEFT_LIMIT )
    p_J22_p_nrmlz_IC_1[ via_pt_num : 2*via_pt_num ] = p_J22_p_nrmlz_IC_1[ via_pt_num : 2*via_pt_num ] * ( SPACE_UP_LIMIT - SPACE_DOWN_LIMIT )
    p_J22_p_nrmlz_IC_1[ 2*via_pt_num : ] = p_J22_p_nrmlz_IC_1[ 2*via_pt_num : ] * ( v_avg_max - v_avg_min )
    
    return p_J22_p_nrmlz_IC_1


## given normalized initial condition, calculate total time of the task
def calc_t_total( nrmlz_X_test ):

    via_x = nrmlz_X_test[0:via_pt_num] * ( SPACE_RIGHT_LIMIT - SPACE_LEFT_LIMIT ) + SPACE_LEFT_LIMIT
    via_y = nrmlz_X_test[via_pt_num:2*via_pt_num] * ( SPACE_UP_LIMIT - SPACE_DOWN_LIMIT ) + SPACE_DOWN_LIMIT
    v_avg = nrmlz_X_test[2*via_pt_num:3*via_pt_num-1] * ( v_avg_max - v_avg_min ) + v_avg_min
    
    t_total = 0.
    for i in range(len(v_avg)):
        t_total = t_total + np.sqrt( ( via_x[i+1] - via_x[i] )**2 + ( via_y[i+1] - via_y[i] )**2 ) / v_avg[i]
    
    return t_total


## Adam optimizer
def adam_opt( grdnt, reset, itr_idx ):
    
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-08
    
    if not hasattr( adam_opt, "m" ):
        adam_opt.m = np.zeros( len(grdnt), dtype=float )
    
    if not hasattr( adam_opt, "v" ):
        adam_opt.v = 0.
    
    if( reset == True ):
        adam_opt.m = 0.
        adam_opt.v = 0.
    
    adam_opt.m = beta1 * adam_opt.m + ( 1 - beta1 ) * grdnt
    adam_opt.v = beta2 * adam_opt.v + ( 1 - beta2 ) * grdnt.dot( grdnt.T )
    m_hat = adam_opt.m / ( 1 - beta1**(itr_idx+1) )
    v_hat = adam_opt.v / ( 1 - beta2**(itr_idx+1) )
    
    return m_hat / ( np.sqrt( v_hat ) + epsilon )


def permut( nrmlz_X, mask ):
    
    # segment = 4
    segment = 3 # for paper 4-5-3-2
    
    mask_idx = np.nonzero( mask )[0]
    permut_set = list( itertools.product( np.arange( 1/segment, 1-1e-3, 1/segment ), repeat=len(mask_idx) ) )
    # permut_set = list( itertools.product( np.arange( 0, 1+1e-8, 1/segment ), repeat=len(mask_idx) ) )
    nrmlz_X_set = []
    nrmlz_X_ = np.copy( nrmlz_X )
    
    for i in range( len(permut_set) ):
        nrmlz_X_[mask_idx] = permut_set[i]
        nrmlz_X_set.append( np.copy(nrmlz_X_) )
    
    return nrmlz_X_set, len(permut_set)


def calc_J2( t_prime_and_x, x_mB_0, t_total ):
    J2 = 0.
    for j in range ( 0, len(t_prime_and_x), 2 ):
            
        if( j == 0 ):
            J2 = J2 + t_total * ( t_prime_and_x[0] * ( 13 * t_prime_and_x[1]**2 + 9 * t_prime_and_x[1] * x_mB_0 + 13 * x_mB_0**2 ) ) / 35

        else:
            J2 = J2 + t_total * ( t_prime_and_x[j] * ( 13 * t_prime_and_x[j+1]**2 + 9 * t_prime_and_x[j+1] * t_prime_and_x[j-1] + 13 * t_prime_and_x[j-1]**2 ) ) / 35
        
    return J2

## do optimization of J2 w.r.t. IC and find the IC which leads to minimum J2
def optmz_J2_wrt_IC( nrmlz_X_test_, DNN_model, jacobian_t_x_to_IC, mask, nrmlz_obs_set, Worst_case ):
    
    DNN_x_mB_o_dim = DNN_model.output.shape.as_list()[1]
    lr_rate = 0.03
    step_num = 500
    # X : via_x*4, via_y*4, v_avg*3, theta_B_0, x_mB_0
    nrmlz_X = np.copy( nrmlz_X_test_ )
    
    J2_process = []
    GD_process = []
    adam_opt( nrmlz_X, True, 0 )
    # print( "\ninitial opt pt:" )
    # print( nrmlz_X )
    # print( "initial task time: ", np.round( calc_t_total( nrmlz_X ), 4 ) )
#    print(nrmlz_obs_set[0])
    
    # if( len(nrmlz_obs_set) > 0 ):
        # fig, ax = plt.subplots()
        # for obs_idx in range(len(nrmlz_obs_set)):
            # ax.add_artist( plt.Circle( nrmlz_obs_set[obs_idx,0:2], nrmlz_obs_set[obs_idx,2], fill=False) ) 
        
    # plt.plot( nrmlz_X[0:4], nrmlz_X[4:8] )
    # plt.show()
    for i in range( step_num ):
        
        # if( i % 50 == 0 ):
            # fig, ax = plt.subplots()
            # for obs_idx in range(len(nrmlz_obs_set)):
                # ax.add_artist( plt.Circle( nrmlz_obs_set[obs_idx,0:2], nrmlz_obs_set[obs_idx,2], fill=False) ) 
        
            # plt.plot( nrmlz_X[0:4], nrmlz_X[4:8] )
            # plt.show()
            # plt.scatter( i, nrmlz_X[-2] )
        
        # GD_process.append( ( 2 * nrmlz_X[-2] - 1 ) * theta_B_0_range ) # for paper 4-5-1
        # GD_process.append( ( 2 * nrmlz_X[-1] - 1 ) * BOX_HALF_LENGTH ) # for paper 4-5-2
        # GD_process.append( np.copy( nrmlz_X[1:3] ) * ( SPACE_RIGHT_LIMIT - SPACE_LEFT_LIMIT ) + SPACE_LEFT_LIMIT ) # for paper 4-5-3-1
        # GD_process.append( np.copy( nrmlz_X[[1,2,5,6]] ) * ( SPACE_RIGHT_LIMIT - SPACE_LEFT_LIMIT ) + SPACE_LEFT_LIMIT ) # for paper 4-5-3-2 ~ 4-5-4-2
        
        x_mB_0 = nrmlz_X[-1] * ( 2 * BOX_HALF_LENGTH ) - BOX_HALF_LENGTH # [1/1] -> [mm]
        t_total = calc_t_total( nrmlz_X )
        
        t_prime_and_x = DNN_model.predict( nrmlz_X.reshape(1,-1) ).reshape(-1)                
        for j in range( 0, len(t_prime_and_x), 2 ):    
            t_prime_and_x[j+1] = t_prime_and_x[j+1] * ( 2 * BOX_HALF_LENGTH ) - BOX_HALF_LENGTH # x_mB_hat_feat, [1/1] -> [mm]
        
        J2 = calc_J2( t_prime_and_x, x_mB_0, t_total )
        J2_process.append( J2 )
        
        # two via points are almost the same...
        if( np.abs( nrmlz_X[1] - nrmlz_X[2] ) < 1e-6 and np.abs( nrmlz_X[5] - nrmlz_X[6] ) < 1e-6 ):
            grad_via1_add, grad_via2_add, Collision_occur = check_via_pts_in_obs( nrmlz_X, nrmlz_obs_set )
            nrmlz_X[[1,5]] = nrmlz_X[[1,5]] + grad_via1_add
            nrmlz_X[[2,6]] = nrmlz_X[[2,6]] + grad_via2_add
            nrmlz_X = np.clip( nrmlz_X, 0., 1. )
            
        else:
            p_J2_p_nrmlz_xmB_0 = t_total * ( 26 * t_prime_and_x[0] * x_mB_0 + 9 * t_prime_and_x[0] * t_prime_and_x[1] ) / 35
            p_J2_p_nrmlz_xmB_0 = p_J2_p_nrmlz_xmB_0 * ( 2 * BOX_HALF_LENGTH )
            
            jacob = np.vstack( jacobian_t_x_to_IC( [ nrmlz_X.reshape(1,-1), LEARNING_PHASE ] ) ).reshape( DNN_x_mB_o_dim, -1 )
            
            p_J2_p_nrmlz_t_x, weight = prtl_J2_prtl_t_and_x( t_prime_and_x[0::2], t_prime_and_x[1::2], x_mB_0, t_total, DNN_x_mB_o_dim )
            p_J2_p_nrmlz_t_x[0,1::2] = p_J2_p_nrmlz_t_x[0,1::2] * ( 2 * BOX_HALF_LENGTH ) # x_mB_hat => nrmlz_x_mB_hat
            p_J2_p_nrmlz_t_x[0,0::2] = p_J2_p_nrmlz_t_x[0,0::2] * t_total # x_mB_hat => nrmlz_x_mB_hat
            
            p_J2_p_IC = p_J2_p_nrmlz_t_x.dot( jacob )
            
            p_J2_p_IC[0,-1] = p_J2_p_IC[0,-1] + p_J2_p_nrmlz_xmB_0
            p_J2_p_IC = p_J2_p_IC.reshape(-1)
            p_J2_p_IC[:-2] = p_J2_p_IC[:-2] + weight * prtl_J22_prtl_nrmlz_x_y_v_avg( nrmlz_X )
            
            grad_via1_add, grad_via2_add, Collision_occur = check_via_pts_in_obs( nrmlz_X, nrmlz_obs_set )
            grad_nrmlz_X = - lr_rate * adam_opt( p_J2_p_IC, False, i ) * mask
            
            if( Collision_occur == False ):
                nrmlz_X = nrmlz_X + ( -2.0 * Worst_case + 1.0 ) * grad_nrmlz_X
            
            else:
                nrmlz_X[[1,5]] = nrmlz_X[[1,5]] + grad_via1_add
                nrmlz_X[[2,6]] = nrmlz_X[[2,6]] + grad_via2_add
            
            nrmlz_X = np.clip( nrmlz_X, 0., 1. )
            # check_via_pts_out_obs( nrmlz_X, grad_nrmlz_X, nrmlz_obs_set )

    # final check whether collision occurs
    for i in range( int(step_num/5) ):
        
        grad_via1_add, grad_via2_add, Collision_occur = check_via_pts_in_obs( nrmlz_X, nrmlz_obs_set )
        if( Collision_occur == False ):
            break
        
        else:
            if( i == int(step_num/5) - 1 ):
                J2 = 1e8 # invalid traj with obstacle collision
                break
                # print("a trajectory with collision")
                # fig, ax = plt.subplots()
                # for obs_idx in range(len(nrmlz_obs_set)):
                    # ax.add_artist( plt.Circle( nrmlz_obs_set[obs_idx,0:2], nrmlz_obs_set[obs_idx,2], fill=False) ) 
        
                # plt.plot( nrmlz_X[0:4], nrmlz_X[4:8] )
                # plt.show()
            GD_process.append( np.copy( nrmlz_X[[1,2,5,6]] ) * ( SPACE_RIGHT_LIMIT - SPACE_LEFT_LIMIT ) + SPACE_LEFT_LIMIT )
            
            x_mB_0 = nrmlz_X[-1] * ( 2 * BOX_HALF_LENGTH ) - BOX_HALF_LENGTH # [1/1] -> [mm]
            t_total = calc_t_total( nrmlz_X )
            t_prime_and_x = DNN_model.predict( nrmlz_X.reshape(1,-1) ).reshape(-1)                
            for j in range( 0, len(t_prime_and_x), 2 ):    
                t_prime_and_x[j+1] = t_prime_and_x[j+1] * ( 2 * BOX_HALF_LENGTH ) - BOX_HALF_LENGTH # x_mB_hat_feat, [1/1] -> [mm]
        
            J2 = calc_J2( t_prime_and_x, x_mB_0, t_total )
            J2_process.append( J2 )
            
            nrmlz_X[[1,5]] = nrmlz_X[[1,5]] + grad_via1_add
            nrmlz_X[[2,6]] = nrmlz_X[[2,6]] + grad_via2_add
            nrmlz_X = np.clip( nrmlz_X, 0., 1. )
            
    process_len = step_num + i
    t_total = calc_t_total( nrmlz_X )
    
    if(Worst_case == False ):
        GD_process = np.vstack( GD_process ).T
        GD_process = np.concatenate( ( GD_process , np.zeros( ( 4, int( 1.2*step_num-len(J2_process) ) ) ) ), axis=1 )
    
    J2_process = np.array( J2_process )
    J2_process = np.append( J2_process, [ 0. for i in range( int( 1.2*step_num-len(J2_process) ) ) ] )
    if( J2 == 1e8 ):
        J2_process = np.append( J2_process, -1 )
    else:
        J2_process = np.append( J2_process, 1 )
    # plt.plot( nrmlz_X[0:4], nrmlz_X[4:8] )
    
    # plt.axis('equal')
    # plt.show()
    # plt.plot( J2_set )
    # plt.show()
#    print(t_total)
    return nrmlz_X, t_total, J2, J2_process, GD_process, process_len
        

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


## calculate time stamp of each via point, given via point set and averaged velocity set
def calc_t_stamp( via_x, via_y, v_avg ):
    t_stamp = [0.]
    
    for i in range(len(v_avg)):
        t_stamp.append( t_stamp[-1] + np.sqrt( ( via_x[i+1] - via_x[i] )**2 + ( via_y[i+1] - via_y[i] )**2 ) / v_avg[i] )
        
    return np.array( t_stamp )    
    

## check a point is at the right side or left side of a line segment
def calc_via_mov_dir( pt, seg_e, seg_s ):
    
    temp = ( seg_e[0] - seg_s[0] ) * pt[1] - ( seg_e[1] - seg_s[1] ) * ( pt[0] - seg_s[0] ) - seg_s[1] * ( seg_e[0] - seg_s[0] )
    vec = np.array( [ seg_e[1] - seg_s[1], seg_s[0] - seg_e[0] ] )
    
    if( temp * vec[1] > 0 ):
        return ( - vec / norm( vec, 2 ) )
    else:
        return ( vec / norm( vec, 2 ) )


## check whether a segment of line is totally outside of a circle
def line_seg_outside_circle( p1, p2, cntr, R ):
    
    
    if( ( p1[0] - cntr[0] )**2 + ( p1[1] - cntr[1] )**2 <= R**2 ): return False
        
    if( ( p2[0] - cntr[0] )**2 + ( p2[1] - cntr[1] )**2 <= R**2 ): return False
        
    a = ( p2[0] - p1[0] )**2 + ( p2[1] - p1[1] )**2
    b = 2 * ( ( p2[0] - p1[0] ) * ( p1[0] - cntr[0] ) + ( p2[1] - p1[1] ) * ( p1[1] - cntr[1] ) )
    c = ( p1[0] - cntr[0] )**2 + ( p1[1] - cntr[1] )**2 - R**2
    Delta = b**2 - 4 * a * c
    
    if( Delta >= 0 ):
        
        t1 = ( - b + np.sqrt( Delta ) ) / ( 2 * a )
        t2 = ( - b - np.sqrt( Delta ) ) / ( 2 * a )
        
        if( 0 <= t1 and t1 <= 1 ): return False
        if( 0 <= t2 and t2 <= 1 ): return False   
    
    return True


## if via points in the circle or line segments cross the circle, output the additional gradient to push them out
def check_via_pts_in_obs( nrmlz_X, nrmlz_obs_set ):
    
    mov_rate = 0.02
    via_0 = np.array( [ nrmlz_X[0], nrmlz_X[4] ] )
    via_1 = np.array( [ nrmlz_X[1], nrmlz_X[5] ] )
    via_2 = np.array( [ nrmlz_X[2], nrmlz_X[6] ] )
    via_3 = np.array( [ nrmlz_X[3], nrmlz_X[7] ] )
    grad_via1 = np.zeros( 2, dtype=float )
    grad_via2 = np.zeros( 2, dtype=float )
    Collision_occur = False
    
    for i in range( len(nrmlz_obs_set) ):
        
        cntr = nrmlz_obs_set[i,0:2]
        r = nrmlz_obs_set[i,2]
        vec_11 = cntr - via_1
        vec_12 = via_0 - via_1
        vec_21 = cntr - via_2
        vec_22 = via_3 - via_2
        
        if( norm( vec_12 ) > 1e-8 ):
            if( ( norm( vec_11 ) < r ) or ( norm( np.cross( vec_11, vec_12 ) ) / norm( vec_12 ) < r and np.inner( vec_11, vec_12 ) > 0 ) ):
            
                grad_via1 = grad_via1 + mov_rate * ( calc_via_mov_dir( cntr, via_1, via_0 ) + vec_12 / norm( vec_12 ) )
                Collision_occur = True
                
        if( norm( vec_22 ) > 1e-8 ):
            if( ( norm( vec_21 ) < r ) or ( norm( np.cross( vec_21, vec_22 ) ) / norm( vec_22 ) < r and np.inner( vec_21, vec_22 ) > 0 ) ):    
            
                grad_via2 = grad_via2 + mov_rate * ( calc_via_mov_dir( cntr, via_2, via_3 ) + vec_22 / norm( vec_22 ) )
                Collision_occur = True
        
        if( norm( via_1 - via_2 ) > 1e-8 and line_seg_outside_circle( via_1, via_2, cntr, r ) == False ):
            if( norm( via_1 - via_2 ) >= 1e-8 ):
                
                if( norm( cntr - via_1 ) < norm( cntr - via_2 ) ):
                    grad_via1 = grad_via1 + mov_rate * calc_via_mov_dir( cntr, via_1, via_2 )
                    grad_via2 = grad_via2 + 0.5 * mov_rate * calc_via_mov_dir( cntr, via_1, via_2 )
                
                elif( norm( cntr - via_2 ) < norm( cntr - via_1 ) ):
                    grad_via1 = grad_via1 + 0.5 * mov_rate * calc_via_mov_dir( cntr, via_1, via_2 )
                    grad_via2 = grad_via2 + mov_rate * calc_via_mov_dir( cntr, via_1, via_2 )
                
                else:
                    grad_via1 = grad_via1 + 0.5 * mov_rate * calc_via_mov_dir( cntr, via_1, via_2 )
                    grad_via2 = grad_via2 + 0.5 * mov_rate * calc_via_mov_dir( cntr, via_1, via_2 )
                
                Collision_occur = True
                
#    print(grad_via_1)
#    print(grad_via_2)
    
    return grad_via1, grad_via2, Collision_occur
    

## check
def check_via_pts_out_obs( nrmlz_X, grad_nrmlz_X, nrmlz_obs_set ):
    
    mov_rate = 0.02
    via_0 = np.array( [ nrmlz_X[0], nrmlz_X[4] ] )
    via_1 = np.array( [ nrmlz_X[1], nrmlz_X[5] ] )
    via_2 = np.array( [ nrmlz_X[2], nrmlz_X[6] ] )
    via_3 = np.array( [ nrmlz_X[3], nrmlz_X[7] ] )
    grad_via1 = np.array( [ grad_nrmlz_X[1], grad_nrmlz_X[5] ] )
    grad_via2 = np.array( [ grad_nrmlz_X[2], grad_nrmlz_X[6] ] )
    
    for i in range(len(nrmlz_obs_set)):
        
        cntr = nrmlz_obs_set[i,0:2]
        r = nrmlz_obs_set[i,2]
        
        if( norm( via_1 + grad_via1 - cntr ) <= r ):
            dir = np.flip( cntr - via_1 ) * [1,-1]
            grad_via1 = np.inner( grad_via1, dir ) / norm( dir, dir ) * dir
        
        if( norm( via_2 + grad_via2 - cntr ) <= r ):
            dir = np.flip( cntr - via_2 ) * [1,-1]
            grad_via1 = np.inner( grad_via2, dir ) / norm( dir, dir ) * dir


## smooth invert step function for weighting use, centered at 0.5
def smooth_step( x ):
    
    return np.exp(-100*x)*np.exp(50) / (1.0+np.exp(-100*x)*np.exp(50))
    

## calculate friction force
def calc_frctn_f( nrml_f, a_mB, mu_s, v_mB, m ):
    
    if( v_mB > 1e-8 ): # moving forward
        return np.clip( - np.abs( nrml_f * mu_s ), -np.inf, 0. )
    elif( v_mB < -1e-8 ):  # moving backward
        return np.clip( np.abs( nrml_f * mu_s ), 0., np.inf )
    else:
        return np.clip( - m * a_mB, - nrml_f * mu_s, nrml_f * mu_s )


'''
plt.style.use('grayscale')
#nrmlz_X = np.random.uniform( 0.25,0.75,size=8)
#nrmlz_X[[0,4]] = [0,0]
#nrmlz_X[[3,7]] = [1,1]
nrmlz_X = np.array([1.000,  1.000,  1.000,  0.000,  1.000,  1.000,  1.000,  0.000])
print(nrmlz_X)
nrmlz_obs_set = np.array( [[0.5,0.5,0.3]] )
fig, ax = plt.subplots()
ax.add_artist(plt.Circle(nrmlz_obs_set[0,0:2], nrmlz_obs_set[0,2], fill=False))
plt.plot(nrmlz_X[0:4],nrmlz_X[4:])
for i in range(100):
    grad_via1, grad_via2 = check_via_pts_in_obs( nrmlz_X, nrmlz_obs_set )
    nrmlz_X[[1,5]] = nrmlz_X[[1,5]] + grad_via1
    nrmlz_X[[2,6]] = nrmlz_X[[2,6]] + grad_via2
    plt.plot(nrmlz_X[0:4],nrmlz_X[4:])
plt.axis('equal')
plt.show()
'''
