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
from TrajectoryPlanning import CubicTrajetoryPlanning_v2

np.set_printoptions( formatter={'float': '{: 0.3f}'.format} )

MODEL_NUMBER = 4
seeds = [ 1234 , 5460 , 10 , 445 , 2500 ]
Training = True

#tf.set_random_seed( seeds[MODEL_NUMBER-1] )
np.random.seed( seeds[MODEL_NUMBER-1] )

## Global condition
dt = 0.012 # [sec]
grvty = 9810 # [mm/sec^2]
SPACE_UP_LIMIT = 650 # [mm]
SPACE_DOWN_LIMIT = 0 # [mm]
SPACE_RIGHT_LIMIT = 650 # [mm]
SPACE_LEFT_LIMIT = 0 # [mm]
BOX_HALF_LENGTH = 150 # [mm]

## physical conditions
m = 1 # [kg]
std_tau = 120 # [N-mm] of torque sensor
v_avg_min = 100 # [mm/sec]
v_avg_max = 150 # [mm/sec]
theta_B_0_range = 30 * pi/180 # [rad]
via_pt_num = 4
intrpltn_num = 1
P = (3e-10) * np.array([[ (dt**3)/6, 0,  0 ], # covariance matrix for Kalman filter
                       [ 0, (dt**2)/2,  0 ],
                       [ 0,         0, dt ]], dtype=float)
param_num = intrpltn_num * ( via_pt_num - 1 ) # parameters: theta1, theta2, ...
X_BOUND = np.array([ -pi/6 * np.ones( param_num, dtype=float ), pi/6 * np.ones( param_num, dtype=float ) ]) # x upper and lower bounds
via_theta = np.zeros( param_num, dtype=float )
#via_theta = np.array( [ -pi/6, pi/6, 0 ], dtype=float ) # for hard cond. and Syntec Report

## parameters for iLQR
ep_num = 15
dim_x = 4
dim_u = 1
w_x = 1e-9
w_omega = 1e-4

#PARENT_NUM = 8 # for another deleted algorithm



####################
### FUNCTION SET ###
####################

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
#    P_coefficient = 0.5 #[deg/mm]
#    I_coefficient = 0.0
#    D_coefficient = 0.00
    
    ## perfect ##
#    P_coefficient = 0.7 #[deg/mm]
#    I_coefficient = 0.2
#    D_coefficient = 0.034
    
    ## perfect with using EKF ##
    P_coefficient = 0.5 #[deg/mm]
    I_coefficient = 0.2
    D_coefficient = 0.0003
    
#    P_coefficient = 0.5 #[deg/mm]
#    I_coefficient = 0.
#    D_coefficient = 0.014
     
    PID_controller.integral_term = PID_controller.integral_term + error * dt
    delta_x_command = P_coefficient * ( error + I_coefficient * PID_controller.integral_term + D_coefficient * ( error - PID_controller.error_prvs ) / dt )
    PID_controller.error_prvs = error
    
#    if( delta_x_command > 90.0 ):
#        return 90.0
#    elif( delta_x_command < -90.0 ):
#        return -90.0
#    else:
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

## Run the dynamic simulation
def run_dynamic_model( x_Bx, x_By, theta_B_, t_duration, x_mB_0 ):
    
    x_B = np.vstack((x_Bx, x_By)).T # [mm]
    v_B = np.vstack( ( [0,0], np.diff( x_B, axis=0 ) ) ) # [mm/sec]
    a_B = np.vstack( ( [0,0], np.diff( v_B, axis=0 ) ) ) # [mm/sec^2]
    
    ## variable space initialization ##
    itr_num = len( x_Bx )
    t_axis = np.arange( 0, itr_num*dt, dt )
    x_mB, v_mB, a_mB =  np.zeros( ( 3, itr_num ), dtype=float ) # [mm], [mm/sec], [mm/sec^2]
    theta_B_PID, omega_B, alpha_B = np.zeros( ( 3, itr_num ), dtype=float ) # [rad], [rad/sec], [rad/sec^2]
    theta_B = np.copy( theta_B_ )

    ## kinematics initial conditions
    x_mB[0] = x_mB_0
    v_mB[0] = ( 0 - v_B[0,0] ) * np.cos( theta_B[0] ) + ( 0 - v_B[0,1] ) * np.sin( theta_B[0] )
    PID_controller( 0 , True ) # reset PID controller

    ### Run the dynamic simulation ###
    for idx in range( itr_num - 1 ):
    
        ## update the dynamic state of the sliding box ##
        a_mB[idx+1] = omega_B[idx]**2 * x_mB[idx] - ( a_B[idx,0] * np.cos(theta_B[idx]) + a_B[idx,1] * np.sin(theta_B[idx]) + grvty * np.sin(theta_B[idx]) )           
        v_mB[idx+1] = v_mB[idx] + a_mB[idx+1] * dt
        x_mB[idx+1] = x_mB[idx] + v_mB[idx+1] * dt + 0.5 * a_mB[idx+1] * dt**2
    
        ## Boundary constraints ##
        if( x_mB[idx+1] >= BOX_HALF_LENGTH or x_mB[idx+1] <= - BOX_HALF_LENGTH ):
                
            if( np.sign( x_mB[idx+1] ) > 0 ):
                x_mB[idx+1] = BOX_HALF_LENGTH
                v_mB[idx+1] = np.clip( v_mB[idx+1], -np.inf, 0. )
                
            elif( np.sign( x_mB[idx+1] ) < 0 ):
                x_mB[idx+1] = - BOX_HALF_LENGTH
                v_mB[idx+1] = np.clip( v_mB[idx+1], 0., np.inf )
            
        theta_B_PID[idx+1] = PID_controller( x_mB[idx+1] , False ) * pi/180
        theta_B[idx+1] = theta_B[idx+1] + theta_B_PID[idx+1]
        theta_B[idx+1] = np.clip( theta_B[idx+1], -pi/3, pi/3 )
            
        omega_B[idx+1] = ( theta_B[idx+1] - theta_B[idx] ) / dt
        alpha_B[idx+1] = ( omega_B[idx+1] - omega_B[idx] ) / dt
        
        return t_axis, x_B, v_B, a_B, x_mB, v_mB, a_mB, theta_B, omega_B, alpha_B, theta_B_PID


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
    Q = 2e-2 * np.array([[ (dt**3)/6,         0,  0 ],
                         [         0, (dt**2)/2,  0 ],
                         [         0,         0, dt ]], dtype=float )
                   
    I = np.identity( 3, dtype=float )
    
    state_bar = A.dot( state_prvs ) + u
#    print(state_bar)
    P_bar = A.dot( P_prvs.dot( A.T ) ) + Q
    #             3*3    3*1                  1*3  3*3    3*1   1
    KalmanGain = P_bar.dot( C.T ).dot( np.linalg.inv( C.dot( P_bar.dot( C.T ) ) + R ) )
#    print(KalmanGain)
    est_state = state_bar + KalmanGain.dot(  msr_state - C.dot( state_bar ) )
#    print(msr_state-C.dot( state_bar ))
    P = ( I - KalmanGain.dot( C ) ).dot( P_bar )
    
    return est_state, P
        
####################
### FUNCTION END ###
####################

### initialize the environment ###    
## initial kinematic  conditions ##
if( Training == True ):
    via_x, via_y, v_avg, theta_B_0, x_mB_0 = set_random_ini_cond() # !!!Normalized values
    via_x, via_y, v_avg, theta_B_0, x_mB_0 = denrmlz_values( via_x, via_y, v_avg, theta_B_0, x_mB_0 ) # recover values
    
else:
#    ini_cond_path = "./GA_data/ini_cond(" + str(MODEL_NUMBER) + ").csv"
    ini_cond_path = "./ini_cond_hard.csv" # read hard cond. for Syntec report
    via_x, via_y, v_avg, theta_B_0, x_mB_0 = read_ini_cond( ini_cond_path )

# x_mB_0 = 0


x_Bx, x_By, theta_B, t_duration = CubicTrajetoryPlanning_v2( via_x, via_y, np.concatenate( ( [theta_B_0], via_theta ) ) , v_avg , dt )
x_B = np.vstack((x_Bx, x_By)).T # [mm]
v_B = np.vstack( ( [0,0], np.diff( x_B, axis=0 ) ) ) / dt # [mm/sec]

## blending the velocity plot
t_idx = 0
for i in t_duration[0:-1]:
    t_idx = t_idx + int(i/dt)
    v_B[t_idx] = ( v_B[t_idx-1] + v_B[t_idx+1] ) /2

a_B = np.vstack( ( [0,0], np.diff( v_B, axis=0 ) ) ) / dt # [mm/sec^2]

## variable space initialization ##
itr_num = len( x_Bx )
t_axis = np.arange( 0, itr_num*dt, dt )
x_mB, v_mB, a_mB =  np.zeros( ( 3, itr_num ), dtype=float ) # [mm], [mm/sec], [mm/sec^2]
#a_mB_cntr, a_mB_box, a_mB_grvty = np.zeros( ( 3, itr_num ), dtype=float ) # for Syntec report
theta_B_PID, omega_B, alpha_B = np.zeros( ( 3, itr_num ), dtype=float ) # [rad], [rad/sec], [rad/sec^2]
nrml_f = np.zeros( itr_num, dtype=float ) # [N]
msr_tau, tau = np.zeros( ( 2, itr_num ), dtype=float ) # [N-mm]

## kinematics and force initial conditions
x_mB[0] = x_mB_0
v_mB[0] = ( 0 - v_B[0,0] ) * np.cos( theta_B[0] ) + ( 0 - v_B[0,1] ) * np.sin( theta_B[0] )
nrml_f[0] = m * alpha_B[0] * x_mB[0] + 2 * m * omega_B[0] * v_mB[0] + m * ( a_B[0,1] * np.cos(theta_B[0]) + grvty * np.cos(theta_B[0]) - a_B[0,0] * np.sin(theta_B[0]) )
nrml_f[0] = 0.001 * nrml_f[0]

## initialize variable space for iLQR
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

for ep_idx in range( ep_num+2 ):
    
    PID_controller( 0 , True ) # reset PID controller    
    est_state = np.array( [ [ x_mB[0] ], [ v_mB[0] ], [ a_mB[0] ] ] )
    est_x_mB = np.copy( x_mB )
    
    ### Run the dynamic simulation ###
    for idx in range( itr_num - 1 ):
        
        if( ep_idx > 0 and ep_idx < ep_num + 1 ):           
            ## state feedback control for iLQR ##
            state = np.array([ [x_mB[idx]], [v_mB[idx]], [a_mB[idx]], [theta_B[idx]] ])
            omega_B[idx] = K_set[idx].dot( state ) + k_set[idx]        
#            omega_B[idx] = np.clip( omega_B[idx], -0.15, 0.15 )
            theta_B[idx+1] = theta_B[idx] + omega_B[idx] * dt
#            theta_B[idx+1] = np.clip( theta_B[idx+1], -pi/3, pi/3 )
            
        ## update the dynamic state of the sliding box ##
        a_mB[idx+1] = omega_B[idx]**2 * x_mB[idx] - ( a_B[idx,0] * np.cos(theta_B[idx]) + a_B[idx,1] * np.sin(theta_B[idx]) + grvty * np.sin(theta_B[idx]) )
		
        nrml_f[idx+1] = m * alpha_B[idx] * x_mB[idx] + 2 * m * omega_B[idx] * v_mB[idx] + m * ( a_B[idx,1] * np.cos(theta_B[idx]) + grvty * np.cos(theta_B[idx]) - a_B[idx,0] * np.sin(theta_B[idx]) )
        nrml_f[idx+1] = 0.001 * nrml_f[idx+1] # unit translation to [N]
		
        ## for Syntec Report ##
#        a_mB_cntr[idx+1] = omega_B[idx]**2 * x_mB[idx]
#        a_mB_box[idx+1] = - a_B[idx,0] * np.cos(theta_B[idx]) - a_B[idx,1] * np.sin(theta_B[idx])
#        a_mB_grvty[idx+1] = - grvty * np.sin(theta_B[idx])
        
        v_mB[idx+1] = v_mB[idx] + a_mB[idx+1] * dt
        x_mB[idx+1] = x_mB[idx] + v_mB[idx+1] * dt + 0.5 * a_mB[idx+1] * dt**2
    
        ## Boundary constraints ##
        if( x_mB[idx+1] >= BOX_HALF_LENGTH or x_mB[idx+1] <= - BOX_HALF_LENGTH ):
                
            if( np.sign( x_mB[idx+1] ) > 0 ):
                x_mB[idx+1] = BOX_HALF_LENGTH
                v_mB[idx+1] = np.clip( v_mB[idx+1], -np.inf, 0. )
                
            elif( np.sign( x_mB[idx+1] ) < 0 ):
                x_mB[idx+1] = - BOX_HALF_LENGTH
                v_mB[idx+1] = np.clip( v_mB[idx+1], 0., np.inf )
        
        tau[idx+1] = x_mB[idx+1] * nrml_f[idx+1] # [N-mm]=[kg-m/sec^2 - mm]
        msr_tau[idx+1] = tau[idx+1] * np.random.normal( 1., 0.03 )
        
        if( idx>100 ):
            
            est_state, P = EKF( est_state, 1000*np.array([[msr_tau[idx+1]]]), P,  m, theta_B[idx], omega_B[idx], alpha_B[idx], a_B[idx,:] )
            
            ## boundary condition for EKF
            if( est_state[0,0] >= BOX_HALF_LENGTH or est_state[0,0] <= - BOX_HALF_LENGTH ):
                
                if( np.sign( est_state[0,0] ) > 0 ):
                    est_state[0,0] = BOX_HALF_LENGTH
                    est_state[1,0] = np.clip( est_state[1,0], -np.inf, 0. )
                
                elif( np.sign( est_state[0,0] ) < 0 ):
                    est_state[0,0] = - BOX_HALF_LENGTH
                    est_state[1,0] = np.clip( est_state[1,0], 0., np.inf )
            
        else:
            est_state = np.array( [ [ x_mB[idx+1] ], [ v_mB[idx+1] ], [ a_mB[idx+1] ] ] )
    
        est_x_mB[idx+1] = est_state[0,0]
#        est_x_mB[idx+1] = x_mB[idx+1]
        
        if( ep_idx == ep_num+1 ):
            ## PID controller part ##
            theta_B_PID[idx+1] = PID_controller( est_x_mB[idx+1] , False ) * pi/180
            theta_B[idx+1] = theta_B[idx+1] + theta_B_PID[idx+1]
            theta_B[idx+1] = np.clip( theta_B[idx+1], -pi/3, pi/3 )
                
            omega_B[idx+1] = ( theta_B[idx+1] - theta_B[idx] ) / dt
                
        alpha_B[idx+1] = ( omega_B[idx+1] - omega_B[idx] ) / dt
    
    ## iLQR part
    for i in range( itr_num-2, -1, -1 ):
        
        F_set[i] = prtl_f_prtl_x_and_u( theta_B[i], omega_B[i], a_B[i], x_mB[i] ) 
        f_set[i] = np.array([ [x_mB[i+1]], [v_mB[i+1]], [a_mB[i+1]], [theta_B[i+1]]]) - F_set[i].dot( np.array([ [x_mB[i]], [v_mB[i]], [a_mB[i]], [theta_B[i]], [omega_B[i]] ]) )
        
        Q_set[i] = F_set[i].T.dot( V_set[i+1].dot( F_set[i] ) )
        Q_set[i,0,0] = Q_set[i,0,0] + w_x # specialized for this case
        Q_set[i,dim_x,dim_x] = Q_set[i,dim_x,dim_x] + w_omega # specialized for this case
        q_set[i] = F_set[i].T.dot( V_set[i+1].dot( f_set[i] ) ) + F_set[i].T.dot( v_set[i+1] )
        
        K_set[i] = - Q_set[ i, dim_x:, :dim_x ] / ( Q_set[ i, dim_x:, dim_x: ] + 1e-9 )
        k_set[i] = - q_set[ i, dim_x:, : ] / ( Q_set[ i, dim_x:, dim_x: ] + 1e-9 )
#        try:
#            K_set[i] = - np.linalg.solve( Q_set[ i, dim_x:, dim_x: ], Q_set[ i, dim_x:, :dim_x ] )
#        except LinAlgError:
#            K_set[i] = - 1e3 * Q_set[ i, dim_x:, :dim_x ]
#            print("time ", i , ": Q_uu = 0")
#        
#        try:
#            k_set[i] = - np.linalg.solve( Q_set[ i, dim_x:, dim_x: ], q_set[ i, dim_x:, : ] )
#        except LinAlgError:
#            k_set[i] = - 1e3 * q_set[ i, dim_x:, : ]
            
        
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
    print( "Ep ", ep_idx, ": %.4f" %dist_RMSE[ep_idx] )
    
    
   
### record experiment results
#now = datetime.now()
'''
if( Training == True ):
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
plt.plot( t_axis, x_mB )
plt.plot( t_axis, est_x_mB )
plt.show()
#plt.plot(est_x_mB-x_mB)
#plt.show()
print( np.sqrt( np.average( (est_x_mB-x_mB)**2 ) ) )
#plt.plot( t_axis, theta_B*180/pi )
#plt.show()
plt.plot( t_axis[1000:], x_mB[1000:] )    
plt.show()
#plt.plot( t_axis[4:], alpha_B[4:]*180/pi )
    

## Syntec Report ##
window_half = 2
for i in range( window_half, len(msr_tau)-window_half ):
    msr_tau[i] = np.median( msr_tau[ (i - window_half) : (i + window_half) ] )
        
np.savetxt( "./SyntecReport/x_Bx.csv", x_Bx )
np.savetxt( "./SyntecReport/x_By.csv", x_By )
np.savetxt( "./SyntecReport/theta_B.csv", theta_B )
np.savetxt( "./SyntecReport/t_axis.csv", t_axis )
np.savetxt( "./SyntecReport/x_mB.csv", x_mB)
np.savetxt( "./SyntecReport/est_x_mB.csv", est_x_mB)
np.savetxt( "./SyntecReport/msr_tau.csv", msr_tau)
#np.savetxt( "./SyntecReport/a_mB_cntr.csv", a_mB_cntr )
#np.savetxt( "./SyntecReport/a_mB_box.csv", a_mB_box)
#np.savetxt( "./SyntecReport/a_mB_grvty.csv",a_mB_grvty )
