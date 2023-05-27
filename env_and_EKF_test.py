# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:41:59 2019

@author: brl-pc5
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from TrajectoryPlanning import CubicTrajetoryPlanning

SEED = 11
np.random.seed(SEED)

dt = 0.012 # [sec]
grvty = 9810 # [mm/sec^2]

####################
### FUNCTION SET ###
####################

## another version 
def get_acc_at_m( x_m, v_m, x_B, v_B, a_B, alpha_B, omega_B ):
    
    return np.array([[ a_B[0] - alpha_B * ( x_m[1] - x_B[1] ) - omega_B**2 * ( x_m[0] - x_B[0] ) - 2 * omega_B * ( v_m[1] - v_B[1] )],
                     [ a_B[1] + alpha_B * ( x_m[0] - x_B[0] ) - omega_B**2 * ( x_m[1] - x_B[1] ) + 2 * omega_B * ( v_m[0] - v_B[0] )] 
                    ])

## 
def state_transition( x, dt ):
    A = np.array([[              1, dt,  0.5*(dt**2) ],
                  [              0,  1,           dt ],
                  [ 3*(omega_B**2),  0,            0 ]], dtype=float )
    
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
    Q = 3e-4 * np.array([[ (dt**3)/6,         0,  0 ],
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


def PID_controller( error , reset ):

    if not hasattr( PID_controller, "error_prvs" ):
        PID_controller.error_prvs = error
    
    if not hasattr( PID_controller, "integral_term" ):
        PID_controller.integral_term = 0
    
    if( reset == True ):
        PID_controller.error_prvs = error
        PID_controller.integral_term = 0

#    P_coefficient = 0.1 #[deg/mm]
#    I_coefficient = 0.0
#    D_coefficient = 0.00
    P_coefficient = 0.5 #[deg/mm]
    I_coefficient = 0.00
    D_coefficient = 0.007
     
    PID_controller.integral_term = PID_controller.integral_term + error * dt
    delta_x_command = ( P_coefficient * error ) + I_coefficient * PID_controller.integral_term + D_coefficient * ( error - PID_controller.error_prvs ) / dt
    PID_controller.error_prvs = error
    
#    if( delta_x_command > 90.0 ):
#        return 90.0
#    elif( delta_x_command < -90.0 ):
#        return -90.0
#    else:
    return delta_x_command


### initialize the environment ###

## initial condition setting ##
# physical conditions
SPACE_UP_LIMIT = 650 # [mm]
SPACE_RIGHT_LIMIT = 650 # [mm]
BOX_HALF_LENGTH = 150 # [mm]
m = 1 # [kg]
std_tau = 120 # [N-mm]
# covariance matrix for Kalman filter
P = (1e-8) * np.array([[ (dt**3)/6, 0,  0 ],
                       [ 0, (dt**2)/2,  0 ],
                       [ 0,         0, dt ]], dtype=float)

## kinematics conditions ##
via_x = [ SPACE_UP_LIMIT, 0, SPACE_UP_LIMIT, 0 ]
via_y = [ SPACE_UP_LIMIT, 2/3*SPACE_UP_LIMIT, 1/3*SPACE_UP_LIMIT, 0 ]
via_theta = [ 0, 0, 0, 0 ]
x_Bx, x_By, theta_B = CubicTrajetoryPlanning( via_x , via_y , via_theta , 150 , dt )
x_B = np.vstack((x_Bx, x_By)).T

## for test
#total_t = 12 #[sec]
#itr_num = int( total_t / dt )
itr_num = len( x_Bx )

## variable space initialization ##
t_axis = np.arange( 0, itr_num*dt, dt )
x_m, v_m, a_m = np.zeros( ( 3, itr_num, 2 ), dtype=float ) # [mm], [mm/sec], [mm/sec^2]
#x_B, v_B, a_B = np.zeros( ( 3, itr_num, 2 ), dtype=float ) # [mm], [mm/sec], [mm/sec^2]
x_mB, v_mB, a_mB =  np.zeros( ( 3, itr_num ), dtype=float ) # [mm], [mm/sec], [mm/sec^2]
#theta_B, omega_B, alpha_B = np.zeros( ( 3, itr_num ), dtype=float ) # [rad], [rad/sec], [rad/sec^2]
omega_B, alpha_B = np.zeros( ( 2, itr_num ), dtype=float ) # [rad], [rad/sec], [rad/sec^2]
nrml_f = np.zeros( itr_num, dtype=float ) # [N]
msr_tau, tau = np.zeros( ( 2, itr_num ), dtype=float ) # [N-mm]

v_B = np.vstack( ( [0,0], np.diff( x_B, axis=0 ) ) )
a_B = np.vstack( ( [0,0], np.diff( v_B, axis=0 ) ) )
x_m[0] = x_B[0] + 1.0 * BOX_HALF_LENGTH * np.array([ np.cos(theta_B[0]) , np.sin(theta_B[0]) ])

# kinematics conditions
#omega_B[0] = pi / 12 # [rad/sec]
#v_m[0,:] = [ 0.25, 0. ] # [m/sec]


#v_m_ = np.copy(v_m)
#x_m_ = np.copy(x_m)
#a_m_ = np.copy(a_m)
#a_mB_ = np.copy(a_mB)

x_mB[0] = ( x_m[0,0] - x_B[0,0] ) * np.cos( theta_B[0] ) + ( x_m[0,1] - x_B[0,1] ) * np.sin( theta_B[0] )
v_mB[0] = ( v_m[0,0] - v_B[0,0] ) * np.cos( theta_B[0] ) + ( v_m[0,1] - v_B[0,1] ) * np.sin( theta_B[0] )
nrml_f[0] = m * alpha_B[0] * x_mB[0] + 2 * m * omega_B[0] * v_mB[0] + m * ( a_B[0,1] * np.cos(theta_B[0]) + grvty * np.cos(theta_B[0]) - a_B[0,0] * np.sin(theta_B[0]) )
nrml_f[0] = 0.001 * nrml_f[0]
PID_controller( 0 , True )
est_state = np.array( [ [ x_mB[0] ], [ v_mB[0] ], [ a_mB[0] ] ] )
est_x_mB = np.copy( x_mB )
### Run the dynamic simulation ###
for idx in range(itr_num-1):
    
    ## update kinematic state of the plate ##
#    omega_B[idx+1] = omega_B[idx] + alpha_B[idx] * dt
#    alpha_B[idx+1] = theta_B[idx] + omega_B[idx] * dt
    
#    v_B[idx+1,:] = v_B[idx,:] + a_B[idx,:] * dt
#    x_B[idx+1,:] = x_B[idx,:] + v_B[idx,:] * dt
     
    
    ## another method ##
#    acc_at_m_ = get_acc_at_m( x_m_[idx], v_m_[idx], x_B[idx], v_B[idx], a_B[idx], alpha_B[idx], omega_B[idx] )
#    a_mB_[idx] = - acc_at_m_[0,0] * np.cos( theta_B[idx] ) - ( acc_at_m_[1,0] + grvty ) * np.sin( theta_B[idx] )
#    
#    a_m_[idx+1,:] = acc_at_m_[:,0] + ( a_mB_[idx] + 2 * omega_B[idx]**2 * x_mB[idx] ) * np.array([ np.cos(theta_B[idx]) , np.sin(theta_B[idx]) ])
#    v_m_[idx+1,:] = v_m_[idx,:] + a_m_[idx+1,:] * dt
#    x_m_[idx+1,:] = x_m_[idx,:] + v_m_[idx+1,:] * dt + 0.5 * a_m_[idx+1,:] * dt**2
    ####################
    
    ## update the dynamic state of the sliding box ##
#    acc_at_m = get_acc_at_m( x_m[idx], v_m[idx], x_B[idx], v_B[idx], a_B[idx], alpha_B[idx], omega_B[idx] )    
#    a_mB[idx] = - acc_at_m[0,0] * np.cos( theta_B[idx] ) - ( acc_at_m[1,0] + grvty ) * np.sin( theta_B[idx] )
#    nrml_f[idx] = m * ( acc_at_m[0,0] * np.sin( theta_B[idx] ) + ( acc_at_m[1,0] + grvty ) * np.cos( theta_B[idx] ) )    
    a_mB[idx+1] = 3 * omega_B[idx]**2 * x_mB[idx] - ( a_B[idx,0] * np.cos(theta_B[idx]) + a_B[idx,1] * np.sin(theta_B[idx]) + grvty * np.sin(theta_B[idx]) )
    nrml_f[idx+1] = m * alpha_B[idx] * x_mB[idx] + 2 * m * omega_B[idx] * v_mB[idx] + m * ( a_B[idx,1] * np.cos(theta_B[idx]) + grvty * np.cos(theta_B[idx]) - a_B[idx,0] * np.sin(theta_B[idx]) )
    nrml_f[idx+1] = 0.001 * nrml_f[idx+1] # unit translation to [N]
    
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
    
    v_m[idx+1,:] = v_mB[idx+1] * np.array([ np.cos(theta_B[idx+1]) , np.sin(theta_B[idx+1]) ]) + v_B[idx+1,:] + omega_B[idx+1] * x_mB[idx+1] * np.array([ -np.sin(theta_B[idx+1]) , np.cos(theta_B[idx+1]) ])
    x_m[idx+1,:] = x_mB[idx+1] * np.array([ np.cos(theta_B[idx+1]) , np.sin(theta_B[idx+1]) ]) + x_B[idx+1,:]
    a_m[idx+1] = ( v_m[idx+1,:] - v_m[idx,:] ) / dt
    
    tau[idx+1] = x_mB[idx+1] * nrml_f[idx+1] # [N-mm]=[kg-m/sec^2 - mm]
    msr_tau[idx+1] = tau[idx+1] * np.random.normal( 1., 0.03 )
    
    if( idx>100 ):
        est_state, P = EKF( est_state, 1000*np.array([[msr_tau[idx+1]]]), P,  m, theta_B[idx], omega_B[idx], alpha_B[idx], a_B[idx,:] )
    
    else:
        est_state = np.array( [ [ x_mB[idx+1] ], [ v_mB[idx+1] ], [ a_mB[idx+1] ] ] )
    
    est_x_mB[idx+1] = est_state[0,0]
    
#    temp_a = m * alpha_B[idx]
    
    
#    alpha_B[idx+1] = alpha_B[idx+1] + PID_controller( x_mB[idx+1] , False ) * pi/180
#    omega_B[idx+1] = omega_B[idx] + alpha_B[idx+1] * dt
#    theta_B[idx+1] = theta_B[idx] + omega_B[idx+1] * dt + 0.5 * alpha_B[idx+1] * dt**2
      
    theta_B[idx+1] = theta_B[idx+1] + PID_controller( est_x_mB[idx+1] , False ) * pi/180            
    theta_B[idx+1] = np.clip( theta_B[idx+1], -pi/3, pi/3 )
    omega_B[idx+1] = ( theta_B[idx+1] - theta_B[idx] ) / dt
#    omega_B[idx+1] = np.mean( omega_B[idx-1:idx+2] )
    alpha_B[idx+1] = ( omega_B[idx+1] - omega_B[idx] ) / dt
#    alpha_B[idx+1] = np.mean( alpha_B[idx-1:idx+2] )
    

    
    
### plot simulation result ###
plt.plot( t_axis, x_mB )
plt.plot( t_axis, est_x_mB )
plt.show()
plt.plot(est_x_mB-x_mB)
plt.show()
plt.plot( t_axis, theta_B*180/pi )
plt.show()
#plt.plot( t_axis[:250], theta_B[:250]*180/pi )    
#plt.show()
#plt.plot( t_axis[4:], alpha_B[4:]*180/pi )
    
    