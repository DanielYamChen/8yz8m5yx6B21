# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:41:59 2019

@author: brl-pc5
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi


def get_acc_at_m( x_m, v_m, x_B, v_B, a_B, alpha_B, omega_B ):
    
    return np.array([[ a_B[0] - alpha_B * ( x_m[1] - x_B[1] ) - omega_B**2 * ( x_m[0] - x_B[0] ) - 2 * omega_B * ( v_m[1] - v_B[1] )],
                     [ a_B[1] + alpha_B * ( x_m[0] - x_B[0] ) - omega_B**2 * ( x_m[1] - x_B[1] ) + 2 * omega_B * ( v_m[0] - v_B[0] )] 
                    ])
    

    
dt = 0.012 # [sec]
#total_t = 4 * np.sqrt(3) # [sec]
total_t = 12 #[sec]
itr_num = int( total_t / dt )
grvty = 0. # [m/sec^2]
m = 1. # [kg]
x_m, x_B, v_m, v_B, a_m, a_B = np.zeros( ( 6, itr_num, 2 ), dtype=float ) # [m], [m/sec], [m/sec^2]
x_mB, v_mB, a_mB =  np.zeros( ( 3, itr_num ), dtype=float ) # [m], [m/sec], [m/sec^2]
nrml_f = np.zeros( itr_num, dtype=float ) # [N]
theta_B, omega_B, alpha_B = np.zeros( ( 3, itr_num ), dtype=float ) # [rad], [rad/sec], [rad/sec^2]

#theta_B[0] = 37 * pi / 180 # [rad]
#for i in range(itr_num):
#    a_B[i,0] = - 0.5 * grvty
omega_B[0] = pi / 12 # [rad/sec]
v_m[0,:] = [ 0.25, 0. ]

x_mB[0] = ( x_m[0,0] - x_B[0,0] ) * np.cos( theta_B[0] ) + ( x_m[0,1] - x_B[0,1] ) * np.sin( theta_B[0] )
v_mB[0] = ( v_m[0,0] - v_B[0,0] ) * np.cos( theta_B[0] ) + ( v_m[0,1] - v_B[0,1] ) * np.sin( theta_B[0] )

#v_m_ = np.copy(v_m)
#x_m_ = np.copy(x_m)
#x_m__ = np.copy(x_m)
#v_m__ = np.copy(v_m)
#a_m_ = np.copy(a_m)
#a_mB_ = np.copy(a_mB)
for idx in range(itr_num-1):
    
    omega_B[idx+1] = omega_B[idx] + alpha_B[idx] * dt
    theta_B[idx+1] = theta_B[idx] + omega_B[idx] * dt
    
    v_B[idx+1,:] = v_B[idx,:] + a_B[idx,:] * dt
    x_B[idx+1,:] = x_B[idx,:] + v_B[idx,:] * dt
    
    ## another method ##
#    acc_at_m_ = get_acc_at_m( x_m_[idx], v_m_[idx], x_B[idx], v_B[idx], a_B[idx], alpha_B[idx], omega_B[idx] )
#    a_mB_[idx] = - acc_at_m_[0,0] * np.cos( theta_B[idx] ) - ( acc_at_m_[1,0] + grvty ) * np.sin( theta_B[idx] )
#    
#    a_m_[idx+1,:] = acc_at_m_[:,0] + ( a_mB_[idx] + 2 * omega_B[idx]**2 * x_mB[idx] ) * np.array([ np.cos(theta_B[idx]) , np.sin(theta_B[idx]) ])
#    v_m_[idx+1,:] = v_m_[idx,:] + a_m_[idx+1,:] * dt
#    x_m_[idx+1,:] = x_m_[idx,:] + v_m_[idx+1,:] * dt + 0.5 * a_m_[idx+1,:] * dt**2
    ####################
    
#    acc_at_m = get_acc_at_m( x_m[idx], v_m[idx], x_B[idx], v_B[idx], a_B[idx], alpha_B[idx], omega_B[idx] )    
#    a_mB[idx] = - acc_at_m[0,0] * np.cos( theta_B[idx] ) - ( acc_at_m[1,0] + grvty ) * np.sin( theta_B[idx] )
#    nrml_f[idx] = m * ( acc_at_m[0,0] * np.sin( theta_B[idx] ) + ( acc_at_m[1,0] + grvty ) * np.cos( theta_B[idx] ) )
    a_mB[idx] = 3 * omega_B[idx]**2 * x_mB[idx] - ( a_B[idx,0] * np.cos(theta_B[idx]) + a_B[idx,1] * np.sin(theta_B[idx]) + grvty * np.sin(theta_B[idx]) )
    nrml_f[idx] = m * alpha_B[idx] * x_mB[idx] + 2 * m * omega_B[idx] * v_mB[idx] + m * ( a_B[idx,1] * np.cos(theta_B[idx]) + grvty * np.cos(theta_B[idx]) - a_B[idx,0] * np.sin(theta_B[idx]) )
   
    v_mB[idx+1] = v_mB[idx] + a_mB[idx] * dt
    x_mB[idx+1] = x_mB[idx] + v_mB[idx] * dt + 0.5 * a_mB[idx] * dt**2
        
    v_m[idx+1,:] = v_mB[idx+1] * np.array([ np.cos(theta_B[idx+1]) , np.sin(theta_B[idx+1]) ]) + v_B[idx+1,:] + omega_B[idx+1] * x_mB[idx+1] * np.array([ -np.sin(theta_B[idx+1]) , np.cos(theta_B[idx+1]) ])
#    x_m__[idx+1,:] = x_m__[idx,:] + v_m[idx,:] * dt
    x_m[idx+1,:] = x_mB[idx+1] * np.array([ np.cos(theta_B[idx+1]) , np.sin(theta_B[idx+1]) ]) + x_B[idx+1,:]
#    v_m__[idx+1] = ( x_m[idx+1,；] - x_m[idx,:] ) / dt
    a_m[idx+1] = ( v_m[idx+1,:] - v_m[idx,:] ) / dt
    
    
    
    
    
    
    
    
    