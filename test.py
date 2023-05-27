# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:52:41 2019

@author: brl-pc5
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import sympy as sp

## initial condition ##
m, x_m0, v_m = 0, 0, 0
frctn_s, frctn_k = 0, 0
grvty = sp.var('g')

def smblc_cross( A, B ):
    
    return sp.Matrix([[ A[1]*B[2] - A[2]*B[1] ],
                      [ A[2]*B[0] - A[0]*B[2] ],    
                      [ A[0]*B[1] - A[1]*B[0] ]])
 
a_B = sp.Matrix( [ [sp.var('a_Bx')], [sp.var('a_By')], [0] ])
alpha_B = sp.Matrix( [ [0], [0], [sp.var('alpha_B')] ] )
omega_B = sp.Matrix( [ [0], [0], [sp.var('omega_B')] ] )
theta_B = sp.var('theta_B')
R_BG = sp.Matrix([[ sp.cos(theta_B), -sp.sin(theta_B), 0 ],
                  [ sp.sin(theta_B),  sp.cos(theta_B), 0 ],
                  [               0,                0, 1 ]])
R_BG_inv = sp.Matrix([[  sp.cos(theta_B), sp.sin(theta_B), 0 ],
                      [ -sp.sin(theta_B), sp.cos(theta_B), 0 ],
                      [                0,               0, 1 ]])
x_mB = sp.Matrix( [[sp.var('x_mBx')], [sp.var('x_mBy')], [0] ] )
v_mB = sp.Matrix( [[sp.var('v_mBx')], [sp.var('v_mBy')], [0] ] )
a_mB = sp.Matrix( [[sp.var('a_mBx')], [sp.var('a_mBy')], [0] ] )

tang_acc = smblc_cross( alpha_B, x_mB )
nrml_acc = smblc_cross( omega_B, smblc_cross( omega_B, x_mB ) )
coriolis = 2 * smblc_cross( omega_B, R_BG * v_mB )

a_m = a_B + tang_acc + nrml_acc + coriolis + a_mB

m = sp.var('m')
f_img =  - m * a_m

f_m = f_img + sp.Matrix([0,-m*grvty,0])
f_m_at_B = R_BG_inv * f_m
#f_m_at_B[1] = 0
f_m = R_BG * f_m_at_B
a_m =  - f_m / m + a_B

a_m_arg = sp.Matrix( [ sp.var('a_Bx'), sp.var('a_By'), sp.var('alpha_B'),
                       sp.var('omega_B'), sp.var('theta_B'), sp.var('x_mBx'),
                       sp.var('x_mBy'), sp.var('v_mBx') ,sp.var('v_mBy'),
                       sp.var('a_mBx'), sp.var('a_mBy'), grvty ] )

func_a_m = sp.lambdify( a_m_arg, a_m, "numpy" )


dt = 0.012 # [sec]
total_t = 12 #[sec]
itr_num = int( total_t / dt )
grvty = 0.

x_m, x_B, v_m, v_B, a_m, a_B = np.zeros( ( 6, itr_num, 2 ), dtype=float ) #[m], [m/sec], [m/sec^2]
theta_B, omega_B, alpha_B = np.zeros( ( 3, itr_num ), dtype=float ) #[rad], [rad/sec], [rad/sec^2]

omega_B[0] = 0.5236 # [rad/sec]
v_m[0,:] = [ 0.25, 0. ]

for idx in range(itr_num-1):
    x_mB = x_m[idx] - x_B[idx]
    v_mB = v_m[idx] - v_B[idx]
    a_mB = [0., 0.]
#    a_mB = a_m[idx] - a_B[idx]
#    a_mB = -2.5 * x_mB
    
    a_m[idx+1] = func_a_m( a_B[idx,0], a_B[idx,1], alpha_B[idx],
                           omega_B[idx], theta_B[idx], x_mB[0],
                           x_mB[1], v_mB[0] ,v_mB[1],
                           a_mB[0], a_mB[1], grvty )[:2,0]
#    a_m[idx+1] = func_a_m( 1, 1, 1,
#                           1, 1, 1,
#                           1, 1, 1,
#                           1, 1, grvty )[:2,0]
    
#    a_m[idx+1] = np.asarray(temp,dtype=float)
#    print(a_m[idx])
    v_m[idx] = omega_B[idx] * 
    v_m[idx+1] = v_m[idx] + a_m[idx+1,:]*dt + omega_B[idx] * np.no
    x_m[idx+1] = x_m[idx] + v_m[idx+1,:]*dt
    v_B[idx+1] = v_B[idx] + a_B[idx+1,:]*dt
    x_B[idx+1] = x_B[idx] + v_B[idx+1,:]*dt
    omega_B[idx+1] = omega_B[idx] + alpha_B[idx+1]*dt
    theta_B[idx+1] = theta_B[idx] + omega_B[idx+1]*dt
    


