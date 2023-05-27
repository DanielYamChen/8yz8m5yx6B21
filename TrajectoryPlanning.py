# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 17:38:35 2018

@author: Biorola
"""

import numpy as np
from math import pi
from scipy import optimize


### cos wave via point ###
def cos_wave_via( T , amplitude , delta_x ):
    
    pt_num = int( T / delta_x ) + 1    
    temp_x = np.linspace( 0 , T , pt_num )
    temp_y = np.zeros( pt_num )
    temp_theta = np.zeros( pt_num )
    for i in range( pt_num ):
        temp_y[i] = amplitude * ( 1.0 - np.cos( (2*pi/T) * temp_x[i] ) )
        slope = amplitude * (2*pi/T) * np.sin( (2*pi/T) * temp_x[i] )
        temp_theta[i] = np.arctan2( slope , 1.0 ) - pi/2
        
    return temp_x , temp_y , temp_theta


### via-points of Triangle wave ###
def triangle_wave_via( length , height , duty_cycle , delta_x ):
    
    pt_num = int( length / delta_x ) + 1
    temp_x = np.linspace( 0 , length , pt_num )
    temp_y = np.zeros( pt_num )
    temp_theta = np.zeros( pt_num )
    for i in range( pt_num ):
        x_normalized = temp_x[i] / length
        if( x_normalized < duty_cycle ):
            temp_y[i] = ( x_normalized / duty_cycle ) * height
            slope = height / ( length * duty_cycle )
            
        elif( x_normalized > duty_cycle ):
            temp_y[i] = height - ( (x_normalized-duty_cycle) / (1-duty_cycle) ) * height
            slope = - height / ( length * ( 1 - duty_cycle ) )
            
        else:
            temp_y[i] = height
            slope = height / ( 2 * length ) * ( ( 1 - 2 * duty_cycle ) / ( duty_cycle * ( 1 - duty_cycle ) ) )
        
        temp_theta[i] = np.arctan2( slope , 1.0 ) - pi/2   
            
    return temp_x , temp_y , temp_theta


### half oval ###
def half_oval_via( length , height , delta_x ):
    
    pt_num = int( length / delta_x ) + 1
    temp_x = np.linspace( 0 , length , pt_num )
    temp_y = np.zeros( pt_num )
    temp_theta = np.zeros( pt_num )
    for i in range( pt_num ):
        
        temp_y[i] = height * np.sqrt( 1.0 - ( ( 2 * temp_x[i] - length ) / length )**2 )
        if( temp_x[i] == 0. ):
            temp_theta[i] = 0
        elif( temp_x[i] == length ):
            temp_theta[i] = -pi
        else:
            slope = ( height / length ) * ( ( 2 * length - 4 * temp_x[i] ) / length ) * 1.0 / np.sqrt( 1 - ( ( 2 * temp_x[i] - length ) / length )**2 )
            temp_theta[i] = np.arctan2( slope , 1.0 ) - pi/2
            
    return temp_x , temp_y , temp_theta


def rectangle_via( length , height , via_pts_num ):
    
    if( via_pts_num == 5 ):
        temp_x = np.array( [ 0. , 0. , length/2 , length , length ] ) # [mm]
        temp_y = np.array( [ 0. , height , height , height , 0. ] ) # [mm]
        temp_theta = np.array( [ 0. , -pi/4 , -pi/2 , -3*pi/4 , -pi ] ) # [rad]
            
    return temp_x , temp_y , temp_theta


### cubic polynomial trajectory generator ###
def CubicTrajetoryPlanning( via_x , via_y , via_theta , v_avg , dt ):
    
    traj_x = np.array( [ via_x[0] ] )
    traj_y = np.array( [ via_y[0] ] )
    traj_theta = np.array( [ via_theta[0] ] )
    
    # Construct t_duration , via_v and via_omega array
    t_duration , via_v_x , via_v_y , via_v_theta = np.zeros( ( 4 , len(via_x)-1 ) )

    for k in range( len(t_duration) ):
        t_duration[k] = np.sqrt( (via_x[k+1] - via_x[k])**2 + (via_y[k+1] - via_y[k])**2 ) / v_avg
        via_v_x[k] = ( via_x[k+1] - via_x[k] ) / t_duration[k]
        via_v_y[k] = ( via_y[k+1] - via_y[k] ) / t_duration[k]
        via_v_theta[k] = ( via_theta[k+1] - via_theta[k] ) / t_duration[k]
    via_v_x[-1] = 0.
    via_v_y[-1] = 0.
    via_v_theta[-1] = 0.
    via_v_x = np.append( 0. , via_v_x )
    via_v_y = np.append( 0. , via_v_y )
    via_v_theta = np.append( 0. , via_v_theta )
    
    # Calculate cubic pynomial coefficients and append trajectory points to trajectory array
    for k in range( len(via_x)-1 ):
        
        # Calculate cubic pynomial coefficients
        delta_t = t_duration[k]
        
        # A * temp1 = temp2 <=> temp1 = A^-1 * temp2 
#        A = np.array([[ 1. , 0. , 0. , 0. ],
#                      [ 0. , 1. , 0. , 0. ],
#                      [ 1. , delta_t , delta_t**2 , delta_t**3 ],
#                      [ 0. , 1 , 2*delta_t , 3*(delta_t**2) ]])
        invA = np.array([[ 1. , 0. , 0. , 0. ],
                         [ 0. , 1. , 0. , 0. ],
                         [ -3/delta_t**2 , -2/delta_t , 3/delta_t**2 , -1/delta_t ],
                         [ 2/delta_t**3 , 1/delta_t**2 , -2/delta_t**3 , 1/delta_t**2 ]])
        
        temp2 = np.array([ [ via_x[k] ], [ via_v_x[k] ] , [ via_x[k+1] ] , [ via_v_x[k+1] ] ])
        temp1 = np.dot( invA , temp2 )        
        # append trajectory points to trajectory array
        for i in range( 1 , int(delta_t/dt)+1 ):
            traj_x = np.append( traj_x , temp1[0] + temp1[1] * (i*dt) + temp1[2] * (i*dt)**2 + temp1[3] * (i*dt)**3 )
        
        temp2 = np.array([ [ via_y[k] ], [ via_v_y[k] ] , [ via_y[k+1] ] , [ via_v_y[k+1] ] ])
        temp1 = np.dot( invA , temp2 )        
        # append trajectory points to trajectory array
        for i in range( 1 , int(delta_t/dt)+1 ):
            traj_y = np.append( traj_y , temp1[0] + temp1[1] * (i*dt) + temp1[2] * (i*dt)**2 + temp1[3] * (i*dt)**3 )
        
        temp2 = np.array([ [ via_theta[k] ], [ via_v_theta[k] ] , [ via_theta[k+1] ] , [ via_v_theta[k+1] ] ])
        temp1 = np.dot( invA , temp2 )        
        # append trajectory points to trajectory array
        for i in range( 1 , int(delta_t/dt)+1 ):
            traj_theta = np.append( traj_theta , temp1[0] + temp1[1] * (i*dt) + temp1[2] * (i*dt)**2 + temp1[3] * (i*dt)**3 )
        
    return traj_x , traj_y , traj_theta

### cubic polynomial trajectory generator with multiple theta interpolated among position points ###
def CubicTrajetoryPlanning_v2( via_x , via_y , via_theta , v_avg , dt ):
    
    # via_theta equally interpolated via_x and via_y
    division_num = int( ( len(via_theta) - 1 ) / ( len(via_x) - 1 ) )
    traj_x = np.array( [] )
    traj_y = np.array( [] )
    traj_theta = np.array( [] )
    
    # Construct t_duration , via_v and via_omega array
    t_duration , via_v_x , via_v_y = np.zeros( ( 3 , len(via_x)-1 ) )
    via_v_theta = np.zeros( len(via_theta)-1 )
    
    for k in range( len(t_duration) ): # k = 0, 1, 2
        t_duration[k] = np.sqrt( (via_x[k+1] - via_x[k])**2 + (via_y[k+1] - via_y[k])**2 ) / v_avg[k]
        via_v_x[k] = ( via_x[k+1] - via_x[k] ) / t_duration[k]
        via_v_y[k] = ( via_y[k+1] - via_y[k] ) / t_duration[k]
        
        for j in range( division_num ): # (0 1 2) (3 4 5) (6 7 8)
            k_ = division_num * k + j
            via_v_theta[k_] = ( via_theta[k_+1] - via_theta[k_] ) / (t_duration[k] / division_num)
    
    via_v_x[-1] = 0.
    via_v_y[-1] = 0.
    via_v_theta[-1] = 0.
    via_v_x = np.append( 0. , via_v_x )
    via_v_y = np.append( 0. , via_v_y )
    via_v_theta = np.append( 0. , via_v_theta )
    
    # Calculate cubic pynomial coefficients and append trajectory points to trajectory array
    for k in range( len(via_x)-1 ):
        
        # Calculate cubic pynomial coefficients
        delta_t = t_duration[k]
        
        # A * temp1 = temp2 <=> temp1 = A^-1 * temp2 
#        A = np.array([[ 1. , 0. , 0. , 0. ],
#                      [ 0. , 1. , 0. , 0. ],
#                      [ 1. , delta_t , delta_t**2 , delta_t**3 ],
#                      [ 0. , 1 , 2*delta_t , 3*(delta_t**2) ]])
        invA = np.array([[ 1. , 0. , 0. , 0. ],
                         [ 0. , 1. , 0. , 0. ],
                         [ -3/delta_t**2 , -2/delta_t , 3/delta_t**2 , -1/delta_t ],
                         [ 2/delta_t**3 , 1/delta_t**2 , -2/delta_t**3 , 1/delta_t**2 ]])
        
        temp2 = np.array([ [ via_x[k] ], [ via_v_x[k] ] , [ via_x[k+1] ] , [ via_v_x[k+1] ] ])
        temp1 = np.dot( invA , temp2 )        
        # append trajectory points to trajectory array
        for i in range( 0 , int(delta_t/dt) ):
            traj_x = np.append( traj_x , temp1[0] + temp1[1] * (i*dt) + temp1[2] * (i*dt)**2 + temp1[3] * (i*dt)**3 )
        
        temp2 = np.array([ [ via_y[k] ], [ via_v_y[k] ] , [ via_y[k+1] ] , [ via_v_y[k+1] ] ])
        temp1 = np.dot( invA , temp2 )        
        # append trajectory points to trajectory array
        for i in range( 0 , int(delta_t/dt) ):
            traj_y = np.append( traj_y , temp1[0] + temp1[1] * (i*dt) + temp1[2] * (i*dt)**2 + temp1[3] * (i*dt)**3 )
        
        for j in range( division_num ):
            
            delta_t = t_duration[k] / division_num
            invA = np.array([[ 1. , 0. , 0. , 0. ],
                             [ 0. , 1. , 0. , 0. ],
                             [ -3/delta_t**2 , -2/delta_t , 3/delta_t**2 , -1/delta_t ],
                             [ 2/delta_t**3 , 1/delta_t**2 , -2/delta_t**3 , 1/delta_t**2 ]])
            k_ = division_num * k + j
            temp2 = np.array([ [ via_theta[k_] ], [ via_v_theta[k_] ] , [ via_theta[k_+1] ] , [ via_v_theta[k_+1] ] ])
            temp1 = np.dot( invA , temp2 )      
            # append trajectory points to trajectory array
            for i in range( 0 , int(delta_t/dt) ):
                traj_theta = np.append( traj_theta , temp1[0] + temp1[1] * (i*dt) + temp1[2] * (i*dt)**2 + temp1[3] * (i*dt)**3 )
        
        ## fill up short trajectories
        if( len(traj_theta) < len(traj_x) ):
            for _ in range( len(traj_x) - len(traj_theta) ):
                traj_theta = np.append( traj_theta, traj_theta[-1] )

        elif( len(traj_theta) > len(traj_x) ):
            for _ in range( len(traj_theta) - len(traj_x) ):
                traj_x = np.append( traj_x, traj_x[-1] )
                traj_y = np.append( traj_y, traj_y[-1] )
            
    return traj_x , traj_y , traj_theta, t_duration


## Given via points which are extreme points, output cubic polynomial trajectory ##
def cbc_traj_extrm_pts( via_t , via_x , v_0, dt, itr_num ):
    
    traj_x = np.array( [], dtype=float )
    
    for k in range( len(via_x)-1 ):
        
        # Calculate cubic pynomial coefficients
        delta_t = via_t[k+1] - via_t[k]
        
        if( delta_t >= 1e-10 ):
            
            if( k == 0 ):            
                invA = np.array([[ 1. , 0. , 0. , 0. ],
                                 [ 0. , 1. , 0. , 0. ],
                                 [ -3/delta_t**2 , -2/delta_t , 3/delta_t**2 , -1/delta_t ],
                                 [ 2/delta_t**3 , 1/delta_t**2 , -2/delta_t**3 , 1/delta_t**2 ]])
                temp2 = np.array([ [ via_x[k] ], [ v_0 ] , [ via_x[k+1] ] , [ 0 ] ])
                temp1 = np.dot( invA , temp2 )  
            
            else:
                temp1 = [ via_x[k],
                          0,
                          3 * ( via_x[k+1] - via_x[k] ) / ( delta_t**2 ),
                          - 2 * ( via_x[k+1] - via_x[k] ) / ( delta_t**3 )
                        ]
                
            # append trajectory points to trajectory array
            for i in range( 0 , int(delta_t/dt)+1 ):
                traj_x = np.append( traj_x , temp1[0] + temp1[1] * (i*dt) + temp1[2] * (i*dt)**2 + temp1[3] * (i*dt)**3 )
        
    if( itr_num != np.nan ):
        if( len(traj_x) > itr_num ):
            traj_x = traj_x[:itr_num]
        elif( len(traj_x) < itr_num ):
            traj_x = np.append( traj_x, np.array( ( itr_num - len(traj_x) ) * [traj_x[-1]] ) )
    
    return traj_x
 
 
# Solve the cross point of the axial line and the cosine wave, and calculuate the distance between measured and cross point
def solve_cos_wave( Amplitude , T , msr_x , msr_y , msr_theta , Type ):
    
    func = lambda x : Amplitude * ( 1.0 - np.cos( ( 2 * pi / T ) * x ) ) - ( np.tan(msr_theta) * ( x - msr_x ) + msr_y )
    sol = optimize.root( func , msr_x )
    sol_x = sol.x
    sol_y = Amplitude * ( 1.0 - np.cos( ( 2 * pi / T ) * sol_x ) )
    if( ( sol_x - msr_x ) * np.cos( msr_theta ) > 0 ):
        distance = np.sqrt( ( msr_x - sol_x )**2 + ( msr_y - sol_y )**2 )
    else:
        distance = - np.sqrt( ( msr_x - sol_x )**2 + ( msr_y - sol_y )**2 )
    
    if(Type=='desired'):
        return distance
    else: 
        return sol_x , sol_y , distance
    

# Solve the cross point of the axial line and the triangle wave, and calculuate the distance between measured and cross point
def solve_triangle_wave( height , length , msr_x , msr_y , msr_theta , duty_cycle , Type ):
    
    if( msr_theta >= -pi/2 ):
        func = lambda x : ( ( x / ( length * duty_cycle ) ) * height ) - ( np.tan( msr_theta ) * ( x - msr_x ) + msr_y )
    else:
        func = lambda x : ( height - ( ( x / length - duty_cycle ) / ( 1 - duty_cycle ) ) * height ) - ( np.tan( msr_theta ) * ( x - msr_x ) + msr_y )
    
    sol = optimize.root( func , msr_x )
    sol_x = sol.x
    sol_y = ( np.tan( msr_theta ) * ( sol_x - msr_x ) + msr_y )
    if( ( sol_x - msr_x ) * np.cos( msr_theta ) > 0 ):
        distance = np.sqrt( ( msr_x - sol_x )**2 + ( msr_y - sol_y )**2 )
    else:
        distance = - np.sqrt( ( msr_x - sol_x )**2 + ( msr_y - sol_y )**2 )
    
    if(Type=='desired'):
        return distance
    else: 
        return sol_x , sol_y , distance
    
    
# Solve the cross point of the axial line and the half oval, and calculuate the distance between measured and cross point
def solve_half_oval( height , length , msr_x , msr_y , msr_theta , Type ):

    in_x , in_y , out_x , out_y , middle_x , middle_y = np.zeros(6)
    LENGTH = 0.

    if( ( length / 2 ) >= height ):
        LENGTH = length / 2;
    else:
        LENGTH = height;

    if( ( ( 2 * msr_x - length ) / length )**2 + ( msr_y / height )**2 < 1.0 ):
        out_x = msr_x + LENGTH * np.cos( msr_theta + pi )
        out_y = msr_y + LENGTH * np.sin( msr_theta + pi )
        in_x = msr_x
        in_y = msr_y

    else:
        in_x = msr_x + LENGTH * np.cos( msr_theta );
        in_y = msr_y + LENGTH * np.sin( msr_theta );
        out_x = msr_x
        out_y = msr_y

    while( np.abs( in_x - out_x ) > 0.001 or np.abs( in_y - out_y ) >   0.001 ):
        middle_x = ( out_x + in_x ) / 2.0
        middle_y = ( out_y + in_y ) / 2.0
        if( ( ( 2 * middle_x - length ) / length )**2 + ( middle_y / height )**2 < 1.0 ):
            in_x = middle_x;
            in_y = middle_y;
    
        else:
            out_x = middle_x;
            out_y = middle_y;
    sol_x , sol_y = out_x , out_y
    
    if( ( sol_x - msr_x ) * np.cos( msr_theta ) > 0 ):
        distance = np.sqrt( ( msr_x - sol_x )**2 + ( msr_y - sol_y )**2 )
    else:
        distance = - np.sqrt( ( msr_x - sol_x )**2 + ( msr_y - sol_y )**2 )
    
    if(Type=='desired'):
        return distance
    else: 
        return sol_x , sol_y , distance
    
    
# Solve the cross point of the axial line and the rectangle, and calculuate the distance between measured and cross point
def solve_rectangle( height , length , msr_x , msr_y , msr_theta , Type ):

    if( msr_theta >= -pi/4 ):
        sol_x = 0.
        sol_y = ( np.tan( msr_theta ) * ( sol_x - msr_x ) + msr_y )
        
    elif( -pi/4 > msr_theta and msr_theta >= -pi*3/4 ):
        sol_y = height
        sol_x = ( ( sol_y - msr_y ) / np.tan( msr_theta ) + msr_x )
    else:
        sol_x = length
        sol_y = ( np.tan( msr_theta ) * ( sol_x - msr_x ) + msr_y )
    
    if( ( sol_x - msr_x ) * np.cos( msr_theta ) + ( sol_y - msr_y ) * np.sin( msr_theta ) > 0 ):
        distance = np.sqrt( ( msr_x - sol_x )**2 + ( msr_y - sol_y )**2 )
    else:
        distance = - np.sqrt( ( msr_x - sol_x )**2 + ( msr_y - sol_y )**2 )
    
    if(Type=='desired'):
        return distance
    else: 
        return sol_x , sol_y , distance
