clc;
clear all;
close all;

model_idx = [ 5461 , 123 , 445 , 2500 , 1111 ];

% path_idx = '1';
% path_idx = '2';
% path_idx = '3';
path_idx = '4';

t_axis = csvread( strcat( 'data/t_axis_' , path_idx , '.csv' ) );

tolerance = 2.5; % [mm]

for i = 1 : length( model_idx )
    
    x_mB_NN_set( i , : ) = csvread( strcat( 'data/x_mB_NN(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) );
    theta_B_NN_set( i , : ) = csvread( strcat( 'data/theta_B_NN(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) ) * 180 / pi;
    
end

possilbe_settle_time_set = ones( 5 , 1 );
for i = 1 : length( model_idx )
    
    temp = x_mB_NN_set( i , : );
    
    % RMSE
    RMSE( i ) = sqrt( mean( temp.^2 ) );
    
    % percentage overshoot
    step_info = stepinfo( temp , t_axis , 0 );
    if( x_mB_NN_set( i , 1 ) > 0 )
        min(  temp );
        prcnt_ovrsht( i ) = min(  temp ) /  temp(1) * 100;
    else
        max(  temp );
        prcnt_ovrsht( i ) = max(  temp ) / temp(1) * 100;
    end
    
    % settling time
%     count = 0;
    possilbe_settle_time = [];
    for j = 1 : length(t_axis) - 1
        if( ( ( temp(j) - tolerance ) * ( temp(j+1) - tolerance ) <= 0 ) || ( ( temp(j) + tolerance ) * ( temp(j+1) + tolerance ) <= 0 ) )
%             count = count + 1;
%             if( size( possilbe_settle_time_set , 2 ) < count )
%                possilbe_settle_time_set = horzcat( possilbe_settle_time_set , ones( 5 , 1 ) );
%                 possilbe_settle_time_set( i , count ) =  j+1;
%             else
%                 possilbe_settle_time_set( i , count ) = j+1;
%             end
            possilbe_settle_time = [ possilbe_settle_time , j+1 ];
        end
    end
    [ extrm_idxs, nttn ] = extrct_feat_pts( temp );
    
    % consecutive max-min or min-max pairs are within tolerance strand
    flag1 = 0;
    for j = 1 : length(extrm_idxs)-1
        if( x_mB_NN_set( i , extrm_idxs(j) ) <= tolerance && x_mB_NN_set( i , extrm_idxs(j) ) >=  - tolerance )
            if( x_mB_NN_set( i , extrm_idxs(j+1) ) <= tolerance && x_mB_NN_set( i , extrm_idxs(j+1) ) >=  - tolerance )
                i;
               t_axis( extrm_idxs(j) );
                t_axis( extrm_idxs(j+1) );
                flag1 = flag1 +1;
                if( flag1 == 2 )
                    break;
                end
                [ ~ , argmin ] = min( abs( possilbe_settle_time - extrm_idxs(j) ) );
                settle_pt1 = possilbe_settle_time( argmin );
                t_axis( settle_pt1 );
            else
                flag1 = 0;
            end
        end
    end
    
    % consecutive max-min or min-max pairs' range are within tolerance range
    flag2 = 0; 
    for j = 1 : length(extrm_idxs)-1
        if( nttn(j) * nttn(j+1) < 0 )
            if( ( nttn(j) * x_mB_NN_set( i , extrm_idxs(j) ) - 2 * tolerance ) <= ( -1 * nttn(j+1) * x_mB_NN_set( i , extrm_idxs(j+1) ) ) )
                flag2 = flag2 +1;
                if( flag2 == 2 )
                    break;
                end
                settle_pt2 = round( ( extrm_idxs(j) + extrm_idxs(j+1) ) / 2 );
                t_axis( settle_pt2 );
            else
                flag2 = 0;
            end
        end
    end
    
    figure();
    plot(t_axis,(x_mB_NN_set(i,:)))
    hold on;
    scatter(t_axis( possilbe_settle_time ) , x_mB_NN_set( i , possilbe_settle_time ) ,'filled', 'MarkerFaceColor' , 'r' );
    scatter(t_axis( extrm_idxs ) , x_mB_NN_set( i ,extrm_idxs ) ,'filled', 'MarkerFaceColor' , 'b' );
    ezplot( '2.5' , [ t_axis(1) , t_axis(end) ] );
    ezplot( '-2.5' , [ t_axis(1) , t_axis(end) ] );
    if( flag1==2 || flag2==2 )
        if( exist('settle_pt1') && ~exist('settle_pt2') )
            scatter( t_axis( settle_pt1 ) , x_mB_NN_set( i ,settle_pt1 ) , 'filled', 'MarkerFaceColor' , 'k' );
        elseif( exist('settle_pt2') && ~exist('settle_pt1') )
            scatter( t_axis( settle_pt2 ) , x_mB_NN_set( i ,settle_pt2 ) , 'filled', 'MarkerFaceColor' , 'k' );
        else
            if( settle_pt1 < settle_pt2 )
                scatter( t_axis( settle_pt1 ) , x_mB_NN_set( i ,settle_pt1 ) , 'filled', 'MarkerFaceColor' , 'k' );
            else
                scatter( t_axis( settle_pt2 ) , x_mB_NN_set( i ,settle_pt2 ) , 'filled', 'MarkerFaceColor' , 'k' );
            end
        end
        clear settle_pt;
    end
%     possilbe_settle_time_set = vertcat( possilbe_settle_time_set , possilbe_settle_time );
end

% RMSE
% prcnt_ovrsht

RMSE_mean = round( mean( RMSE ) , 2 );
RMSE_std = round( std( RMSE ) , 2 );
prcnt_ovrsht_mean = round( mean( prcnt_ovrsht ) , 1 );
prcnt_ovrsht_std = round( std( prcnt_ovrsht ) , 1 );


function [ extrm_idxs, nttn ] = extrct_feat_pts( traj )
    
    wndw_size = 7;
    tol = 0.001; % [mm]
    
    % find extreme point
    extrm_idxs = [];
    nttn = [];
    for i = wndw_size : length( traj ) - wndw_size
        
        head = i - wndw_size + 1;
        tail = i + wndw_size;
        segment = traj( head : tail );
        
        if( max( segment ) - min( segment ) >= tol ) % avoid flate segment
            [~, argmax] = max(segment);
            [~, argmin] = min(segment);
            if( ( head + argmax ) == i )
                extrm_idxs = [ extrm_idxs , i ];
                nttn = [ nttn , 1 ];
            end
            if( ( head + argmin ) == i )
                extrm_idxs = [ extrm_idxs , i ];
                nttn = [ nttn , -1 ];
            end
        end
    
    end
end


