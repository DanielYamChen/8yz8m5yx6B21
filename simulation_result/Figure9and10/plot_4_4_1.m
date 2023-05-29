clear all;
% close all;

model_idx = [ 5461 , 123 , 445 , 2500 , 1111 ];
dt = 0.012; % [sec]

% path_idx = '1';
% path_idx = '2';
% path_idx = '3';
path_idx = '4';

t_axis = csvread( strcat( 'data/t_axis_' , path_idx , '.csv' ) );
x_mB_iLQR = csvread( strcat( 'data/x_mB_iLQR_' , path_idx , '.csv' ) );
theta_B_iLQR = csvread( strcat( 'data/theta_B_iLQR_' , path_idx , '.csv' ) ) * 180 / pi;

% Load Data
for i = 1 : length( model_idx )
    
    x_mB_NN_set( i , : ) = csvread( strcat( 'data/x_mB_NN(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) );
    SSE_set( i ) = sum( x_mB_NN_set( i , : ).^2 ) * dt;
    theta_B_NN_set( i , : ) = csvread( strcat( 'data/theta_B_NN(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) ) * 180 / pi;
    
end

% Find the argmin with the minimum J2 of x_mB_NN_hat
[ ~ , argmin ] = min( SSE_set );
x_mB_NN = x_mB_NN_set( argmin, : )';
theta_B_NN = theta_B_NN_set( argmin, : )';

% fill( [ t_axis ; flipud( t_axis ) ] , [ ( x_mB_NN_mean - x_mB_NN_std ) ; flipud( x_mB_NN_mean + x_mB_NN_std ) ],[0 , 0.75 , 0.75 ] , 'linestyle' , 'none' );
% fill( [ t_axis ; flipud( t_axis ) ] , [ ( theta_B_NN_mean - theta_B_NN_std ) ; flipud( theta_B_NN_mean + theta_B_NN_std ) ],[0 , 0.75 , 0.75 ] , 'linestyle' , 'none' );

hold on;
% plot( t_axis , x_mB_NN , 'Color', 'b', 'LineWidth' , 3 );
% plot( t_axis , theta_B_NN , 'Color', 'b', 'LineWidth' , 3 );

% plot( t_axis , x_mB_iLQR ,':k' , 'LineWidth' , 4 );
% plot( t_axis , theta_B_iLQR ,':k' , 'LineWidth' , 4 );

% xlabel({'time (sec)'});
% ylabel({'\x_{mB} (mm)'});
% ylabel({'\theta_{B} (mm)'});

% legend({ 'x_{mB,NN}' , 'x_{mB,iLQR}' });
% legend({ '\theta_{B,NN}' , '\theta_{B,iLQR}' });

% set(gca,'FontSize',40);
