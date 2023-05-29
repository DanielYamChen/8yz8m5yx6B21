clear all;
% close all;

model_idx = [ 5461 , 123 , 445 , 2500 , 1111 ];

% path_idx = '2';
path_idx = '3';

t_axis = csvread( strcat( 'data/t_axis_' , path_idx , '.csv' ) );
x_mB_iLQR = csvread( strcat( 'data/x_mB_iLQR_' , path_idx , '.csv' ) );
theta_B_iLQR = csvread( strcat( 'data/theta_B_iLQR_' , path_idx , '.csv' ) ) * 180 / pi;

for i = 1 : length( model_idx )
    
    x_mB_NN_set( i , : ) = csvread( strcat( 'data/x_mB_NN(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) );
    theta_B_NN_set( i , : ) = csvread( strcat( 'data/theta_B_NN(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) ) * 180 / pi;
    
end

x_mB_NN_mean = mean( x_mB_NN_set )';
x_mB_NN_std = std( x_mB_NN_set )';
theta_B_NN_mean = mean( theta_B_NN_set )';
theta_B_NN_std = std( theta_B_NN_set )';


% fill( [ t_axis ; flipud( t_axis ) ] , [ ( x_mB_NN_mean - x_mB_NN_std ) ; flipud( x_mB_NN_mean + x_mB_NN_std ) ],[0 , 0.75 , 0.75 ] , 'linestyle' , 'none' );
fill( [ t_axis ; flipud( t_axis ) ] , [ ( theta_B_NN_mean - theta_B_NN_std ) ; flipud( theta_B_NN_mean + theta_B_NN_std ) ],[0 , 0.75 , 0.75 ] , 'linestyle' , 'none' );

hold on;
% plot( t_axis , x_mB_NN_mean , 'Color', 'b', 'LineWidth' , 3 );
plot( t_axis , theta_B_NN_mean , 'Color', 'b', 'LineWidth' , 3 );

% plot( t_axis , x_mB_iLQR ,':k' , 'LineWidth' , 3 );
plot( t_axis , theta_B_iLQR ,':k' , 'LineWidth' , 3 );

% xlabel({'time (sec)'});
% ylabel({'\theta_{B} (mm)'});
% set(gca,'FontSize',40);
