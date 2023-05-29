clear all;
% close all;
idx = '_1';
% idx = '_2';
% idx = '_3';
% idx = '_4';

t_axis = csvread( strcat( '/data/t_axis' , idx , '.csv' ) );
x_Bx = csvread( strcat( '/data/x_Bx' , idx , '.csv' ) );
x_By = csvread( strcat( '/data/x_By' , idx , '.csv' ) );
theta_B_PID = csvread( strcat( '/data/theta_B_PID' , idx , '.csv' ) );
theta_B_iLQR = csvread( strcat( 'data/theta_B_iLQR' , idx , '.csv' ) );
x_mB_PID = csvread( strcat( '/data/x_mB_PID' , idx , '.csv' ) );
x_mB_iLQR = csvread( strcat( '/data/x_mB_iLQR' , idx , '.csv' ) );

L = 125; % [mm]
l = 40;
tolerance = 2.5; % [mm]

possilbe_settle_time_iLQR = [];
possilbe_settle_time_PID = [];
for i = 1 : length(t_axis) - 1
    if( ( ( x_mB_iLQR(i) - tolerance ) * ( x_mB_iLQR(i+1) - tolerance ) <= 0 ) || ( ( x_mB_iLQR(i) + tolerance ) * ( x_mB_iLQR(i+1) + tolerance ) <= 0 ) )
        possilbe_settle_time_iLQR = [ possilbe_settle_time_iLQR , i+1 ];
    end
    if( ( ( x_mB_PID(i) - tolerance ) * ( x_mB_PID(i+1) - tolerance ) <= 0 ) || ( ( x_mB_PID(i) + tolerance ) * ( x_mB_PID(i+1) + tolerance ) <= 0 ) )
        possilbe_settle_time_PID = [ possilbe_settle_time_PID , i+1 ];
    end
end

hold on;
% plot( t_axis, theta_B_iLQR*180/pi, 'Color', 'b', 'LineWidth', 3 );
% plot( t_axis, theta_B_PID*180/pi,  'Color', [ 0 , 0.68 , 0.31 ], 'LineWidth', 3 );
% 
% legend('£c_{B,PID}','£c_{B,iLQR}');

plot( t_axis, x_mB_iLQR, 'Color', 'b', 'LineWidth', 3 );
plot( t_axis, x_mB_PID, 'Color', [ 0 , 0.68 , 0.31 ], 'LineWidth', 3 );

legend('x_{mB,iLQR}','x_{mB,PID}');

% plot( x_Bx, x_By, 'Color', 'b', 'LineWidth', 3 );
% plot( [ x_Bx(1) + L * cos( theta_B_PID(1) ) , x_Bx(1) - L * cos( theta_B_PID(1) ) ], ...
%        [ x_By(1) + L * sin( theta_B_PID(1) ) , x_By(1) - L * sin( theta_B_PID(1) ) ], 'r', 'LineWidth', 3 );
% draw_rectangle( [ x_Bx(1) + x_mB_PID(1) * cos( theta_B_PID(1) ) - l/2 * sin( theta_B_PID(1) ), ...
%                            x_By(1) + x_mB_PID(1) * sin( theta_B_PID(1) ) + l/2 * cos( theta_B_PID(1) ) ], l, l, theta_B_PID(1), [0,1,0] );
% axis equal;

function[]= draw_rectangle(center_location,L,H,theta,rgb)
    
    center1=center_location(1);
    center2=center_location(2);
    R= ([cos(theta), -sin(theta); sin(theta), cos(theta)]);
    X=([-L/2, L/2, L/2, -L/2]);
    Y=([-H/2, -H/2, H/2, H/2]);
    
    for i=1:4
        T(:,i)=R*[X(i); Y(i)];
    end
    
    x_lower_left=center1+T(1,1);
    x_lower_right=center1+T(1,2);
    x_upper_right=center1+T(1,3);
    x_upper_left=center1+T(1,4);
    y_lower_left=center2+T(2,1);
    y_lower_right=center2+T(2,2);
    y_upper_right=center2+T(2,3);
    y_upper_left=center2+T(2,4);
    x_coor=[x_lower_left x_lower_right x_upper_right x_upper_left];
    y_coor=[y_lower_left y_lower_right y_upper_right y_upper_left];

    fill(x_coor, y_coor,rgb);
    
end
