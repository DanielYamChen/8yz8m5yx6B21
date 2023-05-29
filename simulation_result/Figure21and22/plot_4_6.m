%{
clear all;
% close all;

model_idx = [ 5461 , 123 , 445 , 2500 , 1111 ];
% model_idx = [ 2500 ];

path_idx = 'worst';

line_color_deep = [    0 ,    0 ,    0 ;
                                 1 ,    0 ,    0 ;
                                 0 ,    0 ,    1 ;
                                 1 ,    0 , 0.5 ;
                                 0 ,    1 ,    0 ]; % [ black , red , blue , pink , green ]

line_color_light = [ 0.8 , 0.8 , 0.8 ;
                             1.0 , 0.8 , 0.8 ;
                             0.8 , 0.8 , 1.0 ;
                             1.0 , 0.8 , 0.9 ;
                             0.8 , 1.0 , 0.8 ];

                         
segment_num = 8192;
line_style = [ "-" , "--" , ":" ];
L = 125; % [mm]
l = 40;
% if( path_idx == '1_obs' )
%     obs_set = 650 * [ 0.5,0.5,0.3 ]; % [mm]
% elseif( path_idx == '2_obs' )
%     obs_set = 650 * [ 0.18, 0.34, 0.28 ; 0.82, 0.66, 0.28 ]; % [mm]
% end


for i = 1 : length( model_idx )
    
    x_mB_iLQR_set{ i } = csvread( strcat( 'data/x_mB_iLQR_' , path_idx , '_' , num2str( i ) , '.csv' ) ); % [mm]
    x_mB_NN_set{ i } = csvread( strcat( 'data/x_mB(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) ); % [mm]
    SSE_set( i ) = sum( x_mB_NN_set{ i }.^2 ) * 0.012; % [mm^2-sec]
    x_mB_NN_hat_set{ i } = csvread( strcat( 'data/x_mB_hat(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) ); % [mm]
    theta_B_iLQR_set{ i } = csvread( strcat( 'data/theta_B_iLQR_' , path_idx , '_' , num2str( i ) , '.csv' ) ) * 180 / pi; % [mm]
    theta_B_NN_set{ i } = csvread( strcat( 'data/theta_B(', num2str( model_idx( i ) ), ')_' , path_idx , '.csv' ) ) * 180 / pi; % [deg]
    theta_B_NN_hat_set{ i } = csvread( strcat( 'data/theta_B_hat(', num2str( model_idx( i ) ), ')_' , path_idx , '.csv' ) ) * 180 / pi; % [deg]
    best_idx_set( i ) = csvread( strcat( 'data/best_ini_cond(' , num2str( model_idx( i ) ) ,')_' , path_idx , '.csv' ) , 13 , 0 ) + 1;
    process_len_set( i , : ) = csvread( strcat( 'data/process_len(' , num2str( model_idx( i ) ) ,')_' , path_idx , '.csv' ) );
    
end

for i = 1 : length( model_idx )
    
    temp = csvread( strcat( 'data/J2_process(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) ); % [mm^2-sec]
    J2_process_set( i , : ) = temp( best_idx_set( i ) , : );
    
    process_len( i ) = process_len_set( i , best_idx_set( i ) );
    
end


% via_points = csvread( strcat( 'data/ini_cond_hard_' , path_idx , '.csv' ) , 0 , 0 , [ 0 , 0 , 0 , 7 ] );
% via_points = reshape( via_points, [], 2 )';


%%% PART 2
% for i = 1 : length( model_idx )
%     epoch_num = process_len( i );
%     plot(  J2_process_set( i , 1 : epoch_num ) , 'Color', line_color_deep(i,:), 'LineWidth' , 3 );
%     hold on;
% end
%}

%%% PART 3
arg_min = 5;
x_mB_NN = x_mB_NN_set{ arg_min };
x_mB_NN_hat = x_mB_NN_hat_set{ arg_min };
x_mB_iLQR = x_mB_iLQR_set{ arg_min };

theta_B_NN = theta_B_NN_set{ arg_min };
theta_B_NN_hat = theta_B_NN_hat_set{ arg_min };
theta_B_iLQR = theta_B_iLQR_set{ arg_min };

if( length( theta_B_iLQR ) > length( theta_B_NN ) )
    theta_B_iLQR = theta_B_iLQR( 1 : length( theta_B_NN ) );
    x_mB_iLQR = x_mB_iLQR( 1 : length( x_mB_NN ) );
else
    theta_B_NN = theta_B_NN( 1 : length( theta_B_iLQR ) );
    theta_B_NN_hat = theta_B_NN_hat( 1 : length( theta_B_iLQR ) );
    x_mB_NN = x_mB_NN( 1 : length( x_mB_iLQR ) );
    x_mB_NN_hat = x_mB_NN_hat( 1 : length( x_mB_iLQR ) );
end
    
t_axis = 0 : 0.012 : ( length( x_mB_NN ) - 1 ) * 0.012;

hold on;

% plot( t_axis , x_mB_iLQR , 'Color', 'r', 'LineWidth' , 3 );
% plot( t_axis , x_mB_NN , 'Color', 'b', 'LineWidth' , 3 );
% plot( t_axis , x_mB_NN_hat , ':k' , 'LineWidth' , 4 );
% legend( { '$x_{mB,iLQR}$' , '$x_{mB,NN}$' , '$\hat{x}_{mB,NN}$'} , 'Interpreter' , 'latex' );

% plot( t_axis , theta_B_iLQR , 'Color', 'r', 'LineWidth' , 3 );
% plot( t_axis , theta_B_NN , 'Color', 'b', 'LineWidth' , 3 );
% plot( t_axis , theta_B_NN_hat , ':k' , 'LineWidth' , 4 );
% legend( { '$\theta_{B,iLQR}$' , '$\theta_{B,NN}$' , '$\hat{\theta}_{B,NN}$'} , 'Interpreter' , 'latex' );

% xlabel({'time (sec)'});
% ylabel({'\x_{mB} (mm)'});
% ylabel({'\theta_{B} (mm)'});

% set(gca,'FontSize',40);


%%PART 4
via_points = csvread( strcat( 'data/best_ini_cond(' , num2str( model_idx( arg_min ) ) ,')_' , path_idx , '.csv' ) , 0 , 0 , [ 0 , 0 , 7 , 0 ] )';
via_points = reshape( via_points, [], 2 )';
x_Bx = via_points( 1 , : );
x_By = via_points( 2 , : );

plot( x_Bx , x_By , 'Color', 'b', 'LineWidth' , 3 );
hold on;
plot( [ x_Bx(1) + L * cosd( theta_B_NN(1) ) , x_Bx(1) - L * cosd( theta_B_NN(1) ) ], ...
       [ x_By(1) + L * sind( theta_B_NN(1) ) , x_By(1) - L * sind( theta_B_NN(1) ) ], 'r', 'LineWidth', 3 );
draw_rectangle( [ x_Bx(1) + x_mB_NN(1) * cosd( theta_B_NN(1) ) - l/2 * sind( theta_B_NN(1) ), ...
                           x_By(1) + x_mB_NN(1) * sind( theta_B_NN(1) ) + l/2 * cosd( theta_B_NN(1) ) ], l, l, theta_B_NN(1), [0,1,0] );
scatter( x_Bx , x_By , 100 , 'filled' , 'MarkerFaceColor' , 'k' );
axis equal;


function [] = draw_rectangle(center_location,L,H,theta,rgb)
    
    center1=center_location(1);
    center2=center_location(2);
    R= ([cosd(theta), -sind(theta); sind(theta), cosd(theta)]);
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


function [ color ] = line_color( i , j , tot , line_color_deep , line_color_light )
    
    color = ( line_color_deep( i , : ) - line_color_light( i , : ) ) / tot * j + line_color_light( i , : );
    color(color>1) = 1;
    color(color<0) = 0;
    
end

