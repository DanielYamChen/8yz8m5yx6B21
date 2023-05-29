clear all;
% close all;

model_idx = [ 5461 , 123 , 445 , 2500 , 1111 ];
% model_idx = [ 2500 ];

% path_idx = '1_obs';
path_idx = '2_obs';


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

                         
segment_num = 256;
line_style = [ "-" , "--" , ":" ];
L = 125; % [mm]
l = 40;
if( path_idx == '1_obs' )
    obs_set = 650 * [ 0.5,0.5,0.3 ]; % [mm]
elseif( path_idx == '2_obs' )
    obs_set = 650 * [ 0.18, 0.34, 0.28 ; 0.82, 0.66, 0.28 ]; % [mm]
end


for i = 1 : length( model_idx )

    x_mB_NN_set{ i } = csvread( strcat( 'data/x_mB(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) ); % [mm]
    SSE_set( i ) = sum( x_mB_NN_set{ i }.^2 ) * 0.012; % [mm^2-sec]
    theta_B_NN_set{ i } = csvread( strcat( 'data/theta_B(', num2str( model_idx( i ) ), ')_' , path_idx , '.csv' ) ) * 180 / pi; % [deg]
    best_idx_set( i ) = csvread( strcat( 'data/best_ini_cond(' , num2str( model_idx( i ) ) ,')_' , path_idx , '.csv' ) , 13 , 0 ) + 1;
    process_len_set( i , : ) = csvread( strcat( 'data/process_len(' , num2str( model_idx( i ) ) ,')_' , path_idx , '.csv' ) );
    
end

[ min_val , arg_min ] =  min( SSE_set );


for i = 1 : length( model_idx )
    
    temp = csvread( strcat( 'data/J2_process(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) ); % [mm^2-sec]
    J2_process_set( i , : ) = temp( best_idx_set( i ) , : );
    
    temp = csvread( strcat( 'data/GD_process(', num2str( model_idx(i) ) , ')_' , path_idx , '.csv' ) ); % [mm,mm]
    GD_process_set( i , : , : ) = temp( best_idx_set( i ) * 4 - 3 : best_idx_set( i ) * 4 , : );
    process_len( i ) = process_len_set( i , best_idx_set( i ) );
    
end


via_points = csvread( strcat( 'data/ini_cond_hard_' , path_idx , '.csv' ) , 0 , 0 , [ 0 , 0 , 0 , 7 ] );
via_points = reshape( via_points, [], 2 )';


%%% PART 1
% for i = 1 : size(  obs_set , 1 )
%     rectangle( 'Position' , [ obs_set(i,1) - obs_set(i,3) , obs_set(i,2) - obs_set(i,3) , 2 * obs_set(i,3),  2 * obs_set(i,3) ] , ...
%                     'Curvature' , [1,1] ,...
%                     'LineStyle' , '--' ,...
%                     'LineWidth' , 3 );
%     hold on;
% end
% for i = 1 : length( model_idx )
%     epoch_num = process_len( i );
%     idx_set = 1 : round( epoch_num / 10 ) : epoch_num;
%     
%     for j = 1 : length( idx_set ) + 1 % 1, 51, 101, 151, ..., 451, 500
%         if( j == length( idx_set ) + 1 )
%             via_points = csvread( strcat( 'data/best_ini_cond(' , num2str( model_idx( i ) ) ,')_' , path_idx , '.csv' ) , 0 , 0 , [ 0 , 0 , 7 , 0 ] )';
%             via_points = reshape( via_points, [], 2 )';
%             plot( via_points(1,:) , via_points(2,:) , 'Color', line_color( i , j , length( idx_set ) + 1 , line_color_deep , line_color_light ) , 'LineWidth', 3 );
%             hold on;
%         else    
%             via_points( 1 , 2:3 ) = GD_process_set( i , 1 : 2 , idx_set( j ) ); % x-coodinate
%             via_points( 2 , 2:3 ) = GD_process_set( i , 3 : 4 , idx_set( j ) ); % y-coodinate
%             plot( via_points(1,:) , via_points(2,:) , 'Color', line_color( i , j , length( idx_set ) + 1 , line_color_deep , line_color_light ) , 'LineWidth', 3 );
%             hold on;
%         end
%     end
% end
% 
% ini_via_pts = [ 0 , 216.67 , 433.33 , 650 ];
%  for i = 1 : 4
%     for j = 1 : 4
%         ini_via_pts_set( 1:2 , i*4-4+j ) = [ ini_via_pts(i) , ini_via_pts(j) ];
%     end
%  end
%  
% scatter( ini_via_pts_set(1,:) , ini_via_pts_set(2,:) , 52, 'filled' , 'MarkerFaceColor' , [ 0.5 , 0.5 , 0.5 ] );
%  
% axis equal;


%%% PART 2
% for i = 1 : length( model_idx )
%     epoch_num = process_len( i );;
%     plot(  J2_process_set( i , 1 : epoch_num ) , 'Color', line_color_deep(i,:), 'LineWidth' , 3 );
%     hold on;
% end


%%% PART 3
x_mB_NN = x_mB_NN_set{ arg_min };
theta_B_NN = theta_B_NN_set{ arg_min };
t_axis = 0 : 0.012 : ( length( x_mB_NN ) - 1 ) * 0.012;

% hold on;
% plot( t_axis , x_mB_NN , 'Color', 'b', 'LineWidth' , 3 );
% plot( t_axis , theta_B_NN , 'Color', 'b', 'LineWidth' , 3 );

% xlabel({'time (sec)'});
% ylabel({'\x_{mB} (mm)'});
% ylabel({'\theta_{B} (mm)'});

% legend({ 'x_{mB,NN,std}' , 'x_{mB,NN,mean}' });
% legend({ '\theta_{B,NN,std}' , '\theta_{B,NN,mean}' });

% set(gca,'FontSize',40);


%%%PART 4
via_points = csvread( strcat( 'data/best_ini_cond(' , num2str( model_idx( arg_min ) ) ,')_' , path_idx , '.csv' ) , 0 , 0 , [ 0 , 0 , 7 , 0 ] )';
via_points = reshape( via_points, [], 2 )';
x_Bx = via_points( 1 , : );
x_By = via_points( 2 , : );

for i = 1 : size(  obs_set , 1 )
    rectangle( 'Position' , [ obs_set(i,1) - obs_set(i,3) , obs_set(i,2) - obs_set(i,3) , 2 * obs_set(i,3),  2 * obs_set(i,3) ] , ...
                    'Curvature' , [1,1] ,...
                    'LineStyle' , '--' ,...
                    'LineWidth' , 3 );
    hold on;
end

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

