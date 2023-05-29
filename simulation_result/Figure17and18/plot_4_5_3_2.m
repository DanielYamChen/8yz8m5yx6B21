clear all;
% close all;

model_idx = [ 5461 , 123 , 445 , 2500 , 1111 ];

path_idx = '1__';


line_color = [ 0 , 0 , 0 ; 1 , 0 , 0 ; 0 , 1 , 0 ; 0 , 0 , 1 ; 0.7 , 0.7 , 0.7 ];
segment_num = 256;
line_style = [ "-" , "--" , ":" ];
L = 125; % [mm]
l = 40;


for i = 1 : length( model_idx )
    temp = csvread( strcat( 'data/J2_process(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) ); %[mm^2-sec]
    for j = 1 : segment_num
        J2_process_set( j , i , : ) = temp( j , : );
    end
    
    x_mB_NN_set{ i } = csvread( strcat( 'data/x_mB(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) ); % [mm]
    SSE_set( i ) = sum( x_mB_NN_set{ i }.^2 ) * 0.012; % [mm^2-sec]
    theta_B_NN_set{ i } = csvread( strcat( 'data/theta_B(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) ) * 180 / pi; % [deg]
    
end

[ min_val , arg_min ] =  min( SSE_set );

epoch_num = size( temp , 2 );
idx_set = 1 : epoch_num / 5 : epoch_num;
idx_set = [ idx_set , epoch_num ];

for i = 1 : segment_num
    temp = csvread( strcat( 'data/GD_process(', num2str( model_idx( arg_min ) ) , ')_' , path_idx , '.csv' ) ); %[mm,mm]
    GD_process_set( i , : , : ) = temp( i * 4 - 3 : i * 4 , : );
end

via_points = csvread( strcat( 'data/ini_cond_hard_1_.csv' ) , 0 , 0 , [ 0 , 0 , 0 , 7 ] );
via_points = reshape( via_points, [], 2 )';


%%% PART 1
% for i = 1 : length( idx_set ) % 1, 51, 101, 151, ..., 451, 501
%     for j = 1 : segment_num
%         via_points( 1 , 2:3 ) = GD_process_set( j , 1:2 , idx_set( i ) );
%         via_points( 2 , 2:3 ) = GD_process_set( j , 3:4 , idx_set( i ) );
%         plot( via_points(1,:) , via_points(2,:) , 'Color', ( 0.8 - 0.8 / ( length( idx_set ) ) * i ) * [1,1,1] , 'LineWidth', 3 );
%         hold on;
%     end
% end
% 
%  for j = 1 : segment_num
% %         via_points( 1 , 2:3 ) = GD_process_set( j , : , 500 );
% %         plot( via_points(1,:) , via_points(2,:) , 'Color', [0,0,1] , 'LineWidth', 3 );
%         
%         via_points( 1 , 2:3 ) = GD_process_set( j , 1:2 , 1 );
%         via_points( 2 , 2:3 ) = GD_process_set( j , 3:4 , 1 );
%         scatter( via_points(1,2:3) , via_points(2,2:3) , 52, 'filled' );
%  end
%  
% axis equal;


%%% PART 2
% for i = 1 : length( model_idx )
%     for j = 1 : segment_num
%         plot( reshape( J2_process_set( j , i , : ) , [ 1 , epoch_num ] ) , 'Color', line_color(i,:), 'LineWidth' , 3 );
%         hold on;
%     end
% end

x_mB_NN = x_mB_NN_set{ arg_min };
theta_B_NN = theta_B_NN_set{ arg_min };
t_axis = 0 : 0.012 : ( length( x_mB_NN ) - 1 ) * 0.012;

%%% PART 3
% fill( [ t_axis ; flipud( t_axis ) ] , [ ( x_mB_NN_mean - x_mB_NN_std ) ; flipud( x_mB_NN_mean + x_mB_NN_std ) ],[0 , 0.75 , 0.75 ] , 'linestyle' , 'none' );
% fill( [ t_axis ; flipud( t_axis ) ] , [ ( theta_B_NN_mean - theta_B_NN_std ) ; flipud( theta_B_NN_mean + theta_B_NN_std ) ],[0 , 0.75 , 0.75 ] , 'linestyle' , 'none' );

% hold on;
plot( t_axis , x_mB_NN , 'Color', 'b', 'LineWidth' , 3 );
% plot( t_axis , theta_B_NN , 'Color', 'b', 'LineWidth' , 3 );

% xlabel({'time (sec)'});
% ylabel({'\x_{mB} (mm)'});
% ylabel({'\theta_{B} (mm)'});

% legend({ 'x_{mB,NN,std}' , 'x_{mB,NN,mean}' });
% legend({ '\theta_{B,NN,std}' , '\theta_{B,NN,mean}' });

% set(gca,'FontSize',40);


%%%PART 4
% via_points = csvread( strcat( 'data/best_ini_cond(' , num2str( model_idx( arg_min ) ) ,')_1__.csv' ) , 0 , 0 , [ 0 , 0 , 7 , 0 ] )';
% via_points = reshape( via_points, [], 2 )';
% x_Bx = via_points( 1 , : );
% x_By = via_points( 2 , : );
% plot( x_Bx , x_By , 'Color', 'b', 'LineWidth' , 3 );
% hold on;
% plot( [ x_Bx(1) + L * cosd( theta_B_NN(1) ) , x_Bx(1) - L * cosd( theta_B_NN(1) ) ], ...
%        [ x_By(1) + L * sind( theta_B_NN(1) ) , x_By(1) - L * sind( theta_B_NN(1) ) ], 'r', 'LineWidth', 3 );
% draw_rectangle( [ x_Bx(1) + x_mB_NN(1) * cosd( theta_B_NN(1) ) - l/2 * sind( theta_B_NN(1) ), ...
%                            x_By(1) + x_mB_NN(1) * sind( theta_B_NN(1) ) + l/2 * cosd( theta_B_NN(1) ) ], l, l, theta_B_NN(1), [0,1,0] );
% scatter( x_Bx , x_By , 72, 'filled', 'MarkerFaceColor' , 'k' );
% axis equal;



function[]= draw_rectangle(center_location,L,H,theta,rgb)
    
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


