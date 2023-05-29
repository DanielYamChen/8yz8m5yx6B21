clear all;
% close all;

model_idx = [ 5461 , 123 , 445 , 2500 , 1111 ];

% path_idx = '1';
path_idx = '2';
% path_idx = '3';
% path_idx = '4';

t_axis = csvread( strcat( 'data/t_axis_' , path_idx , '.csv' ) );
line_color = [ 0 , 0 , 0 ; 1 , 0 , 0 ; 0 , 1 , 0 ; 0 , 0 , 1 ; 0.7 , 0.7 , 0.7 ];
segment_num = 3;
line_style = [ "-" , "--" , ":" ];

for i = 1 : length( model_idx )
    for j = 1 : segment_num
        temp = csvread( strcat( 'data/GD_process(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) ) * 180 / pi;
        GD_process_set( j , i , : ) = temp( j , : );
        temp = csvread( strcat( 'data/J2_process(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) );
        J2_process_set( j , i , : ) = temp( j , : );
    end
    
    x_mB_NN_set( i , : ) = csvread( strcat( 'data/x_mB(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) );
    theta_B_NN_set( i , : ) = csvread( strcat( 'data/theta_B(', num2str( model_idx(i) ), ')_' , path_idx , '.csv' ) ) * 180 / pi;
    
end

epoch_num = size( temp , 2 );
idx_axis = 1 : 1 : epoch_num;

% for i = 1 : length( model_idx )
%     for j = 1 : 3
%         temp = reshape( GD_process_set(j,i,:), [1,500] );
%         plot( temp , line_style(j) , 'Color', line_color(i,:), 'LineWidth' , 3 );
%         hold on;
% %         plot( idx_axis(1:25:end) , temp(1:25:end) , marker_style(i) , 'MarkerEdgeColor', I(j,:) , 'MarkerSize',12 , 'MarkerFaceColor', I(j,:) );
%     end
% end


% for i = 1 : length( model_idx )
%     for j = 1 : 3
%         plot( reshape( J2_process_set( j , i , : ) , [ 1 , epoch_num ] ) , line_style(j) , 'Color', line_color(i,:), 'LineWidth' , 3 );
%         hold on;
%     end
% end


x_mB_NN_mean = mean( x_mB_NN_set )';
x_mB_NN_std = std( x_mB_NN_set )';
theta_B_NN_mean = mean( theta_B_NN_set )';
theta_B_NN_std = std( theta_B_NN_set )';


% fill( [ t_axis ; flipud( t_axis ) ] , [ ( x_mB_NN_mean - x_mB_NN_std ) ; flipud( x_mB_NN_mean + x_mB_NN_std ) ],[0 , 0.75 , 0.75 ] , 'linestyle' , 'none' );
fill( [ t_axis ; flipud( t_axis ) ] , [ ( theta_B_NN_mean - theta_B_NN_std ) ; flipud( theta_B_NN_mean + theta_B_NN_std ) ],[0 , 0.75 , 0.75 ] , 'linestyle' , 'none' );

hold on;
% plot( t_axis , x_mB_NN_mean , 'Color', 'b', 'LineWidth' , 3 );
plot( t_axis , theta_B_NN_mean , 'Color', 'b', 'LineWidth' , 3 );

% xlabel({'time (sec)'});
% ylabel({'\x_{mB} (mm)'});
% ylabel({'\theta_{B} (mm)'});

% legend({ 'x_{mB,NN,std}' , 'x_{mB,NN,mean}' });
legend({ '\theta_{B,NN,std}' , '\theta_{B,NN,mean}' });

% set(gca,'FontSize',40);

