clear all;
% close all;

model_idx = [ 5461 , 123 , 445 , 2500 , 1111 ];

 x_mB_diff_set = csvread('data/x_mB_diff(123).csv');
 for i = 1 : size(x_mB_diff_set,1)
    j = size(x_mB_diff_set,2);
    while( x_mB_diff_set(i,j) == 0 )
        j = j - 1;
    end
    length_set( i ) = j;
 end
 
 clear x_mB_diff_set

for i = 1 : length( model_idx )
    
    x_mB_diff_set(:,i,:) = csvread( strcat( 'data/x_mB_diff(', num2str( model_idx(i) ) , ').csv' ) );
    theta_B_diff_set(:,i,:) = csvread( strcat( 'data/theta_B_diff(', num2str( model_idx(i) ) , ').csv' ) ) * 180 / pi;

end

for i = 1 : 100
    
    x_mB_diff_mean_set{i} = mean( x_mB_diff_set( i , : , 1 : length_set( i ) ) );
    x_mB_diff_mean_set{i} = x_mB_diff_mean_set{i}(:)';
%     x_mB_diff_std_set{i} = std( x_mB_diff_set( i , : , 1 : length_set( i ) ) );
%     x_mB_diff_std_set{i} = x_mB_diff_std_set{i}(:)';
    theta_B_diff_mean_set{i} = mean( theta_B_diff_set( i , : , 1 : length_set( i ) ) );
    theta_B_diff_mean_set{i} = theta_B_diff_mean_set{i}(:)';
%     theta_B_diff_std_set{i} = std( theta_B_diff_set( i , : , 1 : length_set( i ) ) );
%     theta_B_diff_std_set{i} =theta_B_diff_std_set{i}(:)';
    nrmlz_idx_set{i} = 0 : 1/length_set( i ) : 1;
end

for i = 1 : 100
%     fill( [ t_axis ; flipud( t_axis ) ] , [ ( x_mB_NN_mean - x_mB_NN_std ) ; flipud( x_mB_NN_mean + x_mB_NN_std ) ],[0 , 0.75 , 0.75 ] , 'linestyle' , 'none' );
%      fill( [ t_axis ; flipud( t_axis ) ] , [ ( theta_B_NN_mean - theta_B_NN_std ) ; flipud( theta_B_NN_mean + theta_B_NN_std ) ],[0 , 0.75 , 0.75 ] , 'linestyle' , 'none' );

%     hold on;
%     plot( nrmlz_idx_set{i}(1:length_set( i )), x_mB_diff_mean_set{i} , 'LineWidth' , 1.5 );
    plot( nrmlz_idx_set{i}(1:length_set( i )), theta_B_diff_mean_set{i} , 'LineWidth' , 1.5 );
    hold on;
end

% xlabel({'time (sec)'});
% ylabel({'\theta_{B} (mm)'});
set(gca,'FontSize',36);
