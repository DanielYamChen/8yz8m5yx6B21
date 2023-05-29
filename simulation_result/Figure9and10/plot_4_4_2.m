clear all;
% close all;

model_idx = [ 5461 , 123 , 445 , 2500 , 1111 ];

iLQR_set = csvread( strcat( 'data/x_mB_iLQR_RMSE.csv' ) )';
for i = 1 : length( model_idx )
     NN_set_set( i , : ) = csvread( strcat( 'data/x_mB_NN_RMSE(', num2str( model_idx(i) ), ').csv' ) );
end

NN_set = min( NN_set_set , [] , 1 );
cmpr_set = NN_set ./ iLQR_set;

% [ y_cmpr , x_cmpr ] = hist(  cmpr_set , 0 : 0.5 : 5 );
hist(  cmpr_set , 0 : 0.25 : 5 );


% for i = 1 : length( model_idx )
%     
%     [ y( i , : ) , x( i , : ) ] = hist(  NN_set( i , : ) , 0 : 5 : 40 );
%     plot( x( i , : ) , y( i , : ), 'LineWidth' , 3 );
%     hold on;
%     
% end
% 
% plot( x_cmpr , y_cmpr, 'b' , 'LineWidth' , 4 );
