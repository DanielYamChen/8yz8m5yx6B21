clear all;
% close all;

model_idx = [ 5461 , 123 , 445 , 2500 , 1111 ];
bar_color = [ 1 , 0 , 1 ; 1 , 0 , 0 ; 0 , 1 , 0 ; 0 , 0 , 1 ; 0.7 , 0.7 , 0.7 ];

for i = 1 : length( model_idx )
    
    bst_ini_cond( i , : , : ) = csvread( strcat( 'data/best_ini_cond_set(', num2str( model_idx(i) ), ').csv' ) , 0 , 11 );
    bfr_set_set( i , : ) = csvread( strcat( 'data/bfr_opt_SSE_set(', num2str( model_idx(i) ), ').csv' ) );
    aftr_set_set( i , : ) = csvread( strcat( 'data/aftr_opt_SSE_set(', num2str( model_idx(i) ), ').csv' ) );
    
%     [ y( i , : ) , x( i , : ) ] = hist(  imprv_set( i , : ) , -1 : 0.2 : 0 );
%     h  = histogram(  imprv_set( i , : ) , -1 : 0.2 : 0 );
%     y( i , : ) = h.Values;
%     x( i , : ) = ( h.BinEdges(1:end-1) + h.BinEdges(2:end) ) / 2;
%     plot( x( i , : ) , y( i , : ), 'LineWidth' , 3 );
%     scatter( 1:1:50 , - imprv_set(i,:) , 'filled' );

end

bfr_set = min( bfr_set_set , [] , 1 );
[ aftr_set , argmin ] = min( aftr_set_set , [] , 1 );
imprv_set = - ( aftr_set - bfr_set ) ./ bfr_set;

% hist(  imprv_set , -0.1 : 0.1 : 1.1 );

% for i = 1 : size( imprv_set , 2 )
%     [ temp , temp_idx ] = sort( - imprv_set(:,i) , 'descend' );
%     for j = 1 : length( model_idx )
%         if( j == 1 )
%             bar( i , temp(j) , 'FaceColor', bar_color( temp_idx(j) , : ) );
%         else
%             bar( i , temp(j) , 'FaceColor', bar_color( temp_idx(j) , : ) , 'EdgeAlpha' , 0 );
%         end
%         hold on;
%     end
% end

% for i = 1 : length( model_idx )
% 
%     plot( x( i , : ) , y( i , : ), 'LineWidth' , 3 );
%     hold on;
%     
% end


for j = 1 : size( bst_ini_cond , 2 )
    
    scatter( bst_ini_cond( argmin(j) , j , 1 ) * 180 / pi , bst_ini_cond( argmin(j) , j , 2 ) , 52 , 'filled' , 'MarkerFaceColor' , 'b' );
    hold on;
    
end

