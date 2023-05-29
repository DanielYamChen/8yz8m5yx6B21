clear all;
% close all;
model_idx = [ 5461 , 123 , 445 , 2500 , 1111 ];
for i = 1 : length( model_idx )
    
    progress_path = strcat( 'data/ISOSC_theta(', num2str( model_idx(i) ), ')_extreme_loss_theta.csv' );
%     progress_path = strcat( 'data/ISOSC_theta(', num2str( model_idx(i) ), ')_extreme_val_loss_theta.csv' );
%     progress_path = strcat( 'data/ISOSC_x_mB(', num2str( model_idx(i) ), ')_extreme_loss_x_mB.csv' );
%     progress_path = strcat( 'data/ISOSC_x_mB(', num2str( model_idx(i) ), ')_extreme_val_loss_x_mB.csv' );
    all_ep_r(i,:)=csvread( progress_path );
    
end

all_ep_r_mean = mean(all_ep_r)';
all_ep_r_std = std(all_ep_r)';
x = 1:1:100; x=x';

fill( [ x ; flipud(x) ] , [ (all_ep_r_mean-all_ep_r_std) ; flipud(all_ep_r_mean+all_ep_r_std) ],[0 , 0.75 , 0.75 ],'linestyle','none');
% fill( [ x ; flipud(x) ] , [ (all_ep_r_mean-all_ep_r_std) ; flipud(all_ep_r_mean+all_ep_r_std) ],[0 , 1.00 , 0.75 ],'linestyle','none');

hold on;
plot( x , all_ep_r_mean , 'Color', 'b', 'LineWidth' , 3 );
% plot( x , all_ep_r_mean , 'Color', [ 0 , 0.7 , 0 ], 'LineWidth' , 3 );

% xlabel({'episode'});
% ylabel({'moving-averaged return'});
% set(gca,'FontSize',40,'PlotBoxAspectRatio',[ 16 , 9 , 1 ]);

