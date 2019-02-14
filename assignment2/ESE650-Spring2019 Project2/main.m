 addpath('imu')
addpath('vicon')

load('imuRaw1.mat')
tIMU = ts;
t1 = ts(1);
load('viconRot1.mat')
tVICON = ts - ts(1);
t2 = ts(1); % 16 or 17


% g = [0;0;-1];%.*ones(3,1,numel(rots(1,1,:)));
% 
% a = sum(rots.*g',2);
% b = norm(a(:,1));
% 
% idx = 3;
% scatter(a(idx,16:end),vals(idx,1:end-99))
% P = polyfit(a(idx,16:end),vals(idx,1:end-99),1);



