clear all; close all;
addpath('imu')
addpath('vicon')

load('imuRaw2.mat')
tIMU = ts(52:end-46);
t1 = tIMU(1);
load('viconRot2.mat')
tVicon = ts - t1;
tIMU = tIMU - ts(1);

% g = [0;0;-9.80665];%.*ones(3,1,numel(rots(1,1,:)));
% 
% a = sum(rots.*g',2);
% b = norm(a(:,1));
% 
% sensitivity = -33;
% factor = 3300/1023/sensitivity;
% accelVals = (vals(1:3,:) - [510.80714286;500.99428571;505.15857143])*factor;
% 
% i = 1:5546;
% idx = 1;
% figure(1)
% plot(i,a(1,16:end),i,accelVals(1,1:end-99))
% figure(2)
% plot(i,a(2,16:end),i,accelVals(2,1:end-99))
% figure(3)
% plot(i,a(3,16:end),i,accelVals(3,1:end-99))
% [Pa,Sa,mua] = polyfit(a(idx,16:end),vals(idx,1:end-99),1);
 
% Rt = rots(:,:,1:end-1);
% Rtp1 = rots(:,:,2:end);
% dt = tVicon(2:end) - tVicon(1:end-1);
% Rtp1T = permute(Rtp1,[2,1,3]);
% dR = zeros(size(Rt));
% for k=1:numel(dt)
%    dR(:,:,k) =  Rtp1T(:,:,k)*Rt(:,:,k);
% end
% axang = rotm2axang(dR);
% theta = axang(:,4)';
% r = axang(:,1:3)';
% w = (1./dt).*r.*theta;
% n = vecnorm(w,2,1);
% [v,i] = max(n);

% gyro = 3;
% plot(i,vals(idx+gyro,1:end-100))
% [Pg,Sb,mub] = polyfit(w(idx,16:end),vals(idx+gyro,1:end-100),1);



