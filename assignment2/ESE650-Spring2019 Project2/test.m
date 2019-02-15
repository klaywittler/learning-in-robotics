addpath('imu')
addpath('vicon')

load('imuRaw1.mat')
tIMU = ts;
t1 = ts(1);
load('viconRot1.mat')
tVicon = ts - ts(1);
t2 = ts(16); % 16 or 17


g = [0;0;-1];%.*ones(3,1,numel(rots(1,1,:)));

a = sum(rots.*g',2);
b = norm(a(:,1));

idx = 3;
scatter(a(idx,16:end),vals(idx,1:end-99))
[Pa,Sa,mua] = polyfit(a(idx,16:end),vals(idx,1:end-99),1);
 
Rt = rots(:,:,1:end-1);
Rtp1 = rots(:,:,2:end);
dt = tVicon(2:end) - tVicon(1:end-1);
Rtp1T = permute(Rtp1,[2,1,3]);
dR = zeros(size(Rt));
for k=1:numel(dt)
   dR(:,:,k) =  Rtp1T(:,:,k)*Rt(:,:,k);
end
axang = rotm2axang(dR);
theta = axang(:,4)';
r = axang(:,1:3)';
w = (1./dt).*r.*theta;
n = vecnorm(w,2,1);
[v,i] = max(n);

gyro = 3;
scatter(w(idx,16:end),vals(idx+gyro,1:end-100))
[Pg,Sb,mub] = polyfit(w(idx,16:end),vals(idx+gyro,1:end-100),1);


