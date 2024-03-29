xtm1 = [3;4];
u = [1;2];
R = [5 1;1 3];
sigm1 = [8 3;3 4];
Q = 10;
G = [7 2;2/3 1];
xt1 = xtm1(1) + u(2)*xtm1(2) +u(1)*xtm1(1)^2;
xt2 = u(1)*xtm1(2) + u(2)*log(xtm1(1));
H = [1/xt2 -xt1/xt2^2];
sig = G*sigm1*G' + R;
K = sig*H'*inv(H*sig*H' + Q);

%     qi = zeros(4,12);
%     qi(:,1) = [0.865918, 0.2939987, 0.3276866, 0.2374283];
%     qi(:,2) = [0.8571044,0.3052557, 0.3172266, 0.2675038];
%     qi(:,3) = [0.8641028,0.3039343, 0.3015601, 0.2645975];
%     qi(:,4) = [0.8498449,0.3064842, 0.3327965, 0.2703286];
%     qi(:,5) = [0.8550644,0.2916056, 0.3374637, 0.2644793];
%     qi(:,6) = [0.8443666,0.3212693, 0.3280279, 0.2760955];
%     qi(:,7) = [0.8586668,0.3189687, 0.2968963, 0.2698201];
%     qi(:,8) = [0.8675603,0.3084128, 0.3078473, 0.2396887];
%     qi(:,9) = [0.8709171,0.2788416, 0.3317804, 0.2316732];
%     qi(:,10) = [0.8692756,0.2888073, 0.306132, 0.2592942];
%     qi(:,11) = [0.8637737,0.2796507, 0.347435, 0.2344769];
%     qi(:,12) = [0.8534878,0.3096256, 0.3390402, 0.2464594];
%     qbar = [0.360125,0.557931,-0.55474,-0.501302];
%     
%     qi = quaternion(qi');
%     
%     quatAverage = meanrot(qi',1)