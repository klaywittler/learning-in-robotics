#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter
import numpy as np
import scipy.io as sio
import math as m


def estimate_rot(data_num=1):
    #your code goes here
    file = 'imu/imuRaw' + str(data_num) + '.mat'
    imu = sio.loadmat(file)
    accVals = imu['vals'][0:3]
    gyroVals = imu['vals'][3::]
    ts = imu['ts'][0]
    roll, pitch = accelerometer(accVals[0],accVals[1],accVals[2])
    yaw = 0
    Droll,Dpitch,Dyaw = gyro(gyroVals[0],gyroVals[1],gyroVals[2],ts)
    return roll,pitch,yaw


def accelerometer(x,y,z):
    roll = np.rad2deg(np.arctan2(y, z))
    pitch = np.rad2deg(np.arctan2(-x, np.linalg.norm([y, z],axis=0)))
    return roll, pitch    


def gyro(x,y,z,t):
    dt = t[1::] - t[0:-1]
    Droll = np.trapz([x[1::],x[0:-1]],dx=dt,axis=0)
    Dpitch = np.trapz([y[1::],y[0:-1]],dx=dt,axis=0)
    Dyaw = np.trapz([z[1::],z[0:-1]],dx=dt,axis=0)
    print(Droll)
    print(Dpitch)
    print(Dyaw)
    return Droll,Dpitch,Dyaw 

if __name__ == "__main__":
    data_num = 1
    viconFile = 'vicon/viconRot' + str(data_num) + '.mat'
    vicon = sio.loadmat(viconFile)
    viconRot = vicon['rots']
    viconTS = vicon['ts']
    print(viconRot[0][0])
    [r,p,y] = estimate_rot(data_num)
    print(r)
    print(p)
    print(y)
    # a = np.array([1,2,3,4])
    # print(a[0:2])