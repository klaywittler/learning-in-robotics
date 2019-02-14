#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter
import numpy as np
import scipy.io as sio
import math as m
from bias import *
from motion import *
import matplotlib.pyplot as plt


def estimate_rot(data_num=1):
    #your code goes here
    file = 'imu/imuRaw' + str(data_num) + '.mat'
    imu = sio.loadmat(file)
    accelVals = imu['vals'][0:3]
    gyroVals = imu['vals'][3::]
    ts = imu['ts'][0] - imu['ts'][0,0]

    accelVals = calibrate(ts,accelVals,'accelerometer',calibrate=True)
    gyroVals = calibrate(ts,gyroVals,'gyro',calibrate=True)

    roll, pitch = accelerometer(accelVals)
    yaw = 0
    # print(roll)
    # print(pitch)

    Droll,Dpitch,Dyaw = gyro(gyroVals,ts)
    # print(Droll)
    # print(Dpitch)
    # print(Dyaw)

    roll,pitch,yaw = UKF()
    return roll,pitch,yaw


def UKF():
    roll = 0
    pitch = 0
    yaw = 0
    return roll,pitch,yaw


def plotMeasure(accelVals,gyroVals):
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(accelVals[0])
    axs[0].plot(accelVals[1]) 
    axs[0].plot(accelVals[2])
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('accelerometer')
    axs[0].grid(True)

    axs[1].plot(gyroVals[0])
    axs[1].plot(gyroVals[1])
    axs[1].plot(gyroVals[2])
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('gyro')
    axs[1].grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_num = 1
    viconFile = 'vicon/viconRot' + str(data_num) + '.mat'
    vicon = sio.loadmat(viconFile)
    viconRot = vicon['rots']
    viconTS = vicon['ts']  
    g = np.tile(np.array([[0],[0],[-1]]),len(viconRot[0,0,:]))
    a = np.dot(viconRot[:,:,:],g[:,:,np.newaxis])
    print(len(a[:,0,0]))
    [r,p,y] = estimate_rot(data_num)
    # print(r)
    # print(p)
    # print(y)
    # a = np.array([1,2,3,4])
    # print(a[0:2])
    # file = 'imu/imuRaw' + str(data_num) + '.mat'
    # imu = sio.loadmat(file)
    # accelVals = imu['vals'][0:3]
    # gyroVals = imu['vals'][3::]
    