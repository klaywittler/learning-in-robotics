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
from filter import *
import matplotlib.pyplot as plt


def estimate_rot(data_num=1):
    #your code goes here
    file = 'imu/imuRaw' + str(data_num) + '.mat'
    imu = sio.loadmat(file)
    accelVals = imu['vals'][0:3,0:5545]
    gyroVals = imu['vals'][3::,0:5545]
    tsI = imu['ts'][0,0:5545]
    dtI = tsI - tsI[0]

    viconFile = 'vicon/viconRot' + str(data_num) + '.mat'
    vicon = sio.loadmat(viconFile)
    viconRot = vicon['rots'][:,:,16::]
    tsV = vicon['ts'][0,16::]

    accelVals, accelVar = calibrate(tsI,accelVals,'accelerometer',calibrate=False)
    gyroVals, gyroVar = calibrate(tsI,gyroVals,'gyro',calibrate=False)

    # plotMeasure(accelVals,gyroVals)
    roll, pitch = accelerometer(accelVals)
    yaw = np.zeros(roll.shape)

    Droll,Dpitch,Dyaw = gyro(gyroVals,dtI)
    # print(roll)
    # print(pitch)
    # print(Droll)
    # print(Dpitch)
    # print(Dyaw)

    R0 = viconRot[:,:,15]
    q0 = rot2quat(R0)
    x = np.array([q0[0],q0[1],q0[2],q0[3],Droll[0],Dpitch[0],Dyaw[0]])
    # print(x[4::])
    p = 0.1*np.eye(6)
    q = 0.1*np.eye(6)
    # z = np.array([roll,pitch,yaw,Droll,Dpitch,Dyaw])
    r = np.array([accelVar, gyroVar])
    for i in range(len(tsI)-1):
        dt = tsI[i+1]-tsI[i]
        z = np.array([roll[i],pitch[i],yaw[i],Droll[i],Dpitch[i],Dyaw[i]])
        x,p = UKF(dt,x,p,q,z,r)

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

    [r,p,y] = estimate_rot(data_num)
    # print(r)
    # print(p)
    # print(y)

    # file = 'imu/imuRaw' + str(data_num) + '.mat'
    # imu = sio.loadmat(file)
    # accelVals = imu['vals'][0:3]
    # gyroVals = imu['vals'][3::]
    