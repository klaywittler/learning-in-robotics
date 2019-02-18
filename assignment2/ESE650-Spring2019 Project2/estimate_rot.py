#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter
import time
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
    gyroVals = np.array([imu['vals'][4,0:5545],imu['vals'][5,0:5545],imu['vals'][3,0:5545]])
    tsI = imu['ts'][0,0:5545]
    dtI = tsI - tsI[0]

    viconFile = 'vicon/viconRot' + str(data_num) + '.mat'
    vicon = sio.loadmat(viconFile)
    R0 = vicon['rots'][:,:,15]
    viconRot = vicon['rots'][:,:,16::]
    tsV = vicon['ts'][0,16::]
    vroll = np.zeros(len(viconRot[0,0,:]))
    vpitch = np.zeros(len(viconRot[0,0,:]))
    vyaw = np.zeros(len(viconRot[0,0,:]))
    for i in range(len(viconRot[0,0,:])):
        vroll[i], vpitch[i], vyaw[i] = rot2eul(viconRot[:,:,i])

    accelVals, accelVar = calibrate(tsI,accelVals,'accelerometer',calibrate=False)
    gyroVals, gyroVar = calibrate(tsI,gyroVals,'gyro',calibrate=True)

    zroll, zpitch, zyaw = accelerometer(accelVals)
    Droll,Dpitch,Dyaw = gyro(gyroVals,dtI)
    rz = np.zeros((3,len(zroll)))
    for i in range(len(zroll)):
        rz[:,i] = eul2axang(zroll[i],zpitch[i],zyaw[i])

    plot(accelVals,[vroll,vpitch,vyaw])
    # plot([zroll,zpitch,zyaw],[vroll,vpitch,vyaw])
    # dr2r = vroll[0] + np.cumsum(Droll)
    # dp2p = vpitch[0] + np.cumsum(Dpitch)
    # dy2y = vyaw[0] + np.cumsum(Dyaw)
    # plot([dr2r, dp2p, dy2y],[vroll,vpitch,vyaw])

    q0 = rot2quat(R0)
    x = np.array([q0[0],q0[1],q0[2],q0[3],Droll[0],Dpitch[0],Dyaw[0]])
    P = 10*np.eye(6)
    Q = 10*np.eye(6)
    R = 10*np.eye(6)
    roll = np.zeros(zroll.shape)
    pitch = np.zeros(zpitch.shape)
    yaw = np.zeros(zyaw.shape)

    for i in range(len(tsI)-1): # len(tsI)-1
        # print(i)
        dt = tsI[i+1]-tsI[i]
        # z = np.array([zroll[i],zpitch[i],zyaw[i],Droll[i],Dpitch[i],Dyaw[i]])
        z = np.array([accelVals[0,i],accelVals[1,i],accelVals[2,i],Droll[i],Dpitch[i],Dyaw[i]])
        x,P = UKF(dt,x,P,Q,z,R)
        # print(P)
        # time.sleep( 2 )
        roll[i], pitch[i], yaw[i] = quat2eul(x[0:4])

    return roll,pitch,yaw


def plot(accelVals=None,gyroVals=None):
    if accelVals is not None and gyroVals is not None:
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(accelVals[0])
        axs[0].plot(accelVals[1]) 
        axs[0].plot(accelVals[2])
        axs[0].set_xlabel('time')
        axs[0].set_ylabel('estimated')
        axs[0].grid(True)

        axs[1].plot(gyroVals[0])
        axs[1].plot(gyroVals[1])
        axs[1].plot(gyroVals[2])
        axs[1].set_xlabel('time')
        axs[1].set_ylabel('real')
        axs[1].grid(True)
        fig.tight_layout()
    elif accelVals is not None:
        plt.plot(accelVals[0])
        plt.plot(accelVals[1]) 
        plt.plot(accelVals[2])
    elif gyroVals is not None:
        plt.plot(gyroVals[0])
        plt.plot(gyroVals[1]) 
        plt.plot(gyroVals[2])
    plt.show()


if __name__ == "__main__":
    data_num = 1
    viconFile = 'vicon/viconRot' + str(data_num) + '.mat'
    vicon = sio.loadmat(viconFile)
    R0 = vicon['rots'][:,:,15]
    viconRot = vicon['rots'][:,:,16::]
    tsV = vicon['ts'][0,16::]
    vroll = np.zeros(len(viconRot[0,0,:]))
    vpitch = np.zeros(len(viconRot[0,0,:]))
    vyaw = np.zeros(len(viconRot[0,0,:]))
    for i in range(len(viconRot[0,0,:])):
        vroll[i], vpitch[i], vyaw[i] = rot2eul(viconRot[:,:,i])

    [r,p,y] = estimate_rot(data_num)
    # print(r)
    # print(p)
    # print(y)

    plot([r,p,y],[vroll,vpitch,vyaw])

    # file = 'imu/imuRaw' + str(data_num) + '.mat'
    # imu = sio.loadmat(file)
    # accelVals = imu['vals'][0:3]
    # gyroVals = imu['vals'][3::]
    