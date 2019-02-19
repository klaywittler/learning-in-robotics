import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from bias import *
from motion import *
from filter import *


def calibrate(ts,vals,sensor):
    if sensor == 'accelerometer':
        # bias = np.transpose(np.array([[510.80714286,500.99428571,605.15857143]]))
        bias = np.transpose(np.array([[510.80714286,500.99428571,505.15857143]]))
        sensitivity = -33.0
        factor = 3300/1023/sensitivity
    elif sensor == 'gyro':
        bias =  np.transpose(np.array([[373.74337241,375.59278629,370.04075744]]))
        # bias =  np.transpose(np.array([[369.68571429,373.57142857,375.37285714]]))
        sensitivity = 218.0 
        factor = 3300/1023/sensitivity
    else:
        return 'error'

    corrected = (vals - bias)*factor
    return corrected


def accelerometer(accelVals):
    x = accelVals[0]
    y = accelVals[1]
    z = accelVals[2]
    roll = np.arctan(y, z)
    pitch = np.arctan(-x, np.linalg.norm(np.array([y, z]),axis=0))
    yaw = np.zeros(roll.shape)
    return roll, pitch, yaw 


def gyro(gyroVals,t):
    x = gyroVals[0]
    y = gyroVals[1]
    z = gyroVals[2]
    dt = t[1::] - t[0:-1]
    Droll = np.trapz([x[1::],x[0:-1]],dx=dt,axis=0)
    Dpitch = np.trapz([y[1::],y[0:-1]],dx=dt,axis=0)
    Dyaw = np.trapz([z[1::],z[0:-1]],dx=dt,axis=0)
    return Droll,Dpitch,Dyaw 


def plot(valsE=None,valsR=None):
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(valsE[0])
    axs[0].plot(valsR[0]) 
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('roll')
    axs[0].grid(True)

    axs[1].plot(valsE[1])
    axs[1].plot(valsR[1])
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('pitch')
    axs[1].grid(True)

    axs[2].plot(valsE[2])
    axs[2].plot(valsR[2])
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('yaw')
    axs[2].grid(True)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_num = 1
    file = 'imu/imuRaw' + str(data_num) + '.mat'
    imu = sio.loadmat(file)
    accelVals = imu['vals'][0:3,0:5545]
    gyroVals = np.array([imu['vals'][4,0:5545],imu['vals'][5,0:5545],imu['vals'][3,0:5545]])
    tsI = imu['ts'][0,0:5545]
    dtI = tsI - tsI[0]

    viconFile = 'vicon/viconRot' + str(data_num) + '.mat'
    vicon = sio.loadmat(viconFile)
    viconRot = vicon['rots'][:,:,16::]
    tsV = vicon['ts'][0,16::]
    vroll = np.zeros(len(viconRot[0,0,:]))
    vpitch = np.zeros(len(viconRot[0,0,:]))
    vyaw = np.zeros(len(viconRot[0,0,:]))
    for i in range(len(viconRot[0,0,:])):
        vroll[i], vpitch[i], vyaw[i] = rot2eul(viconRot[:,:,i])

    # g = np.tile(np.array([[0],[0],[-9.80665]]),len(viconRot[0,0,:]))
    # a = np.dot(viconRot[:,:,0],g[:,0,np.newaxis])
    g = np.array([0,0,-9.80665])
    a = np.zeros((len(g),len(viconRot[0,0,:])))
    for i in range(len(viconRot[0,0,:])):
        a[:,i] = np.dot(viconRot[:,:,i],g)

    accelVals = calibrate(tsI,accelVals,'accelerometer')
    gyroVals = calibrate(tsI,gyroVals,'gyro')

    zroll, zpitch, zyaw = accelerometer(accelVals)
    Droll,Dpitch,Dyaw = gyro(gyroVals,dtI)
    
    # dr2r = vroll[0] + np.cumsum(Droll)
    # dp2p = vpitch[0] + np.cumsum(Dpitch)
    # dy2y = vyaw[0] + np.cumsum(Dyaw)
    # plot([dr2r, dp2p, dy2y],[vroll,vpitch,vyaw])
    # plot([zroll,zpitch,zyaw],[vroll,vpitch,vyaw])
    # plot(accelVals,a)