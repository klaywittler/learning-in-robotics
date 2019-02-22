import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from bias import *
from filter import *


def calibrate(vals,sensor,calibrate=False,iteration=700):
    if calibrate:
        bias = np.transpose(np.array([np.average(vals[:, 0:iteration],axis=1)]))
        if sensor == 'accelerometer':
            sensitivity = 33.0
        elif sensor == 'gyro':
            sensitivity = 218.0
        print(bias)
    else:
        if sensor == 'accelerometer':
            # bias = np.transpose(np.array([[510.8436,500.9864,501.0]]))
            bias = np.transpose(np.array([[510.80714286,500.99428571,505.15857143]]))
            sensitivity = 34.5*np.array([1.0,1.0,1.0]) # 33.0
            # sensitivity = np.array([330.5295,330.5295,-338.4347])
        elif sensor == 'gyro':
            # bias =  np.transpose(np.array([[376.0,376.0,381.0]]))
            bias =  np.transpose(np.array([[373.74337241,375.59278629,370.04075744]]))
            # sensitivity = 218.0*np.array([1.0,1.0,1.0]) # 218.0 
            sensitivity = 180/np.pi*np.array([3.8,3.8,3.8])
        else:
            return 'error'

    factor = 3300.0/1023.0/sensitivity
    corrected = (vals - bias)*factor[:,np.newaxis]
    return corrected


def accelerometer(accelVals):
    x = accelVals[0]
    y = accelVals[1]
    z = accelVals[2]
    roll = np.arctan2(y, z)
    pitch = np.arctan2(-x, np.linalg.norm(np.array([y, z]),axis=0))
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


def plot(valsE,valsR):
    plt.close('all')
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
    data_num = 3
    file = 'imu/imuRaw' + str(data_num) + '.mat'
    imu = sio.loadmat(file)

    viconFile = 'vicon/viconRot' + str(data_num) + '.mat'
    vicon = sio.loadmat(viconFile)

    if data_num==1:
        viconRot = vicon['rots'][:,:,16::] # dataset number 1
        s = 0
        e = 5545
        m = 0
    elif data_num==2:
        viconRot = vicon['rots'] # dataset number 2
        s = 51
        e = -47
        m = 51
    elif data_num==3:
        viconRot = vicon['rots'][:,:,0:-64] # dataset number 3
        s = 35
        e = None
        m = 34

    accelVals = imu['vals'][0:3,0:5545]
    gyroVals = np.array([imu['vals'][4,s:e],imu['vals'][5,s:e],imu['vals'][3,s:e]])
    tsI = imu['ts'][0,s:e]
    dtI = tsI - tsI[0]

    vroll = np.zeros(len(viconRot[0,0,:]))
    vpitch = np.zeros(len(viconRot[0,0,:]))
    vyaw = np.zeros(len(viconRot[0,0,:]))
    for i in range(len(viconRot[0,0,:])):
        vroll[i], vpitch[i], vyaw[i] = rot2eul(viconRot[:,:,i])

    g = np.array([0,0,9.80665])
    a = np.zeros((len(g),len(viconRot[0,0,:])))
    for i in range(len(viconRot[0,0,:])):
        a[:,i] = np.dot(viconRot[:,:,i],g)

    accelVals = calibrate(accelVals,'accelerometer')
    gyroVals = calibrate(gyroVals,'gyro')

    zroll, zpitch, zyaw = accelerometer(accelVals)
    # plot([zroll,zpitch,zyaw],[vroll,vpitch,vyaw])

    Droll,Dpitch,Dyaw = gyro(gyroVals,dtI)
    
    dr2r = vroll[0] + np.cumsum(Droll)
    dp2p = vpitch[0] + np.cumsum(Dpitch)
    dy2y = vyaw[0] + np.cumsum(Dyaw)
    # plot([dr2r, dp2p, dy2y],[vroll,vpitch,vyaw])
    # plot(accelVals,a)