#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter
import time
import numpy as np
import scipy.io as sio
from bias import *
from motion import *
from filter import *
import matplotlib.pyplot as plt


def estimate_rot(data_num=1):
    print('estimating ...')
    file = 'imu/imuRaw' + str(data_num) + '.mat'
    imu = sio.loadmat(file)
    accelVals = imu['vals'][0:3,:]
    gyroVals = np.array([imu['vals'][4,:],imu['vals'][5,:],imu['vals'][3,:]])
    tsI = imu['ts'][0,:]
    dtI = tsI - tsI[0]

    viconFile = 'vicon/viconRot' + str(data_num) + '.mat'
    vicon = sio.loadmat(viconFile)
    R0 = vicon['rots'][:,:,15]

    accelVals = calibrate(tsI,accelVals,'accelerometer')
    gyroVals = calibrate(tsI,gyroVals,'gyro')

    # rz = np.zeros((3,len(zroll)))
    # for i in range(len(zroll)):
    #     rz[:,i] = eul2axang(zroll[i],zpitch[i],zyaw[i])

    q0 = rot2quat(R0)
    x = np.array([q0[0],q0[1],q0[2],q0[3],gyroVals[0,0],gyroVals[1,0],gyroVals[2,0]])
    # x = np.array([q0[0],q0[1],q0[2],q0[3],Droll[0],Dpitch[0],Dyaw[0]])
    P = 10*np.eye(6)
    Q = 75*np.eye(6)
    R = 75*np.eye(6)
    roll = np.zeros(accelVals.shape[1])
    pitch = np.zeros(accelVals.shape[1])
    yaw = np.zeros(accelVals.shape[1])


    for i in range(len(tsI)-1): # len(tsI)-1
        dt = tsI[i+1]-tsI[i]
        z = np.array([accelVals[0,i],accelVals[1,i],accelVals[2,i],gyroVals[0,i],gyroVals[1,i],gyroVals[2,i]])
        # print(z)
        x,P = UKF(dt,x,P,Q,z,R)
        # time.sleep( 2 )
        roll[i], pitch[i], yaw[i] = quat2eul(x[0:4])

    return roll,pitch,yaw


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

    plot([r[0:5545],p[0:5545],y[0:5545]],[vroll,vpitch,vyaw])
    