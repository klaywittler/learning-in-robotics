#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter
import numpy as np
import scipy.io as sio
from bias import *
from filter import *
import matplotlib.pyplot as plt


def estimate_rot(data_num=1):
    file = 'imu/imuRaw' + str(data_num) + '.mat'
    imu = sio.loadmat(file)
    accelVals = imu['vals'][0:3,:]
    gyroVals = np.array([imu['vals'][4,:],imu['vals'][5,:],imu['vals'][3,:]])
    tsI = imu['ts'][0,:]
    dt = imu['ts'][0,1::]-imu['ts'][0,0:-1]

    accelVals = calibrate(accelVals,'accelerometer')
    gyroVals = calibrate(gyroVals,'gyro')

    x = np.array([1.0,0,0,0,gyroVals[0,0],gyroVals[1,0],gyroVals[2,0]])
    P = 10.0*np.eye(6)
    Q = 0.1*np.eye(6)
    R = 0.1*np.eye(6)
    roll = np.zeros(accelVals.shape[1])
    pitch = np.zeros(accelVals.shape[1])
    yaw = np.zeros(accelVals.shape[1])
    orient = np.zeros((4,len(accelVals[0,:])))

    for i in range(len(tsI)-1): # len(tsI)-1
        z = np.array([accelVals[0,i],accelVals[1,i],-accelVals[2,i],gyroVals[0,i],gyroVals[1,i],gyroVals[2,i]])
        x,P = UKF(dt[i],x,P,Q,z,R)
        roll[i], pitch[i], yaw[i] = quat2eul(x[0:4])
        orient[:,i] = x[0:4]
        # print(x[0:4])
    # return orient
    return roll, pitch, yaw


if __name__ == "__main__":
    data_num = 1
    viconFile = 'vicon/viconRot' + str(data_num) + '.mat'
    vicon = sio.loadmat(viconFile)
    viconRot = vicon['rots'][:,:,16::]
    R0 = vicon['rots'][:,:,15]
    tsV = vicon['ts'][0,16::]
    vroll = np.zeros(len(viconRot[0,0,:]))
    vpitch = np.zeros(len(viconRot[0,0,:]))
    vyaw = np.zeros(len(viconRot[0,0,:]))
    for i in range(len(viconRot[0,0,:])):
        vroll[i], vpitch[i], vyaw[i] = rot2eul(viconRot[:,:,i])

    [r,p,y] = estimate_rot(data_num)
    plot([r[0:5545],p[0:5545],y[0:5545]],[vroll,vpitch,vyaw])

    # q = estimate_rot(data_num)
    # g = np.array([0,0,9.80665])
    # a  = np.zeros((len(g),len(viconRot[0,0,:])))
    # a_estim = np.zeros((len(g),len(viconRot[0,0,:])))
    # for i in range(len(viconRot[0,0,:])):
    #     temp = quatMult(quatMult(q[:,i],np.array([0,0,0,9.80665])),quatCong(q[:,i]))
    #     a_estim[:,i] = np.array([temp[1],temp[2],temp[3]])
    #     a[:,i] = np.dot(viconRot[:,:,i],g)

    # plot(a_estim,a)
    