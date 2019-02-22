#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter
import numpy as np
import scipy.io as sio
import scipy.linalg as sp
from bias import *
from filter import *
import matplotlib.pyplot as plt


def estimate_rot(data_num=1, P=10.0*np.eye(6), Q=90.0*np.eye(6), R=60.0*np.eye(6)):
    file = 'imu/imuRaw' + str(data_num) + '.mat'
    imu = sio.loadmat(file)
    accelVals = imu['vals'][0:3,:]
    gyroVals = np.array([imu['vals'][4,:],imu['vals'][5,:],imu['vals'][3,:]])
    dt = imu['ts'][0,1::]-imu['ts'][0,0:-1]

    accelVals = calibrate(accelVals,'accelerometer')
    gyroVals = calibrate(gyroVals,'gyro')

    x = np.array([1.0,0,0,0,gyroVals[0,0],gyroVals[1,0],gyroVals[2,0]])

    roll = np.zeros(accelVals.shape[1])
    pitch = np.zeros(accelVals.shape[1])
    yaw = np.zeros(accelVals.shape[1])
    orient = np.zeros((4,len(accelVals[0,:])))

    for i in range(len(dt)): #  len(dt)
        z = np.array([accelVals[0,i],accelVals[1,i],accelVals[2,i],gyroVals[0,i],gyroVals[1,i],gyroVals[2,i]])
        x,P = UKF(dt[i],x,P,Q,z,R)
        roll[i], pitch[i], yaw[i] = quat2eul(x[0:4])
    return roll, pitch, yaw


def estimate_rot4(data_num=1, P=10.0*np.eye(3), Q=40.0*np.eye(3), R=10.0*np.eye(3)):
    file = 'imu/imuRaw' + str(data_num) + '.mat'
    imu = sio.loadmat(file)
    accelVals = imu['vals'][0:3,:]
    gyroVals = np.array([imu['vals'][4,:],imu['vals'][5,:],imu['vals'][3,:]])
    dt = imu['ts'][0,1::]-imu['ts'][0,0:-1]

    accelVals = calibrate(accelVals,'accelerometer')
    gyroVals = calibrate(gyroVals,'gyro')

    x4 = np.array([1.0,0,0,0])

    roll = np.zeros(accelVals.shape[1])
    pitch = np.zeros(accelVals.shape[1])
    yaw = np.zeros(accelVals.shape[1])

    for i in range(len(dt)): 
        z4 = np.array([accelVals[0,i],accelVals[1,i],accelVals[2,i]])
        u4 = np.array([gyroVals[0,i],gyroVals[1,i],gyroVals[2,i]])
        x4,P = UKF4(dt[i],x4,u4,P,Q,z4,R)
        roll[i], pitch[i], yaw[i] = quat2eul(x4)
    return roll, pitch, yaw


def estimate_quat(data_num=1, P=1.0*np.eye(6), Q=0.001*np.eye(6), R=0.0001*np.eye(6)):
    file = 'imu/imuRaw' + str(data_num) + '.mat'
    imu = sio.loadmat(file)
    accelVals = imu['vals'][0:3,:]
    gyroVals = np.array([imu['vals'][4,:],imu['vals'][5,:],imu['vals'][3,:]])
    dt = imu['ts'][0,1::]-imu['ts'][0,0:-1]

    accelVals = calibrate(accelVals,'accelerometer')
    gyroVals = calibrate(gyroVals,'gyro')

    x = np.array([1.0,0,0,0,gyroVals[0,0],gyroVals[1,0],gyroVals[2,0]])

    roll = np.zeros(accelVals.shape[1])
    pitch = np.zeros(accelVals.shape[1])
    yaw = np.zeros(accelVals.shape[1])
    orient = np.zeros((4,len(accelVals[0,:])))

    for i in range(len(dt)):
        z = np.array([-accelVals[0,i],-accelVals[1,i],accelVals[2,i],gyroVals[0,i],gyroVals[1,i],gyroVals[2,i]])
        x,P = UKF(dt[i],x,P,Q,z,R)
        orient[:,i] = x[0:4]
    return orient


def estimate_quat4(data_num=1, P=1.0*np.eye(6), Q=0.001*np.eye(6), R=0.0001*np.eye(6)):
    file = 'imu/imuRaw' + str(data_num) + '.mat'
    imu = sio.loadmat(file)
    accelVals = imu['vals'][0:3,:]
    gyroVals = np.array([imu['vals'][4,:],imu['vals'][5,:],imu['vals'][3,:]])
    dt = imu['ts'][0,1::]-imu['ts'][0,0:-1]

    accelVals = calibrate(accelVals,'accelerometer')
    gyroVals = calibrate(gyroVals,'gyro')

    x4 = np.array([1.0,0,0,0])

    roll = np.zeros(accelVals.shape[1])
    pitch = np.zeros(accelVals.shape[1])
    yaw = np.zeros(accelVals.shape[1])
    orient = np.zeros((4,len(accelVals[0,:])))

    for i in range(len(dt)):
        z4 = np.array([accelVals[0,i],accelVals[1,i],accelVals[2,i]])
        u4 = np.array([gyroVals[0,i],gyroVals[1,i],gyroVals[2,i]])
        x4,P = UKF4(dt[i],x,u4,P,Q,z4,R)
        orient[:,i] = x4
    return orient


if __name__ == "__main__":
    data_num = 3

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

    vroll = np.zeros(len(viconRot[0,0,:]))
    vpitch = np.zeros(len(viconRot[0,0,:]))
    vyaw = np.zeros(len(viconRot[0,0,:]))
    for i in range(len(viconRot[0,0,:])):
        vroll[i], vpitch[i], vyaw[i] = rot2eul(viconRot[:,:,i])


    [r,p,y] = estimate_rot4(data_num)
    plot([r[s:e],p[s:e],y[s:e]],[vroll,vpitch,vyaw])


    # q = estimate_quat(data_num,P,Q,R)
    # g = np.array([0,0,9.80665])
    # a = np.zeros((len(g),len(viconRot[0,0,:])))
    # a_estim = np.zeros((len(g),len(viconRot[0,0,:])))
    # for i in range(len(viconRot[0,0,:])):
    #     temp = quatMult(quatMult(q[:,i+m],np.array([0,0,0,9.80665])),quatCong(q[:,i+m]))
    #     a_estim[:,i] = np.array([temp[1],temp[2],temp[3]])
    #     a[:,i] = np.dot(viconRot[:,:,i],g)

    # plot(a_estim,a)
