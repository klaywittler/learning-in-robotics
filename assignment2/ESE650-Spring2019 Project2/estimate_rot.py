#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter
import numpy as np
import scipy.io as sio


def estimate_rot(data_num=1):
    #your code goes here
    file = 'imu/imuRaw' + str(data_num) + '.mat'
    imu = sio.loadmat(file)
    imuVals = imu['vals']
    imuts = imu['ts']
    print(imuVals)
    roll = 0
    pitch = 0
    yaw = 0
    return roll,pitch,yaw


if __name__ == "__main__":
    data_num = 1
    viconFile = 'vicon/viconRot' + str(data_num) + '.mat'
    vicon = sio.loadmat(viconFile)
    viconRot = vicon['rots']
    viconTS = vicon['ts']
    # print(vicon)
    [r,p,y] = estimate_rot(data_num)
    print(r)
    print(p)
    print(y)