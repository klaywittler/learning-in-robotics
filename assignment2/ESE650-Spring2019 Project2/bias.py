import numpy as np
import scipy.io as sio


def calibrate(ts,vals,sensor,calibrate=False,iteration=700):
    if calibrate:
        bias = np.transpose(np.array([np.average(vals[:, 0:iteration],axis=1)]))
        sensitivity = 200
        factor = 3300/1023/sensitivity
        scale = np.transpose(np.array([[factor,factor,factor]]))
        variance = np.array([1,1,1])
        # g = np.tile(np.array([[0],[0],[-1]]),len(viconRot[0,0,:]))
        # a = np.dot(viconRot[:,:,:],g[:,:,np.newaxis])
        # print(len(a[:,0,0]))
    else:
        if sensor == 'accelerometer':
            bias = np.transpose(np.array([[510.80714286,500.99428571,605.15857143]]))
            sensitivity = -200
            factor = 3300/1023/sensitivity
            scale = np.transpose(np.array([[factor,factor,factor]]))
            variance = np.array([0.3804**2,0.3425**2,0.3547**2])
        elif sensor == 'gyro':
            bias =  np.transpose(np.array([[369.68571429,373.57142857,375.37285714]]))
            sensitivity = 200
            factor = 3300/1023/sensitivity
            scale = np.transpose(np.array([[factor,factor,factor]]))
            variance = np.array([6.168**2,15.9328**2,2.6225**2])
        else:
            return 'error'

    corrected = np.multiply((vals - bias),scale)
    return corrected, variance


if __name__ == "__main__":
    data_num = 1
    file = 'imu/imuRaw' + str(data_num) + '.mat'
    imu = sio.loadmat(file)
    accelVals = imu['vals'][0:3]
    gyroVals = imu['vals'][3::]
    ts = imu['ts'][0] - imu['ts'][0,0]
    accelVals = imu['vals'][0:3]
    gyroVals = imu['vals'][3::]
    imuAccel = calibrate(ts,accelVals,'accelerometer',calibrate=False)
    imuGyro = calibrate(ts,gyroVals,'gyro',calibrate=True)
    