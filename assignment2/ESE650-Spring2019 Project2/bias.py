import numpy as np
import scipy.io as sio


def calibrate(ts,vals,sensor,calibrate=False,iteration=700):
    if calibrate:
        bias = np.transpose(np.array([np.average(vals[:, 0:iteration],axis=1)]))
        # bias = np.array([np.repeat(bias[0],len(vals[0])), np.repeat(bias[1],len(vals[1])), np.repeat(bias[2],len(vals[2]))])
        scale = np.transpose(np.array([[1,1,1]]))
        variance = np.array([1,1,1])
    else:
        if sensor == 'accelerometer':
            bias = np.transpose(np.array([[510.322936459452,500.225993485033,605.15857143]]))
            scale =  np.transpose(np.array([[-104.047762473456,-102.994632975264,-102.630243325576]]))
            variance = np.array([0.3804**2,0.3425**2,0.3547**2])
        elif sensor == 'gyro':
            bias =  np.transpose(np.array([[370.040275601405,373.743749965801,375.593248984362]]))
            scale =  np.transpose(np.array([[-0.00520804281847987,-0.00213666125103214,-0.0285429019421436]]))
            variance = np.array([6.168**2,15.9328**2,2.6225**2])
        else:
            return 'error'

    corrected = np.divide((vals - bias),scale)
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
    