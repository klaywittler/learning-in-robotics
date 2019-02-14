import numpy as np
import scipy.io as sio

def calibrate(ts,vals,sensor,calibrate=False,iteration=700):
    if calibrate:
        bias = np.transpose(np.array([np.average(vals[:, 0:iteration],axis=1)]))
        # bias = np.array([np.repeat(bias[0],len(vals[0])), np.repeat(bias[1],len(vals[1])), np.repeat(bias[2],len(vals[2]))])
        scale = 1
    else:
        if sensor == 'accelerometer':
            bias = np.transpose(np.array([510.80714286,500.99428571,605.15857143]))
            scale = 1
        elif sensor == 'gyro':
            bias = np.transpose(np.array([369.68571429,373.57142857,375.37285714]))
            scale = 1
        else:
            return 'error'

    corrected = (vals - bias)/scale
    return corrected


if __name__ == "__main__":
    data_num = 1
    file = 'imu/imuRaw' + str(data_num) + '.mat'
    imu = sio.loadmat(file)
    accelVals = imu['vals'][0:3]
    gyroVals = imu['vals'][3::]
    ts = imu['ts'][0] - imu['ts'][0,0]
    accelVals = imu['vals'][0:3]
    gyroVals = imu['vals'][3::]
    imuAccel = calibrate(ts,accelVals,'accelerometer',calibrate=True)
    imuGyro = calibrate(ts,gyroVals,'gyro',calibrate=True)
    