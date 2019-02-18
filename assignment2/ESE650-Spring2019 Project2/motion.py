import numpy as np


def accelerometer(accelVals):
    x = accelVals[0]
    y = accelVals[1]
    z = accelVals[2]
    roll = np.arctan2(y, z)
    pitch = np.arctan2(-x, np.linalg.norm([y, z],axis=0))
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
    