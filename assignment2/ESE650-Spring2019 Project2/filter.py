import numpy as np


def UKF(t,x,w,z,r):
    q = axang2quat(x[0:3])
    qw = axang2quat(w[0:3])
    qtilde = quatMult(q,qw)
    dq = axang2quat(qw,t)
    wtilde = x[3::] + w[3::]
    roll = 0
    pitch = 0
    yaw = 0
    return roll,pitch,yaw


def quatMult(q1,q2):
    u0 = q1[0]
    v0 = q2[0]
    u = q1[1::]
    v = q2[1::]
    q0 = u0*v0 - np.dot(u,v)
    q = u0*v + v0*u + np.cross(u,v)
    return np.array([q0,q])


def quat2rot(q):
    u0 = q[0]
    u = q[1::]
    R = np.matmult((u0**2 - np.inner(np.transpose(u),u)),np.eye(3)) +2*u0,vec2skew(u) + 2*np.outer(u,np.transpose(u))
    return R

def rot2quat(R):
    theta = np.arccos((np.trace(R) - 1)/2)
    what = (R - np.transpose(R))/(2*np.sin(theta))
    w = skew2vec(what)
    q = axang2quat(theta,w)
    return  


def axang2quat(w,t=None):
    if t==None: 
        theta = np.norm(w)
        u = w/theta
    else:
        theta = np.multiply(np.norm(w),t)
        u = w/np.norm(w)   
    return np.array([np.cos(theta/2),np.sin(theta/2)*u])


def skew2vec(x):
    return np.array([x[1,2], x[0,2],x[1,0]])


def vec2skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, x[0]], [x[1], x[0], 0]])
