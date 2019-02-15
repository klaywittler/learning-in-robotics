import numpy as np
import scipy.sparse.linalg as ssp
import scipy.linalg as sp


def UKF(dt,x,p,q,z,r):
    # q = axang2quat(x[0:3])
    # qw = axang2quat(w[0:3])
    # qtilde = quatMult(q,qw)
    # dq = axang2quat(qw,t)
    # wtilde = x[3::] + w[3::]

    S = sp.sqrtm(p)
    print(S)
    dq = axang2quat(x[4::],dt)

    xk = x
    pk = p
    return xk,pk


def quatMult(q1,q2):
    u0 = q1[0]
    v0 = q2[0]
    u = q1[1::]
    v = q2[1::]
    q0 = u0*v0 - np.dot(u,v)
    q = u0*v + v0*u + np.cross(u,v)
    return np.array([q0,q[0],q[1],q[2]])


def quat2rot(q):
    u0 = q[0]
    u = q[1::]
    R = (u0**2 - np.inner(np.transpose(u),u))*np.eye(3) + 2*u0*vec2skew(u) + 2*np.outer(u,np.transpose(u))
    return R 


def rot2quat(R):
    angle = np.arccos((np.trace(R) - 1)/2)
    # what = (R - np.transpose(R))/(2*np.sin(theta))
    # w = skew2vec(what)
    eigVals, eigVec = ssp.eigs(R, k=1, sigma=1)
    axis = np.array([eigVec[0,0],eigVec[1,0],eigVec[2,0]])
    q = axang2quat(axis*angle)
    return  q


def axang2quat(w,t=None):
    if t is None: 
        angle = np.linalg.norm(w)
        axis = w/angle
    else:
        angle = np.multiply(np.linalg.norm(w),t)
        axis = w/np.linalg.norm(w)  
    u = np.sin(angle/2)*axis
    return np.array([np.cos(angle/2),u[0],u[1],u[2]])


def quat2axang(q):
    angle = 2*np.arccos(q[0])
    axis = q[1::]/np.sqrt(1-q[0]**2)
    return axis*angle


def skew2vec(x):
    return np.array([x[1,2], x[0,2],x[1,0]])


def vec2skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


if __name__ == "__main__":
    q1 = np.array([0.5**0.5,0,0.5**0.5,0])
    r = quat2rot(q1)
    q = rot2quat(r)
    