import numpy as np
import scipy.sparse.linalg as ssp
import scipy.linalg as sp


def UKF(dt,x,P,Q,z,R):
    xq = x[0:4]
    xw = x[4::]
    dq = axang2quat(xw,dt)

    S = sp.sqrtm(2*len(P[0,:])*(P+Q))
    Sq1 = S[0:3,0:3]
    Sq2 = S[0:3,3::]
    Wq1 = axang2quat(Sq1)
    Wq2 = axang2quat(Sq2)
    Ww = S[3::,::]

    Xqp1 = quatMult(xq,Wq1)
    Xqp2 = quatMult(xq,Wq2)
    Xqm1 = quatMult(xq,-Wq1)
    Xqm2 = quatMult(xq,-Wq2)
    Xwp = xw[:,np.newaxis] + Ww
    Xwm = xw[:,np.newaxis] - Ww

    Yqp1 = quatMult(Xqp1,dq)
    Yqp2 = quatMult(Xqp2,dq)
    Yqm1 = quatMult(Xqm1,dq)
    Yqm2 = quatMult(Xqm2,dq)
    Yq = np.concatenate((Yqp1,Yqp2,Yqm1,Yqm2),axis=1)
    Yw = np.concatenate((Xwp,Xwm),axis=1)
    Xq_ = quat2axang(Yq)
    xq_ = np.mean(Xq_,axis=1)
    W_ = Xq_ - xq_[:,np.newaxis]
    xw_ = np.mean(Yw,axis=1)
    W_ = np.concatenate((W_,Yw-xw_[:,np.newaxis]),axis=0)
    Pk_ = np.dot(W_,np.transpose(W_))/(2*len(W_[:,0]))
    x_ = np.concatenate((xq_,xw_),axis=0)

    g = np.array([0,0,0,9.80665])
    Zqp1 = quatMult(quatMult(Yqp1,g),quatCong(Yqp1))
    Zqp2 = quatMult(quatMult(Yqp2,g),quatCong(Yqp2))
    Zqm1 = quatMult(quatMult(Yqm1,g),quatCong(Yqm1))
    Zqm2 = quatMult(quatMult(Yqm2,g),quatCong(Yqm2))
    Zq = np.concatenate((Zqp1,Zqp2,Zqm1,Zqm2),axis=1)
    Zq_ = quat2axang(Zq)
    zq_ = np.mean(Zq_,axis=1)
    Z_ = np.concatenate((Zq_,Yw),axis=0)
    z_ = np.concatenate((zq_,xw_),axis=0)
    v = z - z_

    Pzz = np.dot(Z_-z_[:,np.newaxis],np.transpose(Z_-z_[:,np.newaxis]))/(2*len(W_[:,0]))
    Pvv = Pzz + R
    Pxz = np.dot(W_,np.transpose(Z_-z_[:,np.newaxis]))/(2*len(W_[:,0]))
    K = np.matmul(Pxz,np.linalg.inv(Pvv))
    xk = x_ + np.dot(K,v)
    Pk = Pk_ - np.matmul(np.matmul(K,Pvv),np.transpose(K))
    xkq = axang2quat(xk[0:3])
    # xk = np.concatenate((xkq,xk[3::]))
    xk = np.concatenate((xkq,z[3::]))

    return xk,Pk


def quatMult(q1,q2, normalize = False):
    u0 = q1[0]
    v0 = q2[0]
    u = q1[1::]
    v = q2[1::]
    q0 = u0*v0 - (u*v).sum(axis=0) # np.dot(u,v)
    # print(u)
    # print(v0)
    # print(np.multiply(v0,u))
    # print(np.cross(u,v,axis=0))
    # q0 = u0*v0 - np.inner(u,v.T)
    q = u0*v + v0*u + np.cross(u,v,axis=0)
    q = np.array([q0,q[0],q[1],q[2]])
    if normalize:
        q = q/np.linalg.norm(q)
    return q


def quatCong(q):
    if q.size > 4:
        p = np.array([[1],[-1],[-1],[-1]])
    else:
        p = np.array([1,-1,-1,-1])
    return np.multiply(q,p)


def quat2rot(q):
    u0 = q[0]
    u = q[1::]
    R = (u0**2 - np.inner(np.transpose(u),u))*np.eye(3) + 2*u0*vec2skew(u) + 2*np.outer(u,np.transpose(u))
    return R.real


def rot2quat(R):
    angle = np.arccos((np.trace(R) - 1)/2)
    eigVals, eigVec = ssp.eigs(R, k=1, sigma=1)
    axis = np.array([eigVec[0,0],eigVec[1,0],eigVec[2,0]])
    q = axang2quat(axis*angle)
    return  q


def eul2axang(roll,pitch,yaw):
    R = eul2rot(roll,pitch,yaw)
    angle = np.arccos((np.trace(R) - 1)/2)
    axis = skew2vec(R-np.transpose(R)/(2*np.sin(angle)))
    # print(angle)
    # eigVals, eigVec = ssp.eigs(R, k=1, sigma=1)
    # axis = np.array([eigVec[0,0],eigVec[1,0],eigVec[2,0]])
    return axis*angle


def quat2eul(q):
    R = quat2rot(q)
    return rot2eul(R.real)


def eul2rot(roll,pitch,yaw):
    phi   = roll
    theta = pitch
    psi   = yaw
    # Rz(yaw)Ry(pitch)Rx(roll)
    R = np.array([[np.multiply(np.cos(psi),np.cos(theta)) - np.multiply(np.multiply(np.sin(phi),np.sin(psi)),np.sin(theta)), -np.multiply(np.cos(phi),np.sin(psi)), np.multiply(np.cos(psi),np.sin(theta)) + np.multiply(np.multiply(np.cos(theta),np.sin(phi)),np.sin(psi))],
         [np.multiply(np.cos(theta),np.sin(psi)) + np.multiply(np.multiply(np.cos(psi),np.sin(phi)),np.sin(theta)),  np.multiply(np.cos(phi),np.cos(psi)), np.multiply(np.sin(psi),np.sin(theta)) - np.multiply(np.multiply(np.cos(psi),np.cos(theta)),np.sin(phi))],
         [-np.multiply(np.cos(phi),np.sin(theta)), np.sin(phi), np.multiply(np.cos(phi),np.cos(theta))]])
    return R


def rot2eul(R):
    if R[2,1] < 1:
        if R[2,1] > -1:
            roll = np.arcsin(R[2,1])
            yaw = np.arctan2(-R[0,1], R[1,1])
            pitch = np.arctan2(-R[2,0], R[2,2]);
        else: # R(3,2) == -1
            roll = -np.pi/2;
            yaw = -np.arctan2(R[0,2],R[0,0])
            pitch = 0
    else: # R(3,2) == +1
        roll = np.pi/2
        yaw = np.arctan2(R[0,2],R[0,0])
        pitch = 0
    return roll, pitch, yaw


def axang2quat(w,t=None):
    if t is None: 
        if w.size > 3:
            angle = np.linalg.norm(w,axis=0)
            a = np.copy(angle)
            a[a==0] = 1
            axis = w/a
            u = np.multiply(np.sin(angle/2),axis)
            return np.array([np.cos(angle/2),u[0,:],u[1,:],u[2,:]])
        else:
            angle = np.linalg.norm(w)
            axis = w/angle
    else:
        angle = np.multiply(np.linalg.norm(w),t)
        axis = np.divide(w,np.linalg.norm(w))  
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
    # q1 = np.array([[0.5**0.5, 1, 0,0.5**0.5 ],[0, 0, 1, 0.5**0.5],[0.5**0.5, 0, 0, 0],[0, 0, 0, 0]])
    q2 = np.array([0.5**0.5, 0, 0.5**0.5, 0])
    # q = quatMult(q1,q2)

    # print(q2[0,:])
    r = quat2rot(q2)
    q = rot2quat(r)
    print(r)
    print(q)
    # roll,pitch,yaw = rot2eul(r)
    # r2 = eul2rot(roll,pitch,yaw)
    # Q = np.arange(4).reshape(2,2)
    # print(Q)
    # P = np.dot(Q,np.transpose(Q))
    # print(P)

    # q1 = np.array([1,0,0,1])
    # q2 = np.array([[0,1,0,0],[1,0,0,0],[0,0,1,1]])
    # print(q1)
    # print(q2)
    # print(q1*q2)
    