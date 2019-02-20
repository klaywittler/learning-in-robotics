import numpy as np
import scipy.sparse.linalg as ssp
import scipy.linalg as sp


def UKF(dt,x,P,Q,z,R):
    nrm = False
    xq = x[0:4]
    xw = x[4::]
    dq = axang2quat(xw,dt,normalize=nrm)
    n = len(P[0,:])

    # getting sigma points
    S = sp.cholesky(0.01*n*(P+Q))
    Sq = S[0:3,:]
    Sw = S[3::,:]
    Sq = np.concatenate((Sq,-Sq),axis=1)
    Wq = axang2quat(Sq,normalize=nrm)
    Ww = np.concatenate((Sw,-Sw),axis=1)
    Xq = quatMult(xq,Wq,normalize=nrm)

    # projecting forward sigma points and recharacterizing distribution
    Yq = quatMult(Xq,dq,normalize=nrm)
    Yw = xw[:,np.newaxis] + Ww
    xq_k, xeVec = quatMean(Yq,xq)
    xw_k = np.mean(Yw,axis=1)
    W_k = np.concatenate((xeVec,Ww),axis=0)
    P_k = np.dot(W_k,np.transpose(W_k))/(2.0*n)

    # expected measurement characterization
    g = np.array([0,0,0,9.80665])
    Zq = quatMult(quatMult(Yq,g),quatCong(Yq))
    Zq_k = Zq[1::,:]
    # print(Zq_k)
    zq_k = np.mean(Zq_k,axis=1)
    # print(zq_k)
    # print(np.linalg.norm(zq_k,axis=0))
    Z_k = np.concatenate((Zq_k-zq_k[:,np.newaxis],Ww),axis=0)
    z_k = np.concatenate((zq_k,xw_k),axis=0)

    v = z - z_k
    # print(np.linalg.norm(Zq,axis=0))
    # print(np.linalg.norm((quat2axang(zq_k))))
    
    # Covariance update
    Pzz = np.dot(Z_k,np.transpose(Z_k))/(2.0*n)
    Pvv = Pzz + R
    Pxz = np.dot(W_k,np.transpose(Z_k))/(2.0*n)

    #Kalman gain update
    K = np.matmul(Pxz,np.linalg.inv(Pvv))
    Kp = np.dot(K,v)
    Kq = axang2quat(Kp[0:3])
    xkq = quatMult(xq_k,Kq, normalize=nrm)
    xkw = xw_k + Kp[3::]
    Pk = P_k - np.matmul(np.matmul(K,Pvv),np.transpose(K))
    xk = np.concatenate((xkq,xkw))

    return xk,Pk


def quatMean(Yq,xq,normalize=False):
    qBar = quatCong(xq)
    error = 1.0
    count = 0
    while error >= 10**-2:
        Eq = quatMult(Yq,qBar,normalize=normalize)
        eVec = quat2axang(Eq)
        eMean = np.mean(eVec, axis=1)
        error = np.linalg.norm(eMean,axis=0)
        eq = axang2quat(eMean)
        qBar = quatMult(eq,qBar,normalize=normalize)
        count += 1
        if count >= 100:
            break
    return qBar, eVec


def quatMult(q1,q2, normalize=False):
    qw = q2[0]*q1[0] - q2[1]*q1[1] - q2[2]*q1[2] - q2[3]*q1[3]
    qi = q2[0]*q1[1] + q2[1]*q1[0] - q2[2]*q1[3] + q2[3]*q1[2]
    qj = q2[0]*q1[2] + q2[1]*q1[3] + q2[2]*q1[0] - q2[3]*q1[1]
    qk = q2[0]*q1[3] - q2[1]*q1[2] + q2[2]*q1[1] + q2[3]*q1[0]
    q = np.array([qw,qi,qj,qk])
    if normalize:
        q = q/np.linalg.norm(q,axis=0)
    return q


def quatCong(q):
    if q.size > 4:
        p = np.array([[1.0],[-1.0],[-1.0],[-1.0]])
    else:
        p = np.array([1.0,-1.0,-1.0,-1.0])
    return np.multiply(q,p)


def quat2rot(q):
    u0 = q[0]
    u = q[1::]
    R = (u0**2 - np.inner(np.transpose(u),u))*np.eye(3) + 2.0*u0*veemap(u) + 2.0*np.outer(u,np.transpose(u))
    return R


def rot2quat(R):
    angle = np.arccos((np.trace(R) - 1)/2.0)
    eigVals, eigVec = ssp.eigs(R, k=1, sigma=1)
    axis = np.array([eigVec[0,0],eigVec[1,0],eigVec[2,0]])
    q = axang2quat(axis*angle)
    q = q/np.linalg.norm(q,axis=0)
    return  q


def eul2axang(roll,pitch,yaw):
    R = eul2rot(roll,pitch,yaw)
    angle = np.arccos((np.trace(R) - 1)/2.0)
    axis = veemap(R-np.transpose(R)/(2.0*np.sin(angle)))
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
            roll = -np.pi/2.0
            yaw = -np.arctan2(R[0,2],R[0,0])
            pitch = 0
    else: # R(3,2) == +1
        roll = np.pi/2.0
        yaw = np.arctan2(R[0,2],R[0,0])
        pitch = 0
    return roll, pitch, yaw


def axang2quat(w,t=None, normalize=False):
    if t is None: 
        if w.size > 3:
            angle = np.linalg.norm(w,axis=0)
            a = np.copy(angle)
            a[a==0] = 1.0
            axis = w/a
            u = np.multiply(np.sin(angle/2.0),axis)
            q = np.array([np.cos(angle/2.0),u[0,:],u[1,:],u[2,:]])
            if normalize:
                q = q/np.linalg.norm(q,axis=0)
            return q
        else:
            angle = np.linalg.norm(w)
            axis = w/angle
    else:
        angle = np.multiply(np.linalg.norm(w),t)
        a = np.linalg.norm(w,axis=0)
        axis = np.divide(w,a)  
    u = np.sin(angle/2.0)*axis
    q = np.array([np.cos(angle/2.0),u[0],u[1],u[2]])
    if normalize:
        q = q/np.linalg.norm(q,axis=0)
    return q


def quat2axang(q):
    angle = 2.0*np.arccos(q[0])
    axis = q[1::]/np.sqrt(1-np.power(q[0],2))
    return axis*angle


def veemap(x):
    if x.size <= 3:
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    else:
        return np.array([x[1,2], x[0,2],x[1,0]])    


if __name__ == "__main__":
    q1 = np.array([[0.5**0.5, 1, 0,0.5**0.5 ],[0, 0, 1, 0.5**0.5],[0.5**0.5, 0, 0, 0],[0, 0, 0, 0]])
    q2 = np.array([0.5**0.5, 0, 0.5**0.5, 0])
    print(10**-2)
    # print(q2[0,:])
    # r = quat2rot(q2)
    # q = rot2quat(r)
    # print(r)
    # print(q1)
    # print(q2)
    # print(q)
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
    # a = np.array([0,2.0,0,2],[])
    # a[a==0] = 1
    # print(a)