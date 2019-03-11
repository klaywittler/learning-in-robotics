from __future__ import division
import numpy as np
import slam_utils
import tree_extraction

def motion_model(u, dt, ekf_state, vehicle_params):
    '''
    Computes the discretized motion model for the given vehicle as well as its Jacobian

    Returns:
        f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.

        df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi)
    '''

    a = vehicle_params['a']
    b = vehicle_params['b']
    L = vehicle_params['L']
    H = vehicle_params['H']
    ve = u[0]
    alpha = u[1]
    vc = ve/(1.0-np.tan(alpha)*(H/L))
    x = ekf_state['x']

    motion = np.array([dt*(vc*np.cos(x[2]) - (vc/L)*np.tan(alpha)*(a*np.sin(x[2])+b*np.cos(x[2]))),
                        dt*(vc*np.sin(x[2]) + (vc/L)*np.tan(alpha)*(a*np.cos(x[2])-b*np.sin(x[2]))),
                        dt*(vc/L)*np.tan(alpha)])

    G = np.array([[1.0, 0, dt*(-vc*np.sin(x[2]) - (vc/L)*np.tan(alpha)*(a*np.cos(x[2])-b*np.sin(x[2])))],
                    [0, 1.0, dt*(vc*np.cos(x[2]) + (vc/L)*np.tan(alpha)*(-a*np.sin(x[2])-b*np.cos(x[2])))],
                    [0,0,1.0]])

    return motion, G

def odom_predict(u, dt, ekf_state, vehicle_params, sigmas):
    '''
    Perform the propagation step of the EKF filter given an odometry measurement u 
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.

    Returns the new ekf_state.
    '''
    # print('odom')
    motion, G = motion_model(u, dt, ekf_state, vehicle_params)
    R = np.diag([sigmas['xy'],sigmas['xy'],sigmas['phi']])

    ekf_state['x'] = ekf_state['x'] + motion
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
    ekf_state['P'] = slam_utils.make_symmetric(np.matmul(np.matmul(G,ekf_state['P']),G.T) + R)

    return ekf_state


def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''
    # print('gps')
    x = ekf_state['x']
    z = np.array([gps[0],gps[1],0])
    H = np.diag([1.0,1.0,0])
    Q = np.diag([sigmas['gps'],sigmas['gps'],1.0])
    P = ekf_state['P']
    S = np.matmul(np.matmul(H,P),H.T)+Q.T
    r = z-np.dot(H,x)

    if mahalanobisDist(r,S) <= 13.8:
        K = np.matmul(np.matmul(P,H.T),np.linalg.inv(S))
        ekf_state['x'] = x + np.dot(K,r)
        ekf_state['P'] = slam_utils.make_symmetric(np.matmul((np.eye(3) - np.matmul(K,H)),P))
    
    return ekf_state

def laser_measurement_model(ekf_state, landmark_id):
    ''' 
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian. 

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
    '''
    
    ###
    # Implement the measurement model and its Jacobian you derived
    ###
    a = vehicle_params['a']
    b = vehicle_params['b']
    L = vehicle_params['L']
    H = vehicle_params['H']
    ve = u[0]
    alpha = u[1]
    vc = ve/(1-np.tan(alpha)*(H/L))
    x = ekf_state['x']
    xL = x[3 + 2*(landmark_id - 1):3 + 2*(landmark_id - 1) +1]
    H = np.zeros((2,3+2*ekf_state['num_landmarks']))

    H[0:3,:] = np.array([[(x[0]-xL[0])/np.linalg.norm(xL - x[0:2]), (x[1]-xL[1])/np.linalg.norm(xL - x[0:2]), 0],
                             [(xL[1]-x[1])/((xL[0]-x[0])**2 + (xL[1]-x[1])**2), -1.0/((1+((xL[1]-x[1])/(xL[0]-x[0]))**2)*(xL[0]-x[0])), -1]])

    H[3 + 2*(landmark_id - 1):3 + 2*(landmark_id - 1) +1,:] = np.array([[(xL[0]-x[0])/np.linalg.norm(xL - x[0:2]), (xL[1]-x[1])/np.linalg.norm(xL - x[0:2])],
                                                                [(x[1]-xL[1])/((xL[0]-x[0])**2 + (xL[1]-x[1])**2), 1.0/((1+((xL[1]-x[1])/(xL[0]-x[0]))**2)*(xL[0]-x[0]))]])

    zhat = np.array([np.linalg.norm(xL - x[0:2]), np.arctan2((xL[1]-x[1])/(xL[0]-x[0])) - x[2] + np.pi/2.0])

    return zhat, H

def initialize_landmark(ekf_state, tree):
    '''
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''
    print('init')
    ekf_state['num_landmarks'] += 1
    x = ekf_state['x']
    xL = np.sqrt((tree[0]**2 - np.tan(tree[1]+x[2]-np.pi/2)**2)/2.0) + x[0]
    yL = np.tan(tree[1]+x[2]-np.pi/2)*(xL-x[0]) + x[1]
    ekf_state['x'] = np.concatenate((x,np.array([xL,yL])),axis=0)
    N = ekf_state['P'].shape
    pTemp = np.zeros((N[0]+1,N[1]+1))
    pTemp[:-1,:-1] = ekf_state['P']
    pTemp[N,N] = 10.0
    ekf_state['P'] = slam_utils.make_symmetric(pTemp)

    return ekf_state

def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from 
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''
    # print('assoc')
    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for m in measurements]

    ###
    # Implement this function.
    ###
    for i in range(ekf_state['num_landmarks']):
        z, H = laser_measurement_model(ekf_state, landmark_id)
        Q = np.diag([sigmas['range'],sigmas['bearing']])
        P = ekf_state['P']
        S = np.matmul(np.matmul(H,P),H.T)+Q.T

    slam_utils.solve_cost_matrix_heuristic(M)

    return assoc

def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    Perform a measurement update of the EKF state given a set of tree measurements.

    trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    in the state for measurement i. If assoc[i] == -2, discard the measurement as 
    it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''

    ###
    # Implement the EKF update for a set of range, bearing measurements.
    ###
    # print('laser')

    measurements = [tree[0:2] for tree in trees]
    assoc = compute_data_association(ekf_state,measurements,sigmas,params)
    # ekf_state1 = [initialize_landmark(ekf_state,measurements[x]) for i, x in enumerate(np.where(np.array(assoc) == -1)[0])]

    # x = ekf_state['x']
    # z = np.array([gps[0],gps[1],0])
    # H = np.diag([1.0,1.0,0])
    # Q = np.diag([sigmas['xy'],sigmas['xy'],sigmas['phi']])
    # P = ekf_state['P']
    # S = np.matmul(np.matmul(H,P),H.T)+Q.T
    # r = z-np.dot(H,x)

    # if mahalanobisDist(r,S): 
    #     K = np.matmul(np.matmul(P,H.T),np.linalg.inv(S))
    #     ekf_state['x'] = x + np.dot(K,r)
    #     ekf_state['P'] = slam_utils.make_symmetric(np.matmul((np.eye(3) - np.matmul(K,H)),P))

    return ekf_state


def mahalanobisDist(r,S):
    d = np.dot(np.dot(r.T,np.linalg.inv(S)),r)
    return d


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }
    
    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)

        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t

        else:
            # Laser
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)

        
        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3,:3])))
        state_history['t'].append(t)

    return state_history


def main():
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key = lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50, 
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75, # meters

        # general...
        "do_plot": True,
        "plot_raw_laser": True,
        "plot_map_covariances": True

        # Add other parameters here if you need to...
    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5*np.pi/180,

        # Measurement noise
        "gps": 3,
        "range": 0.4,
        "bearing": 3*np.pi/180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array( [gps[0,1], gps[0,2], 36*np.pi/180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0
    }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)

if __name__ == '__main__':
    # p = np.array([[0,1],[2,3]])
    # d = np.diag([p.shape)
    # print(p.shape[0])
    main()
