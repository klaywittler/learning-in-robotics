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
    motion, G = motion_model(u, dt, ekf_state, vehicle_params)
    R = np.diag([sigmas['xy'],sigmas['xy'],sigmas['phi']])

    ekf_state['x'][0:3] = ekf_state['x'][0:3] + motion
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])

    ekf_state['P'][0:3,0:3] = np.matmul(np.matmul(G,ekf_state['P'][0:3,0:3]),G.T) + R
    ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'])

    return ekf_state


def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''
    # print('x: ', x, ' gps: ', gps)
    H = np.array([[1.0,0,0],[0,1.0,0]])
    Q = np.diag([sigmas['gps'],sigmas['gps']])
    P = ekf_state['P'][0:3,0:3]

    r = gps[::-1]-ekf_state['x'][0:2]
    Sinv = np.linalg.inv(np.matmul(np.matmul(H,P),H.T)+Q.T)

    if mahalanobisDist(r,Sinv) <= 13.816:
        K = np.matmul(np.matmul(P,H.T),Sinv)
        ekf_state['x'][0:3] = ekf_state['x'][0:3] + np.dot(K,r)
        ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
        ekf_state['P'][0:3,0:3] = slam_utils.make_symmetric(np.matmul((np.eye(3) - np.matmul(K,H)),P))
    
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
    x = ekf_state['x']
    xL = x[3 + 2*(landmark_id - 1):3 + 2*(landmark_id - 1) +2]
    H = np.zeros((2,3+2*ekf_state['num_landmarks']))

    H[:,0:3] = np.array([[(x[0]-xL[0])/np.linalg.norm(xL - x[0:2]), (x[1]-xL[1])/np.linalg.norm(xL - x[0:2]), 0],
                             [(xL[1]-x[1])/((xL[0]-x[0])**2 + (xL[1]-x[1])**2), -1.0/((1+((xL[1]-x[1])/(xL[0]-x[0]))**2)*(xL[0]-x[0])), -1]])

    H[:,3 + 2*(landmark_id - 1):3 + 2*(landmark_id - 1) +2] = np.array([[(xL[0]-x[0])/np.linalg.norm(xL - x[0:2]), (xL[1]-x[1])/np.linalg.norm(xL - x[0:2])],
                                                                [(x[1]-xL[1])/((xL[0]-x[0])**2 + (xL[1]-x[1])**2), 1.0/((1+((xL[1]-x[1])/(xL[0]-x[0]))**2)*(xL[0]-x[0]))]])

    # zhat = np.array([np.linalg.norm(xL - x[0:2]), np.arctan2((xL[1]-x[1]),(xL[0]-x[0])) - x[2] + np.pi/2.0])
    zhat = np.array([np.linalg.norm(xL - x[0:2]), np.arctan2((xL[1]-x[1]),(xL[0]-x[0])) - x[2]])

    return zhat, H


def initialize_landmark(ekf_state, tree):
    '''
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''
    ekf_state['num_landmarks'] += 1
    ekf_state['x'] = np.concatenate((ekf_state['x'],tree),axis=0)
    N = ekf_state['P'].shape
    pTemp = 0.1*np.eye((N[0]+2))
    pTemp[:-2,:-2] = ekf_state['P']
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
    else:
        x = ekf_state['x']
        M = np.zeros((ekf_state['num_landmarks'],len(measurements))) 
        Q = np.diag([sigmas['range'],sigmas['bearing']])
        P = ekf_state['P']
        Zm = np.array(measurements)[:,0:2]
        Zm_xy = slam_utils.tree_to_global_xy(Zm, ekf_state).T
        for i in range(ekf_state['num_landmarks']):
            zhat, H = laser_measurement_model(ekf_state, i+1)
            ztest = np.array([zhat[0]*np.cos(zhat[1]+x[2]) + x[0] , zhat[0]*np.sin(zhat[1]+x[2]) + x[1]])
            Sinv = np.linalg.inv(np.matmul(np.matmul(H,P),H.T)+Q.T)
            # Sinv = np.linalg.inv(H @ P @ H.T + Q.T)
            # r = Zm - zhat
            print(Zm.shape)
            r = Zm_xy - ztest
            M[i,:] = mahalanobisDist(r,Sinv)

        C = slam_utils.solve_cost_matrix_heuristic(np.copy(M))
        assoc = [-1]*(len(measurements))
        for v in C:
            if M[v] > 13.816:
                assoc[v[1]] = -1
            elif M[v] > 5.991:
                assoc[v[1]] = -2
            else:
                assoc[v[1]] = v[0] + 1

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
    # print('laser')
    measurements = slam_utils.tree_to_global_xy(trees, ekf_state)
    # print(measurements)
    for i, v in enumerate(assoc):
        if v == -1:
            ekf_state = initialize_landmark(ekf_state, measurements[:,i])
        elif v > 0:
            x = ekf_state['x']
            z = np.array([trees[i][0],trees[i][1]])
            zhat, H = laser_measurement_model(ekf_state, v)
            # xL = tree[0]*np.cos(tree[1]+x[2]) + x[0]
            # yL = tree[0]*np.sin(tree[1]+x[2]) + x[1]
            ztest = np.array([zhat[0]*np.cos(zhat[1]+x[2]) + x[0] , zhat[0]*np.sin(zhat[1]+x[2]) + x[1]])
            # print(ztest)
        
            Q = np.diag([sigmas['range'],sigmas['bearing']])
            P = ekf_state['P']

            r = measurements[:,i]-ztest
            Sinv = np.linalg.inv(np.matmul(np.matmul(H,P),H.T)+Q.T)

            K = np.matmul(np.matmul(P,H.T),Sinv)

            ekf_state['x'] = x + np.dot(K,r)
            ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
            ekf_state['P'] = slam_utils.make_symmetric(np.matmul((np.eye(P.shape[0]) - np.matmul(K,H)),P))

    return ekf_state


def mahalanobisDist(r,Sinv):
    if len(r.shape) > 1:
        r = r.reshape((r.shape[0],2,1))
        d = np.matmul(np.matmul(np.transpose(r,axes=(0,2,1)),Sinv[np.newaxis,:,:]),r)
        return d[:,0,0] 
    else:
        d = np.dot(np.dot(r.T,Sinv),r)
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
        "plot_raw_laser": False,
        "plot_map_covariances": False,
        "plot_vehicle_covariances": False,
        "plot_map": True,
        # "plot_tree_measurements": True

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
    # p = np.array([[0,1,6],[2,3,6]])
    # d = np.diag([p.shape)
    # print(p[0:2,0:2])
    main()
