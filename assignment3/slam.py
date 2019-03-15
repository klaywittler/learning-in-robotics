from __future__ import division
from scipy.stats import chi2
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
    R = np.diag([sigmas['xy']**2,sigmas['xy']**2,sigmas['phi']**2])

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
    H = np.zeros((2,3+2*ekf_state['num_landmarks']))
    H[0:2,0:2] = np.eye(2)
    Q = np.diag([sigmas['gps']**2,sigmas['gps']**2])
    P = ekf_state['P']

    r = gps-ekf_state['x'][0:2]
    Sinv = np.linalg.inv(np.matmul(np.matmul(H,P),H.T)+Q.T)

    if mahalanobisDist(r,Sinv) <= 13.816: 
        K = np.matmul(np.matmul(P,H.T),Sinv)
        ekf_state['x'] = ekf_state['x'] + np.dot(K,r)
        ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
        ekf_state['P'] = slam_utils.make_symmetric(np.matmul((np.eye(P.shape[0]) - np.matmul(K,H)),P))
    
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
    L = x[3 + 2*landmark_id:5 + 2*landmark_id]
    H = np.zeros((2,3+2*ekf_state['num_landmarks']))

    dh1dx = (x[0]-L[0])/np.linalg.norm(L - x[0:2])
    dh1dy = (x[1]-L[1])/np.linalg.norm(L - x[0:2])
    dh2dx = (1.0/(1.0+((L[1]-x[1])/(L[0]-x[0]))**2))*((L[1]-x[1])/(L[0]-x[0])**2) # (L[1]-x[1])/((L[0]-x[0])**2 + (L[1]-x[1])**2) # 
    dh2dy = (1.0/(1.0+((L[1]-x[1])/(L[0]-x[0]))**2))*(1.0/(L[0]-x[0])**2)

    H[:,0:3] = np.array([[dh1dx, dh1dy, 0], [dh2dx, dh2dy, -1.0]])

    dh1dxL = -dh1dx # (L[0]-x[0])/np.linalg.norm(L - x[0:2]) # 
    dh1dyL = -dh1dy # (L[1]-x[1])/np.linalg.norm(L - x[0:2]) # 
    dh2dxL = -dh2dx # (x[1]-L[1])/((L[0]-x[0])**2 + (L[1]-x[1])**2) # 
    dh2dyL = -dh2dy # 1.0/((1.0+((L[1]-x[1])/(L[0]-x[0]))**2)*(L[0]-x[0])**2) # 

    H[:,3 + 2*landmark_id:5 + 2*landmark_id ] = np.array([[dh1dxL, dh1dyL], [dh2dxL, dh2dyL]])

    zhat = np.array([np.linalg.norm(L - x[0:2]), slam_utils.clamp_angle(np.arctan2((L[1]-x[1]),(L[0]-x[0])) - x[2])])

    return zhat, H


def initialize_landmark(ekf_state, tree):
    '''
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''
    ekf_state['num_landmarks'] += 1
    x = ekf_state['x']
    xL = np.array([tree[0]*np.cos(tree[1]+x[2]) + x[0] , tree[0]*np.sin(tree[1]+x[2]) + x[1]])
    ekf_state['x'] = np.concatenate((x,xL) ,axis=0)
    N = ekf_state['P'].shape
    pTemp = 10.0**3*np.eye((N[0]+2))
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
        B = 5.991*np.ones((len(measurements),len(measurements)))
        M = np.zeros((len(measurements),ekf_state['num_landmarks'])) 
        Q = np.diag([sigmas['range']**2,sigmas['bearing']**2])
        P = ekf_state['P']
        Zm = np.array(measurements)[:,0:2]
        for i in range(ekf_state['num_landmarks']):
            zhat, H = laser_measurement_model(ekf_state, i)
            Sinv = np.linalg.inv(np.matmul(np.matmul(H,P),H.T)+Q.T)
            r = Zm - zhat
            M[:,i] = mahalanobisDist(r,Sinv)

        Mpad = np.concatenate((M,B),axis=1)
        C = slam_utils.solve_cost_matrix_heuristic(Mpad)
        assoc = [-2]*len(measurements)
        for c in C:
            if c[1] < ekf_state['num_landmarks']:
                assoc[c[0]] = c[1]
            elif np.min(M[c[0],:]) > 9.21:
                assoc[c[0]] = -1
 
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
    Q = np.diag([sigmas['range']**2,sigmas['bearing']**2])
    for i, t in enumerate(trees):
        j = assoc[i]
        if j == -1:
            ekf_state = initialize_landmark(ekf_state, t)
            j = ekf_state['num_landmarks'] - 1
        
        if j >= 0: 
            P = ekf_state['P']
            z = np.array(t[0:2])
            zhat, H = laser_measurement_model(ekf_state, j)

            r = z-zhat
            Sinv = np.linalg.inv(np.matmul(np.matmul(H,P),H.T)+Q.T)

            K = np.matmul(np.matmul(P,H.T),Sinv)

            ekf_state['x'] = ekf_state['x'] + np.matmul(K,r)
            ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
            ekf_state['P'] = slam_utils.make_symmetric(np.matmul((np.eye(P.shape[0]) - np.matmul(K,H)),P))

    return ekf_state


def mahalanobisDist(r,Sinv):
    if len(r.shape) > 1:
        r = r.reshape((r.shape[0],2,1))
        d = np.matmul(np.matmul(np.transpose(r,axes=(0,2,1)),Sinv[np.newaxis,:,:]),r)
        return d[:,0,0] 
    else:
        d = np.matmul(np.matmul(r.T,Sinv),r)
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
            # print("state size = {}".format(ekf_state['x'].shape))

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
            if not trees:
                pass
            else:
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
        "plot_map_covariances": False,
        "plot_vehicle_covariances": True,
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
        "range": 0.5,
        "bearing": 5*np.pi/180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array( [gps[0,1], gps[0,2], 36*np.pi/180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0
    }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)

if __name__ == '__main__':
    main()
