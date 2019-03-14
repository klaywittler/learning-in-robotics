import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        Use starter.npz data given in /data to debug and test your code. starter.npz has sequence of actions and 
        corresponding observations. True belief is also given to compare your filters results with actual results. 
        cmap = arr_0, actions = arr_1, observations = arr_2, true state = arr_3
        '''

        ### Your Algorithm goes Below.

        M = np.copy(cmap.astype(float))
        if observation != 0:
            # print('see color')    
            M[M == 1] = 0.9 # probability that sensor it correct given it has seen a color
            M[M == 0] = 0.1 # probability that sensor it wrong given it has seen a color
        elif observation == 0:
            # print('no color')
            M[M == 1] = 0.1 # probability that sensor it wrong given it didn't see a color
            M[M == 0] = 0.9 # probability that sensor it correct given it didn't see a color

        # 90% probability to move, 10% probability to stay stationary 
        m = np.shape(M)
        T = np.zeros(m)
        b = np.full((1,m[0]-1),0.9, dtype=float)[0]
        T = np.diag(b,1)
        np.fill_diagonal(T,0.1)
        T[-1][-1] = 1 # right movement action transition matrix - 0.1 across diagonal with 0.9 in spot to the right

        if action[0] != 0: # right
            if action[0] == -1: # left
                T=np.flip(T,1)
                T= np.flipud(T)  # left movement is a flipped version of right transition matrix
            belief_T = np.dot(belief,T) # action update
            belief = np.multiply(M,belief_T) # measurement update
        elif action[0] == 0: 
            if action[1] != 0:  # down
                # down movements correspond to premultiplying the traspose of the right transition matrix
                # up movements correspond to premultiplying the traspose of the left transition matrix
                if action[1] == 1: # up
                    T=np.flip(T,1)
                    T= np.flipud(T) # construction the movement left transition matrix
                belief_T = np.dot(np.transpose(T),belief) # action update
                belief = np.multiply(M, belief_T) # measurement update
            elif action[1] == 0: # no movement case
                T = np.eye(m[0]-1,m[0]-1)
                belief = np.multiply(M,np.dot(belief,T))

        # normalization
        n = belief.sum(dtype=float)
        belief = belief/n

        # maximum likelihood estimate
        idx = np.unravel_index(np.argmax(belief), belief.shape)
        belief_state = np.asarray(idx)
        belief_state = [belief_state[1],m[0]-1-belief_state[0]]

        return belief, belief_state

        