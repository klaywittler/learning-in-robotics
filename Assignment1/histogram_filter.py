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

        if action[0] != 0:
            if action[0] == 1:
                # print('right')
                M = np.zeros((20,20))
                b = np.full((1,19),0.9, dtype=float)[0]
                M = np.diag(b,1)
                np.fill_diagonal(M,0.1)
                M[-1][-1] = 1
            elif action[0] == -1:
                # print('left')
                M = np.zeros((20,20))
                b = np.full((1,19),0.9, dtype=float)[0]
                M = np.diag(b,1)
                np.fill_diagonal(M,0.1)
                M[-1][-1] = 1
                M=np.flip(M,1)
            belief = belief*M
        elif action[0] == 0:
            if action[1] != 0:
                if action[1] == 1:
                    # print('up')
                    M = np.zeros((20,20))
                    b = np.full((1,19),0.9, dtype=float)[0]
                    M = np.diag(b,1)
                    np.fill_diagonal(M,0.1)
                    M[-1][-1] = 1
                elif action[1] == -1:
                    # print('down')
                    M = np.zeros((20,20))
                    b = np.full((1,19),0.9, dtype=float)[0]
                    M = np.diag(b,1)
                    np.fill_diagonal(M,0.1)
                    M[-1][-1] = 1
                    M=np.flip(M,1)
                belief_T = belief.T*M
                belief = belief_T.T
            elif action[1] == 0:
                M = 0


        belief_update = np.random.rand(20, 20)

        return belief

        