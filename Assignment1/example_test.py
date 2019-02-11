import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']


    #### Test your code here
    m = np.shape(cmap)
    belief = np.full(m, 1.0/(m[0]*m[1]), dtype=float)
    H = HistogramFilter()

    for i in range(len(actions)):
        # print(i)
        belief,belief_state = H.histogram_filter(cmap, belief, actions[i], observations[i])
        
        print('estimated',belief_state)
        # print('actual',belief_states[i])


    # print(cmap)
    # print(actions)
    # print(observations)
    # print(belief_states)