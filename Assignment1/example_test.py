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
    # size = 0
    # print(cmap.size())
    belief = np.full((20, 20), 1.0/400, dtype=float)
    H = HistogramFilter()

    for i in range(len(actions)):
        # print(i)
        belief = H.histogram_filter(cmap, belief, actions[i], observations[i])
        p = np.amax(belief)
        idx = np.unravel_index(np.argmax(belief), belief.shape)
        belief_state = np.asarray(idx)
        # print(belief)
        print('estimated',belief_state)
        print('actual',belief_states[i])

    # print(cmap)
    # print(actions)
    # print(observations)
    # print(belief_states)