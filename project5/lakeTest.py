import math as m
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import *
import lake_env
from tester import Tester

if __name__ == '__main__':
    ##############################
    # Policy Iteration/Value #
    ##############################
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    tester = Tester()

    gamma = 0.9
    [v, i] = tester.value_iteration(env, gamma)
    policy = tester.policy_selection(env,gamma,v)
    print(v,i)
    print(policy)

    policy, v, c, nv = tester.policy_iteration(env, gamma)
    print(v, c, nv)
    print(policy)

    env.close()