import numpy as np
import math as m
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F

import lake_env
from tester import Tester

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128,self.action_space, bias=False)

        self.gamma = gamma

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall loss and reward history
        self.reward_history = []
        self.loss_history = []
    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLu(),
            self.l2,
            nn.ReLu())
        return model(x)


def select_Action(state)


def policy_improvement(env,gamma,policy,v):
  P = np.zeros([env.nS,env.nS,env.nA])
  R = np.zeros([env.nS,env.nS,env.nA])
  for action in range(env.nA):
    for state in range(env.nS):
      for (prob, nextstate, r, is_terminal) in env.P[state][action]:
        P[state,nextstate,action] += prob  
        R[state,nextstate,action] += r
  
  q = np.sum(np.multiply(P,R + gamma*np.repeat(v[:,np.newaxis],env.nA,axis=1)),axis=1)
  policy = np.argmax(q,axis=1)
  return policy


# def get_trainingData(episodes=10,goal_steps=1000):
#     training_data = np.empty(9)

#     for i_episode in range(episodes):
#         observation = env.reset()
#         print(observation)
#         for t in range(goal_steps):
#             # env.render()
#             action = env.action_space.sample()
#             print(action)
#             observation, reward, done, info = env.step(action)
            
#             if done:
#                 break
                
#     env.close()
#     # training_data = np.delete(training_data,0,axis=0)
    # return 0

# def training():




#   torch.save(model,'model.pt')

episodes = 10
goal_steps = 10
# P : s a probability reward, terminal
env = gym.make('Stochastic-4x4-FrozenLake-v0')
env.render()
for i_episode in range(episodes):
      observation = env.reset()
      # print(observation)
      for t in range(goal_steps):
          # env.render()
          action = env.action_space.sample()
          print(action)
          observation, reward, done, info = env.step(action)
          
          if done:
              break
              
# net = Net()
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
gamma = 0.9

tester = Tester()

[v, i] = tester.value_iteration(env, gamma)
policy = policy = np.random.randint(env.nA, size= env.nS)
policy = policy_improvement(env,gamma,policy,v)
print(v,i)
print(policy)

policy, v, c, nv = tester.policy_iteration(env, gamma)
print(v, c, nv)
print(policy)

env.close()