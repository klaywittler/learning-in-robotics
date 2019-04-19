import numpy as np
import math as m
import gym
from gym import spaces
import torch
import lake_env
# import tester


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
  P = np.zeros([env.nS,env.nS,env.nA])
  R = np.zeros([env.nS,env.nS,env.nA])
  for action in range(env.nA):
    for state in range(env.nS):
      for (prob, nextstate, r, is_terminal) in env.P[state][action]:
        P[state,nextstate,action] += prob  
        R[state,nextstate,action] += r
  v = np.zeros(env.nS)
  eps = 100
  i = 0
  while eps > tol:
    vOld = v
    for action in range(env.nA):
      if action > 0:
        v = np.sum(np.multiply(P[:,:,action],R[:,:,action] + gamma*v),axis=1)
        v = np.maximum(vA,v)
      else:
        vA = np.sum(np.multiply(P[:,:,action],R[:,:,action] + gamma*v),axis=1)

    i += 1
    if i > max_iterations:
      break

    eps = np.linalg.norm(v - vOld)

  return v, i

# P : s a probability reward, terminal
env = gym.make('Stochastic-4x4-FrozenLake-v0')


env.render()

print(env.P[14])
# print(np.zeros([env.nS,env.nS]))
gamma = 0.9
# v = np.zeros(env.nS)
# for action in range(env.nA):
#   P = np.zeros([env.nS,env.nS])
#   R = np.zeros([env.nS,env.nS])
#   for state in range(env.nS):
#     for (prob, nextstate, r, is_terminal) in env.P[state][action]:
#       P[state][nextstate] = prob 
#       R[state][nextstate] = r
# vtemp = np.sum(np.multiply(P,R + gamma*v),axis=1) 

[v, i] = value_iteration(env, gamma)
print(v,i)

# t = np.array([10,6,5,3,2])

# for v, i in enumerate(t):
#   print("v: ", v)
#   print("i: ", i)

# t = np.round(env.nA*np.random.rand(env.nS))
# print(t)
