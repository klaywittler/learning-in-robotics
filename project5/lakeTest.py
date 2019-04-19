import numpy as np
import math as m
import gym
from gym import spaces
# import torch
import lake_env
from tester import Tester


def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray
      The value for the given policy
    """
    P = np.zeros([env.nS,env.nS])
    R = np.zeros([env.nS,env.nS])
    for state, action in enumerate(policy):
      for (prob, nextstate, r, is_terminal) in env.P[state][action]:
          P[state,nextstate] += prob 
          R[state,nextstate] += r

    v = np.zeros(env.nS)
    for i in range(max_iterations):
      vOld = v
      v = np.sum(np.multiply(P,R + gamma*v),axis=1)
      eps = np.linalg.norm(v - vOld)
      if eps < tol:
        break

    return v,i

def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    nV = 0
    policy = np.random.randint(env.nA, size= env.nS)

    P = np.zeros([env.nS,env.nS,env.nA])
    R = np.zeros([env.nS,env.nS,env.nA])
    for action in range(env.nA):
      for state in range(env.nS):
        for (prob, nextstate, r, is_terminal) in env.P[state][action]:
          P[state,nextstate,action] += prob 
          R[state,nextstate,action] += r

    v = np.zeros(env.nS)
    for i in range(max_iterations):
      vOld = v
      pOld = policy
      v, c = evaluate_policy(env, gamma, policy)
      q = np.sum(np.multiply(P,R + gamma*np.repeat(v[:,np.newaxis],env.nA,axis=1)),axis=1)
      policy = np.argmax(q,axis=1)

      nV += c
      eps = np.linalg.norm(v - vOld)
      if eps < tol or np.array_equal(pOld,policy):
        break

    return policy, v, i, nV

def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    P = np.zeros([env.nS,env.nS,env.nA])
    R = np.zeros([env.nS,env.nS,env.nA])
    for action in range(env.nA):
      for state in range(env.nS):
        for (prob, nextstate, r, is_terminal) in env.P[state][action]:
          P[state,nextstate,action] += prob 
          R[state,nextstate,action] += r

    v = np.zeros(env.nS)
    for i in range(max_iterations):
      vOld = v
      vA = np.sum(np.multiply(P,R + gamma*np.repeat(v[:,np.newaxis],env.nA,axis=1)),axis=1)
      v = np.amax(vA,axis=1)
      eps = np.linalg.norm(v - vOld)

      if eps < tol:
        break

    return v, i

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

  # for state in range(env.nS):
  #   q = np.zeros(env.nA)
  #   for action in range(env.nA):
  #     q = np.sum(np.multiply(P[state,:,action],R[state,:,action] + gamma*v))
  #   policy[state] = np.argmax(q)
  return policy


# P : s a probability reward, terminal
env = gym.make('Stochastic-4x4-FrozenLake-v0')
env.render()

gamma = 0.9

[v, i] = value_iteration(env, gamma)
policy = policy = np.random.randint(env.nA, size= env.nS)
policy = policy_improvement(env,gamma,policy,v)
print(v,i)
print(policy)

policy, v, c, nv = policy_iteration(env, gamma)
print(v, c, nv)
print(policy)

