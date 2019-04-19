import numpy as np
import torch
import gym

class Tester(object):

    def __init__(self):
        """
        Initialize the Tester object by loading your model.
        """
        # TODO: Load your pyTorch model for Policy Gradient here.
        pass


    def evaluate_policy(self, env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
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
        eps = 100
        i = 0
        while eps > tol:
          vOld = v
          v = np.sum(np.multiply(P,R + gamma*v),axis=1)
          i += 1
          if i > max_iterations:
            break

          eps = np.linalg.norm(v - vOld)

        return v

    def policy_iteration(self, env, gamma, max_iterations=int(1e3), tol=1e-3):
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
        # TODO:  Your code goes here.
        nV = 0
        policy = np.round(env.nA*np.random.rand(env.nS))

        eps = 100
        i = 0
        while eps > tol:
          vOld = v
          pOld = policy
          v = evaluate_policy(self, env, gamma, policy)
          for state in range(env.nS):
            for action in range(env.nA):
              for (prob, nextstate, r, is_terminal) in env.P[state][action]:
                v[nextstate] = 

          i += 1
          if i > max_iterations or pOld == policy:
            break

          eps = np.linalg.norm(v - vOld)
        
        return policy, v, i, nV

    def value_iteration(self, env, gamma, max_iterations=int(1e3), tol=1e-3):
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

    def policy_gradient_test(self, state):
        """
        Parameters
        ----------
        state: np.ndarray
            The state from the CartPole gym environment.
        Returns
        ------
        np.ndarray
            The action in this state according to the trained policy.
        """
        # TODO. Your Code goes here.
        return 0