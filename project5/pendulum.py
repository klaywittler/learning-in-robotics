import numpy as np
import math as m
import gym
from gym import spaces
import torch


env = gym.make('Pendulum-v0')

# initialization variables
LR = 1e-3 # learning rate
goal_steps = 500 # how long each game runs
initial_games = 1000 # how many games
test_games = 5 # how test games
test_steps = 500 # how long each test game runs

def get_trainingData(episodes=10,goal_steps=1000):
    training_data = np.empty(9)

    for i_episode in range(episodes):
        observation = env.reset()
        # state vector:[ cos(theta)   ,  sin(theta)   ,  theta                , theta dot     ]
        s0 = np.array([observation[0], observation[1], m.acos(observation[0]), observation[2]])
        for t in range(goal_steps):
            # env.render()
            action = env.action_space.sample()
            sa0 = np.hstack((s0,action))
            observation, reward, done, info = env.step(action)
            
            s1 = np.array([observation[0], observation[1], m.acos(observation[0]), observation[2]])
            training_data = np.vstack((training_data,np.hstack((sa0,s1))))
            s0 = s1
            # if done:
            #     break
                
    env.close()
    training_data = np.delete(training_data,0,axis=0)
    return training_data


if __name__ == '__main__':
    