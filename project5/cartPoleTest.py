import numpy as np
import math as m
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import *
import lake_env
from tester import *


def main(episodes=1000,steps=1000):
    running_reward = 10
    for episode in range(episodes):
        state = env.reset()
        done = False
        for time in range(steps):
            # env.render()
            action = select_action(state)

            state, reward, done, _ = env.step(action.data.item())
            policy.reward_episode.append(reward)
            if done: 
                break

        running_reward = (running_reward*gamma) + (time*0.01)

        update_policy()

        if episode % 50 == 0:
            print('Episode {}\tLst length: {:5d}\tAverage length: {:0.2f}'.format(episode, time, running_reward))
        if running_reward > env.spec.reward_threshold:
            print('Solved! Episode {}\t. Running reward is now {} and the last episode runs to {} time steps'.format(episode, running_reward, time))
            break
            


def select_action(state):
    state = torch.from_numpy(state).float()
    prob_action = policy(Variable(state))
    c = Categorical(prob_action)
    action = torch.tensor([c.sample()])
    policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action)])
    return action


def update_policy():
    R = 0
    rewards = []

    # Discount future rewards back to present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + gamma * R
        rewards.insert(0,R)

    # Scale rewards
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean())/(rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss 
    loss = torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1)

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.data.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []


if __name__ == '__main__':
    # # P : s a probability reward, terminal
    ###################
    # Policy Gradient #
    ###################

    s = 'CartPole-v1'
    #### Training ####
    # gamma = 0.99
    # LR = 0.01 # learning rate
    # env = gym.make('CartPole-v1')
    # options = {'gamma':gamma,'lr':LR,'state_space':env.observation_space.shape[0],'action_space':env.action_space.n}
    # policy = Policy(options)
    # optimizer = torch.optim.Adam(policy.parameters(),lr = options['lr'])
    # main()
    # torch.save(policy.state_dict(),'CartPolePolicy.pt')
    # env.close()


    #### Testing ####
    env = gym.make('CartPole-v1')
    tester = Tester()
    episodes = 1
    steps = 1000
    for episode in range(episodes):
        state = env.reset()
        done = False
        for time in range(steps):
            env.render()

            action = tester.policy_gradient_test(state)

            state, reward, done, _ = env.step(action)
            if done:
                print(time)
                break

    env.close() 
