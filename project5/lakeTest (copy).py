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
from tester import Tester

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128,self.action_space, bias=False)

        # self.gamma = options['gamma']
        self.gamma = gamma

        # Episode policy and reward history
        self.policy_history = torch.tensor(-9999) # Variable(torch.Tensor())
        self.reward_episode = []
        # Overall loss and reward history
        self.reward_history = []
        self.loss_history = []
    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
            )
        return model(x)


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

        running_reward = (running_reward*0.99) + (time*0.01)

        update_policy()

        if episode % 50 == 0:
            print('Episode {}\tLst length: {:5d}\tAverage length: {:0.2f}'.format(episode, time, running_reward))
        if running_reward > env.spec.reward_threshold:
            print('Solved! Running reward is now {} and the last episode runs to {} time steps'.format(running_reward, time))
            break


def select_action(state):
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    c = Categorical(state)
    action = c.sample()

    if policy.policy_history.dim() != 0:
        log_probA = torch.tensor([c.log_prob(action)],requires_grad=True)
        policy.policy_history = torch.cat([policy.policy_history, log_probA])
    else:
        policy.policy_history = torch.tensor([c.log_prob(action)],requires_grad=True)
    return action

def update_policy():
    R = 0
    rewards = []

    # Discount future rewards back to present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)

    # Scale rewards
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean())/(rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss 
    loss = torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1)

    # Update network weights
    optimizer = torch.optim.Adam(policy.parameters(),lr = options['lr'])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.data.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = torch.tensor(-9999)
    policy.reward_episode = []


if __name__ == '__main__':
    # # P : s a probability reward, terminal
    ###################
    # Policy Gradient #
    ###################
    # print(torch.tensor(1).dim())
    gamma = 0.99
    LR = 0.01 # learning rate
    options = {'gamma':gamma,'lr':LR}
    # env = gym.make('Stochastic-4x4-FrozenLake-v0')
    env = gym.make('CartPole-v0')
    policy = Policy()
    main()

    # print(policy.reward_history)
    episodes = 1
    steps = 1000
    for episode in range(episodes):
        state = env.reset()
        done = False
        for time in range(steps):
            env.render()
            action = select_action(state)
            state, reward, done, _ = env.step(action.data.item())
            policy.reward_episode.append(reward)
            if done: 
                break

    env.close()

    torch.save(policy,'CartPolePolicy.pt') 

    ##############################
    # Policy Iteration/Value #
    ##############################
    # tester = Tester()

    # [v, i] = tester.value_iteration(env, gamma)
    # policy = policy = np.random.randint(env.nA, size= env.nS)
    # policy = tester.policy_selection(env,gamma,policy,v)
    # print(v,i)
    # print(policy)

    # policy, v, c, nv = tester.policy_iteration(env, gamma)
    # print(v, c, nv)
    # print(policy)

    # env.close()