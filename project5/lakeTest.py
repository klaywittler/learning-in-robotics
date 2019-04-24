import numpy as np
import math as m
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import lake_env
from tester import Tester

class policy(nn.Module):
    def __init__(self):
        super(policy, self).__init__()
        self.state_space = env.observation_space.n
        self.action_space = env.action_space.n
        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128,self.action_space, bias=False)

        self.gamma = options['gamma']

        self.gamma = 0.9
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


def main(episodes=4,steps=100):
    running_reward = 1
    for episode in range(episodes):
        state = env.reset()
        done = False
        for time in range(steps):
            action = select_action(state)

            state, reward, done, _ = env.step(action.data[0])
            policy.reward_episode.append(reward)
            if done: 
                break

        running_reward = (running_reward*0.99) + (time*0.01)

        update_policy()

        if episode % 50 == 0:
            print('Episode {}\tLst length: {:5d}\tAverage length: {:0.2f}'.format(episode, time, running_reward))
        if running_reward > end.spec.reward_threshold:
            print('Solved! Running reward is now {} and the last episode runs to {} time steps'.format(running_reward, time))
            break

def select_action(state):
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    c = Categorical(state)
    actoin = c.sample()

    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_gistory, c.log_prob(action)])
    else:
        policy.policy_history = (c.log_prob(action))
    return action

def update_policy():
    R = 0
    rewards = []

    # Discount future rewards back to present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.inter(0,R)

    # Scale rewards    
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean())/(rewards.std() + np.finfo(npfloat32).eps)

    # Calculate loss 
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # SAve and intialize episode history counters
    policy.loss_history.append(loss.data[0])
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []


def getData(episodes=10,steps=1000):
    inputs = np.zeros([episodes*steps,5])
    labels = np.zeros([episodes*steps,4])

    for episode in range(episodes):
        observation = env.reset()
        # state vector:[ cos(theta)   ,  sin(theta)   ,  theta                , theta dot     ]
        s0 = np.array([observation[0], observation[1], m.acos(observation[0]), observation[2]])
        for t in range(steps):
            # env.render()
            action = env.action_space.sample()
            inputs[(episode*steps)+t,:] = np.hstack((s0,action))
            observation, reward, done, info = env.step(action)
            
            s1 = np.array([observation[0], observation[1], m.acos(observation[0]), observation[2]])
            labels[(episode*steps)+t,:] = s1
            s0 = s1

    env.close()
    training_data = {'inputs':inputs, 'labels':labels}
    return training_data


if __name__ == '__main__':
    # # P : s a probability reward, terminal
    ###################
    # Policy Gradient #
    ###################
    gamma = 0.9
    episodes = 10
    steps = 10
    # env = gym.make('Stochastic-4x4-FrozenLake-v0')
    env = gym.make('CartPole-v0')
    env.reset()
    env.render()
    main()
    # torch.save(model,'model.pt')
     
    # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
    


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