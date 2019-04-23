import numpy as np
import math as m
import gym
from gym import spaces
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.state_space = env.observation_space.shape[0]
        # self.action_space = env.action_space.shape[0]
        self.state_space = 4
        self.action_space = 1
        self.l1 = nn.RNN(self.state_space + self.action_space, 128)
        self.l2 = nn.RNN(128,self.self.state_space)
    def forward(self, x):
        model = nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLu(),
            self.l2,
            nn.ReLu())

        return model(x)


class Trainer(object):
    def __init__(self, options):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model()
        # define loss function and optimizer
        self.loss = torch.nn.MSELoss().to(self.device) # for Global loss
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = options['lr'])
        self.train_inputs = options['training_data']['training_inputs']
        self.train_labels = options['training_data']['training_labels']

    def train(self):
        self.total_step_count = 0
        start_time = time()
        for epoch in range(1,self.options['num_epochs']+1):
            elapsed_time = time() - start_time
            seconds_per_epoch = elapsed_time / epoch
            remaining_time = (self.options['num_epochs'] - epoch) * seconds_per_epoch

            print("Epoch %d/%d - Elapsed Time %f - Remaining Time %f" %
                    (epoch, self.options.num_epochs,
                        elapsed_time, remaining_time))

            for step, inputs in enumerate(self.train_inputs):
                self.model.train()                
                self.optimizer.zero_grad()
            
                pred = self.model(inputs)
                loss = self.loss(pred[-1], self.training_labels[step,:])
                loss.backward()
                self.optimizer.step()


def get_trainingData(episodes=10,goal_steps=1000):
    training_inputs = np.zeros([episodes*goal_steps,5])
    training_labels = np.zeros([episodes*goal_steps,4])

    for i_episode in range(episodes):
        observation = env.reset()
        # state vector:[ cos(theta)   ,  sin(theta)   ,  theta                , theta dot     ]
        s0 = np.array([observation[0], observation[1], m.acos(observation[0]), observation[2]])
        for t in range(goal_steps):
            # env.render()
            action = env.action_space.sample()
            training_inputs[(i_episode*goal_steps)+t,:] = np.hstack((s0,action))
            observation, reward, done, info = env.step(action)
            
            s1 = np.array([observation[0], observation[1], m.acos(observation[0]), observation[2]])
            training_labels[(i_episode*goal_steps)+t,:] = s1
            s0 = s1

    env.close()
    training_data = {'training_inputs':training_inputs, 'training_labels':training_labels}
    return training_data


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    observation = env.reset()

    # initialization variables
    LR = 1e-3 # learning rate
    num_epochs = 5
    goal_steps = 50 # how long each game runs
    initial_games = 10 # how many games
    test_games = 5 # how test games
    test_steps = 500 # how long each test game runs
    # print(env.observation_space.shape[])
    training_data = get_trainingData(initial_games,goal_steps)
    np.save('saved_training_data0.npy',training_data)

    options = {'lr':LR, 'num_epochs':num_epochs,'training_data':training_data}
    # trainer = Trainer(options)
    # trainer.train()
    # for i, v in enumerate(training_data['training_inputs']):
    #     print(v)
    model = Model()
    