import numpy as np
import math as m
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.state_space = env.observation_space.shape[0]
        # self.action_space = env.action_space.shape[0]
        self.l1 = nn.Linear(5, 128)
        self.l2 = nn.Linear(128,4)
    def forward(self, x):
        # x = F.relu(self.l1(x))
        # x = self.l2(x)
        # return x
        model = nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.ReLU())
        return model(x)


class Trainer(object):
    def __init__(self, options):
        self.options = options
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.model = Model().to(self.device)
        # define loss function and optimizer
        self.loss = torch.nn.MSELoss().to(self.device) # for Global loss
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = options['lr'])
        self.train_inputs = torch.from_numpy(options['training_data']['inputs']).float()
        self.train_labels = torch.from_numpy(options['training_data']['labels']).float()
        self.test_inputs = torch.from_numpy(options['testing_data']['inputs']).float()
        self.test_labels = torch.from_numpy(options['testing_data']['labels']).float()

        self.train_loss = []
        self.test_loss = []

    def train(self):
        self.total_step_count = 0
        start_time = time()
        for epoch in range(1,self.options['num_epochs']+1):
            elapsed_time = time() - start_time
            seconds_per_epoch = elapsed_time / epoch
            remaining_time = (self.options['num_epochs'] - epoch) * seconds_per_epoch

            print("Epoch %d/%d - Elapsed Time %f - Remaining Time %f" %
                    (epoch, self.options['num_epochs'],
                        elapsed_time, remaining_time))

            for step, inputs in enumerate(self.train_inputs):
                self.model.train()                
                self.optimizer.zero_grad()
            
                pred = self.model(inputs)
                loss = self.loss(pred[-1], self.train_labels[step,:])
                loss.backward()
                self.optimizer.step()

    def test(self):
        for step, inputs in enumerate(self.test_inputs):             
            self.model.eval()
            with torch.no_grad():
                pred = self.model(inputs)
                loss = self.loss(pred[-1], self.train_labels[step,:])

        return loss


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


def Astar(model, n_theta = 35, n_dtheta = 35, n_u = 35):
    theta_lower = -m.pi
    theta_upper = m.pi
    dtheta_lower = -8
    dtheta_upper = 8
    u_lower = -2
    u_upper = 2

    theta_space = np.linspace(theta_lower,theta_upper,n_theta)
    dtheta_space = np.linspace(dtheta_lower,dtheta_upper,n_dtheta)
    action_space = np.linspace(u_lower,u_upper,n_u)

    cost = np.empty([theta_space.size,dtheta_space.size,action_space.size])


    t=0
    for theta in theta_space:
        dt = 0
        for dtheta in dtheta_space:
            a = 0
            for action in action_space:
                sa0 = np.array([m.cos(theta), m.sin(theta), theta, dtheta, action])
                predict_state = model.predict(sa0.reshape(-1,len(sa0)))
                # heuristic
                h = -(predict_state[0][2]**2 + predict_state[0][3]**2 + 0.001*action**2)
                g = -abs(predict_state[0][2]) #-abs(predict_state[0][3])
                cost[t,dt,a] = h + g
                a += 1
            dt += 1
        t += 1  

    return [cost, theta_space, dtheta_space, action_space]

def simulation(model,games=1,steps=1000):
    [cost, theta_space, dtheta_space, action_space] = get_costMap(model)
    for i_episode in range(games):
        observation = env.reset()
        for t in range(steps):
            env.render()
            theta = m.acos(observation[0]) 
            dtheta = observation[2]
            t = np.where(theta_space>=theta) 
            dt = np.where(dtheta_space>=dtheta)
            a = Astar(model,theta,dtheta)
            a = cost[t[0][0],dt[0][0],:].argmax()
            action = [action_space[a]]
            # if abs(theta) >= 0.75*m.pi:
            #     if np.array(m.copysign(1,dtheta)) != np.array(m.copysign(1,action_space[a])):
            #         action = np.array(m.copysign(1,-1))*action
            # else:
            #     if np.array(m.copysign(1,dtheta)) == np.array(m.copysign(1,action_space[a])):
            #         action = np.array(m.copysign(1,-1))*action
            
            print(action)
            observation, reward, done, info = env.step(action)
                    
    env.close()


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    observation = env.reset()

    # initialization variables
    LR = 1e-3 # learning rate
    num_epochs = 5 # number of times data is seen
    games = 10 # how many games
    steps = 50 # how long each game runs
    test_games = 5 # how test games
    test_steps = 50 # how long each test game runs
    
    training_data = getData(games,steps)
    np.save('saved_training_data0.npy',training_data)

    testing_data = getData(games,steps)
    np.save('saved_testing_data0.npy',testing_data)

    options = {'lr':LR, 'num_epochs':num_epochs,'training_data':training_data,'testing_data': testing_data}
    trainer = Trainer(options)
    trainer.train()
    t = torch.from_numpy(testing_data['inputs']).float()
    

