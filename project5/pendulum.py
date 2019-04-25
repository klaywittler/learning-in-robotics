import numpy as np
import math as m
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from time import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(5, 128)
        self.l2 = nn.Linear(128,4)
    def forward(self, x):
        model = nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.ReLU())
        return model(x)


class LSTM(nn.Module):
    def __init__(self, batch_size):
        super(LSTM, self).__init__()
        self.input_dim = 5
        self.output_dim = 4
        self.hidden_dim = 128
        self.batch_size = batch_size
        self.num_layers = 1

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        x = F.relu(lstm_out[-1].view(self.batch_size, -1))
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(x)
        return y_pred.view(-1)


class PendulumData(Dataset):
    def __init__(self, states, labels):
        self.states = states
        self.labels = labels

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        X = self.states[index,:]
        y = self.labels[index,:]
        return X, y


class Trainer(object):
    def __init__(self, options):
        self.options = options
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        if options['model'] == 'FC':
            self.model = Model().to(self.device)
        elif options['model'] == 'LSTM':
            self.model = LSTM(options['params']['batch_size']).to(self.device)
        else:
            self.model = Model().to(self.device)
        # define loss function and optimizer
        self.loss = torch.nn.MSELoss().to(self.device) # for Global loss
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = options['lr'])

        self.train_inputs = torch.from_numpy(options['training_data']['inputs']).float()
        self.train_labels = torch.from_numpy(options['training_data']['labels']).float()
        self.test_inputs = torch.from_numpy(options['testing_data']['inputs']).float()
        self.test_labels = torch.from_numpy(options['testing_data']['labels']).float()

        self.train_ds = PendulumData(self.train_inputs,self.train_labels)
        self.test_ds = PendulumData(self.test_inputs,self.test_labels)

        self.train_data_loader = DataLoader(self.train_ds, **options['params'])
        self.test_data_loader = DataLoader(self.test_ds, **options['params'])

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
            for local_batch, local_labels in self.train_data_loader:
                self.model.train()                
                self.optimizer.zero_grad()
            
                pred = self.model(local_batch)
                loss = self.loss(pred[-1], local_labels)
                loss.backward()
                self.optimizer.step()
                self.train_loss.append(loss)

    def test(self):
        for local_batch, local_labels in self.test_data_loader:           
            self.model.eval()
            with torch.no_grad():
                pred = self.model(local_batch)
                loss = self.loss(pred[-1], local_labels)
                self.test_loss.append(loss)

def getData(episodes=1000,steps=1000,shuffle=False, render=False):
    inputs = np.zeros([episodes*steps,5])
    labels = np.zeros([episodes*steps,4])

    for episode in range(episodes):
        if not shuffle:
            observation = env.reset()
            s0 = np.array([observation[0], observation[1], np.arctan2(observation[1],observation[0]), observation[2]])
        for t in range(steps):
            if shuffle:
                observation = env.reset()
                s0 = np.array([observation[0], observation[1], np.arctan2(observation[1],observation[0]), observation[2]])
            if render:
                env.render()
            action = env.action_space.sample()
            inputs[(episode*steps)+t,:] = np.hstack((s0,action))
            observation, reward, done, info = env.step(action)
            
            s1 = np.array([observation[0], observation[1], np.arctan2(observation[1],observation[0]), observation[2]])
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
            # env.render()
            theta = m.acos(observation[0]) 
            dtheta = observation[2]
            t = np.where(theta_space>=theta) 
            dt = np.where(dtheta_space>=dtheta)
            a = Astar(model,theta,dtheta)
            a = cost[t[0][0],dt[0][0],:].argmax()
            action = [action_space[a]]
            print(action)
            observation, reward, done, info = env.step(action)
                    
    env.close()


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    observation = env.reset()

    # initialization variables
    model = 'FC'
    params = {'batch_size': 50,
            'shuffle': True,
            'num_workers':6}
    LR = 1e-3 # learning rate
    num_epochs = 5 # number of times data is seen
    games = 100 # how many games
    steps = 100 # how long each game runs
    test_games = 1 # how test games
    test_steps = 500 # how long each test game runs
    
    training_data = getData(games,steps,params['shuffle'])
    np.save('saved_training_data0.npy',training_data)

    testing_data = getData(games,steps,False)
    np.save('saved_testing_data0.npy',testing_data)

    options = {'model':model,'lr':LR, 'num_epochs':num_epochs,'training_data':training_data,'testing_data': testing_data,'params':params}
    trainer = Trainer(options)
    trainer.train()
    trainer.test()

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(trainer.train_loss)
    axs[0].set_ylabel('training loss')

    axs[1].plot(trainer.test_loss)
    axs[1].set_ylabel('testing loss')

    fig.tight_layout()
    plt.show()
    

