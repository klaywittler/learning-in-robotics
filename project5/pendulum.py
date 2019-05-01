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
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l1 = nn.Linear(self.input_dim, 128)
        self.l2 = nn.Linear(128,self.output_dim)
    def forward(self, x):
        model = nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2)
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
        self.input_dim = options['training_data']['inputs'].shape[1]
        self.output_dim = options['training_data']['labels'].shape[1]
        if options['model'] == 'FC':
            self.model = Model(self.input_dim,self.output_dim).to(self.device)
        elif options['model'] == 'LSTM':
            self.model = LSTM(options['params']['batch_size']).to(self.device)
        else:
            self.model = Model(self.input_dim,self.output_dim).to(self.device)
        # define loss function and optimizer
        self.loss = torch.nn.MSELoss(reduction='sum').to(self.device) # for Global loss
        self.testLoss = torch.nn.MSELoss().to(self.device) # for Global loss
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = options['lr'])

        self.train_inputs = torch.from_numpy(options['training_data']['inputs']).float()
        self.train_labels = torch.from_numpy(options['training_data']['labels']).float()
        self.test_inputs = torch.from_numpy(options['testing_data']['inputs']).float()
        self.test_labels = torch.from_numpy(options['testing_data']['labels']).float()

        self.train_ds = PendulumData(self.train_inputs,self.train_labels)
        self.test_ds = PendulumData(self.test_inputs,self.test_labels)

        self.train_data_loader = DataLoader(self.train_ds, **options['params'])
        self.test_data_loader = DataLoader(self.test_ds)

        self.train_loss = []
        self.test_loss = []

    def train(self):
        start_time = time()
        for epoch in range(1,self.options['num_epochs']+1):
            elapsed_time = time() - start_time
            seconds_per_epoch = elapsed_time / epoch
            remaining_time = (self.options['num_epochs'] - epoch) * seconds_per_epoch

            epoch_loss = 0
            if epoch > 1:
                print("Epoch: %d/%d - Elapsed Time: %f - Remaining Time: %f - Loss: %f" %
                        (epoch, self.options['num_epochs'],
                            elapsed_time, remaining_time, self.train_loss[-1]))
            for local_batch, local_labels in self.train_data_loader:
                self.model.train()                
                self.optimizer.zero_grad()
            
                pred = self.model(local_batch)
                loss = self.loss(pred, local_labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss
                pred.detach()
                loss.detach()

            self.train_loss.append(epoch_loss/len(self.train_data_loader.dataset))

    def test(self):
        for local_batch, local_labels in self.test_data_loader:           
            self.model.eval()
            with torch.no_grad():
                pred = self.model(local_batch)
                loss = self.testLoss(pred, local_labels)
                self.test_loss.append(loss)
                # print("pred:",pred)
                # print("label: ", local_labels)
                pred.detach()
                loss.detach()

def getData(episodes=1000,steps=1000,shuffle=False, render=False, theta = False):
    if theta:
        inputs = np.zeros([episodes*steps,5])
        labels = np.zeros([episodes*steps,4])
    else:
        inputs = np.zeros([episodes*steps,4])
        labels = np.zeros([episodes*steps,3])

    for episode in range(episodes):
        if not shuffle:
            observation = env.reset()
            if theta:
                s0 = np.array([observation[0], observation[1], np.arctan2(observation[1],observation[0]), observation[2]])
            else:
                s0 = np.array(observation)
        for t in range(steps):
            if shuffle:
                observation = env.reset()
            if render:
                env.render()
            if theta:
                s0 = np.array([observation[0], observation[1], np.arctan2(observation[1],observation[0]), observation[2]])
            else:
                s0 = np.array(observation)
            action = env.action_space.sample()
            inputs[(episode*steps)+t,:] = np.hstack((s0,action))
            observation, reward, done, info = env.step(action)
            if theta:
                s1 = np.array([observation[0], observation[1], np.arctan2(observation[1],observation[0]), observation[2]])
            else:    
                s1 = np.array(observation)
            labels[(episode*steps)+t,:] = s1
            s0 = s1

    env.close()
    training_data = {'inputs':inputs, 'labels':labels}
    return training_data


def getMap(model, n_theta = 35, n_dtheta = 35, n_u = 35):
    theta_lower = -m.pi
    theta_upper = m.pi
    dtheta_lower = -8
    dtheta_upper = 8
    u_lower = -2
    u_upper = 2

    theta_space = np.linspace(theta_lower,theta_upper,n_theta)
    dtheta_space = np.linspace(dtheta_lower,dtheta_upper,n_dtheta)
    action_space = np.linspace(u_lower,u_upper,n_u)

    return [theta_space, dtheta_space, action_space]

def Astar(model, space, start, goal = np.zeros(3)):
    theta_space = space[0]
    dtheta_space = space[1]
    action_space = space[2]

    front = [start]
    explored = []
    currentIndex = 0

    # while len(front) > 0:
    if np.linalg.norm(start-goal) < 0.01:
        path = np.zeros(1)    
    front.pop(currentIndex)
    explored.append(start)


    theta = np.repeat(start[0],len(action_space),axis=0)
    thetaDot = np.repeat(start[1],len(action_space),axis=0)
    state = np.array([np.cos(theta),np.sin(theta), thetaDot, action_space]).T
    inputs = torch.from_numpy(state).float()
    outputs = model(inputs).detach().numpy()
    neighbors = np.array([np.arctan2(outputs[:,1],outputs[:,0]), outputs[:,2]]).T
    h = -(neighbors[:,0]**2 + 0.1*neighbors[:,1]**2 + 0.001*state[:,2]**2)
    front.append(neighbors)
    print(neighbors)





    
    return np.zeros(1)


def simulation(model,games=1,steps=1000):
    disc_space = getMap(model)
    for i_episode in range(games):
        observation = env.reset()
        path = Astar(model, disc_space, observation)
        for t in range(len(path)):
            env.render()
            action = path[t]
            observation, reward, done, info = env.step(action)
            if done:
                break
                    
    env.close()


if __name__ == '__main__':
    TrainModel = False
    SimModel = True    

    env = gym.make('Pendulum-v0')
    observation = env.reset()

    # initialization variables
    model = 'FC'
    params = {'batch_size': 1,
            'shuffle': True,
            'num_workers':6}
    LR = 1e-3 # learning rate
    num_epochs = 40 # number of times data is seen
    games = 10 # how many games
    steps = 100 # how long each game runs
    test_games = 1 # how test games
    test_steps = 500 # how long each test game runs
    theta = False
    
    training_data = getData(games,steps,shuffle=params['shuffle'],render=False,theta=theta)
    np.save('saved_training_data0.npy',training_data)

    testing_data = getData(test_games,test_steps,shuffle=False,render=False,theta=theta)
    np.save('saved_testing_data0.npy',testing_data)

    options = {'model':model,'lr':LR, 'num_epochs':num_epochs,'training_data':training_data,'testing_data': testing_data,'params':params}

    if TrainModel:
        trainer = Trainer(options)
        trainer.train()
        trainer.test()

        # simulation(trainer.model)
        meanTest = np.mean(np.array(trainer.test_loss))
        print("mean test error: ", meanTest)
        if meanTest < 0.04:
            torch.save(trainer.model.state_dict(),'pendulumModel.pt')
        
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(trainer.train_loss)
        axs[0].set_ylabel('training loss')
        axs[1].plot(trainer.test_loss)
        axs[1].set_ylabel('testing loss')
        fig.tight_layout()
        plt.show()
        plt.close('all')

        model = trainer.model
    else:
        input_dim = options['training_data']['inputs'].shape[1]
        output_dim = options['training_data']['labels'].shape[1]
        model = Model(input_dim,output_dim)
        model.load_state_dict(torch.load('pendulumModel.pt'))
        model.eval()

    if SimModel:
        simulation(model)

    

