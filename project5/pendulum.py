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


def getMeshMap(model, n_theta = 35, n_dtheta = 35, n_u = 35):
    theta_lower = -m.pi
    theta_upper = m.pi
    dtheta_lower = -8
    dtheta_upper = 8
    u_lower = -2
    u_upper = 2

    theta_space = np.linspace(theta_lower,theta_upper,n_theta)
    dtheta_space = np.linspace(dtheta_lower,dtheta_upper,n_dtheta)
    action_space = np.linspace(u_lower,u_upper,n_u)

    grid = np.stack(np.meshgrid(theta_space, dtheta_space, action_space),-1).reshape(-1,3)
    state = np.array([np.cos(grid[:,0]),np.sin(grid[:,0]), grid[:,1], grid[:,2]]).T

    inputs = torch.from_numpy(state).float()
    outputs = model(inputs).detach().numpy()




    return [theta_space, dtheta_space, action_space]

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

def AstarMesh(model, space, start, goal_state = np.zeros(2)):

    return 0

def Astar(model, space, start, goal_state = np.zeros(2)):
    start = np.array([np.arctan2(start[1],start[0]), start[2]])
    theta_space = space[0]
    dtheta_space = space[1]
    action_space = space[2]

    goal_theta_bin = np.digitize(goal_state[0],theta_space)
    goal_dtheta_bin = np.digitize(goal_state[1],dtheta_space)
    goal = tuple(np.array([goal_theta_bin,goal_dtheta_bin]))

    theta_bin = np.digitize(start[0],theta_space)
    dtheta_bin = np.digitize(start[1],dtheta_space)
    current = tuple(np.array([theta_bin,dtheta_bin]))
    front = {current:{'parent':None,'action':None,'value':0}}
    explored = {}

    count = 0
    while len(front) > 0:
        explored.update({current:front[current]})
        del front[current]

        if current == goal or count > 5000:
            print('finding path')
            path = []
            parent = explored[current]['parent']
            trace = 0
            while parent is not None and trace < 5000: 
                path.append(explored[current]['action'])
                current = parent
                parent = explored[current]['parent']
                trace += 1
                print('finding path',trace, parent)
            return path

        theta = np.repeat(theta_space[current[0]],len(action_space),axis=0)
        thetaDot = np.repeat(theta_space[current[1]],len(action_space),axis=0)

        state = np.array([np.cos(theta),np.sin(theta), thetaDot, action_space]).T
        inputs = torch.from_numpy(state).float()
        outputs = model(inputs).detach().numpy()
        neighbors = np.array([np.arctan2(outputs[:,1],outputs[:,0]), outputs[:,2]]).T
        h = -(neighbors[:,0]**2 + 0.1*neighbors[:,1]**2 + 0.001*state[:,2]**2)

        theta_bins = np.digitize(neighbors[:,0],theta_space)
        dtheta_bins = np.digitize(neighbors[:,1],dtheta_space)
        indTup = map(tuple, np.array([theta_bins,dtheta_bins]).T)
        for i, t in enumerate(indTup):
            if t in front:
                if front[t]['value'] > h[i]:
                    front[t] = {'action':action_space[i],'parent':(theta_bin,dtheta_bin),'value':h[i]}
            else:
                front[t] = {'action':action_space[i],'parent':(theta_bin,dtheta_bin),'value':h[i]}

        current = min(front, key=front.get)
        count += 1
        print(count, current)

    return path


def simulation(model,games=1,steps=1000):
    disc_space = getMap(model)
    for i_episode in range(games):
        observation = env.reset()
        path = Astar(model, disc_space, observation)
        for t in range(len(path)):
            env.render()
            action = np.array([path[t]])
            print(action)
            observation, reward, done, info = env.step(action)
            if done:
                break
                    
    env.close()

def simulationMesh(model,games=1,steps=1000):
    disc_space = getMeshMap(model)
    for i_episode in range(games):
        observation = env.reset()
        path = AstarMesh(model, disc_space, observation)
        for t in range(len(path)):
            env.render()
            action = np.array([path[t]])
            print(action)
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
        if meanTest < 0.03:
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
        # simulationMesh(model)
        a = np.array([1,2,3,4])
        b = np.array([10,11,12])
        c = np.array([20,21,22])
        grid = np.stack(np.meshgrid(a,b,c),-1).reshape(-1,3)
        print(grid[:,0:2].shape)

        grid2 = np.array(np.meshgrid(a,b)).T.reshape(-1,2)

        print(grid2.shape)
        
        # grid3 = np.zeros([len(a)*len(b),len(c)])
        grid3 = grid[:,0:2].reshape(len(a)*len(b),len(c),2)
        print(grid3)



    

