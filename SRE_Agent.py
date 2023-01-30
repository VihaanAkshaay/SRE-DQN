import random
import torch
import numpy as np
from collections import deque, namedtuple

import torch
import torch.nn as nn  
import torch.nn.functional as F

device = torch.device("mps" if torch.cuda.is_available() else "cpu")


class ReplayBuffer_Hindsight:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, traj_buffer_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory_traj = deque(maxlen = traj_buffer_size) 
        self.seed = random.seed(seed)
    
    def add_trajectory(self,trajectory):
        """ Add a trajectory to the trajectory list"""
        self.memory_traj.append(trajectory)
    
    def sample_trajectory(self):
        trajectory = random.sample(self.memory_traj,1)
        return trajectory

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory_traj)

class QNetwork_HER(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self,input_units,output_units,seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork_HER, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(2*input_units, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_units)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

########################### SRE Agent ##############################
import random

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("mps" if torch.cuda.is_available() else "cpu")


TRAJECTORY_BUFFER_SIZE = 1000    # trajectory memory size
GAMMA = 0.95                     # discount factor
LR = 5e-3                        # learning rate 
UPDATE_EVERY = 30                # how often to update the network (When Q target is present)

class Agent_SREDQN():
    """
    Encompases Scrambler and Repeater
    """

    def __init__(self, state_size, action_size, seed):

        # Agent Environment Interaction
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork_HER(state_size, action_size, seed).to(device) # Policy network used by the repeater
        self.qnetwork_target = QNetwork_HER(state_size, action_size, seed).to(device) # We use network_target as a less_frequently updated NN which is used by the scrambler
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer_Hindsight(action_size,TRAJECTORY_BUFFER_SIZE, seed)
        self.scramble_dict = []

        # Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets
        self.t_step = 0

    def scramble_action(self,state,eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_target.eval()
        with torch.no_grad():
            action_values = self.qnetwork_target(state)
        self.qnetwork_target.train()

        # Epsilon-antiGreedy action selection
        if random.random() > eps:
            return np.argmin(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

        

    def step(self, trajectory):

        # Save experience in replay memory
        #self.memory.add_trajectory(trajectory)
        
        # If enough samples are available in memory, get random subset and learn

        #trajectory = self.memory.sample_trajectory()
        #print(trajectory)
        self.learn(trajectory, GAMMA)

        # Updating the Network every 'UPDATE_EVERY' steps taken       
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def agent_act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self,trajectory,gamma):

        #Traverse reverse in a trajectory and train with each sample
        for experience in reversed(trajectory):
            print(experience)
            states = experience[0]
            actions = experience[1]
            rewards = experience[2]
            next_states = experience[3]
            dones = experience[4]

            states = torch.from_numpy(np.vstack([states])).float().to(device)
            actions = torch.from_numpy(np.vstack([actions])).long().to(device)
            rewards = torch.from_numpy(np.vstack([rewards])).float().to(device)
            next_states = torch.from_numpy(np.vstack([next_states])).float().to(device)
            dones = torch.from_numpy(np.vstack([dones]).astype(np.uint8)).float().to(device)
            #states, actions, rewards, next_states, dones = experience

            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states 
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)

            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            
            #Gradiant Clipping
            """ +T TRUNCATION PRESENT """
            for param in self.qnetwork_local.parameters():
                param.grad.data.clamp_(-1, 1)
                
            self.optimizer.step()
