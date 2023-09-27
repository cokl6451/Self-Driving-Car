import numpy as np
import random
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Create neural network architecture, inheriting from torch.nn
class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        # Number of inputs
        self.input_size = input_size
        # Number of outputs
        self.nb_action = nb_action
        # Add full connections
        self.fc1 = nn.Linear(input_size, 64) #input -> hidden
        self.fc2 = nn.Linear(64, nb_action) #hidden -> output

    def forward(self, state):
        # Forward propagation
        # Rectifier function to activate hidden neurons
        x = F.relu(self.fc1(state))
        # Get Q-values
        q_values = self.fc2(x)
        return q_values
    
# Implement Experience Replay
class ReplayMemory(object):

    def __init__(self, capacity):
        # Initialize capacity
        self.capacity = capacity
        # Initialize memory
        self.memory = []

    # Push event to memory
    def push(self, event):
        # Append event to memory
        self.memory.append(event)
        # If memory is full, remove first event
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        # Get random samples from memory                               action  reward    
        samples = zip(*random.sample(self.memory, batch_size)) #zip*: (1,2,3),(4,5,6) -> (1,4),(2,5),(3,6)
        # Convert samples to torch variables
        return map(lambda x: Variable(torch.cat(x, 0)), samples) #concatenate samples into one tensor
    
# Implement Deep Q-Learning
class Dqn():
    
    # Initialize parameters and neural network
    def __init__(self, input_size, nb_action, gamma):
        # Discount factor
        self.gamma = gamma
        # Sliding window of mean of last rewards
        self.reward_window = []
        # Create model
        self.model = Network(input_size, nb_action)
        # Create memory
        self.memory = ReplayMemory(100000)
        # Create optimizer (stochastic gradient descent)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        # Create last state
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        # Create last action
        self.last_action = 0
        # Create last reward
        self.last_reward = 0
        # Create device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    # Select action
    def select_action(self, state):
        # Get probabilities of actions
        with torch.no_grad():
            # Get probabilities of actions
            probs = torch.softmax(self.model(state.to(self.device)) * 100, dim=1) #Temperature parameter T = 100
            # Get action
            action = probs.multinomial(num_samples=1)
            return action.item()
    
    # Learn from new state
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # Get outputs from model
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # Get next outputs from model
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        # Get target
        target = self.gamma * next_outputs + batch_reward
        # Compute loss
        td_loss = F.smooth_l1_loss(outputs, target)
        # Reset optimizer
        self.optimizer.zero_grad()
        # Backpropagate loss
        td_loss.backward(retain_graph = True)
        # Update weights
        self.optimizer.step()
        
    # Update state
    def update(self, reward, new_signal):
        # Get new state
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # Push new transition to memory
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        # Get new action
        action = self.select_action(new_state)
        # If memory is full, learn
        if len(self.memory.memory) > 100:
            # Learn
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        # Update last state
        self.last_state = new_state
        # Update last action
        self.last_action = action
        # Update last reward
        self.last_reward = reward
        # Update reward window
        self.reward_window.append(reward)
        # If reward window is full, remove first reward
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        # Return action
        return action
    
    # Get score
    def score(self):
        # Return mean of reward window
        return sum(self.reward_window)/(len(self.reward_window) + 1)
    
    # Save model
    def save(self):
        # Save model
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                   }, 'last_brain.pth')

    # Load model
    def load(self):
        # Check if file exists
        if os.path.isfile('last_brain.pth'):
            # Load model
            print("=> Loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loading Complete.")
        else:
            # Print error
            print("No checkpoint found.")
