# AI for Self Driving Car

# Importing the libraries

import numpy as np #allows us to work with arrays
import random #we will be taking some random samples from different batches while using experience replay
import os #used to load model
import torch #used to implement neural networks and it is preffered over others as it can handle dynamic graphs
import torch.nn as nn #has all the tools to implement neural networks
import torch.nn.functional as F  #contains different functions to implement neural networks specially loss function
import torch.optim as optim #to import optimizers for stochastic gradient descent
import torch.autograd as autograd #to put tensor into a variable that will also contain a gradient (tensors are more advanced arrays) 
from torch.autograd import Variable

#creating the architecture of the neural network

class Network(nn.Module): #inheritance has been used to inherit all the tools of module class
    
    def __init__(self, input_size, nb_action): #self is used to specify that this is a variable of the object
        super(Network, self).__init__() #to use all the tools of module(inheritance has been used)
        self.input_size = input_size #every time if the size of our input neuron is 5 our object will also have size 5
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
        
    def forward(self, state): #this function will activate the neurons and will also return the q value of each and every possible action depending upon the input state
       x = F.relu(self.fc1(state)) #x are the hidden neurons and relu function is used to transfer input neurons from the input state to hidden layer
       q_values = self.fc2(x)
       return q_values  
   
#implementing experiance replay
       
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event): #event has 4 tuple form
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        #creating one batch for each of the state, action, reward
        samples = zip(*random.sample(self.memory, batch_size)) #we are taking random samples from the memory of fixed batch_size and zip function is used to just reshape the list, for instance if list = ((1,2,3), (4,5,6)), then zip(*list) = ((1,4), (2,5), (3,6))
        return map(lambda x: Variable(torch.cat(x, 0)), samples) #lambda function will take samples, concatenate them w.r.t first dimension and then eventually  convert them into torch variables that contain both torch and the gradient
    
#implementing deep q learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma  = gamma
        self.input_size = input_size
        self.nb_action = nb_action
        self.reward_window = []
        self.model = Network(input_size, nb_action) #created one neural network for our deep q learning model
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0) #we created fake dimension with the help of unsqueeze and it is our first dimension corresponding to index 0. We also have a dimension corresponding to tensor
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs  = F.softmax(self.model(Variable(state, volatile = True))*122) #temperature parameter is 7 and state is a torch tensor and we will convert into variable
        #higher the temeprature parameter higher will be the probability of wining q value
        action = probs.multinomial(1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.long().unsqueeze(1)).squeeze(1) #will select only one best possible action for the selected state
        next_outputs = self.model(batch_next_state).detach().max(1)[0] #we get maximum of the q values of the next states represented by 0 according to all the actions that are represented by index 1
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True) #backpropagtes the error into the neural network
        self.optimizer.step() #update the weights
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))) #all the arguments in push must be tensors
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1) #added 1 so that the denominator is never 0 otherwise if den = 0 it will crash our program
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict
                    }, 'last_brain.pth') #we have created keys to save our model and optimizers and all this will saved in the file last_brain.pth
        
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print(" laoding checkpoint........")
            checkpoint = torch.laod('last_brain.pth') #will load the file if it exists
            self.model.load_state_dict(checkpoint['state_dict']) #will update the model from last checkpoint
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found......")
    
            

       
   
       
        
        
        
    
    
           
            
            
            
            