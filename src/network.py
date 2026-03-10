import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    def __init__(self,state_size = 8,action_size = 4,seed = 42):
        super(DuelingQNetwork,self).__init__()
        self.seed = torch.manual_seed(seed)
        #Hidden Layers for DQN
        self.fc1 = nn.Linear(state_size,256)
        self.fc2 = nn.Linear(256,256)
        
        #Value Head
        self.val_fc = nn.Linear(256,128)
        self.val_out = nn.Linear(128,1)
        
        #Advantage Stream Layer
        self.adv_fc = nn.Linear(256,128)
        self.adv_out = nn.Linear(128,action_size)
        
    def forward(self,state):
        #Relu for DQN
        feature = F.relu(self.fc1(state))
        feature = F.relu(self.fc2(feature))
        
        #Relu for Value Stream
        val = F.relu(self.val_fc(feature))
        val = self.val_out(val)
        
        #Advantage Stream Relu
        adv = F.relu(self.adv_fc(feature))
        adv = self.adv_out(adv)
        
        #Recombination
        q_vals = val + (adv - adv.mean(dim=1,keepdim = True))
        return q_vals
        
                
        
        
        