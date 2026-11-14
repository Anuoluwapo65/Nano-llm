import os 
import random
import rich 
import torch
import torch.nn as nn 
import torch.utils.data as data 
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
       vocab_size: int = 10000
       embedding_dim:int = 300
       drop_out:float = 0.5
       hidden_dim:int = 1024
       num_layers:int = 6
       attention_drop_out:float = 0.1
       head_dim:int = 8
       feedforward_dim:int = 8
       max_seq_len:int = 512
       model_dim:int = 512
       
       
       
class feedforeward(nn.Module):
    def __init__(self, config:ModelConfig):
        super(feedforeward, self).__init__()
        
        
        self.fc1 = nn.Linear(config.model_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.model_dim)
        
        
        
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))
    