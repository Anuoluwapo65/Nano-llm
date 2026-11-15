import os 
import random
import rich 
import torch
import torch.nn as nn 
import torch.utils.data as data 
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-7):
        super(LayerNorm, self).__init__()
        
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
        
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        mean = x.mean(-1, keepdim = True)
        var = x.var(-1, keepdim = True, unbiased = False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.gamma + self.beta 
    
    