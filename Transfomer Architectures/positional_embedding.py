
import os 
import random
import rich 
import torch
import torch.nn as nn 
import torch.utils.data as data 
import torch.nn.functional as F 
import torchvision 
from rich.console import Console
from dataclasses import dataclass
import random 
import tokenizers 
from tokenizers import Tokenizer, trainers, models, pre_tokenizers, decoders, normalizers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFD 
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelDecoder
from datasets import load_dataset
import math 


@dataclass
class ModelConfig:
       vocab_size: int = 10000
       embedding_dim:int = 300
       drop_out:float = 0.5
       hidden_dim:int = 1024
       num_layers:int = 6
       attention_drop_out:float = 0.1
       feedforward_dim:int = 8
       head_dim:int = 8
       max_seq_len:int = 512
       model_dim:int = 512



class Positional_embedding(nn.Module):
    def __init__(self, config:ModelConfig):
        super(Positional_embedding, self).__init__()
        
        
        pos_index = torch.arange(0, config.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.model_dim, 2) * (-math.log(10000.0)/config.model_dim))
        
        pe = torch.zeros(config.max_seq_len, config.model_dim)
        
        
        
        pe[:, 0::2] = torch.sin(pos_index * div_term)
        pe[:, 1::2] = torch.cos(pos_index * div_term)
        
        
        
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent = True)
        
    def forward(self, x : torch.Tensor)-> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return x 
    