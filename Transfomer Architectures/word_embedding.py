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



def get_seed(verbose, seed = 42):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if verbose:
        print("using this device", device)
    elif torch.backends.mps.is_available():
        values = torch.values("mps")
    if verbose:
        print("using slicion gpu", values)
    else:
        torch.device("cpu")
    if verbose:
        print("using cpu")
        
   
   
   
   
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
       d_model:int = 512
       
       
       
       
       
class WordEmbedding(nn.Module):
    def __init__(self, vocal_size:int, embed_size:int):
        super(WordEmbedding, self).__init__()
        
        self.embedding = nn.Embedding(vocal_size, embed_size)
        
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = self.embedding(x)
        return x 
        
           
        
        
        
        
    
        
        
        

