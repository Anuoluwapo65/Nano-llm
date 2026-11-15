import torch
import torch.nn as nn 
import torch.utils.data as data 
import torch.nn.functional as F 
import torchvision 
from rich.console import Console
from dataclasses import dataclass
from multiheadattention import MultiheadattentionModule
from layer_norm import LayerNorm
from feed_forward import feedforeward
from encoder import Encoder as EncoderLayer
from decoder import Decoder as DecoderLayer
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
       
       
       
class Encoder(nn.Module):
    def __init__(self, config:ModelConfig):
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_layers)]
        )
        
    def forward(self, x: torch.Tensor, pad_mask = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x,
                      pad_mask = pad_mask
                
            )
        
        return x 
    
    
    
class Decoder(nn.Module):
    def __init__(self, config:ModelConfig):
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_layers)]
        )
        
        
        
    def forward(self, x :torch.Tensor, enc_output : torch.Tensor, src_pad_mask = None, tgt_pad_mask = None)-> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x, 
                enc_output,
                src_pad_mask = src_pad_mask,
                tgt_pad_mask = tgt_pad_mask
            )
            
        return x 
        
        
        
        
        