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
        
        self.attn  = MultiheadattentionModule(config, is_causal = False)
        self.ffn_1 = LayerNorm(config.model_dim)
        
        
        self.linear = feedforeward(config)
        self.ffn_2 = LayerNorm(config.model_dim)
        
        self.drop_out = nn.Dropout(config.drop_out)
        
        
        
    def forward(self, x: torch.Tensor, pad_mask = None):
        _, attn_output = self.attn(x, x, x, pad_mask=pad_mask)
        attn = self.ffn_1(x + self.drop_out(attn_output))
        ffn_output = self.linear(attn)
        return self.ffn_2(attn + self.drop_out(ffn_output))
        
        
        
        
        
        