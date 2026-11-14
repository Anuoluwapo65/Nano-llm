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
       
       
       
       
class Decoder(nn.Module):
    def __init__(self, config:ModelConfig):
        super(Decoder, self).__init__()
        
        
        self.attn = MultiheadattentionModule(config, is_causal=True)
        self.layer_1 = LayerNorm(config.model_dim)
        
        
        self.cross_attn = MultiheadattentionModule(config, is_causal= False)
        self.layer_2 = LayerNorm(config.model_dim)
        
        
        
        self.linear = feedforeward(config)
        self.layer_3 = LayerNorm(config.model_dim)
        
        
        
    def forward(self, x : torch.Tensor, enc_output : torch.Tensor,
                src_pad_mask=None, tgt_pad_mask=None) -> torch.Tensor:
        _, attn_output = self.attn(x, x, x, pad_mask=tgt_pad_mask)
        x = self.layer_1(x + attn_output)
        
        
        _, cross_output = self.cross_attn(x, enc_output, enc_output, pad_mask=src_pad_mask)
        x = self.layer_2(x + cross_output)
        
        
        x = self.layer_3(x + self.linear(x))
        
        
        return x 