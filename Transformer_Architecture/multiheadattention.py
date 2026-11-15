import torch.nn as nn 
import torch.utils.data as data 
import torch.nn.functional as F 
import torchvision
from dataclasses import dataclass
import torch
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
       
       
       
##### scaled dot product attention #######

def scaled_dot_product_attention(q, k, v, mask = None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(q.new_tensor(d_k, dtype=torch.float32))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights



def create_mask(key_len:int, query_len:int):
    return torch.tril(
        torch.ones(query_len, key_len, dtype=torch.bool)
    )[None, None, :, :]




### Multihead attention module #### 

class MultiheadattentionModule(nn.Module):
    def __init__(self,config:ModelConfig, is_causal:bool):
        super(MultiheadattentionModule, self).__init__()
        self.causal = is_causal
        self.config = config 
        self.hidden_dim = config.hidden_dim
        self.model_dim = config.model_dim
        self.head_dim = config.head_dim
        self.num_head = config.model_dim // config.head_dim
        
        
        self.q = nn.Linear(self.model_dim, self.model_dim)
        self.k = nn.Linear(self.model_dim, self.model_dim)
        self.v = nn.Linear(self.model_dim, self.model_dim)
        
        self.out = nn.Linear(self.model_dim, self.model_dim)
        
    def create_casual(self, pad_mask, key_len:int, query_len:int, device):
        mask = None
        if self.causal:
            mask = create_mask(key_len, query_len).to(device)
        
    
    
        if pad_mask is not None:
            pad_mask = pad_mask[:, None, None,:].to(device)
            mask = pad_mask if mask is None else(pad_mask&mask)
        return mask
    
    
    def forward(self, q, k, v, pad_mask=None):
        device = q.device
        b, q_len,_ =  q.shape
        _,kv_len,_ = k.shape
        mask = self.create_casual(
            pad_mask=pad_mask, key_len=kv_len, query_len=q_len, device=device
        )
        
        
        q = self.q(q).view(b, q_len, self.num_head, self.head_dim).transpose(1, 2)
        v = self.v(v).view(b, kv_len, self.num_head, self.head_dim).transpose(1, 2)
        k = self.k(k).view(b, kv_len, self.num_head, self.head_dim).transpose(1, 2)
        
        
        
        out, attn_values = scaled_dot_product_attention(q, k, v, mask=mask)
        attn_output = out.transpose(1, 2).contiguous().view(b, q_len, self.num_head * self.head_dim)
        output = self.out(attn_output)
        
        return attn_values, output

            
        
        
        
        
        
        
        
            
            
            
        
        