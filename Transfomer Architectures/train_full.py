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
from word_embedding import WordEmbedding
from positional_embedding import Positional_embedding
from encoder_decoder import Encoder, Decoder
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
       
       
       
       
class Full_train(nn.Module):
    def __init__(self, config:ModelConfig):
           super(Full_train, self).__init__()
           
           self.word_embeddings = WordEmbedding(config.vocab_size, config.model_dim)
           self.positional_encoding = Positional_embedding(config)
           
           self.encoder = Encoder(config)
           self.decoder = Decoder(config)
           
           
           self.out_proj = nn.Linear(config.model_dim, config.vocab_size)
           
           self.apply(self._init_weights)
           self._values_weight()
           
           
    def _values_weight(self):
           self.out_proj.weight = self.word_embeddings.embedding.weight
           
    def _init_weights(self, module):
           if isinstance(module, nn.Linear):
                  nn.init.xavier_uniform_(module.weight)
                  if module.bias is not None:
                         nn.init.constant_(module.bias, 0)
                         
           elif isinstance(module, nn.Embedding):
                  nn.init.normal_(module.weight, mean = 0, std = 0.02)
                  
            
           elif isinstance(module, LayerNorm):
                  nn.init.ones_(module.gamma)
                  nn.init.zeros_(module.beta)
                  
                  
                  
                  
    def forward(
        self,
        src_input: torch.Tensor,
        tgt_input: torch.Tensor,
        src_pad_mask=None,
        tgt_pad_mask=None,
    ) -> torch.Tensor:
        # Get Src and Tgt embeddings
        src_embeddings = self.positional_encoding(
            self.word_embeddings(src_input)
            * math.sqrt(self.word_embeddings.embedding.embedding_dim)
        )
        tgt_embeddings = self.positional_encoding(
            self.word_embeddings(tgt_input)
            * math.sqrt(self.word_embeddings.embedding.embedding_dim)
        )

        # Feed through Encoder
        enc_output = self.encoder(src_embeddings, pad_mask=src_pad_mask)

        # Feed through Decoder with Encoder output
        dec_output = self.decoder(
            tgt_embeddings,
            enc_output,
            src_pad_mask=src_pad_mask,
            tgt_pad_mask=tgt_pad_mask,
        )

        logits = self.out_proj(dec_output)
        return torch.log_softmax(logits, dim=-1)
 
if __name__ == "__main__":
    config = ModelConfig()
    src = torch.randint(0, config.vocab_size, (8, 10))
    tgt = torch.randint(0, config.vocab_size, (8, 20))
    model = Full_train(config)
    logits = model(src, tgt)
    assert logits.shape == (8, 20, config.vocab_size)
    print("Full model test passed. Logits shape:", logits.shape)
    print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model structure:", model)
 
 
 
       
           
                  
           
                


              
           
           