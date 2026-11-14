import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len = 1024):
        super(PositionalEncoding, self).__init__()
        
        self.model_dim = model_dim 
        self.max_len = max_len 
        
        
        pos = torch.arange(0, max_len).unsqueeze(1) ### underscore(1, num_head)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        
        
        pe = torch.zeros(max_len, model_dim)
        
        
        
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer("pe", pe , persistent = True)
        
        
    def forward(self,x):
        x = x + self.pe[:, :x.size(1), :]
        return x 
    
    
    
    
    
model_dim =  32
max_len = 240
seq_len = 128 
dims_to_plot = [4, 5, 18, 19]




pe = PositionalEncoding(model_dim = model_dim, max_len = max_len).pe[0]
pe = pe[:seq_len, :]
palette = {4: "#2ca02c", 5: "#98df8a", 18: "#d62728", 19: "#ff9896"}

df = pd.concat([
    pd.DataFrame({
            "embedding":pe[:, dim],
            "position":list(range(seq_len)),
            "dimension": dim}
     )
    
    for dim in dims_to_plot
])

sns.set_theme(context = "notebook", style = "whitegrid")
plt.figure(figsize = (10, 17), dpi = 200)


sns.lineplot(
    data = df,
    x = "position",
    y = "embedding",
    hue = "dimension",
    palette= palette
    
)

plt.title("positional_encoding for transfomer model.")
plt.show()