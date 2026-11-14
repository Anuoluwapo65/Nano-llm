import torch 
import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import seaborn as sns 




###Helper function####

def softmax(x):
    z = x - np.max(x)
    u = np.exp(z)
    return u / u.sum()


def grad_of_softmax(p):
    return p * (1 - p)


def demo_grad(dk, n = 6, seed=42):
    rng = np.random.default_rng(seed = seed)
    return rng.normal(0.0, np.sqrt(dk), size = n)




###settings####    
dks = [64, 256, 1024]
n = 6
keys = [f"k{i}" for i in range(6)]
pallete = sns.color_palette("crest", n)



fig, axes = plt.subplots(
    2,
    len(dks),
    figsize = (18, 19),
    sharey = "row",
    constrained_layout = True
    
)
for col, dk in enumerate(dks):
    logits = demo_grad(dk, n = n, seed = 42)
    
    rows = [
        ("without scaling factor", logits),
        ("with scaling factor", logits / np.sqrt(dk)),
    ]

    for row,(title, i) in enumerate(rows):
        p = softmax(i)
        grad = grad_of_softmax(p)
    
    
    
        df = pd.DataFrame({
            "key":keys,
            "probabilities":p,
            "gradients":grad
        })
    
        ax = axes[row, col]
        sns.barplot(
            data = df,
            x = "key",
            y = "gradients",
            hue = "key",
            ax = ax,     
            palette = pallete
        )
    
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        
        
        ax.set_title(f"{title}\n$d_k = {dk}$", fontsize=14)
        ax.set_ylim(0, 0.3)
        ax.set_xlabel("")
        if col == 0:
            ax.set_ylabel("diag(softmax-jacobian)")
        
        
        for i, v in enumerate(grad):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)



plt.suptitle(
    "scaling factor of each value of the attention-layer - $k_d$\n"
    "Without scaling gradients vanish; $1/\\sqrt{d_k}$ restores healthy scale",
    fontsize=16,
)

sns.despine(fig = fig, ax = ax)

plt.show()
        
        


##### mask-attention #######

seq_len = 8
token = [f"token_{i}" for i in range(seq_len)]


causal_mask = torch.triu(
    torch.full((seq_len, seq_len), float("-inf")),
    diagonal=1,
)
sns.set_theme(context = "notebook", style = "whitegrid")
plt.figure(figsize = (10, 18))

sns.heatmap(causal_mask,
            cmap = "coolwarm",
            xticklabels = token,
            yticklabels = token,
            vmax = 0.7,
            vmin = 0.5,
            linewidths= 0.6
            
    
)

plt.xlabel("the value represent :(Key-value)")
plt.ylabel("the value represent :(query_value)")
plt.title("self-attention-mechanism")
plt.tight_layout()
plt.show()

        
    
    
    
        
    