def main()->None:
    import torch
    import torch.nn as nn
    
    class InputEmbeddings(nn.Module):
        
        def __init__(self,d_model: int,vocab_size: int):
            super().__init__()
            self.d_model=d_model
            self.vocab_size=vocab_size
            self.embeddings=nn.Embedding(vocab_size,d_model)
        
        def forward(self,x):
            return self.embeddings(x)*torch.sqrt(torch.tensor(self.d_model))
    class PositionalEncoding(nn.Module):
        
        def __init__(self,d_model:int,seq_len: int,dropout:float)->None:
            super().__init__()
            self.d_model=d_model
            self.seq_len=seq_len
            self.dropout=nn.Dropout(dropout)
            
            # Create a matrix of shape seq_len x d_model
            pe = torch.zeros(seq_len, d_model)
            # Create a position tensor of shape seq_len x 1
            position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)        
            div_term = torch.exp(torch.arange(0,d_model,2).float()*(-torch.log(10000.0)/d_model))
            pe[:,0::2] = torch.sin(position*div_term)
            pe[:,1::2] = torch.cos(position*div_term)#Odd indices
            pe = pe.unsqueeze(0)#Add a batch dimension (1,seq_len,d_model)
            self.register_buffer('pe',pe)
    
        def forward(self,x):
            x = x + (self.pe[:,:x.size(1)]).requires_grad_(False)
            return self.dropout(x)
    class LayerNormalization(nn.Module):
    
        def __init__(self,eps:float=1e-6):
            super().__init__()
            self.eps=eps
            self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative parameter
            self.bias = nn.Parameter(torch.zeros(1))# Additive parameter
    
        def forward(self,x):
            mean = x.mean(-1,keepdim=True)
            std = x.std(-1,keepdim=True)
            return self.alpha*(x-mean)/(std+self.eps)+self.bias
    class FeedForwardBlock(nn.Module):
        def __init__(self,d_model:int,d_ff:int,dropout:float):
            super().__init__()
            self.linear1=nn.Linear(d_model,d_ff)
            self.dropout=nn.Dropout(dropout)
            self.linear2=nn.Linear(d_ff,d_model)
    
        def forward(self,x):
            # x: batch_size x seq_len x d_model --> batch_size x seq_len x d_ff --> batch_size x seq_len x d_model
            return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    class MultiHeadAttention(nn.Module):
        def __init__(self,d_model:int,n_heads:int,dropout:float):
            super().__init__()
            self.d_model=d_model
            assert d_model%n_heads==0, "d_model should be divisible by n_heads"
            self.n_heads=n_heads
            self.d_k=d_model//n_heads
            self.dropout=nn.Dropout(dropout)
            self.linear_q=nn.Linear(d_model,d_model)
            self.linear_k=nn.Linear(d_model,d_model)
            self.linear_v=nn.Linear(d_model,d_model)
            self.linear_o=nn.Linear(d_model,d_model)
            
            
            
            

if __name__ == "__main__":
    main()