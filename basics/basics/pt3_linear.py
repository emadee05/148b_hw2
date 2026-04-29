import math
import torch
import torch.nn as nn


class Linear(nn.Module): 
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        construct a linear transformation module 
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0/(in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''
        construct an embedding module 
        '''
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        std = math.sqrt(1.0)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3, b=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # embedding lookup
        # embedding matrix is token_id -> row in matrix -> vector
        return self.weight[x]

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        '''
        construct a layer normalization module 
        '''
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        result = self.weight * x_hat + self.bias
        return result.to(in_dtype)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        '''
        construct a feedforward module 
        '''
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.fc2 = Linear(d_ff, d_model, device=device, dtype=dtype)
    
    def relu(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.zeros_like(x), x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x 

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, device=None, dtype=None):
        '''
        construct a sinusoidal positional encoding module and precompute positional embedding buffer
        '''
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # precompute tensor of shape (max_seq_len, d_model) containing positional embeddings
        # store as a non-persistent buffer using self.register_buffer
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_embedding", pe, persistent=False)
    
    def forward(self, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        given token positions of shape (.., sequence_length), return positional 
        embeddings of shape (.., sequence_length, d_model)
        '''
        return self.positional_embedding[token_positions]

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    '''
    handle keys and queries of shape (batch_size, ..., seq_len, d_k)
    and values of shape (batch_size, ..., seq_len, d_v)
    where ... is any number of other batch-like dimensions 

    return output wiht shape (batch_size, ..., seq_len, d_v)

    should use optional boolean mask of shape (seq_len, seq_len)
    attention probabilities of positions with mask value of True should sum to 1, attention probabilities
    of positions with a mask value of False should be 0.

    '''
    d_k = K.shape[-1]
    QK_T = Q @ K.transpose(-2, -1)
    QK_T = QK_T / math.sqrt(d_k)
    if mask is not None:
        QK_T = QK_T.masked_fill(~mask, float("-inf"))
    return softmax(QK_T, dim=-1) @ V

    
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        '''
        construct a multihead self-attention module 
        '''
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_v = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: (batch_size, sequence_length, d_model)
        return: (batch_size, sequence_length, d_model)
        '''

        B, T, D = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # split into heads
        # (B, T, d_model) -> (B, T, num_heads, d_k) -> (B, num_heads, T, d_k)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_v).transpose(1, 2)

        # compute attention
        # causal attention (lower triangular mask)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))

        # scaled dot product attention on the heads
        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)
        # shape: (B, num_heads, T, d_v)

        # merging heads back (transpose and contiguous)
        # (B, h, T, d_v) -> (B, T, h, d_v) -> (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)

        # output projection
        out = self.output_proj(attn_out) 

        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        '''
        construct a transformer block module 
        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn = MultiheadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln1 = LayerNorm(d_model, device=device, dtype=dtype)
        self.ffn = FeedForward(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = LayerNorm(d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: (batch_size, sequence_length, d_model)
        return: (batch_size, sequence_length, d_model)
        '''
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_length: int, num_layers: int,  device=None, dtype=None):
        '''
        vocab_size: size of vocabulary, for determining the dimensionality of token embedding matrix and LM head output
        context_length: maximum context length for size of sinusoidal positional encoding
        num_layers: number of transformer blocks
        '''
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, context_length, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = LayerNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        '''
        tokens: (batch_size, sequence_length)
        return: (batch_size, sequence_length, vocab_size)
        '''
        B, T = tokens.shape
        embed = self.token_embedding(tokens)
        # add positional encoding
        x = embed + self.positional_encoding(torch.arange(T, device=tokens.device, dtype=tokens.dtype))
        # pass through transformer blocks
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    '''
    transformer language model: distribution p_theta(x_i+1 | x1:i) for each sequence x of length m+1 and i=1..m
    given training set D of sequences of length m, use standard cross-entropy (negative log likelihood) loss:
    L(theta) = -1/N * sum_{x in D} sum_{i=1}^{m} log p_theta(x_i+1 | x1:i)
    '''
    # subtract largest element for numerical stability 
    # cancel out log and exp when possible
    # handle additional batch dim and return average across batch 
    # batch always comes first before vocabulary size
    
    # for each row, find largest logit and subtract it from the row
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logsumexp = torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))
    # each entry is log of sum of exponents of all logits for that position

    # get correct token logit 
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # shape: (batch_size, sequence_length)

    # compute cross-entropy loss
    loss = logsumexp - target_logits
    # shape: (batch_size, sequence_length)

    # return average across batch
    return loss.mean()