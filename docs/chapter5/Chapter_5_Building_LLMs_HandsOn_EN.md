# Chapter 5: Building a Large Language Model

## 5.1 Implementing LLaMA2 from Scratch

Meta (formerly Facebook) released LLaMA, its first large language model based on the Transformer architecture, in February 2023, and released LLaMA2, a model in the same series, in July of the same year. In Chapter 4 we learned about LLMs and how they are trained. In this section, we will learn how to implement an LLaMA2 model hands-on.

The LLaMA2 model structure is shown in Figure 5.1:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/5-images/LLama2.png" alt="alt text" width="100%">
    <p>Figure 5.1 LLaMA2 structure</p>
</div>

### 5.1.1 Defining Hyperparameters

First of all, we need to define some hyperparameters, including the model size, number of layers, number of heads, word embedding dimensions, hidden layer dimensions, etc. These hyperparameters can be adjusted according to actual conditions.

Here we customize a `ModelConfig` class to store and record our hyperparameters. Here we inherit the `PretrainedConfig` class, which is a parameter class in the `transformers` library. We can easily use some functions in the `transformers` library by inheriting this class, and it also makes it easy to export the model in Hugging Face format later.

```python
from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"
    def __init__(
            self,
            dim: int = 768, # Model dimension
            n_layers: int = 12, # Number of Transformer layers
            n_heads: int = 16, # Number of attention heads
            n_kv_heads: int = 8, # Number of key/value heads
            vocab_size: int = 6144, # Vocabulary size
            hidden_dim: int = None, # Hidden layer dimension
            multiple_of: int = 64, 
            norm_eps: float = 1e-5, # eps for the normalization layer
            max_seq_len: int = 512, # Maximum sequence length
            dropout: float = 0.0, # Dropout probability
            flash_attn: bool = True, # Whether to use Flash Attention
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)
```

> When `args` appears in the following code, the default is to the above `ModelConfig` parameter configuration.

Let's take a look at the meaning of some of the hyperparameters, such as `dim` is the model dimension, `n_layers` is the number of layers of Transformer, `n_heads` is the number of attention heads, `vocab_size` is the vocabulary size, `max_seq_len` is the maximum sequence length of the input, etc. The above code also comments on each parameter in detail. In the subsequent code, we will build our model based on these hyperparameters.

### 5.1.2 Building RMSNorm

`RMSNorm` can be expressed by the following mathematical formula:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \cdot \gamma
$$

in:
- $x_i$ is the $i$-th element of the input vector
- $\gamma$ is a learnable scaling parameter (corresponding to `self.weight` in the code)
- $n$ is the number of dimensions of the input vector
- $\epsilon$ is a small constant for numerical stability (to avoid dividing by zero)

This normalization helps stabilize the learning process by ensuring that the scale of the weights does not become too large or too small, which is particularly useful in deep learning models with many layers.

We can implement `RMSNorm` with the following code:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # eps prevents division by zero
        self.eps = eps
        # weight is a learnable parameter, initialized to all ones
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Compute the core part of RMSNorm
        # x.pow(2).mean(-1, keepdim=True) computes the mean of the squares of input x
        # torch.rsqrt is the reciprocal of the square root, which gives the denominator of RMSNorm; eps is added to keep the denominator from being 0
        # Finally multiply by x to get the RMSNorm result
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # The forward function is the forward pass of the model
        # First convert input x to float, apply RMSNorm, then convert back to the original data type
        # Finally multiply by weight, a learnable scaling factor of RMSNorm
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

Moreover, we can use the following code to test the `RMSNorm` module. You can see that the final output of the code is `torch.Size([1, 50, 288])`, which is consistent with the shape of our input, indicating that the module implementation is correct, and normalization will not change the shape of the input.

```python
norm = RMSNorm(args.dim, args.norm_eps)
x = torch.randn(1, 50, args.dim)
output = norm(x)
print(output.shape)

out:
torch.Size([1, 50, 768])
```

### 5.1.3 Building LLaMA2 Attention

In the LLaMA2 model, although only the LLaMA2-70B model uses Grouped-Query Attention (GQA), we still choose to use GQA to build our LLaMA Attention module, which can improve the efficiency of the model and save some GPU memory.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/5-images/llama2-attention.png" alt="alt text" width="50%">
    <p>Figure 5.2 LLaMA2 Attention Structure</p>
</div>

#### 5.1.3.1 repeat_kv

In the LLaMA2 model, we need to extend the dimensions of keys and values to the same dimensions as those of the query so that attention calculations can be performed. We can implement `repeat_kv` with the following code:

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # Get the shape of the input tensor: batch size, sequence length, number of key/value heads, dimension of each head
    bs, slen, n_kv_heads, head_dim = x.shape
    
    # If the number of repetitions is 1, no repetition is needed; return the original tensor
    if n_rep == 1:
        return x
    
    # Expand and reshape the tensor to repeat the key/value heads
    return (
        x[:, :, :, None, :]  # Add a new dimension at the fourth position (before the head dimension)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # Expand the new dimension to size n_rep to achieve repetition
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # Reshape, merging the key/value head dimension and the repetition dimension
    )
```

In the above code:

- First, get the shape of the input tensor: First, the code uses x.shape to obtain the shape of the input tensor, including the batch size (bs), the sequence length (slen), the number of key/value heads (n_kv_heads), and the dimension size of each head (head_dim).

- Then, check the number of repetitions: Next, the code checks whether the number of repetitions n_rep is 1. If it is 1, it means that no need to repeat the key and value, and the original tensor x is returned directly.

- Finally, expand and reshape the tensor:
  - Add a new dimension after the third dimension (i.e. the dimension of the key/value heads), forming `x[:, :, :, None, :]`.
  - Use the `expand` method to expand the newly added dimension to the `n_rep` size to achieve the repetition effect of key/value pairs.
  - Finally, the `reshape` method reshapes the tensor, merging the expanded dimension back into the number of key/value heads, that is, `x.reshape(bs, slen, n_kv_heads * n_rep, head_dim)`, so that the final tensor shape achieves the effect of consistency with the query dimension.

#### 5.1.3.2 Rotary Embedding

Next, we implement the rotary embedding, which is an important component in the LLaMA2 model, which can provide stronger context information for the attention mechanism, thereby improving the performance of the model.

First, we want to construct a function that obtains the real and imaginary parts of the rotary embedding:

```python
# Note: dim here should be dim//n_head, because we apply the rotary embedding to each head
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float() generates a sequence starting at 0 with step 2, of length dim/2
    # Then divide each element by dim and take the reciprocal of theta raised to it, giving the frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Generate a sequence from 0 to end, of length end
    t = torch.arange(end, device=freqs.device)
    # Compute the outer product to get a 2D matrix; each row is an element of t multiplied by the elements of freqs
    freqs = torch.outer(t, freqs).float()
    # Compute the cosine of the frequencies to get the real part
    freqs_cos = torch.cos(freqs)
    # Compute the sine of the frequencies to get the imaginary part
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin
```

- Calculate frequency sequence:
  - `torch.arange(0, dim, 2)[: (dim // 2)].float()` generates a sequence starting from 0 and step size 2, with a length of half of `dim`.
  - Divide each element by `dim` and take the reciprocal of `theta` to obtain a frequency sequence `freqs`. This step generates frequencies suitable for the rotary embedding.
- Generate the position sequence:
  - `t = torch.arange(end, device=freqs.device)` Generates a sequence from `0` to `end` with a length of `end`. `end` is usually the maximum length of the sequence.
- Calculate the outer product of frequency
  - `freqs = torch.outer(t, freqs).float()` calculates the outer product of the time series `t` and the frequency series `freqs` to obtain a two-dimensional matrix `freqs`. Each row is an element of the time series `t` multiplied by the element of the frequency series `freqs`.
- Compute the real and imaginary parts
  - `freqs_cos = torch.cos(freqs)` calculates the cosine value of the frequency matrix `freqs` to obtain the real part of the rotary embedding.
  - `freqs_sin = torch.sin(freqs)` calculates the sine value of the frequency matrix `freqs` to obtain the imaginary part of the rotary embedding.

Finally, the function returns two matrices `freqs_cos` and `freqs_sin`, respectively, representing the real and imaginary parts of the rotary embedding for subsequent calculations.

Next, we will construct the `reshape_for_broadcast` function that adjusts the shape of tensor. The main purpose of this function is to adjust the shape of `freqs_cis` so that it aligns with the dimension of `x` when performing broadcast operations, so that it can perform correct tensor operations.

```python
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # Get the number of dimensions of x
    ndim = x.ndim
    
    # Assert that 1 is within the dimension range of x
    assert 0 <= 1 < ndim
    
    # Assert that the shape of freqs_cis matches the second and last dimensions of x
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    
    # Build a new shape where every dimension except the second and last is 1, so that freqs_cis can be broadcast against x
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    
    # Reshape freqs_cis to the new shape and return it
    return freqs_cis.view(shape)
```

Finally, we can implement rotary embedding through the following code:

```python
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # Convert the query and key tensors to float and reshape them to separate the real and imaginary parts
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # Reshape the frequency tensors for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # Apply the rotation, computing the rotated real and imaginary parts separately
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Merge the last two dimensions and restore the original tensor shape
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

Here we give the code that can test the `apply_rotary_emb` function. You can also try to add breakpoints to the code to view the calculation results of each step.

```python
xq = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim
xk = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim

# Use the precompute_freqs_cis function to get sin and cos
cos, sin = precompute_freqs_cis(288//6, 50)
print(cos.shape, sin.shape)
xq_out, xk_out = apply_rotary_emb(xq, xk, cos, sin)

xq_out.shape, xk_out.shape
```

OUT:
```
torch.Size([50, 24]) torch.Size([50, 24])

(torch.Size([1, 50, 6, 48]), torch.Size([1, 50, 6, 48]))
```

#### 5.1.3.3 Assembling LLaMA2 Attention

We have completed the implementation of rotary embedding above, and now we can build the LLaMA2 Attention module.

```python
class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        # Determine the number of heads used for keys and values, depending on whether n_kv_heads is specified.
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Make sure the total number of heads is divisible by the number of key/value heads.
        assert args.n_heads % self.n_kv_heads == 0

        # Model-parallel size, defaults to 1.
        model_parallel_size = 1
        # Number of local heads, equal to the total number of heads divided by the model-parallel size.
        self.n_local_heads = args.n_heads // model_parallel_size
        # Number of local key/value heads, equal to the number of key/value heads divided by the model-parallel size.
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # Number of repetitions, used to expand the size of keys and values.
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # Dimension of each head, equal to the model dimension divided by the total number of heads.
        self.head_dim = args.dim // args.n_heads

        # Define the weight matrices.
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # Output weight matrix.
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Define dropout.
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # Store the dropout probability.
        self.dropout = args.dropout

        # Check whether to use Flash Attention (requires PyTorch >= 2.0).
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # If Flash Attention is not supported, use a manually implemented attention mechanism and set up the mask.
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Create an upper-triangular matrix to mask out future positions.
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # Register it as a buffer of the model
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        # Get the batch size and sequence length, [batch_size, seq_len, dim]
        bsz, seqlen, _ = x.shape

        # Compute the queries (Q), keys (K) and values (V).
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # Reshape to fit the head dimension.
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply rotary positional embedding (RoPE).
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # Expand keys and values to match the number of repetitions.
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Treat the heads as a batch dimension.
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Choose the implementation depending on whether Flash Attention is supported.
        if self.flash:
            # Use Flash Attention.
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # Use the manually implemented attention mechanism.
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        # Restore the time dimension and merge the heads.
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # Final projection back into the residual stream.
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
```

Similarly, you can use the following code to test the attention module. You can see that the final output shape of the code is `torch.Size([1, 50, 768])`, which is consistent with the shape we input, indicating that the implementation of the module is correct.

```python
# Create an Attention instance
attention_model = Attention(args)

# Simulated input data
batch_size = 1
seq_len = 50  # Assume the actual sequence length used is 50
dim = args.dim
x = torch.rand(batch_size, seq_len, dim)  # Randomly generate the input tensor
# freqs_cos = torch.rand(seq_len, dim // 2)  # Simulated cos frequencies for RoPE
# freqs_sin = torch.rand(seq_len, dim // 2)  # Simulated sin frequencies for RoPE

freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)

# Run the Attention model
output = attention_model(x, freqs_cos, freqs_sin)

# The output shape of attention is still [batch_size, seq_len, dim]
print("Output shape:", output.shape)
```

OUT:
```
Output shape: torch.Size([1, 50, 768])
```

### 5.1.4 Building the LLaMA2 MLP module

Compared with the LLaMA2 Attention module we implemented earlier, the implementation of the LLaMA2 MLP module is simpler. We can implement `MLP` with the following code:

```python
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # If the hidden dimension is not specified, set it to 4 times the input dimension
        # then reduce it to 2/3, and finally make sure it is a multiple of multiple_of
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # Define the first linear layer, from the input dimension to the hidden dimension
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # Define the second linear layer, from the hidden dimension to the input dimension
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # Define the third linear layer, from the input dimension to the hidden dimension
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # Define the dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Forward pass function
        # First, input x passes through the first linear layer and the SiLU activation
        # Then the result is multiplied by the output of passing x through the third linear layer
        # Finally, pass through the second linear layer and the dropout layer
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

We focus on observing the implementation of the `forward` function. First, the input `x` passes through the first linear layer `self.w1` and the `SILU` activation function; then the result is multiplied by the output of passing `x` through the third linear layer `self.w3`; finally, it passes through the second linear layer `self.w2` and the `dropout` layer to produce the final output.

Similarly, you can use the following code to test the `LLaMAMLP` module. You can see that the final output shape of the code is `torch.Size([1, 50, 768])`, which is consistent with the shape we input, indicating that the implementation of the module is correct.

```python
# Create an MLP instance
mlp = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
# Randomly generate data
x = torch.randn(1, 50, args.dim)
# Run the MLP model
output = mlp(x)
print(output.shape)
```

OUT:
```
torch.Size([1, 50, 768])
```

### 5.1.5 LLaMA2 Decoder Layer

At this point, we have implemented the Attention module and the MLP module of the `LLaMA2` model. Next, we can build the `LLaMA2` Decoder Layer.

```python
class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        # Number of heads for multi-head attention
        self.n_heads = args.n_heads
        # Input dimension
        self.dim = args.dim
        # Dimension of each head, equal to the input dimension divided by the number of heads
        self.head_dim = args.dim // args.n_heads
        # LLaMA2Attention object for the multi-head attention computation
        self.attention = Attention(args)
        # LLaMAMLP object for the feed-forward network computation
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        # Layer ID
        self.layer_id = layer_id
        # Normalization layer for the attention computation
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization layer for the feed-forward network computation
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # Forward pass function
        # First, input x goes through the attention normalization layer and then attention; the result is added to x to get h
        # Then h goes through the feed-forward normalization layer and the feed-forward network; the result is added to h to get the output
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
```

`DecoderLayer` combines the Attention module and the MLP module we completed above to implement a complete `Transformer` module.

Similarly, you can use the following code to test the `DecoderLayer` module. You can see that the final output shape of the code is `torch.Size([1, 50, 768])`, which is consistent with the shape we input, indicating that the implementation of the module is correct.

```python
# Create a LLaMADecoderLayer instance
decoderlayer = DecoderLayer(0, args)

# Simulated input data
dim = args.dim
seq_len = 50

x = torch.randn(1, seq_len, dim) # [bs, seq_len, dim]

freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)

out = decoderlayer(x, freqs_cos, freqs_sin)

print(out.shape) # Same shape as the input x: [batch_size, seq_len, dim]
```

OUT:
```
torch.Size([1, 50, 768])
```

### 5.1.6 Building the LLaMA2 model

OK, we have completed the implementation of all the modules mentioned above, and the next is an exciting moment. We can build the `LLaMA2` model. The `LLaMA2` model is to stack the `DecoderLayer` modules to form a complete `Transformer` model.

```python
class Transformer(PreTrainedModel):
    config_class = ModelConfig  # Configuration class
    last_loss: Optional[torch.Tensor] # Stores the most recently computed loss

    def __init__(self, args: ModelConfig = None):
        super().__init__(args)
        # Initialize model parameters
        self.args = args
        # Vocabulary size
        self.vocab_size = args.vocab_size
        # Number of layers
        self.n_layers = args.n_layers

        # Token embedding layer
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # Dropout layer
        self.dropout = nn.Dropout(args.dropout)
        # Decoder layers
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        # Normalization layer
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Output layer
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Share (tie) the weights of the embedding layer and the output layer
        self.tok_embeddings.weight = self.output.weight 

        # Precompute the frequencies for the relative positional embedding
        freqs_cos, freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Initialize all weights
        self.apply(self._init_weights)
        # Apply special scaled initialization to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

        # Initialize the attribute holding the loss of the last forward pass
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()  # Output container
        self._no_split_modules = [name for name, _ in self.named_modules()]  # List of modules not to split

    def _init_weights(self, module):
        # Weight initialization function
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **keyargs) -> torch.Tensor:
        """
        - tokens: Optional[torch.Tensor], input token tensor.
        - targets: Optional[torch.Tensor], target token tensor.
        - kv_cache: bool, whether to use the key/value cache.
        - keyargs: other keyword arguments.

        - self.OUT: CausalLMOutputWithPast, contains the logits and the loss.
        """

        if 'input_ids' in keyargs:
            tokens = keyargs['input_ids']
        if 'attention_mask' in keyargs:
            targets = keyargs['attention_mask']

        # Forward pass function
        _bsz, seqlen = tokens.shape
        # Pass through the embedding layer and the dropout layer
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        # Get the frequencies for the relative positional embedding
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # Pass through the decoder layers
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        # Pass through the normalization layer
        h = self.norm(h)

        if targets is not None:
            # If targets are given, compute the loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        else:
            # Small inference-time optimization: only run the output layer on the last position
            logits = self.output(h[:, [-1], :]) 
            self.last_loss = None

        # Set the output
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    
    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        Given an input sequence idx (a long tensor of shape (bz,seq_len)), complete the sequence by generating new tokens repeatedly.
        Runs in model.eval() mode. This is a less efficient sampling version that does not use a k/v cache.
        """
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # If the sequence context is too long, truncate it to the maximum length
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            
            # Forward pass to get the logits for the last position in the sequence
            logits = self(idx_cond).logits
            logits = logits[:, -1, :] # Keep only the output of the last time step
            
            if temperature == 0.0:
                # Pick the most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # Scale the logits and apply softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            

            if idx_next == stop_id:
                break

            # Append the sampled index to the sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:] # Return only the generated tokens
```

Similarly, you can use the following code to test the `Transformer` module. You can see that the final output shape of the code is `torch.Size([1, 1, 6144])`, which is consistent with the shape we input, indicating that the implementation of the module is correct.

```python
# LLaMA2Model.forward takes two arguments, tokens and targets, where tokens is the input tensor and should be of int type
x = torch.randint(0, 6144, (1, 50)) # [bs, seq_len]
# Instantiate LLaMA2Model
model = Transformer(args=args)
# Count all parameters of the model
num_params = sum(p.numel() for p in model.parameters())
print('Number of parameters:', num_params)

out = model(x)
print(out.logits.shape) # [batch_size, 1, vocab_size]
```

OUT:
```
Number of parameters: 82594560
torch.Size([1, 1, 6144])
```

## 5.2 Training Tokenizer

In natural language processing (NLP), Tokenizer is a tool that breaks text into smaller units (called tokens). These tokens can be words, subwords, characters, or even specific symbols. Tokenization is the first step in NLP, which directly affects the effectiveness of subsequent processing and analysis. Different types of tokenizers are suitable for different application scenarios. The following are several common tokenizers and their characteristics.

### 5.2.1 Word-based Tokenizer

**Word-based Tokenizer** is the simplest and most intuitive tokenization method. It splits text into words by spaces and punctuation. The advantage of this approach is that it is simple and direct, easy to implement, and consistent with human intuition about language. However, it also has some obvious disadvantages, such as the inability to handle out-of-vocabulary (OOV) words and rare words, and the processing of compound words (such as "New York") or abbreviations (such as "don't") is not fine enough. In addition, Word-based Tokenizer also encounters challenges when dealing with different languages, because some languages (such as Chinese and Japanese) do not have explicit word separators.

Example:
```
Input: "Hello, world! There is Datawhale."
Output: ["Hello", ",", "world", "!", "There", "is", "Datawhale", "."]
```

In this example, the input sentence is divided into a series of words and punctuation marks, each word or punctuation mark serving as an independent token.

### 5.2.2 Character-based Tokenizer

**Character-based Tokenizer** treats each character in the text as a separate token. This method handles text at a very fine granularity and is suitable for handling spelling errors, out-of-vocabulary words or new words. Since each character is an independent token, this method can capture very nuanced language features. This is especially useful for specific application scenarios such as generative tasks or tasks that require handling a large number of out-of-vocabulary words. However, this approach also causes the token sequence to become very long, increasing the computational complexity and training time of the model. Furthermore, character-level segmentation may lose some word-level semantic information, making it difficult for the model to understand the context.

Example:
```
Input: "Hello"
Output: ["H", "e", "l", "l", "o"]
```

In this example, the word "Hello" is divided into single characters, each character serving as an independent token. This method can handle any language and character set with great flexibility.

### 5.2.3 Subword Tokenizer

**Subword Tokenizer** sits between words and characters and better balances tokenization granularity against the ability to handle out-of-vocabulary words. The key idea of Subword Tokenizer is to split text into smaller units than words, but larger than characters, which can not only handle unknown words, but also maintain certain semantic information. Common subword tokenization methods include BPE, WordPiece, and Unigram.

#### (1) Byte Pair Encoding (BPE)

**BPE** is a statistical method that generates a subword dictionary by repeatedly combining the most frequent pairs of characters or character sequences. The advantage of this approach is its simplicity and efficiency, and its ability to effectively handle unknown and rare words while maintaining a low dictionary size. The merging process of BPE is bottom-up, gradually combining the most frequent character pairs into new subwords until the predetermined dictionary size is reached or there are no more high-frequency character pairs.

Example:
```
Input: "lower"
Output: ["low", "er"]

Input: "newest"
Output: ["new", "est"]
```

In this example, the word "lower" is divided into subwords "low" and "er", while "newest" is divided into "new" and "est". This method effectively deals with stems and affixes, maintaining the basic semantic structure of the word.

#### (2) WordPiece

**WordPiece** is another subword-based word segmentation method, originally used in Google's BERT model. Similar to BPE, WordPiece generates dictionaries by maximizing the likelihood function of subword sequences, but pays more attention to the optimization of language models when merging subwords. WordPiece will give priority to subwords that can maximize the probability of the overall sentence, so that the tokenization results have higher probability in the language model.

Example:
```
Input: "unhappiness"
Output: ["un", "##happiness"]
```

In this example, the word "unhappiness" is divided into subwords "un" and "##happiness", where "##" means that this is a suffix subword. In this way, WordPiece can better handle compound words and derivative words, retaining more semantic information.

#### (3) Unigram

**Unigram** tokenization is based on a probability model and segments text by selecting subwords with the highest probability. Unigram dictionary is generated by training language models and can handle text in multiple languages and different types. The Unigram model assigns a probability to each subword and then performs an optimal segmentation based on these probability.

Example:
```
Input: "unhappiness"
Output: ["un", "happiness"]

Input: "newest"
Output: ["new", "est"]
```

In this example, the word "unhappiness" is divided into subwords "un" and "happiness", while "newest" is divided into "new" and "est". This method effectively processes subword segmentation through a probability model, making the segmentation result more in line with language usage habits.

Each Tokenizer method has its specific application scenarios and advantages and disadvantages, and choosing the right Tokenizer is crucial to the success of natural language processing tasks.

### 5.2.4 Training a Tokenizer

Here we choose to use the BPE algorithm to train a Subword Tokenizer. BPE is a simple and effective word segmentation method that can handle out-of-vocabulary words and rare words while maintaining a smaller dictionary size. We will use Hugging Face's `tokenizers` library to train a BPE Tokenizer.


#### Step 1: Installing and importing dependency libraries

First, we need to install the `tokenizers` library, in addition to this, we need to install the `datasets` and `transformers` libraries for loading training data and loading the Tokenizer after training is completed.

```bash
pip install tokenizers datasets transformers
```

Then, import the required library.

```python
import random
import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator
```

#### Step 2: Loading training data

Here we train the tokenizer using the same dataset as pre-training (the Mobvoi Sequence Monkey open-source dataset); you can download and pre-process the dataset using `code/download_dataset.sh` and `code/deal_dataset.py`.

> Note: Due to the large dataset, it may cause insufficient memory during training. Because this project is for learning purposes, it is recommended that learners manually divide a small part of the data set for training and verification. The author also stores the trained tokenizer in the Github repository, which can be used directly.

```python
def read_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    """Read a JSONL file and safely extract the text data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if 'text' not in data:
                    raise KeyError(f"Missing 'text' field in line {line_num}")
                yield data['text']
            except json.JSONDecodeError:
                print(f"Error decoding JSON in line {line_num}")
                continue
            except KeyError as e:
                print(e)
                continue
```

#### Step 3: Create a configuration file

Before training BPE Tokenizer, we need to create a complete `Tokenizer` configuration file, including `tokenizer_config.json` and `special_tokens_map.json`. These configuration files define the parameters and special tokens of the `Tokenizer`, used to train and load `Tokenizer`. Here we keep the `chat_template` consistent with the `Qwen2.5` model.

```python
def create_tokenizer_config(save_dir: str) -> None:
    """Create the complete tokenizer configuration files"""
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # Save the main configuration file
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # Create special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)
```

#### Step 4: Training BPE Tokenizer

Before training BPE Tokenizer, we need to define a training function to train the Tokenizer and save the trained Tokenizer file. Here we use the `Tokenizer` class in the `tokenizers` library to train BPE Tokenizer.

You can see that when we train the Tokenizer, we have configured some special tokens, such as `<unk>`, `<s>`, `</s>`, `<|im_start|>` and `<|im_end|>`. These tokens are used to mark unknown words, the beginning and end of sentences, and the beginning and end of conversations. These special tokens can help the model better understand text data and improve the generalization ability and effectiveness of the model.

```python
def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192) -> None:
    """Train and save a custom tokenizer"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize the tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()  # Add text normalization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Configure special tokens
    special_tokens = [
        "<unk>", 
        "<s>", 
        "</s>", 
        "<|im_start|>", 
        "<|im_end|>"
    ]

    # Configure the trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,  # Filter out more low-frequency tokens
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # Train the tokenizer
    print(f"Training tokenizer with data from {data_path}")
    texts = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=os.path.getsize(data_path))

    # Verify the special token mapping
    try:
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        assert tokenizer.token_to_id("<|im_start|>") == 3
        assert tokenizer.token_to_id("<|im_end|>") == 4
    except AssertionError as e:
        print("Special tokens mapping error:", e)
        raise

    # Save the tokenizer files
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    
    # Create the configuration files
    create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")
```


#### Step 5: Use trained Tokenizer

We can use trained Tokenizer to process text data, such as encoding, decoding, generating dialogues, etc. Here is a simple example showing how to use a trained Tokenizer to process text data.

```python
def eval_tokenizer(tokenizer_path: str) -> None:
    """Evaluate the tokenizer's functionality"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Test basic properties
    print("\n=== Tokenizer basic info ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    # Test the chat template
    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]
    
    print("\n=== Chat template test ===")
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        # add_generation_prompt=True
    )
    print("Generated prompt:\n", prompt, sep="")

    # Test encoding and decoding
    print("\n=== Encode/decode test ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    print("Decoded text matches original:", decoded == prompt)

    # Test special token handling
    print("\n=== Special token handling ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print("Special tokens preserved:", decoded == test_text)
```

```python
eval_tokenizer('your tokenizer path')
```

OUT:
```
=== Tokenizer basic info ===
Vocab size: 6144
Special tokens: ['<|im_start|>', '<|im_end|>', '<unk>', '<s>', '</s>']
Special token IDs: [3, 4, 0, 1, 2]

=== Chat template test ===
Generated prompt:
<|im_start|>system
You are an AI assistant.<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm fine, thank you. and you?<|im_end|>
<|im_start|>user
I'm good too.<|im_end|>
<|im_start|>assistant
That's great to hear!<|im_end|>


=== Encode/decode test ===
Decoded text matches original: False

=== Special token handling ===
Original: <|im_start|>user
Hello<|im_end|>
Decoded:  <|im_start|> user
Hello<|im_end|>
Special tokens preserved: False
```

## 5.3 Pre-training a small LLM

In the previous chapters, we became familiar with the model structures of various large language models, as well as how to train a Tokenizer. In this section, we will train an LLM with 80 million parameters ourselves.

### 5.3.1 Data download

First, we need to download the pre-trained dataset. Here, we use two open source data sets that contain a large amount of Chinese conversation data that can be used to train conversation generation models.

- Mobvoi Sequence Monkey open-source dataset: the Sequence Monkey general text dataset is a large language model pre-training corpus built by aggregating and cleaning a wide range of publicly available data, including web pages, encyclopedias, blogs, Q&A, open-source code, books, newspapers, patents, textbooks and exam questions. It totals about 10B tokens.

- BelleGroup: 3.5 million Chinese dialogue data sets, including human-machine dialogue, human-human dialogue, character dialogue and other dialogue data, which can be used to train dialogue generation models.


```python
# Download the pre-training dataset
os.system("modelscope download --dataset ddzhu123/seq-monkey mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 --local_dir your_local_dir")
# Extract the pre-training dataset
os.system("tar -xvf your_local_dir/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2")

# Download the SFT dataset
os.system(f'huggingface-cli download --repo-type dataset --resume-download BelleGroup/train_3.5M_CN --local-dir BelleGroup')



# 1 Process the pre-training data
def split_text(text, chunk_size=512):
    """Split the text into chunks of the specified length"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

input_file = 'mobvoi_seq_monkey_general_open_corpus.jsonl'

with open('seq_monkey_datawhale.jsonl', 'a', encoding='utf-8') as pretrain:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in tqdm(data, desc=f"Processing lines in {input_file}", leave=False):  # Add a line-level progress bar
            line = json.loads(line)
            text = line['text']
            chunks = split_text(text)
            for chunk in chunks:
                pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')

# 2 Process the SFT data

def convert_message(data):
    """
    Convert the raw data into the standard format
    """
    message = [
        {"role": "system", "content": "你是一个AI助手"},  # "You are an AI assistant" (kept in Chinese: the SFT data is Chinese)
    ]
    for item in data:
        if item['from'] == 'human':
            message.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            message.append({'role': 'assistant', 'content': item['value']})
    return message

with open('BelleGroup_sft.jsonl', 'a', encoding='utf-8') as sft:
    with open('BelleGroup/train_3.5M_CN.json', 'r') as f:
        data = f.readlines()
        for item in tqdm(data, desc="Processing", unit="lines"):
            item = json.loads(item)
            message = convert_message(item['conversations'])
            sft.write(json.dumps(message, ensure_ascii=False) + '\n')
```

### 5.3.2 Training Tokenizer

First, we need to train a Tokenizer for text processing. The purpose of Tokenizer is to convert text into sequences of numbers so that the model can understand and process it. The dataset we use is [Mobvoi Sequence Monkey Open-Source Dataset](https://www.modelscope.cn/datasets/ddzhu123/seq-monkey/files). This dataset contains a large amount of Chinese text data and can be used to train Tokenizer.

> Note: Because the data set is large, if you train on your local computer, the progress will be slower. So here we provide a trained Tokenizer, which you can use directly. If you want to train yourself, you can refer to the following code.

```bash
python code/train_tokenizer.py
```

```python
import random
import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator

random.seed(42)

def read_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    """Read a JSONL file and safely extract the text data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if 'text' not in data:
                    raise KeyError(f"Missing 'text' field in line {line_num}")
                yield data['text']
            except json.JSONDecodeError:
                print(f"Error decoding JSON in line {line_num}")
                continue
            except KeyError as e:
                print(e)
                continue

def create_tokenizer_config(save_dir: str) -> None:
    """Create the complete tokenizer configuration files"""
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # Save the main configuration file
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # Create special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192) -> None:
    """Train and save a custom tokenizer"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize the tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()  # Add text normalization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Configure special tokens
    special_tokens = [
        "<unk>", 
        "<s>", 
        "</s>", 
        "<|im_start|>", 
        "<|im_end|>"
    ]

    # Configure the trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,  # Filter out more low-frequency tokens
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # Train the tokenizer
    print(f"Training tokenizer with data from {data_path}")
    texts = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=os.path.getsize(data_path))

    # Verify the special token mapping
    try:
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        assert tokenizer.token_to_id("<|im_start|>") == 3
        assert tokenizer.token_to_id("<|im_end|>") == 4
    except AssertionError as e:
        print("Special tokens mapping error:", e)
        raise

    # Save the tokenizer files
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    
    # Create the configuration files
    create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")

def eval_tokenizer(tokenizer_path: str) -> None:
    """Evaluate the tokenizer's functionality"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Test basic properties
    print("\n=== Tokenizer basic info ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    # Test the chat template
    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]
    
    print("\n=== Chat template test ===")
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        # add_generation_prompt=True
    )
    print("Generated prompt:\n", prompt, sep="")

    # Test encoding and decoding
    print("\n=== Encode/decode test ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    print("Decoded text matches original:", decoded == prompt)

    # Test special token handling
    print("\n=== Special token handling ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print("Special tokens preserved:", decoded == test_text)

def main():
    # Configure paths
    data_path = "your data path"
    save_dir = "tokenizer_k"

    # Train the tokenizer
    train_tokenizer(
        data_path=data_path,
        save_dir=save_dir,
        vocab_size=6144
    )

    # Evaluate the tokenizer
    eval_tokenizer(save_dir)

if __name__ == '__main__':
    main()
```

After the training is completed, you can use `eval_tokenizer()` to test the function of Tokenizer to ensure that the Tokenizer works normally. In this function, we first load the trained Tokenizer, and then test the basic properties, chat templates, encoding and decoding functions of Tokenizer. These tests can help us verify the correctness of Tokenizer and make sure it works properly. The correct output is:

OUT:
```
=== Tokenizer basic info ===
Vocab size: 6144
Special tokens: ['<|im_start|>', '<|im_end|>', '<unk>', '<s>', '</s>']
Special token IDs: [3, 4, 0, 1, 2]

=== Chat template test ===
Generated prompt:
<|im_start|>system
You are an AI assistant.<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm fine, thank you. and you?<|im_end|>
<|im_start|>user
I'm good too.<|im_end|>
<|im_start|>assistant
That's great to hear!<|im_end|>


=== Encode/decode test ===
Decoded text matches original: False

=== Special token handling ===
Original: <|im_start|>user
Hello<|im_end|>
Decoded:  <|im_start|> user
Hello<|im_end|>
Special tokens preserved: False
```

### 5.3.3 Dataset

#### PretrainDataset

Before we can feed the data into a model, we need to do some processing to convert text data into tokens that the model can understand. Here we are using Pytorch's Dataset class, which is used to load datasets. We define a `PretrainDataset` class to load preprocessed datasets. We inherited `torch.utils.data.IterableDataset` to define this dataset, which allows us to process data more flexibly and efficiently.

```python
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = json.loads(self.data[index])
        text = f"{self.tokenizer.bos_token}{sample['text']}"
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # The remaining part that does not fill the maximum length
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0 means no loss is computed
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
```

As can be seen in the above code and Figure 5.3, `Pretrain Dataset` mainly converts `text` to `input_id` through `tokenizer`, and then splits `input_id` into `X` and `Y`, where `X` is the first n-1 elements of `input_id` and `Y` is the last n-1 elements of `input_id`. `loss_mask` is mainly used to mark which positions need to calculate the loss and which positions do not need to calculate the loss.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/5-images/pretrain_dataset.png" alt="alt text" width="100%">
    <p>Figure 5.3 Pre-training loss function computation</p>
</div>

The example in the figure shows the processing when `max_length=9`:
- **Input sequence**: `[BOS, T1, T2, T3, T4, T5, T6, T7, EOS]`
- **Sample Split**:
  - X: `[BOS, T1, T2, T3, T4, T5, T6, T7]` → Model input context
  - Y: `[T1, T2, T3, T4, T5, T6, T7, EOS]` → Model prediction target
- **Loss mask**:
  - Valid location: `[0, 1, 1, 1, 1, 1, 1, 1, 1]` → Loss is calculated only for T1-EOS

#### SFTDataset

`SFTDataset` is actually a multi-turn conversation dataset. Our goal is to teach the model how to hold multi-turn conversations. At this stage, our input is the conversation content of the previous round, and the output is the conversation content of the current round.

```python
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def generate_loss_mask(self, input_ids):
        # Generate the loss mask: 0 means no loss is computed, 1 means loss is computed
        mask = [0] * len(input_ids)
        a_sequence = [3, 1074, 537, 500, 203]  # <|im_start|>assistant\n
        a_length = len(a_sequence)
        n = len(input_ids)
        i = 0
        
        while i <= n - a_length:
            # Check whether the current position matches the target subsequence
            match = True
            for k in range(a_length):
                if input_ids[i + k] != a_sequence[k]:
                    match = False
                    break
            if match:
                # Starting from the end of the subsequence, find the first 4; 4 is the <|im_end|> EOS id
                j = None
                for idx in range(i + a_length, n):
                    if input_ids[idx] == 4:
                        j = idx
                        break
                if j is not None:
                    start = i + a_length
                    end = j  # Set the end position to j (including the 4)
                    # Mark the span as 1 (from start to end, inclusive)
                    if start <= end:
                        for pos in range(start, end + 1):
                            if pos < len(mask):
                                mask[pos] = 1
                # Skip the current subsequence to avoid overlapping matches
                i += a_length
            else:
                i += 1
        return mask

    def __getitem__(self, index: int):
        sample = json.loads(self.data[index])
        text = self.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # The remaining part that does not fill the maximum length
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0 means no loss is computed
        loss_mask = self.generate_loss_mask(input_id)

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
```

In the SFT stage, a multi-turn conversation dataset is used here, so it is necessary to distinguish which positions need to calculate the loss and which positions do not need to calculate the loss. In the above code, I used a `generate_loss_mask` function to generate `loss_mask`. This function is mainly used to generate `loss_mask`, where the generation rule of `loss_mask` is: when encountering `|<im_start|>assistant\n`, the loss is started to be calculated until it encounters `|<im_end|>`. This ensures that our model only calculates the conversation content of the current round in the SFT stage, as shown in Figure 5.4.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/5-images/sftdataset.png" alt="alt text" width="90%">
    <p>Figure 5.4 SFT loss function calculation</p>
</div>

As you can see, the `X` and `Y` of SFT Dataset and Pretrain Dataset are actually the same, except that in SFT Dataset, we need to generate a `loss_mask` to mark which positions need to calculate the loss and which positions do not need to calculate the loss. The small blue square in `Input ids` in the picture is the answer of AI, so it is a place where model learning is needed. So in `loss_mask`, the corresponding position of the small blue square is yellow, and the other position is gray. The loss is calculated at the position corresponding to 1 in the code `loss_mask`, and the loss is not calculated at the position corresponding to 0.


### 5.3.4 Pre-training

After the data preprocessing is completed, we can start training the model. The model we use is a Decoder only Transformer model with the same structure as the LLama2, implemented using Pytorch. The relevant code is in the `code/k_model.py` file. I won't repeat it here. There are detailed comments (in Chinese) in the source code, and we have also introduced them in detail in previous articles.

In this part of the model, you can focus on how the generative model generates tokens. You can check the `generate` method in the `Transforerm` class in the `k_model.py` file.

```python
@torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        Given an input sequence idx (a long tensor of shape (bz,seq_len)), complete the sequence by generating new tokens repeatedly.
        Runs in model.eval() mode. This is a less efficient sampling version that does not use a k/v cache.
        """
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # If the sequence context is too long, truncate it to the maximum length
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            
            # Forward pass to get the logits for the last position in the sequence
            logits = self(idx_cond).logits
            logits = logits[:, -1, :] # Keep only the output of the last time step
            
            if temperature == 0.0:
                # Pick the most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # Scale the logits and apply softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            

            if idx_next == stop_id:
                break

            # Append the sampled index to the sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:] # Return only the generated tokens

```

In the `generate` method, we first get the `logits` at the last position in the sequence and then generate a new `token` based on these `logits`. Next, the generated new `token` is added to the sequence, and the model then continues to generate the next `token`. Through this iterative process, we are able to generate complete text.

Next is the most important part, training the model!

> Note: When using the following code for model training, you need to specify the `--data_path` parameter as the preprocessed dataset path, such as `--data_path seq_monkey_datawhale.jsonl`, and you also need to specify which GPUs to use for training, such as `--gpus 0,1`.

```python
def get_lr(it, all):
    """
    Compute the learning rate for the current iteration using a cosine annealing schedule
    
    Learning-rate schedule:
    1. Warmup phase: the learning rate grows linearly from 0 to the target learning rate
    2. Cosine annealing phase: the learning rate decays along a cosine curve to the minimum learning rate
    3. Beyond the training steps: keep the minimum learning rate
    
    Args:
        it (int): current iteration step
        all (int): total number of iteration steps
        
    Returns:
        float: learning rate for the current step
    """
    warmup_iters = args.warmup_iters  # Number of warmup iterations
    lr_decay_iters = all  # Total number of iterations for learning-rate decay
    min_lr = args.learning_rate / 10  # Minimum learning rate, 1/10 of the initial learning rate

    # Warmup phase: linear increase
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    # Beyond the training steps: keep the minimum learning rate
    if it > lr_decay_iters:
        return min_lr
    
    # Cosine annealing phase
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine coefficient
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch):
    """
    Function that trains one epoch
    
    Implements the full training loop, including:
    1. Data loading and moving data to the device
    2. Dynamic learning-rate adjustment
    3. Forward pass and loss computation
    4. Gradient accumulation and backpropagation
    5. Gradient clipping and optimizer updates
    6. Logging and model saving
    
    Args:
        epoch (int): current epoch number
    """
    start_time = time.time()  # Record the start time
    
    # Iterate over each batch in the data loader
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # Move the data to the specified device (GPU/CPU)
        X = X.to(args.device)  # Input sequence
        Y = Y.to(args.device)  # Target sequence
        loss_mask = loss_mask.to(args.device)  # Loss mask, used to ignore padding tokens

        # Compute the learning rate for the current step
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        # Update the learning rate of all parameter groups in the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Use the mixed-precision training context
        with ctx:
            # Forward pass
            out = model(X, Y)
            # Compute the loss and divide by the number of accumulation steps (for gradient accumulation)
            loss = out.last_loss / args.accumulation_steps
            # Flatten loss_mask to one dimension
            loss_mask = loss_mask.view(-1)
            # Apply the mask to compute the effective loss (ignoring padding positions)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        # Use the scaler for mixed-precision backpropagation
        scaler.scale(loss).backward()

        # Perform an optimizer update every accumulation_steps steps
        if (step + 1) % args.accumulation_steps == 0:
            # Unscale the gradients in preparation for gradient clipping
            scaler.unscale_(optimizer)
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # Take an optimizer step
            scaler.step(optimizer)
            # Update the scaler's scale factor
            scaler.update()

            # Zero the gradients; set_to_none=True saves memory
            optimizer.zero_grad(set_to_none=True)

        # Log every log_interval steps
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            # Print training progress information
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min;'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # Restore the true loss value
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            
            # If SwanLab is enabled, log training metrics
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })

        # Save the model every save_interval steps
        if (step + 1) % args.save_interval == 0:
            model.eval()  # Switch to evaluation mode
            # Build the checkpoint file name
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}.pth'

            # Handle multi-GPU saving: for a DataParallel model, access the .module attribute
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()  # Switch back to training mode
        
        # Every 20000 steps, save a checkpoint tagged with the step number
        if (step + 1) % 20000 == 0:
            model.eval()
            # Build a checkpoint file name that includes the step number
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}_step{step+1}.pth'

            # Save the model state dict
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_model():
    """
    Initialize the model and tokenizer
    
    This includes:
    1. Loading the pretrained tokenizer
    2. Creating the Transformer model
    3. Setting up multi-GPU parallel training (if available)
    4. Moving the model to the specified device
    5. Counting and printing the number of model parameters
    
    Returns:
        tuple: (model, tokenizer) the initialized model and tokenizer
    """
    def count_parameters(model):
        """
        Count the number of trainable parameters in the model
        
        Args:
            model: PyTorch model
            
        Returns:
            int: total number of trainable parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Load the pretrained tokenizer from a local path
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')

    # Create the Transformer model from the config
    model = Transformer(lm_config)
    
    # Multi-GPU initialization: check the number of available GPUs and set up DataParallel
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        Logger(f"Using {num_gpus} GPUs with DataParallel!")
        # Wrap the model with DataParallel to support multi-GPU training
        model = torch.nn.DataParallel(model)
    
    # Move the model to the specified device (GPU or CPU)
    model = model.to(args.device)
    
    # Compute and print the number of model parameters (in millions)
    Logger(f'Total LLM parameters: {count_parameters(model) / 1e6:.3f} million')
    return model, tokenizer


if __name__ == "__main__":
    # ==================== Command-line argument parsing ====================
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")
    
    # Basic training parameters
    parser.add_argument("--out_dir", type=str, default="base_model_215M", help="Model output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    
    # Experiment tracking and data loading parameters
    parser.add_argument("--use_swanlab", action="store_true", help="Whether to use SwanLab for experiment tracking")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes for data loading")
    parser.add_argument("--data_path", type=str, default="./seq_monkey_datawhale.jsonl", help="Training data path")
    
    # Training optimization parameters
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of learning-rate warmup iterations")
    
    # Logging and saving parameters
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")
    
    # Multi-GPU training parameters
    parser.add_argument("--gpus", type=str, default='0,1,2,3,4,5,6,7', help="GPU IDs to use, comma-separated (e.g. '0,1,2')")

    args = parser.parse_args()

    # ==================== GPU environment setup ====================
    # Set the visible GPU devices
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        # Automatically set the main device to the first available GPU
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    # ==================== Experiment tracking initialization ====================
    if args.use_swanlab:
        # Note: log in first with swanlab.login(api_key='your key') before use
        run = swanlab.init(
            project="Happy-LLM",  # Project name
            experiment_name="Pretrain-215M",  # Experiment name
            config=args,  # Save all hyperparameters
        )

    # ==================== Model configuration ====================
    # Define the configuration parameters of the language model
    lm_config = ModelConfig(
        dim=1024,      # Model dimension
        n_layers=18,   # Number of Transformer layers
    )

    # ==================== Training environment setup ====================
    max_seq_len = lm_config.max_seq_len  # Maximum sequence length
    args.save_dir = os.path.join(args.out_dir)  # Model save directory
    
    # Create the necessary directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Set the random seed for reproducible results
    torch.manual_seed(42)
    
    # Determine the device type (used to choose the right context manager)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # Set up the context manager for mixed-precision training
    # Use nullcontext when training on CPU, autocast when training on GPU
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # ==================== Model and data initialization ====================
    # Initialize the model and tokenizer
    model, tokenizer = init_model()
    
    # Create the training dataset
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=max_seq_len)
    
    # Create the data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,  # Batch size
        pin_memory=True,             # Load data into pinned memory to speed up GPU transfer
        drop_last=False,             # Do not drop the last incomplete batch
        shuffle=True,                # Shuffle the data
        num_workers=args.num_workers # Number of parallel worker processes for data loading
    )

    # ==================== Optimizer and training component initialization ====================
    # Initialize the gradient scaler for mixed-precision training
    # Enabled only when using float16 or bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    
    # Initialize the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # ==================== Start training ====================
    # Compute the number of iterations per epoch
    iter_per_epoch = len(train_loader)
    
    # Start the training loop
    for epoch in range(args.epochs):
        train_epoch(epoch)
```

### 5.3.5 SFT training

SFT training and pre-trained code are basically the same, except that the imported Dataset is different. Here we are using SFTDataset, used for multi-turn conversation training.

```python
import os
import platform
import argparse
import time
import warnings
import math
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

from transformers import AutoTokenizer

from k_model import ModelConfig, Transformer
from dataset import SFTDataset

import swanlab

# Ignore warnings
warnings.filterwarnings('ignore')


def Logger(content):
    """Logger"""
    print(content)

def get_lr(it, all):
    """Get the learning rate"""
    # 1) linear warmup for warmup_iters steps
    # 1) Linear warmup during the warmup iterations
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    # 2) if it > lr_decay_iters, return min learning rate
    # 2) If the iteration count exceeds the learning-rate decay iterations, return the minimum learning rate
    if it > lr_decay_iters:
        return min_lr
    
    # 3) in between, use cosine decay down to min learning rate
    # 3) In between, use cosine decay down to the minimum learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch):
    """Train one epoch"""
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # Get the learning rate and update the optimizer
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        # Backward pass
        scaler.scale(loss).backward()

        # Update the weights
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # Print logs
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })

        # Save the model
        if (step + 1) % args.save_interval == 0:
            model.eval()
            ckp = f'{args.save_dir}/sft_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}.pth'

            # Handle multi-GPU saving
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()
        
        # Save the model periodically
        if (step + 1) % 20000 == 0:
            model.eval()
            ckp = f'{args.save_dir}/sft_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}_step{step+1}.pth'

            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_model():
    """Initialize the model"""
    def count_parameters(model):
        """Count the model parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')

    # Initialize the model
    model = Transformer(lm_config)

    # Load the pretrained weights
    ckp = './base_model_215M/pretrain_1024_18_6144.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    
    # Multi-GPU initialization
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        Logger(f"Using {num_gpus} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)
    
    model = model.to(args.device)
    Logger(f'Total LLM parameters: {count_parameters(model) / 1e6:.3f} million')
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")
    parser.add_argument("--out_dir", type=str, default="sft_model_215M", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--use_swanlab", action="store_true", help="Whether to use SwanLab for experiment tracking")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes for data loading")
    parser.add_argument("--data_path", type=str, default="./BelleGroup_sft.jsonl", help="Training data path")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")
    # Add multi-GPU arguments
    parser.add_argument("--gpus", type=str, default='0,1,2,3,4,5,6,7', help="Comma-separated GPU IDs (e.g. '0,1,2')")

    args = parser.parse_args()

    # Set the visible GPUs
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        # Automatically set the main device to the first GPU
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    # Initialize swanlab
    if args.use_swanlab:
        run = swanlab.init(
            project="Happy-LLM",
            experiment_name="SFT-215M",
            config=args,
        )

    # Model configuration
    lm_config = ModelConfig(
        dim=1024,
        n_layers=18,
    )
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # Context manager
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # Initialize the model and tokenizer
    model, tokenizer = init_model()
    
    # Create the dataset and data loader
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers
    )

    # Scaler and optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Start training
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch)
```


### 5.3.6 Generate text using the model

After the model training is completed, the model file will be generated in the `output` directory, which is the model we trained. We can generate text using the following command.

```bash
python model_sample.py
```

Let's look at the code in the `model_sample.py` file, which defines a `TextGenerator` class to generate text.

```python
import os
import pickle
from contextlib import nullcontext
import torch
from k_model import ModelConfig, Transformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

class TextGenerator:
    def __init__(self, 
                 checkpoint='./base_model_215M/pretrain_1024_18_6144.pth',  # Model checkpoint path
                 tokenizer_model_path='./tokenizer_k/',  # Tokenizer model path
                 seed=42,  # Random seed for reproducibility
                 device=None,  # Device: prefer CUDA; fall back to CPU if CUDA is unavailable
                 dtype="bfloat16"):  # Data type; defaults to float32, float16 or bfloat16 can be chosen
        """
        Initialize the TextGenerator class: load the model, set up the device, the tokenizer, etc.
        """
        # Model loading configuration
        self.checkpoint = checkpoint  # Path to the saved model checkpoint
        self.tokenizer_model_path = tokenizer_model_path  # Path to the tokenizer model files
        self.seed = seed  # Random seed, for reproducible generation
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')  # Choose the device based on available hardware
        self.dtype = dtype  # Floating-point type of the model
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'  # Check whether the current device is CUDA
        
        # Set the random seed for reproducible generation
        torch.manual_seed(seed)  # Set the CPU random seed
        torch.cuda.manual_seed(seed)  # Set the CUDA random seed
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow CUDA to use TF32 precision for matrix multiplication
        torch.backends.cudnn.allow_tf32 = True  # Allow cuDNN to use TF32 precision for acceleration
        
        # Choose the appropriate automatic mixed-precision context based on dtype
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        
        # Load the model checkpoint file
        checkpoint_dict = torch.load(self.checkpoint, map_location=self.device)  # Load the model parameters # Initialize model parameters
        self.model = Transformer(ModelConfig(dim=1024, n_layers=18))  # Instantiate the Transformer model
        sunwanted_prefix = '_orig_mod.'
        for k, v in list(checkpoint_dict.items()):
            if k.startswith(sunwanted_prefix):
                checkpoint_dict[k[len(sunwanted_prefix):]] = checkpoint_dict.pop(k)
        self.model.load_state_dict(checkpoint_dict, strict=False)
        
        # Count the model parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model has {num_params / 1e6:.3f} M parameters.")
        # Put the model in evaluation mode so that training-mode operations such as dropout do not affect the results
        self.model.eval()
        # Move the model to the correct device (GPU or CPU)
        self.model.to(self.device)
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model_path)  # Load the tokenizer from the specified path

    def chat_template(self, prompt):
        message = [
            {"role": "system", "content": "你是一个AI助手，你的名字叫小明。"},  # "You are an AI assistant, and your name is Xiaoming." (kept in Chinese: the model is trained on Chinese data)
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    def sft_sample(self, 
               start="Hello!",  # Starting prompt for text generation; can be any string
               num_samples=3,  # Number of samples to generate; 3 by default
               max_new_tokens=256,  # Maximum number of tokens generated per sample; at most 256 by default
               temperature=0.7,  # Controls the randomness of generation; 1.0 is standard, larger values are more random
               top_k=300):  # Keep the top_k most probable tokens, limiting the choices during generation
        """
        Generate samples from the given starting text.
        
        :param start: starting prompt for text generation
        :param num_samples: number of text samples to generate
        :param max_new_tokens: maximum number of tokens generated per sample
        :param temperature: controls the randomness of generation; smaller values are more deterministic, larger values more random
        :param top_k: limits the range of tokens chosen from during generation
        :return: list of generated text samples
        """
        start = self.chat_template(start)
        # Encode the starting text into a sequence of token ids
        start_ids = self.tokenizer(start).data['input_ids']
        # print('start_ids:', start_ids)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])  # Convert the encoded token ids into a PyTorch tensor
        generated_texts = []  # Holds the generated text samples
        with torch.no_grad():  # Disable gradient computation for efficiency
            with self.ctx:  # Enter the automatic mixed-precision context (when on GPU with float16)
                for k in range(num_samples):  # Loop to generate the requested number of samples
                    y = self.model.generate(x, self.tokenizer.eos_token_id, max_new_tokens, temperature=temperature, top_k=top_k)  # Generate text
                    generated_texts.append(self.tokenizer.decode(y[0].tolist()))  # Decode the generated token sequence into readable text
        return generated_texts  # Return the generated text samples


    def pretrain_sample(self, 
               start="Hello!",  # Starting prompt for text generation; can be any string
               num_samples=3,  # Number of samples to generate; 3 by default
               max_new_tokens=256,  # Maximum number of tokens generated per sample; at most 256 by default
               temperature=0.7,  # Controls the randomness of generation; 1.0 is standard, larger values are more random
               top_k=300):  # Keep the top_k most probable tokens, limiting the choices during generation
        """
        Generate samples from the given starting text.
        
        :param start: starting prompt for text generation
        :param num_samples: number of text samples to generate
        :param max_new_tokens: maximum number of tokens generated per sample
        :param temperature: controls the randomness of generation; smaller values are more deterministic, larger values more random
        :param top_k: limits the range of tokens chosen from during generation
        :return: list of generated text samples
        """
        # If start begins with 'FILE:', read the starting text from a file
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                start = f.read()  # Use the file contents as the starting text
        
        # Encode the starting text into a sequence of token ids
        start_ids = self.tokenizer(start).data['input_ids']
        # print('start_ids:', start_ids)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])  # Convert the encoded token ids into a PyTorch tensor
        # print(x.shape)
        generated_texts = []  # Holds the generated text samples
        with torch.no_grad():  # Disable gradient computation for efficiency
            with self.ctx:  # Enter the automatic mixed-precision context (when on GPU with float16)
                for k in range(num_samples):  # Loop to generate the requested number of samples
                    y = self.model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)  # Generate text
                    generated_texts.append(self.tokenizer.decode(y[0].tolist()))  # Decode the generated token sequence into readable text
        
        return generated_texts  # Return the generated text samples
    
if __name__ == "__main__":
    print("------------------- Pretrain Sample ------------------- \n")

    pretrain_prompt_datas = [
        '<|im_start|>北京大学是',  # "Peking University is" (Chinese prompts: the model is trained on Chinese data)
        '<|im_start|>中国矿业大学（北京）地球科学与测绘工程学院',  # "China University of Mining and Technology (Beijing), School of Earth Sciences and Surveying Engineering"
    ]

    generator = TextGenerator(checkpoint='./base_model_215M/pretrain_1024_18_6144.pth')  # Initialize the generator
    for i in range(len(pretrain_prompt_datas)):
        samples = generator.pretrain_sample(start=pretrain_prompt_datas[i], num_samples=1, max_new_tokens=120, temperature=0.75)
        print(f"\nSample {i+1}:\n{pretrain_prompt_datas[i]}{samples[0]}\n{'-'*20}")  # Print the generated sample followed by a separator line

    print("\n ------------------- SFT Sample ------------------- \n")

    sft_prompt_datas = [
        '你好呀',  # "Hi there"
        "中国的首都是哪里？",  # "What is the capital of China?"
        "1+12等于多少？",  # "What is 1+12?"
        "你是谁？"  # "Who are you?"
    ]
    generator = TextGenerator(checkpoint='./sft_model_215M/sft_dim1024_layers18_vocab_size6144.pth')  # Initialize the generator
    for i in range(len(sft_prompt_datas)):
        samples = generator.sft_sample(start=sft_prompt_datas[i], num_samples=1, max_new_tokens=128, temperature=0.6)
        print(f"\nSample {i+1}:\nQuestion: {sft_prompt_datas[i]} \nAI answer: {samples[0]}\n{'-'*20}")  # Print the generated sample followed by a separator line

```

Finally, let’s take a look at the results of the model output:

```
------------------- SFT Sample ------------------- 

Model has 215.127 M parameters.

Sample 1:
Question: 你好呀  ("Hi there")
AI answer: 你好!有什么我可以帮你的吗?  ("Hello! Is there anything I can help you with?")
--------------------

Sample 2:
Question: 中国的首都是哪里？  ("What is the capital of China?")
AI answer: 中国的首都是北京。  ("The capital of China is Beijing.")
--------------------

Sample 3:
Question: 1+1等于多少？  ("What is 1+1?")
AI answer: 1+1等于2。  ("1+1 equals 2.")
--------------------
------------------- Pretrain Sample ------------------- 

Model has 215.127 M parameters.

Sample 1:
<|im_start|>北京大学是中国最早建立的研究型大学之一,是我国最早设置研究生院的高校之一,是第一、二国教育委员会师资培训基地;北京大学是第一、二所国立大学,其校名与北京大学相同。
北京大学录取标准:本科三批1万元,本科一批1万元,本科一批2000元,专科一批2000元,高中起点:非本科一批  (Gloss of Sample 1: "Peking University is one of the earliest research universities founded in China and one of the first to set up a graduate school... Admission standards: ..." -- the pre-trained model continues the prompt with fluent but factually unreliable text.)
--------------------

Sample 2:
<|im_start|>中国矿业大学（北京）地球科学与测绘工程学院副教授黄河流域地质学科带头人古建平教授为大家介绍世界地质变化的概念及工作经验。
古建平教授介绍了最近几年的植物学和地质学的基本概念,尤其是树都黄河、松涛、暗河等都有地质学工作者的身影,其中树都黄河以分布面积最大,是树都黄河中华砂岩公园的主景区。
黄河内蒙古  (Gloss of Sample 2: "...Associate Professor Gu Jianping, a leading geology scholar for the Yellow River basin, introduced the concept of global geological change and his work experience..." -- again fluent but not factual.)
--------------------
```

At this point, our model has been completed. Congratulations on training a large language model of your own!

> You can lower the batch during training, so as to reduce GPU memory usage and avoid running out of GPU memory. Of course, this will increase training time, and you can adjust the size of the batch according to the GPU memory of your graphics card. In the case of the actual Pretrain batch of 4, only 7 GB of GPU memory is required, and the training time is expected to be 533 hours. The author trained on 8 RTX 4090 GPUs. The pre-training took a total of 46 hours, and the SFT stage took 24 hours of training on BelleGroup's 3.5 million Chinese instructions.

The author also uploaded the models trained in this chapter on the ModelScope platform. If your hardware is not sufficient to train a large language model, you can also download the models from ModelScope for debugging and model experience. The model download address is as follows:

> *ModelScope Model Download Address: [🤖 ModelScope](https://www.modelscope.cn/collections/Happy-LLM-e98b91b10b684a)*
> *ModelScope Studio Demo Address: [🤖 ModelScope Studio](https://www.modelscope.cn/studios/kmno4zx/happy_llm_215M_sft)*


**References**

[1] Andrej Karpathy. (2023). *llama2.c: Fullstack Llama 2 LLM solution in pure C*. GitHub repository. https://github.com/karpathy/llama2.c  

[2] Andrej Karpathy. (2023). *llm.c: GPT-2/GPT-3 pretraining in C/CUDA*. GitHub repository. https://github.com/karpathy/llm.c  

[3] Hugging Face. (2023). *Tokenizers documentation*. https://huggingface.co/docs/tokenizers/index  

[4] Skywork Team. (2023). *SkyPile-150B: A large-scale bilingual dataset*. Hugging Face dataset. https://huggingface.co/datasets/Skywork/SkyPile-150B  

[5] BelleGroup. (2022). *train_3.5M_CN: Chinese dialogue dataset*. Hugging Face dataset. https://huggingface.co/datasets/BelleGroup/train_3.5M_CN  

[6] Jingyao Gong. (2023). *minimind: Minimalist LLM implementation*. GitHub repository. https://github.com/jingyaogong/minimind  

[7] Mobvoi. (2023). *seq-monkey-data: Llama2 training/inference data*. GitHub repository. https://github.com/mobvoi/seq-monkey-data