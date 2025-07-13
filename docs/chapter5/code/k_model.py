import math
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from transformers import PreTrainedModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"
    def __init__(
            self,
            dim: int = 768,
            n_layers: int = 12,
            n_heads: int = 16,
            n_kv_heads: int = 8,
            vocab_size: int = 6144,
            hidden_dim: int = None,
            multiple_of: int = 64,
            norm_eps: float = 1e-5,
            max_seq_len: int = 512,
            dropout: float = 0.0,
            flash_attn: bool = True,
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

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # eps is to prevent the situation of dividing by 0
        self.eps = eps
        # Weight is a learnable parameter, all initialized to 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Calculate the core part of RMSNorm
        # x.pow(2).mean(-1, keepdim=True) calculates the mean of the square of input x
        # torch.rsqrt is the reciprocal of the square root, so that the denominator part of RMSNorm is obtained, plus eps prevents the denominator from being 0
        # Finally multiply by x to get the result of RMSNorm
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # The forward function is the forward propagation of the model
        # First convert input x to float type, then perform RMSNorm, and finally return to the original data type
        # Finally multiplied by weight, which is a learnable scaling factor for RMSNorm
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Obtain the real and virtual parts of the rotation embedded
# Note: The dim here should be dim//n_head, because we are rotating and embedding each head
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float() generates a sequence starting from 0 and step size 2, with a length of half of dim
    # Then divide each element by dim, then take the reciprocal of theta to get the frequency
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Generate a sequence from 0 to end, with a length of end
    t = torch.arange(end, device=freqs.device)
    # Calculate the outer product and get a two-dimensional matrix, each row is the element of t multiplied by the element of freqs
    freqs = torch.outer(t, freqs).float()
    # Calculate the cosine value of the frequency and obtain the real part
    freqs_cos = torch.cos(freqs)
    # Calculate the sine value of the frequency to obtain the imaginary part
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

# The purpose of this function is to adjust freqs_cis to the same shape as x so that it can be broadcasted with x
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # Get the number of dimensions of x
    ndim = x.ndim
    # Assertion, make sure 1 is within the dimension range of x
    assert 0 <= 1 < ndim
    # Assert, make sure that the shape of freqs_cis is the same as the second and last dimensions of x
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # Construct a new shape, except for the second and last dimensions, all other dimensions are 1, which is done to enable broadcasting of freqs_cis with x
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # Resize freqs_cis to a new shape and return
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # Convert query and key tensors to floating point numbers and reshape the shape to separate real and imaginary parts
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # Reshape frequency tensors for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # Apply rotation to calculate the real and imaginary parts after rotation respectively
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Merge the last two dimensions and restore to the shape of the original tensor
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # Get the shape of the input tensor: batch size, sequence length, number of key/value pairs of heads, dimension size for each head
    bs, slen, n_kv_heads, head_dim = x.shape
    
    # If the number of repetitions is 1, no repetition is required, and the original tensor is returned directly
    if n_rep == 1:
        return x
    
    # Extend and reshape tensors to repeat key-value pairs
    return (
        x[:, :, :, None, :]  # Add a new dimension to the fourth dimension (before the header dimension)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # Expand the newly added dimension to n_rep size to achieve duplicate effect
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # Reshape, merge the dimensions of the number of key/value pairs and the number of repetitions
    )

class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        # Determines the number of headers used for keys and values based on whether n_kv_heads is specified.
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Make sure that the total number of headers can be divisible by the key value headers.
        assert args.n_heads % self.n_kv_heads == 0

        # The model handles the size in parallel, default is 1.
        model_parallel_size = 1
        # The number of headers is calculated locally, equal to the total number of headers divided by the model parallel processing size.
        self.n_local_heads = args.n_heads // model_parallel_size
        # The number of local key-value headers is equal to the number of key-value headers divided by the model parallel processing size.
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # Number of repetitions, used to extend the size of keys and values.
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # The dimension of each head is equal to the model dimension divided by the total number of headers.
        self.head_dim = args.dim // args.n_heads

        # Define the weight matrix.
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # Output weight matrix.
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Define dropout.
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # Save the dropout probability.
        self.dropout = args.dropout

        # Check if Flash Attention is used (PyTorch >= 2.0 is required).
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # If Flash Attention is not supported, use the manual attention mechanism and set the mask.
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Create an upper triangle matrix to mask future information.
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # Registered as a buffer for the model
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        # Get batch size and sequence length, [batch_size, seq_len, dim]
        bsz, seqlen, _ = x.shape

        # Calculate query (Q), key (K), value (V).
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # Adjust the shape to fit the dimensions of the head.
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply Rotary Position Embedding (RoPE).
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # Extend keys and values to accommodate the number of repetitions.
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Handle the header as a batch dimension.
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Select the implementation method according to whether Flash Attention is supported.
        if self.flash:
            # Use Flash Attention.
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # Use a manual attention mechanism.
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        # Recover the time dimension and merge headers.
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # The final projection is back to the residual flow.
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # If the dimension of the hidden layer is not specified, we set it to 4 times the input dimension
        # Then reduce it to 2/3 and finally make sure it is a multiple of multiple_of
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # Define the first layer of linear transformation from the input dimension to the hidden dimension
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # Define the second layer of linear transformation from hidden dimension to input dimension
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # Define the third layer linear transformation from the input dimension to the hidden dimension
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # Define dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Forward propagation function
        # First, input x is activated by the first layer of linear transformation and SILU
        # Then, the result is multiplied by the result of input x linear transformation through the third layer
        # Finally, through the second layer of linear transformation and dropout layer
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    

class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        # Define the number of heads of attention for a long head
        self.n_heads = args.n_heads
        # Define input dimensions
        self.dim = args.dim
        # Define the dimension of each header, equal to the input dimension divided by the number of headers
        self.head_dim = args.dim // args.n_heads
        # Define the LLaMA2Attention object for multi-head attention calculation
        self.attention = Attention(args)
        # Define LLaMAMLP object for feedforward neural network calculation
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        # Define the ID of the layer
        self.layer_id = layer_id
        # Define the normalization layer of attention calculation
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Define the normalized layer of feedforward neural network computing
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # Forward propagation function
        # First, input x passes through the attention normalization layer, and then performs attention calculation, and the result is added to input x to obtain h
        # Then, h passes through the feedforward neural network normalization layer, and then performs the feedforward neural network calculation, and the result is added with h to obtain the output
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(PreTrainedModel):
    config_class = ModelConfig  # Configuration class
    last_loss: Optional[torch.Tensor] # Record the last calculated loss

    def __init__(self, args: ModelConfig = None):
        super().__init__(args)
        # Initialize model parameters
        self.args = args
        # Glossary size
        self.vocab_size = args.vocab_size
        # Number of layers
        self.n_layers = args.n_layers

        # Word embedding layer
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # Dropout layer
        self.dropout = nn.Dropout(args.dropout)
        # Decoder layer
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        # Normalization layer
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Output layer
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Share the weight of the word embedding layer with the weight of the output layer
        self.tok_embeddings.weight = self.output.weight 

        # Pre-calculate the frequency of relative position embedding
        freqs_cos, freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Initialize all weights
        self.apply(self._init_weights)
        # Special scaling initialization of residual projection
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

        # Initialize the loss attribute of the last forward propagation
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()  # Output container
        self._no_split_modules = [name for name, _ in self.named_modules()]  # List of undivided modules

    def _init_weights(self, module):
        # Functions that initialize weights
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **keyargs) -> torch.Tensor:
        """- tokens: Optional[torch.Tensor], enter the token tensor.
        - targets: Optional[torch.Tensor], Target token tensor.
        - kv_cache: bool, whether to use key-value cache.
        - keyargs: Other keyword parameters.

        - self.OUT: CausalLMOutputWithPast, contains logits and losses."""

        if 'input_ids' in keyargs:
            tokens = keyargs['input_ids']
        if 'attention_mask' in keyargs:
            targets = keyargs['attention_mask']

        # Forward propagation function
        _bsz, seqlen = tokens.shape
        # Through word embedding layer and dropout layer
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        # Get the frequency of the relative position embedding
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # Through the Decoder layer
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        # Through normalization layer
        h = self.norm(h)

        if targets is not None:
            # If a target is given, the loss is calculated
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        else:
            # Small optimization in reasoning: only forward propagation of the output at the last position
            logits = self.output(h[:, [-1], :]) 
            self.last_loss = None

        # Set the output
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    
    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """Given the input sequence idx (long integer tensor with shape (bz,seq_len)), the sequence is completed by generating new tokens multiple times.
        Run in model.eval() mode. Inefficient sampling version, no key k/v cache is used."""
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # If the sequence context is too long, truncate it to the maximum length
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            
            # Forward propagation to get the last position of the sequence
            logits = self(idx_cond).logits
            logits = logits[:, -1, :] # Only the output of the last time step is retained
            
            if temperature == 0.0:
                # Select the most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # Scaling logits and applying softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            

            if idx_next == stop_id:
                break

            # Add the sampled index to the sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:] # Return only the generated token

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("tokenizer_k")
    args = ModelConfig(
        dim=1024,
        n_layers=18,
    )
    # Instantiate LLaMA2Model
    model = Transformer(args=args)
    # Calculate all parameters of the model
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Total LLM parameter quantity: {num_params / 1e6:.3f} million')

    prompt = "Hello, what are you eating today? How are you doing?"
    text = f"{tokenizer.bos_token}{prompt}{tokenizer.eos_token}"
    print(f"Input text: {text}")

    input_id = tokenizer(text).data['input_ids']
    print("input_ids :", input_id)
    print("dcode_str :", tokenizer.decode(input_id))

    X = torch.tensor(input_id[:-1]).unsqueeze(0)
    Y = torch.tensor(input_id[1:]).unsqueeze(0)
    print("X shape :", X.shape)
    print("Y shape :", Y.shape)

    # Pass the input tensor into the model
    output = model(X, Y)