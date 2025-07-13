import torch
import math
from torch import nn
from dataclasses import dataclass
from transformers import BertTokenizer
import torch.nn.functional as F

@dataclass
class ModelArgs:
    n_embd: int # Embed dimensions
    n_heads: int # Number of heads
    dim: int # Model Dimension
    dropout: float
    max_seq_len: int
    vocab_size: int
    block_size: int
    n_layer: int

    

class MultiHeadAttention(nn.Module):

    def __init__(self, args: ModelArgs, is_causal=False):
        # Constructor
        # args: Configure object
        super().__init__()
        # The hidden layer dimension must be an integer multiple of the number of heads, because later we will split the input into header matrix
        assert args.dim % args.n_heads == 0
        # The model handles the size in parallel, default is 1.
        model_parallel_size = 1
        # The number of headers is calculated locally, equal to the total number of headers divided by the model parallel processing size.
        self.n_local_heads = args.n_heads // model_parallel_size
        # The dimension of each head is equal to the model dimension divided by the total number of headers.
        self.head_dim = args.dim // args.n_heads

        # Wq, Wk, Wv parameter matrix, each parameter matrix is n_embd x n_embd
        # Here, three combination matrices are used to replace the combination of n parameter matrices. The logic is that the matrix internal product and re-split are actually equivalent to the splicing matrix internal product.
        # Readers who don’t understand can simulate it themselves. Each linear layer is actually equivalent to the splicing of n parameter matrices.
        self.wq = nn.Linear(args.n_embd, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, args.n_heads * self.head_dim, bias=False)
        # Output weight matrix, dimensions are dim x n_embd(head_dim = n_embeds / n_heads)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        # The dropout of attention
        self.attn_dropout = nn.Dropout(args.dropout)
        # Residual connection dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal

        # Create an upper triangle matrix to mask future information
        # Note that because it is bullish attention, the Mask matrix has one more dimension than we defined before.
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # Registered as a buffer for the model
            self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        # Get batch size and sequence length, [batch_size, seq_len, dim]
        bsz, seqlen, _ = q.shape

        # Calculate query (Q), key (K), value (V), input through the parameter matrix layer, the dimensions are (B, T, n_embed) x (n_embed, n_embed) -> (B, T, n_embed)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # Split Q, K, V into long heads, the dimensions are (B, T, n_head, C // n_head), and then exchange the dimensions and become (B, n_head, T, C // n_head)
        # Because in attention calculation, we take the last two dimensions to participate in the calculation
        # Why do you need to first press B*T*n_head*C//n_head to expand and then swap 1 and 2 dimensions instead of directly expanding according to attention input? It is because the expansion method of view is to directly arrange all the inputs.
        # Then construct as required, and it can be found that only the above operations can achieve the goal of taking out the corresponding part of each head
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Attention Calculation
        # Calculate QK^T / sqrt(d_k), with dimensions as (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # Mask self-attention must have attention mask
        if self.is_causal:
            assert hasattr(self, 'mask')
            # The sequence length is intercepted here, because some sequences may be shorter than max_seq_len
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
        # Calculate softmax with dimensions (B, nh, T, T)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # Do Dropout
        scores = self.attn_dropout(scores)
        # V * Score, dimensions are (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)

        # Recover the time dimension and merge headers.
        # Splice the results of long heads, first exchange the dimensions as (B, T, n_head, C // n_head), and then splice them into (B, T, n_head * C // n_head)
        # Contiguous function is used to reopen a new memory storage, because Pytorch settings will report an error before transpose before viewing.
        # Because the view is obtained directly based on the underlying storage, however, transpose does not change the underlying storage, so additional storage is required
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # The final projection is back to the residual flow.
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class LayerNorm(nn.Module):
    '''Layer Norm'''
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # Mapping of linear matrix
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        # Calculate the values of all dimensions of each sample, find the mean and variance
        mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
        std = x.std(-1, keepdim=True) # std: [bsz, max_len, 1]
        # Note that there is also a broadcast in the last dimension here
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MLP(nn.Module):
    '''Feedforward neural network'''
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # Define the first layer of linear transformation from the input dimension to the hidden dimension
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # Define the second layer of linear transformation from hidden dimension to input dimension
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # Define dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Forward propagation function
        # First, input x is activated by the first layer of linear transformation and RELU
        # Then, the result is multiplied by the result of input x linear transformation through the third layer
        # Finally, through the second layer of linear transformation and dropout layer
        return self.dropout(self.w2(F.relu(self.w1(x))))
    

class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # There are two LayerNorm in a layer, before Attention and before MLP
        self.attention_norm = LayerNorm(args.n_embd)
        # Encoder does not require a mask, pass it in is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x):
        # Layer Norm
        x = self.attention_norm(x)
        # Self-attention
        h = x + self.attention.forward(x, x, x)
        # Through feedforward neural network
        out = h + self.feed_forward.forward(self.fnn_norm(h))
        return out

class Encoder(nn.Module):
    '''Encoder block'''
    def __init__(self, args):
        super(Encoder, self).__init__() 
        # An Encoder consists of N Encoder Layers
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):
        "Through the N-layer Encoder Layer"
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    '''Decoder layer'''
    def __init__(self, args):
        super().__init__()
        # There are three LayerNorm in a Layer, before Mask Attention, before Self Attention, and before MLP
        self.attention_norm_1 = LayerNorm(args.n_embd)
        # The first part of the Decoder is Mask Attention, passing in is_causal=True
        self.mask_attention = MultiHeadAttention(args, is_causal=True)
        self.attention_norm_2 = LayerNorm(args.n_embd)
        # The second part of Decoder is an Attention similar to Encoder, passing in is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.ffn_norm = LayerNorm(args.n_embd)
        # The third part is MLP
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x, enc_out):
        # Layer Norm
        x = self.attention_norm_1(x)
        # Mask self-attention
        x = x + self.mask_attention.forward(x, x, x)
        # Bulls' attention
        x = self.attention_norm_2(x)
        h = x + self.attention.forward(x, enc_out, enc_out)
        # Through feedforward neural network
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Decoder(nn.Module):
    '''Decoder'''
    def __init__(self, args):
        super(Decoder, self).__init__() 
        # A Decoder consists of N Decoder Layers
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)

class PositionalEncoding(nn.Module):
    '''Position encoding module'''

    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        # Dropout layer
        self.dropout = nn.Dropout(p=args.dropout)

        # block size is the maximum length of the sequence
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        # Calculate theta
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        # Calculate the sin and cos results separately
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add position code to Embedding results
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Transformer(nn.Module):
    '''Overall model'''

    def __init__(self, args):
        super().__init__()
        # Must enter the vocabulary size and block size
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(args.vocab_size, args.n_embd),
            wpe=PositionalEncoding(args),
            drop=nn.Dropout(args.dropout),
            encoder=Encoder(args),
            decoder=Decoder(args),
        ))
        # The final linear layer, the input is n_embd, and the output is the vocabulary size
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # Initialize all weights
        self.apply(self._init_weights)

        # View the number of all parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    '''Statistics the number of all parameters'''

    def get_num_params(self, non_embedding=False):
        # non_embedding: Whether to count the parameters of embedding
        n_params = sum(p.numel() for p in self.parameters())
        # If the embedded parameters are not counted, subtract
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    '''Initialize weights'''

    def _init_weights(self, module):
        # Linear and Embedding layers are initialized to regular distribution
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    '''Forward calculation function'''

    def forward(self, idx, targets=None):
        # The input is idx, the dimension is (batch size, sequence length, 1); targets are the target sequence, used to calculate loss
        device = idx.device
        b, t = idx.size()
        assert t <= self.args.block_size, f"This sequence cannot be calculated, the length of the sequence is {t}, and the maximum sequence length is only {self.args.block_size}"

        # By self.transformer
        # First, pass the input idx through the Embedding layer to get the dimensions as (batch size, sequence length, n_embd)
        print("idx", idx.size())
        # Through the Embedding layer
        tok_emb = self.transformer.wte(idx)
        print("tok_emb", tok_emb.size())
        # Then encode by location
        pos_emb = self.transformer.wpe(tok_emb)
        # Dropout again
        x = self.transformer.drop(pos_emb)
        # Then through Encoder
        print("x after wpe:", x.size())
        enc_out = self.transformer.encoder(x)
        print("enc_out:", enc_out.size())
        # Then through Decoder
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:", x.size())

        if targets is not None:
            # During the training phase, if we give targets, we calculate loss
            # First pass the last Linear layer to get the dimensions as (batch size, sequence length, vocab size)
            logits = self.lm_head(x)
            # Calculate cross entropy with targets
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # In the reasoning stage, we only need logits and loss as None
            # Take -1 to only take the last one in the sequence as output
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


def main():
    args = ModelArgs(100, 10, 100, 0.1, 512, 1000, 1000, 2)
    text = "I like to learn big models happily"
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    inputs_token = tokenizer(
        text,
        return_tensors='pt',
        max_length=args.max_seq_len,
        truncation=True,
        padding='max_length'
    )
    args.vocab_size = tokenizer.vocab_size
    transformer = Transformer(args)
    inputs_id = inputs_token['input_ids']
    logits, loss = transformer.forward(inputs_id)
    print(logits)
    predicted_ids = torch.argmax(logits, dim=-1).item()
    output = tokenizer.decode(predicted_ids)
    print(output)

if __name__ == "__main__":
    print("start")
    main()
