# Chapter 2 Transformer Architecture

## 2.1 Attention Mechanism

### 2.1.1 What is the Attention Mechanism

As NLP moves from statistical machine learning to deep learning, text representation methods, as the core problem of NLP, have gradually moved from statistical learning to deep learning. As we introduced in Chapter 1, text representations have entered the era of learning text representation through neural networks from the initial vector spatial model and language model calculated through the statistical learning model. However, there are three core architectures of neural networks developed from Computer Vision (CV) as their origin:

- Feedforward Neural Network (FNN), that is, the neurons in each layer are fully connected to each neuron in the upper and lower layers, as shown in Figure 2.1:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/1-0.png" alt="Image Description" width="90%"/>
  <p>Figure 2.1 Feedforward neural network</p>
</div>

- Convolutional Neural Network (CNN), that is, the convolutional layer with a training parameter much smaller than that of the feedforward neural network to perform feature extraction and learning, as shown in Figure 2.2:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/1-1.png" alt="Image Description" width="90%"/>
  <p>Figure 2.2 Convolutional Neural Network</p>
</div>

- Recurrent Neural Network (RNN), a network that can use historical information as input, including rings and self-repeaters, as shown in Figure 2.3:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/1-2.png" alt="Image Description" width="90%"/>
  <p>Figure 2.3 Recurrent Neural Network</p>
</div>

Since the text that NLP tasks need to process is often sequences, RNNs dedicated to processing sequence and time-series data can often achieve optimal results on NLP tasks. In fact, before the attention mechanism emerged, RNN and RNN's derivative architecture LSTM were well-deserved overlords in the NLP field. For example, the text representation model ELMo, which we talked about in Chapter 1, created the pre-training idea, uses two-way LSTM as the network architecture.

However, although RNN and LSTM have the advantages of capturing timing information and suitable for sequence generation, they have two irreparable shortcomings:

1. The sequential calculation pattern of sequence can simulate timing information well, but limits the ability of computers to calculate in parallel. Since sequences need to be input and calculated in sequence, the ability of parallel computing of Graphics Processing Unit (GPU) is greatly limited, resulting in the fact that although the parameters of the model with RNN as an infrastructure is not particularly large, the calculation time cost is very high;

2. RNNs are difficult to capture long sequence correlations. In the RNN architecture, the farther the relationship between inputs is, the harder it is to capture. At the same time, the RNN needs to read the entire sequence into memory and calculate it in sequence, which also limits the length of the sequence. Although this has been optimized by the gate mechanism in LSTM, RNN is still unsatisfactory for the capture of longer-distance correlations.

In response to such a problem, scholars such as Vaswani refer to the Attention mechanism proposed in the field of CV and often incorporated into RNNs (note that although the attention mechanism has been promoted in NLP, it is indeed proposed in the field of CV), and innovatively built a neural network composed entirely of attention mechanisms - Transformer, the originator and core architecture of the Large Language Model (LLM), so that the attention mechanism has become one of the most core architectures of deep learning.

So, what exactly is the attention mechanism?

The attention mechanism first originated from the field of computer vision. Its core idea is that when we focus on a picture, we often do not need to see the entire content clearly and only focus on the key part. In the field of natural language processing, we can often achieve more efficient and high-quality computing results by focusing on one or several tokens.

There are three core variables in the attention mechanism: **Query**, **Key**, and **Value**. We can understand the meaning of each variable through a case. For example, when we have a news report and we want to find the time of this report, then our Query can be a vector similar to "time" and "date" (for easy understanding, text is used here to represent it, but it is actually a dense vector), and Key and Value will be the entire text. By operating on Query and Key, we can get a weight, which actually reflects the relative amount of attention that should be distributed on each token in the text starting from Query. By computing the weight and Value, the final result is to calculate the entire text attention from Query.

​Specifically, the attention mechanism is characterized by calculating the correlation between **Query** and **Key** to the truth value weighted sum, thereby fitting the correlation between each word in the sequence and other words.

### 2.1.2 Understand the attention mechanism

We just mentioned that the attention mechanism has three core variables: query value, key value, and truth value. Next, we take the dictionary as an example to gradually analyze how the calculation formula of the attention mechanism is obtained, so as to help readers understand the attention mechanism in depth. First, we have a dictionary like this:

```json
{
    "apple":10,
    "banana":5,
    "chair":2
}
```

At this time, the key of the dictionary is the key value Key in the attention mechanism, and the value of the dictionary is the true value Value. The dictionary supports us to perform exact string matching. For example, if the value we want to find is that the query value Query is "apple", then we can directly get the corresponding value by matching Query with Key.

But what if we want the matching Query to be a concept that contains multiple keys? For example, we want to find "fruit", at this point, we should match both apple and banana, but not chair. Therefore, we often choose to combine the value corresponding to the Key to get the final value.

For example, when our Query is "fruit", we can give the following weights to the three keys:

```json
{
    "apple":0.6,
    "banana":0.4,
    "chair":0
}
```

Then, the value we finally query should be:

$$
value = 0.6 * 10 + 0.4 * 5 + 0 * 2 = 8
$$

The different weights given to different keys are what we call attention scores, which means how much attention we should give to each key in order to query Query. But how to calculate the corresponding attention score for each query? Intuitively, we can think that the higher the correlation between Key and Query, the greater the attention weight it should be given. But how can we find a reasonable way to calculate the correct attention score?

In Chapter 1, we mention the concept of word vectors. Through reasonable training fitting, word vectors can represent semantic information, so that words with similar semantics are closer in vector space, and words with more semantics are farther away in vector space. We often use the European distance to measure the similarity of word vectors, but we can also use dot product to measure:

$$
v·w = \sum_{i}v_iw_i
$$

According to the definition of word vectors, the dot product of word vectors corresponding to two words with similar semantics should be greater than 0, while the dot product of word vectors with different semantics should be less than 0.

Then, we can use dot product to calculate the similarity between words. Assuming that our Query is "fruit" and the corresponding word vector is $q$; the corresponding word vector of our Key is $k = [v_{apple} v_{banana} v_{chair}]$ , then we can calculate the similarity between Query and each key:

$$
x = qK^T
$$

Here, K is a matrix formed by stacking word vectors corresponding to all Keys. Based on the definition of matrix multiplication, x is the dot product of q and each k value. Now that we get x reflects the similarity between Query and each Key, we convert it into a sum of weights with 1 through a Softmax layer:

$$
\text{softmax}(x)_i = \frac{e^{xi}}{\sum_{j}e^{x_j}}
$$

In this way, the resulting vector can reflect the similarity between Query and each Key, and at the same time, the sum weight is 1, which is our attention score. Finally, we can use the corresponding product of the obtained attention score and the value vector. Based on the above process, we can get the basic formula for the calculation of attention mechanism:

$$
attention(Q,K,V) = softmax(qK^T)v
$$

However, the value at this time is still a scalar, and at the same time, we only query one Query this time. We can convert the value into a vector with a dimension of $d_v$, query multiple Query at once, and stack the word vectors corresponding to multiple Query together to form a matrix Q, and obtain the formula:

$$
attention(Q,K,V) = softmax(QK^T)V
$$

Currently, we are still one last step away from the standard attention mechanism formula. In the previous formula, if the dimension $d_k$ corresponding to Q and K is relatively large, softmax will be very easily affected when scaling, causing large differences between different values, thus affecting the stability of the gradient. Therefore, we want to scale the result of the product of Q and K:

$$
attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

This is the core calculation formula of the attention mechanism.

### 2.1.3 Implementation of attention mechanism

Based on the above, we can simply use Pytorch to implement the code of attention mechanism:

```python
'''注意力计算函数'''
def attention(query, key, value, dropout=None):
    '''
    args:
    query: 查询值矩阵
    key: 键值矩阵
    value: 真值矩阵
    '''
    # 获取键向量的维度，键向量的维度和值向量的维度相同
    d_k = query.size(-1) 
    # 计算Q与K的内积并除以根号dk
    # transpose——相当于转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # Softmax
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
        # 采样
     # 根据计算结果对value进行加权求和
    return torch.matmul(p_attn, value), p_attn

```

Note that in the above code, we assume that the input q, k, and v are transformed word vector matrices, that is, Q, K, and V in the formula. We only need to use the above few lines of code to implement the core attention mechanism calculation.

### 2.1.4 Self-attention

Based on the above analysis, we can find that the essence of the attention mechanism is to calculate the similarity of the elements of two sequences in sequence, find out the correlation between each element of one sequence and each element of another sequence, and then weight it based on the correlation, that is, allocate attention. These two sequences are the source of Q, K, and V in our calculation process.

However, in our practical applications, we often only need to calculate the attention result between Query and Key, and there is rarely an additional truth value. That is to say, we only need to fit two text sequences. ​In the classical attention mechanism, Q often comes from one sequence, and K and V come from another sequence, and are both calculated through parameter matrix, so that the relationship between these two sequences can be fitted. For example, in the Decoder structure of Transformer, Q comes from the input of Decoder, and K and V come from the output of Encoder, thus fitting the relationship between encoded information and historical information, making it easier to synthesize these two information to achieve future predictions.

​But in the Encoder structure of Transformer, a variant of the attention mechanism is used - self-attention (self-attention) mechanism. The so-called self-attention means calculating the attention distribution of each element in the sequence to other elements. That is, during the calculation process, Q, K, and V are all calculated from the same input through different parameter matrices. In Encoder, Q, K, and V are the input product to the parameter matrix $W_q, W_k, and W_v$, respectively, to fit the relationship between each token in the input statement to all other tokens.

Through the self-attention mechanism, we can find the correlation size of each token in a text and all other tokens, thereby modeling the dependencies between texts. In the implementation in the code, the self-attention mechanism is actually implemented by passing the same parameter to the inputs of Q, K, and V:

```python
# attention 为上文定义的注意力计算函数
attention(x, x, x)
```

### 2.1.5 Mask self-attention

Mask self-attention, i.e. Mask Self-Attention, refers to the self-attention mechanism using attention masks. The function of the mask is to block some tokens at specific locations. During the learning process of the model, the masked tokens will be ignored.

The core motivation for using attention masks is to allow models to use historical information to predict and not see future information. The Transformer model using attention mechanism is also learned through language model tasks similar to n-gram, that is, for a text sequence, it constantly predicts the next token based on the previous token until the entire text sequence is completed.

For example, if the text sequence to be learned is [BOS] I like you [EOS], then the model will predict and learn in the following order:

    Step 1: Input 【BOS】, output I
    Step 2: Input 【BOS】I, output like
    Step 3: Input 【BOS】I like, output you
    Step 4: Input [BOS] I like you, output [EOS]

Theoretically, as long as there are enough corpus to learn, through the above process, the model can learn any text sequence modeling method, that is, it can complete any text.

However, we can find that the above process is a serial process, that is, you need to complete Step 1 first before you can do Step 2, and then gradually complete the completion of the entire sequence. We said at the beginning that one of the core advantages of Transformer over RNN is that it can compute in parallel and has higher computing efficiency. If the model needs to complete the above process in serial to complete the learning for each training corpus, then it is obvious that parallel computing is not achieved and the computing efficiency is very low.

To address this problem, Transformer proposed a method of masking self-attention. Mask self-attention generates a string of masks to obscure future information. For example, the text sequence we want to learn is still [BOS] I like you [EOS], the attention mask we use is [MASK], so the input of the model is:

    <BOS> 【MASK】【MASK】【MASK】【MASK】
    <BOS>    I   【MASK】 【MASK】【MASK】
    <BOS>    I     like  【MASK】【MASK】
    <BOS>    I     like    you  【MASK】
    <BoS>    I     like    you   </EOS>

In each row of input, the model still sees only the previous token and predicts the next token. However, note that the above input is no longer a serial process, but can be input into the model in parallel. The model only needs each sample to predict the next token based on the unblocked token, thereby implementing a parallel language model.

Observing the above mask, we can find that it is actually an upper triangle matrix of equal length to the text sequence. We can simply create an upper triangle matrix of the same length as the input as the attention mask, and then use the mask to mask the input. That is to say, when the input dimension is (batch_size, seq_len, hidden_size), our Mask matrix dimension is generally (1, seq_len, seq_len) (implementing the calculation of different samples in the same batch through broadcast).

In the specific implementation, we generate the Mask matrix through the following code:

```python
# 创建一个上三角矩阵，用于遮蔽未来信息。
# 先通过 full 函数创建一个 1 * seq_len * seq_len 的矩阵
mask = torch.full((1, args.max_seq_len, args.max_seq_len), float("-inf"))
# triu 函数的功能是创建一个上三角矩阵
mask = torch.triu(mask, diagonal=1)
```

The generated Mask matrix will be an upper triangle matrix, all elements at the upper triangle position are -inf, and elements at other positions are set to 0.

When calculating attention, we will sum the calculated attention score with this mask, and then perform the Softmax operation:

```python
# 此处的 scores 为计算得到的注意力分数，mask 为上文生成的掩码矩阵
scores = scores + mask[:, :seqlen, :seqlen]
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
```

By summing, the attention score results of the upper triangle area (that is, the position corresponding to the token that should be obscured) become `-inf`, while the scores of the lower triangle area remain unchanged. If you do the Softmax operation, the value of `-inf` will be set to 0 after passing through Softmax, thus ignoring the attention score calculated in the upper triangle area, thus achieving attention obscurity.

### 2.1.6 Bucks’ Attention

The attention mechanism can achieve parallelization and long-term dependency fit, but one attention calculation can only fit one correlation, and it is difficult for a single attention mechanism to fully fit the correlation in the sentence sequence. Therefore, Transformer uses a multi-head attention mechanism, which is to perform multiple attention calculations on a corpus at the same time. Each attention calculation can fit different relationships. Splicing the last multiple results as the final output, so that language information can be more comprehensive and in-depth.

In the original paper, the author also verified through experiments that in the multi-head attention calculation, each different attention head can fit different information in the statement, as shown in Figure 2.4:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/1-3.jpeg" alt="Image Description" width="90%"/>
  <p>Figure 2.4 Broad attention mechanism</p>
</div>

​The upper and lower layers are the results of two attention heads performing self-attention calculations on the same sentence sequence. It can be seen that for different attention heads, they can fit relevant information at different levels. By simultaneous calculation of multiple attention heads, the statement relationship can be more comprehensively fitted.

In fact, the so-called multi-head attention mechanism is to process the original input sequence in multiple groups of self-attention; then splice the obtained self-attention results of each group, and then process them through a linear layer to obtain the final output. We can express it as:

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ...,
\mathrm{head_h})W^O    \\
    \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

The most intuitive code implementation is not complicated, that is, n heads have 3 parameter matrices of n groups. Each group performs the same attention calculation, but because it is a different parameter matrix, different attention results are achieved through back propagation, and then the n results are spliced together and output.

However, the above implementations have high spatial and temporal complexity. We can cleverly implement parallel multi-head calculations through matrix operations. The core logic is to use three combination matrices to replace the combination of n parameter matrices, that is, the matrix internal product and re-split are actually equivalent to the splicing matrix and re-inner product. For specific implementation, please refer to the following code:

```python
import torch.nn as nn
import torch

'''多头自注意力计算模块'''
class MultiHeadAttention(nn.Module):

    def __init__(self, args: ModelArgs, is_causal=False):
        # 构造函数
        # args: 配置对象
        super().__init__()
        # 隐藏层维度必须是头数的整数倍，因为后面我们会将输入拆成头数个矩阵
        assert args.dim % args.n_heads == 0
        # 模型并行处理大小，默认为1。
        model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并行处理大小。
        self.n_local_heads = args.n_heads // model_parallel_size
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads

        # Wq, Wk, Wv 参数矩阵，每个参数矩阵为 n_embd x n_embd
        # 这里通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接其实等同于拼接矩阵再内积，
        # 不理解的读者可以自行模拟一下，每一个线性层其实相当于n个参数矩阵的拼接
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        # 输出权重矩阵，维度为 dim x n_embd（head_dim = n_embeds / n_heads）
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        # 注意力的 dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)
         
        # 创建一个上三角矩阵，用于遮蔽未来信息
        # 注意，因为是多头注意力，Mask 矩阵比之前我们定义的多一个维度
        if is_causal:
           mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
           mask = torch.triu(mask, diagonal=1)
           # 注册为模型的缓冲区
           self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = q.shape

        # 计算查询（Q）、键（K）、值（V）,输入通过参数矩阵层，维度为 (B, T, n_embed) x (n_embed, n_embed) -> (B, T, n_embed)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, C // n_head)，然后交换维度，变成 (B, n_head, T, C // n_head)
        # 因为在注意力计算中我们是取了后两个维度参与计算
        # 为什么要先按B*T*n_head*C//n_head展开再互换1、2维度而不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开，
        # 然后按要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目标
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)


        # 注意力计算
        # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 掩码自注意力必须有注意力掩码
        if self.is_causal:
            assert hasattr(self, 'mask')
            # 这里截取到序列长度，因为有些序列可能比 max_seq_len 短
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
        # 计算 softmax，维度为 (B, nh, T, T)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 做 Dropout
        scores = self.attn_dropout(scores)
        # V * Score，维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头。
        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, C // n_head)，再拼接成 (B, T, n_head * C // n_head)
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

```

## 2.2 Encoder-Decoder

In the previous section, we introduced in detail the core of Transformer—the attention mechanism. In the article "Attention is All You Need", the author used only the attention mechanism and abandoned the traditional RNN and CNN architectures to build a Transformer model, thus bringing great changes in the NLP field. In Transformer, the attention mechanism is used by its two core components—Encoder (encoder) and Decoder (decoder). In fact, the subsequent pre-trained language models based on the Transformer architecture basically improve the Encoder-Decoder part to build a new model architecture, such as BERT that only uses Encoder, GPT that only uses Decoder, etc.

In this section, we analyze the Encoder-Decoder structure of Transformer based on the attention mechanism introduced in the above section based on the Seq2Seq task targeted by Transformer.

### 2.2.1 Seq2Seq Model

Seq2Seq, i.e. sequence-to-sequence, is a classic NLP task. Specifically, it means that the model inputs a natural language sequence $input = (x_1, x_2, x_3...x_n)$ , and the output is a natural language sequence that may not be equal to each other $output = (y_1, y_2, y_3...y_m)$ . In fact, Seq2Seq is the most classic task in NLP, and almost all NLP tasks can be regarded as Seq2Seq tasks. For example, text classification tasks can be regarded as target sequences with output length 1 (such as $m$ = 1 in the above formula); part-of-speech annotation tasks can be regarded as target sequences with the same length as the output and the input sequence (such as $m$ = $n$ in the above formula).

The machine translation task is a classic Seq2Seq task. For example, our input might be "The weather is so good today" and the output is "Today is a good day." Transformer is a classic Seq2Seq model, that is, the input of the model is a text sequence and the output is another text sequence. In fact, Transformer was first applied to machine translation tasks.

For the Seq2Seq task, the general idea is to encode and then decode natural language sequences. The so-called encoding means encoding the input natural language sequence into a vector (or matrix) that can represent semantics through a hidden layer, which can be simply understood as a more complex word vector representation. Decoding means that the vector or matrix encoded in the input natural language sequence is output through the hidden layer and then decoded into the corresponding natural language target sequence. By encoding and decoding, the Seq2Seq task can be implemented.

The Encoder in Transformer is used for the above encoding process; Decoder is used for the above decoding process. Transformer structure, as shown in Figure 2.5:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/2-0.jpg" alt="Image Description" width="90%"/>
  <p>Figure 2.5 Encoder-decoder structure</p>
</div>

Transformer consists of Encoder and Decoder, and each Encoder (Decoder) is composed of 6 Encoder (Decoder) Layers. The input source sequence will enter the Encoder for encoding, and then output the encoding result to each layer of the Decoder Layer. After decoding through Decoder, you can get the output target sequence.

Next, we will first introduce the classic structure of traditional neural networks within Encoder and Decoder - feedforward neural networks (FNN), layer normalization (Layer Norm) and residual connection (Residual Connection), and then further analyze the internal structure of Encoder and Decoder.

### 2.2.2 Feedforward neural network

Feed Forward Neural Network (FFN), which is the network structure in which each layer of neurons in the upper and lower layers is fully connected to each neuron in the upper and lower layers. Each Encoder Layer includes the attention mechanism mentioned above and a feedforward neural network. The implementation of feedforward neural network is relatively simple:

```python
class MLP(nn.Module):
    '''前馈神经网络'''
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和RELU激活函数
        # 然后，结果乘以输入x通过第三层线性变换的结果
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.relu(self.w1(x))))
    
```

Note that Transformer's feedforward neural network is composed of a RELU activation function added between two linear layers, and the feedforward neural network also adds a Dropout layer to prevent overfitting.

### 2.2.3 Layer normalization

Layer normalization, that is, Layer Norm, is a classic normalization operation in deep learning. There are generally two types of normalization in neural networks, batch normalization (Batch Norm) and layer normalization (Layer Norm).

The normalization core is to enable the range or distribution of input values of different layers to be relatively consistent. Since the input of each layer in a deep neural network is the output of the previous layer, under multi-layer transmission, the parameter changes of all previous neural layers in the network will cause major changes in the distribution of their inputs. That is to say, as the neural network parameters are updated, the output distribution of each layer is different, and the difference will increase with the increase in network depth. However, the conditional distributions that need to be predicted are always the same, which also causes prediction errors.

Therefore, in deep neural networks, normalization operations are often required to normalize the inputs of each layer into a standard normal distribution. Batch normalization refers to normalization on a mini-batch, which is equivalent to splitting a part of the sample from a batch, and first calculating the mean of the sample:

$$
\mu_j = \frac{1}{m}\sum^{m}_{i=1}Z_j^{i}
$$

Where $Z_j^{i}$ is the value of sample i on the j-th dimension, and m is the size of mini-batch.

Then calculate the variance of the sample:

$$
\sigma^2 = \frac{1}{m}\sum^{m}_{i=1}(Z_j^i - \mu_j)^2
$$

Finally, subtract the mean from the value of each sample and divide it by the standard deviation to convert the distribution of this mini-batch sample into a standard normal distribution:

$$
\widetilde{Z_j} = \frac{Z_j - \mu_j}{\sqrt{\sigma^2 + \epsilon}}
$$

Adding this small amount of $\epsilon$ here is to avoid denominator 0.

However, batch normalization has some shortcomings, such as:

- When the video memory is limited and the mini-batch is small, the mean and variance of the samples taken by Batch Norm cannot reflect the global statistical distribution information, resulting in poor results;
- For RNNs that are unfolding in the time dimension, the same distribution of different sentences is likely to be different, so the normalization of Batch Norm will lose its meaning;
- During training, Batch Norm needs to save statistics (mean and variance) for each step. During testing, due to the characteristics of variable-length sentences, sentences that may appear in the test set that are longer than the training set, so there is no training statistic for the step at the subsequent position;
- To apply Batch Norm, each step needs to save and calculate batch statistics, which is time-consuming and labor-intensive.

Therefore, layer normalization (Layer Norm) is more commonly used and more effective in deep neural networks. Compared with Batch Norm counting the mean and variance of all samples on each layer, Layer Norm calculates the mean and variance of all its layers on each sample, thus stabilizing the distribution of each sample. The normalization method of Layer Norm is actually exactly the same as that of Batch Norm, except that the dimensions of statistical statistics are different.

Based on the above normalization formula, we can simply implement a Layer Norm layer:

```python
class LayerNorm(nn.Module):
    ''' Layer Norm 层'''
    def __init__(self, features, eps=1e-6):
	super(LayerNorm, self).__init__()
    # 线性矩阵做映射
	self.a_2 = nn.Parameter(torch.ones(features))
	self.b_2 = nn.Parameter(torch.zeros(features))
	self.eps = eps
	
    def forward(self, x):
	# 在统计每个样本所有维度的值，求均值和方差
	mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
	std = x.std(-1, keepdim=True) # std: [bsz, max_len, 1]
    # 注意这里也在最后一个维度发生了广播
	return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```
Note that in the Layer Norm layer we implemented above, there are two linear matrices for mapping.

### 2.2.4 Residual connection

Since the Transformer model has a complex structure and deeper layers, in order to avoid model degradation, Transformer adopts the idea of residual connection to connect each sublayer. Residual connection, that is, the input of the next layer is not only the output of the previous layer, but also the input of the previous layer. Residual connection allows the lowest level information to be transmitted directly to the highest level, allowing the upper level to focus on the learning of residuals.

For example, in Encoder, in the first sub-layer, the input enters the multi-head self-attention layer and is directly passed to the output of the layer, and the output of the layer will be added to the original input and then normalized. The same is true in the second sub-layer. Right now:

$$
x = x + MultiHeadSelfAttention(LayerNorm(x))
$$

$$
output = x + FNN(LayerNorm(x))
$$

In our code implementation, we implement residual connections by adding the original value to the forward calculation of the layer:

```python
# 注意力计算
h = x + self.attention.forward(self.attention_norm(x))
# 经过前馈神经网络
out = h + self.feed_forward.forward(self.fnn_norm(h))
```

In the above code, self.attention_norm and self.fnn_norm are both LayerNorm layers, self.attn is the attention layer, and self.feed_forward is the feedforward neural network.

### 2.2.5 Encoder


After implementing the above components, we can build the Transformer Encoder. Encoder consists of N Encoder Layers, each of which includes an attention layer and a feedforward neural network. Therefore, we can first implement an Encoder Layer:

```python
class EncoderLayer(nn.Module):
  '''Encoder层'''
    def __init__(self, args):
        super().__init__()
        # 一个 Layer 中有两个 LayerNorm，分别在 Attention 之前和 MLP 之前
        self.attention_norm = LayerNorm(args.n_embd)
        # Encoder 不需要掩码，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args)

    def forward(self, x):
        # Layer Norm
        norm_x = self.attention_norm(x)
        # 自注意力
        h = x + self.attention.forward(norm_x, norm_x, norm_x)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.fnn_norm(h))
        return out
```

Then we build an Encoder, consisting of N Encoder Layers, and at the end we will add a Layer Norm to achieve normalization:

```python
class Encoder(nn.Module):
    '''Encoder 块'''
    def __init__(self, args):
        super(Encoder, self).__init__() 
        # 一个 Encoder 由 N 个 Encoder Layer 组成
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):
        "分别通过 N 层 Encoder Layer"
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```

The output through Encoder is the result after input encoding.

### 2.2.6 Decoder

Similarly, we can build a Decoder Layer first, and then assemble N Decoder Layers into Decoders. But unlike Encoder, Decoder consists of two attention layers and a feedforward neural network. The first attention layer is a masked self-attention layer, that is, using Mask's attention calculation, ensuring that each token can only use the attention score before the token; the second attention layer is a multi-head attention layer, which uses the output of the first attention layer as query and the output of the Encoder as the key and value to calculate the attention score. Finally, through the feedforward neural network:

```python
class DecoderLayer(nn.Module):
  '''解码层'''
    def __init__(self, args):
        super().__init__()
        # 一个 Layer 中有三个 LayerNorm，分别在 Mask Attention 之前、Self Attention 之前和 MLP 之前
        self.attention_norm_1 = LayerNorm(args.n_embd)
        # Decoder 的第一个部分是 Mask Attention，传入 is_causal=True
        self.mask_attention = MultiHeadAttention(args, is_causal=True)
        self.attention_norm_2 = LayerNorm(args.n_embd)
        # Decoder 的第二个部分是 类似于 Encoder 的 Attention，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.ffn_norm = LayerNorm(args.n_embd)
        # 第三个部分是 MLP
        self.feed_forward = MLP(args)

    def forward(self, x, enc_out):
        # Layer Norm
        norm_x = self.attention_norm_1(x)
        # 掩码自注意力
        x = x + self.mask_attention.forward(norm_x, norm_x, norm_x)
        # 多头注意力
        norm_x = self.attention_norm_2(x)
        h = x + self.attention.forward(norm_x, enc_out, enc_out)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
```

Then, in the same way, we build a Decoder block:

```python
class Decoder(nn.Module):
    '''解码器'''
    def __init__(self, args):
        super(Decoder, self).__init__() 
        # 一个 Decoder 由 N 个 Decoder Layer 组成
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)
```

Complete the above-mentioned construction of Encoder and Decoder, and complete the core part of Transformer. Next, splice the Encoder and Decoder and add the Embedding layer to build a complete Transformer model.

## 2.3 Build a Transformer

In the first two chapters, we have analyzed the Attention mechanism and the core of Transformer - Encoder and Decoder structures. Next, we can build a complete Transformer model based on the components implemented in the previous chapter.

### 2.3.1 Embedding layer

As we said in Chapter 1, in NLP tasks, we often need to convert natural language input into vectors that the machine can process. In deep learning, the component that undertakes this task is the Embedding layer.

The Embedding layer is actually an embedded vector lookup table that stores a fixed-size dictionary. In other words, before entering a neural network, we often let the natural language input pass through the word participle tokenizer. The function of the word participle is to divide the natural language input into tokens and convert it into a fixed index. For example, if we set the vocabulary size to 4 and enter "I like you", then the word participle can convert the input to:

```
input: 我
output: 0

input: 喜欢
output: 1

input：你
output: 2
```

Of course, in reality, the work of tokenizer will be more complicated than this. For example, there are many different ways to divide word into words, into subwords, into characters, etc., and the size of the word list is often as high as tens of thousands or hundreds of thousands. We will not elaborate on the details of tokenizer here. We will introduce in detail how tokenizers of large models run and train them later.

Therefore, the input of the Embedding layer is often a matrix of shape (batch_size, seq_len, 1). The first dimension is the number of batches at a time, the second dimension is the length of the natural language sequence, and the third dimension is the index value converted by tokens through tokenizer. For example, for the above input, the input to the Embedding layer would be:

```
[[[0],[1],[2]]]
```

The batch_size is 1 and seq_len is 3. The converted index is as above.

Embedding is actually a trainable weight matrix (Vocab_size, embedding_dim) and each value in the vocabulary table corresponds to a vector with a dimension of embedding_dim. For the input value, it will correspond to the word vector and then spliced into a matrix output (batch_size, seq_len, embedding_dim).

The above implementation is not complicated, we can directly use the Embedding layer in torch:

```python
self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
```

### 2.3.2 Position encoding

Attention mechanisms can achieve good parallel computing, but at the same time, their attention calculation method also leads to the loss of relative positions in the sequence. In RNN and LSTM, the input sequence will be processed recursively in sequence along the order of the statement itself, so the order of the input sequence provides extremely important information, which is very consistent with the characteristics of natural language itself.

However, from the above analysis of the attention mechanism, we can find that in the calculation process of the attention mechanism, for each token in the sequence, the other positions are equal to it, that is, "I like you" and "You like me" seem to be exactly the same in the attention mechanism, but this is undoubtedly a huge problem with the attention mechanism. Therefore, in order to use sequence sequence information and retain relative position information in the sequence, Transformer adopts a position encoding mechanism, which has been used by various models later.

​Position encoding, that is, encode the token according to the relative position of the sequence, and then add the position encoding to the word vector encoding. There are many ways to encode the position. Transformer uses the sine cosine function to encode the position (absolute position encoding Sinusoidal), and the encoding method is:

$$
PE(pos, 2i) = sin(pos/10000^{2i/d_{model}})\\
PE(pos, 2i+1) = cos(pos/10000^{2i/d_{model}})
$$

​In the above formula, pos is the position of token in the sentence, and 2i and 2i+1 indicate whether the token is an odd or even position. From the above formula, we can see that for the token of odd position and the token of even position, Transformer uses different functions for encoding.

Let's use a simple example to illustrate the calculation process of position encoding: If we input a sentence "I like to code" of length 4, we can get the following word vector matrix $\rm x$ , where each line represents a word vector, $\rm x_0=[0.1,0.2,0.3,0.4]$ corresponds to the word vector of "I", its pos is 0, and so on. The second line represents the word vector of "like", and its pos is 1:

$$
\rm x = \begin{bmatrix} 0.1 & 0.2 & 0.3 & 0.4 \\ 0.2 & 0.3 & 0.4 & 0.5 \\ 0.3 & 0.4 & 0.5 & 0.6 \\ 0.4 & 0.5 & 0.6 & 0.7 \end{bmatrix}
$$

​The word vector after position encoding is:

$$
\rm x_{PE} = \begin{bmatrix} 0.1 & 0.2 & 0.3 & 0.4 \\ 0.2 & 0.3 & 0.4 & 0.5 \\ 0.3 & 0.4 & 0.5 & 0.6 \\ 0.4 & 0.5 & 0.6 & 0.7 \end{bmatrix} + \begin{bmatrix} \sin(\frac{0}{10000^0}) & \cos(\frac{0}{10000^0}) & \sin(\frac{0}{10000^{2/4}}) & \cos(\frac{0}{10000^{2/4}}) \\ \sin(\frac{1}{10000^0}) & \cos(\frac{1}{10000^0}) & \sin(\frac{1}{10000^{2/4}}) & \cos(\frac{1}{10000^{2/4}}) \\ \sin(\frac{2}{10000^0}) & \cos(\frac{2}{10000^0}) & \sin(\frac{2}{10000^{2/4}}) & \cos(\frac{2}{10000^{2/4}}) \\ \sin(\frac{3}{10000^0}) & \cos(\frac{3}{10000^0}) & \sin(\frac{3}{10000^{2/4}}) & \cos(\frac{3}{10000^{2/4}}) \end{bmatrix} = \begin{bmatrix} 0.1 & 1.2 & 0.3 & 1.4 \\ 1.041 & 0.84 & 0.41 & 1.49 \\ 1.209 & -0.016 & 0.52 & 1.59 \\ 0.541 & -0.489 & 0.895 & 1.655 \end{bmatrix}
$$

We can use the following code to get the location encoding of the above example:
```python
import numpy as np
import matplotlib.pyplot as plt
def PositionEncoding(seq_len, d_model, n=10000):
    P = np.zeros((seq_len, d_model))
    for k in range(seq_len):
        for i in np.arange(int(d_model/2)):
            denominator = np.power(n, 2*i/d_model)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

P = PositionEncoding(seq_len=4, d_model=4, n=100)
print(P)
```

```python
[[ 0.          1.          0.          1.        ]
 [ 0.84147098  0.54030231  0.09983342  0.99500417]
 [ 0.90929743 -0.41614684  0.19866933  0.98006658]
 [ 0.14112001 -0.9899925   0.29552021  0.95533649]]
```

Such position coding has two main benefits:

1. Enable PE to adapt to sentences that are longer than all sentences in the training set. Assuming that the longest sentence in the training set has 20 words and suddenly a sentence with a length of 21 comes, then the 21st bit can be calculated using the formula calculation method.
2. The model can easily calculate the relative position. For the fixed-length spacing k, PE(pos+k) can be calculated using PE(pos). Because Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B), Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B).

We can also prove the superiority of this encoding method through rigorous mathematical derivation. The original Transformer Embedding can be expressed as:

$$
\begin{equation}f(\cdots,\boldsymbol{x}_m,\cdots,\boldsymbol{x}_n,\cdots)=f(\cdots,\boldsymbol{x}_n,\cdots,\boldsymbol{x}_m,\cdots)\end{equation}
$$

It is obvious that such a function does not have asymmetry, which means that it cannot characterize relative position information. We want to get such a encoding:

$$
\begin{equation}\tilde{f}(\cdots,\boldsymbol{x}_m,\cdots,\boldsymbol{x}_n,\cdots)=f(\cdots,\boldsymbol{x}_m + \boldsymbol{p}_m,\cdots,\boldsymbol{x}_n + \boldsymbol{p}_n,\cdots)\end{equation}
$$

The $p_m$ added here and $p_n$ is the location encoding. Next we will do Taylor expansion in two positions: m,n:

$$
\begin{equation}\tilde{f}\approx f + \boldsymbol{p}_m^{\top} \frac{\partial f}{\partial \boldsymbol{x}_m} + \boldsymbol{p}_n^{\top} \frac{\partial f}{\partial \boldsymbol{x}_n} + \frac{1}{2}\boldsymbol{p}_m^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_m^2}\boldsymbol{p}_m + \frac{1}{2}\boldsymbol{p}_n^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_n^2}\boldsymbol{p}_n + \underbrace{\boldsymbol{p}_m^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_m \partial \boldsymbol{x}_n}\boldsymbol{p}_n}_{\boldsymbol{p}_m^{\top} \boldsymbol{\mathcal{H}} \boldsymbol{p}_n}\end{equation}
$$

It can be seen that the first term is not related to the position, the 2 to 5th term only depends on a single position, and the 6th term (f finds partial derivatives for m and n respectively) is related to the two positions, so we hope that the sixth term ($p_m^THp_n$) expresses the relative position information, that is, find a function g so that:

$$
p_m^THp_n = g(m-n)
$$

We assume that $H$ is a matrix of units, then:

$$
p_m^THp_n = p_m^Tp_n = \langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle = g(m-n)
$$

By treating the vector [x,y] as a complex number x+yi, the equation is constructed based on the complex number operation rules:

$$
\begin{equation}\langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle = \text{Re}[\boldsymbol{p}_m \boldsymbol{p}_n^*]\end{equation}
$$

Then assume that there is a plural $q_{m-n}$ so that:

$$
\begin{equation}\boldsymbol{p}_m \boldsymbol{p}_n^* = \boldsymbol{q}_{m-n}\end{equation}
$$

Use the exponential form of complex numbers to solve this equation to obtain the solution of position code in two-dimensional situation:

$$
\begin{equation}\boldsymbol{p}_m = e^{\text{i}m\theta}\quad\Leftrightarrow\quad \boldsymbol{p}_m=\begin{pmatrix}\cos m\theta \\ \sin m\theta\end{pmatrix}\end{equation}
$$

Since the inner product satisfies linear superposition, even-dimensional position encoding of higher dimensions can be represented as a combination of multiple two-dimensional position encodings:

$$
\begin{equation}\boldsymbol{p}_m = \begin{pmatrix}e^{\text{i}m\theta_0} \\ e^{\text{i}m\theta_1} \\ \vdots \\ e^{\text{i}m\theta_{d/2-1}}\end{pmatrix}\quad\Leftrightarrow\quad \boldsymbol{p}_m=\begin{pmatrix}\cos m\theta_0 \\ \sin m\theta_0 \\ \cos m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \sin m\theta_{d/2-1}  \end{pmatrix}\end{equation}
$$

Then take $\theta_i = 10000^{-2i/d}$ (this form can make the trend towards zero as |m−n| increases, ⟨pm,pn have a tendency toward zero. This can be proved by integrating the position encoding, and taking base as 10000 is the experimental result), the above encoding method is obtained.

When $H$ is not an unit matrix, because the correlation between any two dimensions of the d-dimensional vector formed by the Embedding layer of the model is relatively small, which satisfies a certain degree of decoupling. We can regard it as a diagonal matrix, so use the above encoding:

$$
\begin{equation}\boldsymbol{p}_m^{\top} \boldsymbol{\mathcal{H}} \boldsymbol{p}_n=\sum_{i=1}^{d/2} \boldsymbol{\mathcal{H}}_{2i,2i} \cos m\theta_i \cos n\theta_i + \boldsymbol{\mathcal{H}}_{2i+1,2i+1} \sin m\theta_i \sin n\theta_i\end{equation}
$$

By integrating and deteriorating:

$$
\begin{equation}\sum_{i=1}^{d/2} \frac{1}{2}\left(\boldsymbol{\mathcal{H}}_{2i,2i} + \boldsymbol{\mathcal{H}}_{2i+1,2i+1}\right) \cos (m-n)\theta_i + \frac{1}{2}\left(\boldsymbol{\mathcal{H}}_{2i,2i} - \boldsymbol{\mathcal{H}}_{2i+1,2i+1}\right) \cos (m+n)\theta_i \end{equation}
$$

This means that the encoding can still represent the relative position.

The above encoding results are shown in Figure 2.6:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/3-0.png" alt="Image Description" width="90%"/>
  <p>Figure 2.6 Encoding results</p>
</div>


Based on the above principle, we implement a position encoding layer:

```python

class PositionalEncoding(nn.Module):
    '''位置编码模块'''

    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        # Dropout 层
        self.dropout = nn.Dropout(p=args.dropout)

        # block size 是序列的最大长度
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        # 计算 theta
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        # 分别计算 sin、cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加到 Embedding 结果上
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
```

### 2.3.3 A complete Transformer

All the above components, and then spliced together according to the Transformer structure in the figure below, will be a complete Transformer model, as shown in Figure 2.7:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/3-1.png" alt="Image Description" width="80%"/>
  <p>Figure 2.7 Transformer model structure</p>
</div>

But it should be noted that the above picture is a picture of the original paper "Attention is all you need". The LayerNorm layer is placed behind the Attention layer, which is the "Post-Norm" structure. However, in the source code it published, the LayerNorm layer is placed before the Attention layer, which is the "Pre Norm" structure. Considering that LLM currently generally adopts the "Pre-Norm" structure (which can make loss more stable), this article adopts the "Pre-Norm" structure when implementing it.

As shown in the figure, the output after the tokenizer mapping is first encoded by the Embedding layer and the Positional Embedding layer, and then enter the N Encoders and N Decoders mentioned in the previous section (in the Transformer original model, N is taken as 6), and finally, through a linear layer and a Softmax layer, the final output is obtained.

Based on the components implemented previously, we implement the complete Transformer model:

```python
class Transformer(nn.Module):
   '''整体模型'''
    def __init__(self, args):
        super().__init__()
        # 必须输入词表大小和 block size
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.vocab_size, args.n_embd),
            wpe = PositionalEncoding(args),
            drop = nn.Dropout(args.dropout),
            encoder = Encoder(args),
            decoder = Decoder(args),
        ))
        # 最后的线性层，输入是 n_embd，输出是词表大小
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # 初始化所有的权重
        self.apply(self._init_weights)

        # 查看所有参数的数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    '''统计所有参数的数量'''
    def get_num_params(self, non_embedding=False):
        # non_embedding: 是否统计 embedding 的参数
        n_params = sum(p.numel() for p in self.parameters())
        # 如果不统计 embedding 的参数，就减去
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    '''初始化权重'''
    def _init_weights(self, module):
        # 线性层和 Embedding 层初始化为正则分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    '''前向计算函数'''
    def forward(self, idx, targets=None):
        # 输入为 idx，维度为 (batch size, sequence length, 1)；targets 为目标序列，用于计算 loss
        device = idx.device
        b, t = idx.size()
        assert t <= self.args.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"

        # 通过 self.transformer
        # 首先将输入 idx 通过 Embedding 层，得到维度为 (batch size, sequence length, n_embd)
        print("idx",idx.size())
        # 通过 Embedding 层
        tok_emb = self.transformer.wte(idx)
        print("tok_emb",tok_emb.size())
        # 然后通过位置编码
        pos_emb = self.transformer.wpe(tok_emb) 
        # 再进行 Dropout
        x = self.transformer.drop(pos_emb)
        # 然后通过 Encoder
        print("x after wpe:",x.size())
        enc_out = self.transformer.encoder(x)
        print("enc_out:",enc_out.size())
        # 再通过 Decoder
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:",x.size())

        if targets is not None:
            # 训练阶段，如果我们给了 targets，就计算 loss
            # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
            logits = self.lm_head(x)
            # 再跟 targets 计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段，我们只需要 logits，loss 为 None
            # 取 -1 是只取序列中的最后一个作为输出
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
```

Note that in addition to building the entire Transformer structure, the above code also implements three additional functions:

- get_num_params: The amount of parameters used to statistics the model
- _init_weights: used to randomly initialize all parameters of the model
- forward: forward calculation function

In addition, in the forward calculation function, we use pytorch's cross entropy function to calculate the loss for the model. For different loss functions, readers can consult the official documentation of Pytorch, and I will not repeat it here.

After the above steps, we can "hand rub" a complete, computable Transformer model from zero. Due to the fact that this book focuses on LLM, in this chapter, we will no longer talk about how to train the Transformer model in detail; in the following text, we will similarly "hand rub" a LLaMA model from scratch and lead everyone to train their own Tiny LLaMA step by step.

**References**

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. (2023). *Attention Is All You Need.* arXiv preprint arXiv:1706.03762.

[2] Jay Mody's article "An Intuition for Attention". Source: https://jaykmody.com/blog/attention-intuition/
