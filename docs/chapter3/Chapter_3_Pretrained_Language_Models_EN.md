# Chapter 3 Pre-trained language model

## 3.1 Encoder-only PLM

In the previous chapter, we explained in detail the attention mechanism, which brought huge changes to the NLP field, and the Transformer model built on it. This marked the start of a milestone transformation in NLP models. In the above explanation of Transformer, we can see that the Transformer structure is mainly composed of two parts: Encoder and Decoder, and the two parts have different structures and inputs and outputs respectively.

In view of the characteristics of Encoder and Decoder, ELMo's pre-training ideas have been introduced, and different ideas for optimizing Transformer have begun to appear. For example, Google only selected the Encoder layer, and by stacking the Encoder layer, and then proposed different pre-trained tasks - Masked Language Model (MLM), creating BERT, the representative model that came to dominate natural language understanding (NLU) tasks. OpenAI chose the Decoder layer, and used the original language model (LM) task to continuously increase model parameters and pre-trained corpus to create a GPT series model with obvious advantages in NLG (Natural Language Generation) task, which is also the base model of the popular LLM today. Of course, there is another idea to keep both Encoder and Decoder to create pre-trained Transformer models, such as the T5 model published by Google.

In this chapter, we will introduce the various mainstream pre-trained models of the Transformer era in sequence in the order of Encoder-Only, Encoder-Decoder, and Decoder-Only, respectively, and introduce the three core model architectures, the pre-training tasks selected by each mainstream model and their unique advantages. This is also the model basis of all mainstream LLMs at present.

### 3.1.1 BERT

BERT, full name Bidirectional Encoder Representations from Transformers, is a pre-trained language model released by the Google team in 2018. The model was published in the paper "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", achieving the optimal performance of seven natural language processing evaluation tasks including GLUE and MultiNLI, which is a milestone achievement. Since the launch of BERT, the pre-training + fine-tuning mode has begun to become the mainstream of natural language processing tasks. Not only is BERT itself constantly updated and iterated to improve model performance, but also models based on BERT such as MacBERT and BART have been optimized and improved. It can be said that BERT is a phased achievement of natural language processing, marking the major progress of various natural language processing tasks and the establishment of the dominance of pre-trained models. It was not until the birth of LLM that the dominance of the NLP field migrated from the BERT-based model. Even in the LLM era, to have a deep understanding of LLM and NLP, BERT is an unavoidable link.

#### (1) Ideas Inherited

BERT is a pre-trained model that unifies multiple ideas. The core ideas it follows include:

- Transformer architecture. As we introduced in the previous chapter, the paper "Attention is All You Need" published in 2017 proposed a Transformer model that completely uses the attention mechanism and abandons RNN and LSTM structures, bringing a new model architecture. BERT follows the idea of Transformer and optimizes the model base of Transformer. By stacking the Encoder structure and expanding the model parameters, it creates a model architecture uniquely suited to NLU tasks;
- Pre-training + fine-tuning paradigm. Also in 2018, the birth of ELMo marked the birth of the pre-training + fine-tuning paradigm. The ELMo model is based on the bidirectional LSTM architecture, pre-trained based on the language model on the training data, and then fine-tuned for downstream tasks, showing better performance, and directing the NLP field to pre-training + fine-tuning research ideas. BERT also adopted this paradigm and introduced a pre-training task MLM that is more suitable for text comprehension and can capture deep bidirectional semantic relationships by adjusting the model architecture to Transformer, pushing the pre-training-fine-tuning paradigm to a climax.

Next, we will analyze BERT in-depth from three aspects: model architecture, pre-training tasks and downstream tasks fine-tuning, analyze the core ideas and advantages of BERT, helping everyone understand why BERT can have performance far exceeding previous models, and thus understand more deeply how LLM can defeat BERT and unveil the curtain of a new era.

#### (2) Model architecture——Encoder Only

BERT's model architecture is made up of the Encoder part of Transformer, and its main structure is shown in Figure 3.1:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/1-0.png" alt="Image Description" width="100%"/>
  <p>Figure 3.1 BERT model structure</p>
</div>

BERT is a pre-trained model for NLU tasks. The input is generally a text sequence, while the output is generally a Label, such as a positive or negative Label for sentiment classification. However, just as Transformer is a Seq2Seq model, the BERT stacked using Encoder is essentially a Seq2Seq model, but there is no Decoder for a specific task added. Therefore, to adapt to various NLU tasks, a classification header prediction_heads is added to the top layer of the model to convert the hidden state of multi-dimensionality to the classification dimension through a linear layer (for example, if there are two categories in total, prediction_heads outputs a two-dimensional vector).

The whole model consists of Embedding, Encoder and prediction_heads:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/1-1.png" alt="Image Description" width="70%"/>
  <p>Figure 3.2 BERT model brief structure</p>
</div>

The input text sequence will first be converted into input_ids through the tokenizer (basically, the operation of each model in tokenizer is similar. You can refer to the tokenizer mechanism of Transformer, which will not be described later), and then enter the Embedding layer and convert it into hidden_states of a specific dimension, and then pass the Encoder block. The Encoder block consists of N stacked Encoder Layers. BERT comes in two sizes: a base version (12 Encoder Layers, hidden dimension of 768, 110M parameters in total) and a large version (24 Encoder Layers, hidden dimension of 1024, 340M parameters in total). The top-most hidden_states after Encoder encoding finally gets the final category probability through prediction_heads. Through Softmax calculation, the category predicted by the model can be calculated.

prediction_heads is actually a linear layer plus an activation function. Generally speaking, the output dimension of the last linear layer is equal to the number of classes of the task, as shown in Figure 3.3:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/1-5.png" alt="Image Description" width="20%"/>
  <p>Figure 3.3 prediction_heads structure</p>
</div>

Each layer of Encoder Layer is a layer with a similar structure to the Encoder Layer in Transformer, as shown in Figure 3.4:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/1-2.png" alt="Image Description" width="40%"/>
  <p>Figure 3.4 Encoder Layer Structure</p>
</div>

As shown in Figure 3.5, hidden_states mapped through the Embedding layer enters the core attention mechanism, and then adds the original input through the residual connection mechanism, and then passes through an Intermediate layer to obtain the final output. The Intermediate layer is a special name for BERT, which is actually a linear layer plus an activation function:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/1-3.png" alt="Image Description" width="40%"/>
  <p>Figure 3.5 Intermediate structure</p>
</div>

Note that the activation function used by BERT is the GELU function, which is the full name of the Gaussian error linear unit activation function, an activation function that only began to receive wide attention with BERT. The calculation method of GELU is:

$$GELU(x) = 0.5x(1 + tanh(\sqrt{\frac{2}{\pi}})(x + 0.044715x^3))$$

The core idea of GELU is to introduce the idea of stochastic regularization into the activation function, deciding whether to drop or keep a neuron based on the probability distribution of the input itself. The principles and core ideas of GELU will not be described here. Interested readers can learn it by themselves.

The attention mechanism of BERT is almost exactly the same as the self-attention mechanism of Encoder in Transformer, but BERT integrates relative position coding into the attention mechanism, and regards relative position coding as a trainable weight parameter, as shown in Figure 3.6:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/1-4.png" alt="Image Description" width="40%"/>
  <p>Figure 3.6 BERT Attention Mechanism Structure</p>
</div>

As shown in the figure, the only difference between BERT's attention calculation process and Transformer is that after completing the calculation of attention scores, the relative position information is first integrated through the Position Embedding layer. The Position Embedding layer here is actually a linear matrix. Fitting relative positions through trainable parameters can be relatively richer than the absolute position encoding Sinusoidal used by Transformer, but this also adds a lot of model parameters, and it is completely impossible to process inputs that exceed the training length of the model (for example, the maximum context length that can be processed for BERT is 512 tokens).

It can be seen that BERT's model architecture is based on Transformer's Encoder, which is why BERT follows the idea of Transformer.

#### (3) Pre-training task—MLM + NSP

Compared with the model architecture that basically follows the Transformer, BERT's greater innovation lies in the two new pre-training tasks it proposes - MLM and NSP (Next Sentence Prediction). The core advantage of the pre-training-fine-tuning paradigm is that by separating pre-training and fine-tuning, a model that completes a pre-training can be applied to almost all downstream tasks only through fine-tuning. As long as the cost of fine-tuning is low, even if the pre-training cost is several times or even dozens of times, the model still has greater application value. Therefore, the model parameters and pre-training data can be further expanded, and a large amount of pre-training corpus can be used to allow the model to fit potential semantics and underlying knowledge, so that the model can gain strong language understanding and generation capabilities through long-term and large-scale pre-training.

Therefore, the core requirement of pre-trained data is to require a huge data scale (hundreds of millions of tokens). There is no doubt that it is difficult to reach this scale by manually labeling the fully supervised data output. Therefore, the pre-training data must be obtained from unsupervised corpus. This is also the reason why traditional pre-training tasks are LM - LM predicts the following text from the preceding text, so it can be applied directly to any text: we simply mask what follows, feed the preceding text into the model and ask it to predict the continuation, so all text corpus on the Internet can be used for pre-training.

However, a major drawback of the LM pre-training task is that it directly fits the semantic relationship from left to right, but ignores the two-way semantic relationship. Although the position information in the text sequence is characterized by position encoding in Transformer, this is essentially different from directly fitting the bidirectional semantic relationship. For example, BiLSTM (Bidirectional LSTM model) is often better than LSTM model in semantic representation because BiLSTM fits bidirectional semantic relationships through bidirectional LSTM. Therefore, is there a pre-training task that can not only utilize a large amount of unsupervised corpus, but also train the model's ability to fit bidirectional semantic relationships?

Based on this idea, scholars such as Jacob proposed MLM, that is, the masked language model, as a new pre-training task. Compared to LM, which simulates human writing, MLM simulates a "cloze" (fill-in-the-blank) test. The idea of MLM is also very simple: randomly mask some tokens in a text sequence, then feed all unmasked tokens into the model and require it to predict the masked tokens based on the input. For example, the input and output may be:

    Input: I <MASK> you because you are <MASK>
    Output: <MASK> - love; <MASK> - wonderful

Since the model can use both the preceding and following context of a masked token to understand the semantics and predict it, through such a task, the model can fit bidirectional semantics, which can better realize the understanding of the text. Similarly, MLM tasks do not require any artificial annotation of text, but only need to randomly obscure the text. Therefore, all text corpus on the Internet can be used to pre-train. For example, BERT's pre-training uses a full 3300M word corpus.

However, MLM also has its inherent flaws. The LM task simulates the natural human writing process, and its training is completely consistent with the downstream tasks. That is to say, training predicts the following text from the preceding text, and the same is true for downstream fine-tuning and inference. However, MLM is different. When fine-tuning and inference downstream tasks, there is actually no `<MASK>` that we manually add. We will directly obtain the corresponding hidden state through the original text and then enter the classifier or other components according to the downstream tasks. The inconsistency between pre-training and fine-tuning will greatly affect the performance of fine-tuning of the model in downstream tasks. In response to this problem, the authors have improved the MLM strategy.

When performing specific MLM training, 15% of the tokens in the training corpus will be randomly selected for masking. However, these 15% of tokens are not all replaced by `<MASK>`: each has an 80% chance of being masked, a 10% chance of being replaced with a random token, and a 10% chance of staying the same. The 10% that remain unchanged serve to eliminate the inconsistency between pre-training and fine-tuning, while the core significance of 10% random replacement is to force the model to maintain learning of contextual information. Because if all of them were masked, the model would only need to process the masked positions, thus only learning the token to be predicted and losing learning of the context. By introducing partial random tokens, the model cannot determine the tokens that need to be predicted, and is forced to maintain the contextual representation distribution of each token, thus having the ability to represent the characteristics of sentences. And because the probability of random tokens is very low, it will not affect the actual language comprehension ability of the model.

In addition to MLM, BERT also proposed another pre-training task - NSP, that is, the next sentence prediction. The core idea of NSP is to target sentence-level NLU tasks such as question-answer matching, natural language inference, etc. Question-and-answer matching refers to entering a question and several answers, requiring the model to find the real answer to the problem; natural language inference refers to entering a premise and a hypothesis and judging whether the hypothesis follows from the premise. Such tasks require the model to fit the relationship at the sentence level and judge the relationship between two sentences, rather than the semantic relationship of MLM fitting at the token level. Therefore, BERT proposes NSP task to train the semantic relationship fit of the model at the sentence level.

The core idea of NSP task is to require the model to determine whether the two sentences of a sentence pair are consecutive in context. For example, the input and output may be:

    Input:
        Sentence A: I love you.
        Sentence B: Because you are wonderful.
    Output:
        1 (is a continuous context)

    Input:
        Sentence A: I love you.
        Sentence B: Because today's dinner is so nice.
    Output:
        0 (not continuous context)

By requiring the model to judge the sentence-pair relationship, the model is forced to fit the relationship between sentences to adapt to the sentence-level NLU tasks. Similarly, since the positive samples of NSP can randomly draw any continuous sentences from unsupervised corpus, while the negative samples can randomly draw the sentences after being disrupted (just make sure not to draw the already continuous sentences), it can also have almost unlimited training data.

During specific pre-training, BERT used 800M BooksCorpus corpus and 2500M English Wikipedia corpus. 90% of the data was trained with 128 context length, and the remaining 10% of the data was pre-trained with 512 as context length, and a total of about 3.3B tokens were trained. The hyperparameters of its training are also worthy of attention. BERT's training corpus is 13GB in total, and it trains 1M steps (40 Epochs) on a batch size of 256. In comparison, LLM generally only trains one Epoch and uses a batch size much larger than 256.

It can be seen that compared with traditional non-pre-trained models, the amount of data it trains has increased exponentially. Of course, more massive training data requires greater computing power. The Base version and Large version of BERT used 16 TPUs and 64 TPUs to complete it after 4 days of training.

#### (4) Downstream Task Fine-Tuning

As a milestone achievement in the NLP field, one of the important significance of BERT is to formally establish the two-stage idea of pre-training-fine-tuning, that is, pre-training on a large number of unsupervised corpus to obtain general text understanding and generation capabilities, and then fine-tuning on corresponding downstream tasks. One of the key points of this idea is whether the powerful abilities obtained by pre-training can be quickly transferred to the corresponding downstream tasks through low-cost fine-tuning.

To this end, BERT designed a more general input and output layer to adapt to transfer learning under multitasking. For each input text sequence, BERT will add a special token `<CLS>` to its header. In subsequent encoding, the token represents the state of the entire sentence, which is the semantic representation at the sentence level. When performing NSP pre-training, the eigenvector corresponding to the token is used as the input to the last classifier.

After completing pre-training, for each downstream task, only a certain amount of fully supervised manual annotation data is needed to fine-tune the pre-trained BERT on the task. The so-called fine-tuning is actually consistent with the strategy of updating model parameters during training, but training is performed on specific tasks, less training data, and smaller batch_size, and the updating parameters is smaller. For most downstream tasks, the output of BERT can be directly used. For example, for text classification tasks, you can directly modify the final classification header of prediction_heads in the model structure. For tasks such as sequence annotation, you can integrate BERT multi-layer hidden layer vectors and output the last annotation result. For text generation tasks, you can also take the output of Encoder and directly decode it to obtain the final generated result. Therefore, BERT can be applied very efficiently to a variety of NLP tasks.

Once BERT was proposed, it directly achieved SOTA results on 11 NLP tracks and became the well-deserved overlord in the NLU direction. Several subsequent models that achieved better results on NLU tasks were improved based on BERT. Until the LLM era, BERT could still achieve optimal results on many NLU tasks with rich annotation data. In fact, BERT is more usable than LLM for certain specific tasks with rich training data and emphasis on high throughput.

### 3.1.2 RoBERTa

As an epoch-making masterpiece of NLP, BERT has achieved SOTA results on multiple lists, which has also led the entire NLP field to migrate towards pre-trained models. Based on BERT, optimization is carried out in multiple directions, and a large number of excellent Encoder-Only pre-trained models have emerged. Most of them have model structures similar or completely consistent with BERT, and are optimized in terms of training data, pre-training tasks, training parameters, etc. to obtain pre-training models with stronger capabilities and more eye-catching performance on downstream tasks. One of them is RoBERTa, also released by Facebook.

As we mentioned earlier, one of the core advantages of pre-training-fine-tuning is that it can be pre-trained using a massive amount of unsupervised corpus that is much larger than previous training data. Because in the traditional deep learning paradigm, for each task, we need to train a model from scratch, so we cannot use too large model parameters, otherwise we need to have a large scale of supervised data to make the model fit better, which is too expensive. However, in the pre-training-fine-tuning paradigm, we can use as much training data as possible in the pre-training stage, and only need to pre-train the model once, and then fine-tune them through a small amount of supervised data on each downstream task. BERT uses 13GB (3.3B token) data for pre-training, which is an extremely huge data scale compared to traditional NLP.

But does 13GB of pre-training data allow BERT to achieve a full fit? If we use more pretrained corpus, can we further enhance the model performance? Moreover, are the pre-training tasks and training hyperparameters selected by BERT the optimal ones? RoBERTa came into being.

#### (1) Optimization 1: Remove NSP pre-training tasks

RoBERTa's model architecture is exactly the same as BERT, that is, the model parameters using BERT-large (24-layer Encoder Layer, 1024's hidden layer dimension, total parameter amount 340M). In pre-training tasks, some scholars question that NSP tasks cannot improve model performance because they are too simple. Adding them to pre-training will not significantly benefit downstream tasks when fine-tuning them, and may even bring negative effects. RoBERTa sets up four experimental groups:

    1. Segment-pair MLM + NSP: BERT's original pre-training task; each input is a pair of segments, each containing multiple sentences, used to construct the NSP task;
    2. Sentence-pair MLM + NSP: each input is a single pair of sentences, and the batch size is increased so the total number of tokens matches the original input;
    3. MLM across documents: Remove the NSP task, each input consists of full sentences sampled contiguously from one or more documents. In order to make the input reach the maximum length (512), one input may include multiple documents;
    4. MLM for single document: Remove NSP tasks and limit one input to sample only from one document. Again, the batch size is increased so the total number of tokens matches the original input.

Experimental results prove that the latter two groups are significantly better than the first two groups, and the single-document MLM group performs best when fine-tuning on downstream tasks. Therefore, RoBERTa removes NSP in pre-training and uses only MLM tasks.

At the same time, RoBERTa has also made improvements to the MLM task itself. In BERT, the operation of Mask is completed in the data processing stage, so the `<MASK>` to be predicted by the same sample is always consistent during the later pre-training. Since BERT trained a total of 40 Epochs, in order to make the model's training data more extensive, BERT applied four different random masks to the data, that is, the data trained by every 10 Epoch models are completely consistent. RoBERTa puts the Mask operation in the training stage, which is a dynamic masking strategy, so that the position of the training data Mask of each Epoch is inconsistent. In the experiment, dynamic masking has only a very slight advantage over static masking, but because dynamic masking is more efficient and easy to implement, subsequent MLM tasks basically all use dynamic masking.

#### (2) Optimization 2: Larger-Scale Pre-training Data and More Pre-training Steps

RoBERTa uses a larger amount of unsupervised corpus for pre-training. In addition to BookCorpus and English Wikipedia used by BERT, it also uses CC-NEWS (the English part of the news field of CommonCrawl Dataset), OPENWEBTEXT (English webpage), and STORIES (CommonCrawl Dataset Story Style Subset), a total of 160GB of data, ten times more than BERT.

At the same time, RoBERTa believes that a larger batch size can not only improve optimization speed but also improve end-task performance. Therefore, when the experiment trained 31K Step under 8K batch size (compared to BERT's batch size is 256), that is, with the same total of 3.3B training tokens as BERT, the model performs better, thus proving the significance of large batch size. On this basis, RoBERTa trained a total of 500K Steps (about 66 Epochs). At the same time, RoBERTa no longer uses the strategy of BERT to perform most of the training on length 256 and then complete training on length 512, but to do all training on length 512.

Of course, larger pretraining data, longer sequence lengths and more training Epoch require more computing resources in the pretraining stage. To train RoBERTa, Meta used 1024 V100 GPUs (32GB of GPU memory each) for one day.

#### (3) Optimization 3: A Larger BPE Vocabulary

RoBERTa, BERT and Transformer all use BPE as the coding strategy for Tokenizer. BPE, that is, Byte Pair Encoding, uses subword pairs as the unit of tokenization. For example, the sentence "Hello World" may be divided into four subword pairs: "Hel, lo, Wor, ld". Chinese, whose basic unit is the character, is generally split according to its byte encoding. For example, in UTF-8 encoding, "我" ("I") is encoded as "E68891", so BPE may split it into two subword pairs, "E68" and "891".

Generally speaking, the larger the BPE-encoded dictionary, the better the encoding effect. Of course, since the Embedding layer maps tokens from the dictionary space to hidden space (that is, the shape of Embedding is (vocab_size, hidden_size), the larger the vocab will also increase the model parameters.

BERT's original BPE vocabulary size was 30K, and RoBERTa selected a 50K vocabulary to optimize the coding capabilities of the model.

Through the optimization of the above three parts, RoBERTa successfully refreshed the SOTA of multiple downstream tasks based on the BERT architecture, and for a time became the most popular of the BERT-family pre-trained models. At the same time, RoBERTa's success also proves the importance of larger pre-training data and more pre-training steps, which is also one of the basis for the birth of LLM.

### 3.1.3 ALBERT

Based on BERT, RoBERTa further explores the role of larger-scale pre-training. The ALBERT model, which is also optimized based on the BERT architecture, is explored from the perspective of whether it can reduce the model parameters and maintain the model's ability. By optimizing the model structure and improving the NSP pre-training task, ALBERT successfully achieved capabilities beyond BERT with smaller scale parameters. Although some of the improvement ideas proposed by ALBERT have not been widely adopted in subsequent research, its method of reducing model parameters and the proposed new pre-training task SOP still provide important reference significance for the NLP field.

#### (1) Optimization 1: Decompose the Embedding parameters

Pre-trained models such as BERT have a much greater number of parameters than traditional neural networks. As mentioned earlier, BERT-large has a 24-layer Encoder Layer, 1024 hidden layer dimension, with a total parameter volume of 340M. Among them, the parameter matrix dimension of the Embedding layer is $V*H$, where V is vocabulary size 30K, and H is the hidden layer size 768, which means that the Embedding layer parameters reach 23M. Such a setting will also bring a bigger problem. When Google explores and tries to build a wider (that is, a larger hidden layer dimension) model, it is found that the increase in the hidden layer dimension will bring about a huge increase in the Embedding layer parameters. If the hidden layer dimension is increased to 2048, the Embedding layer parameters will expand to 61M, which undoubtedly greatly increases the model's calculation overhead.

From another perspective, the vector output by the Embedding layer is our dense vector representation of text tokens. Judging from the successful experience of Word2Vec, this word vector does not require a large dimension. Word2Vec achieved good results by using only 100 dimensions. Therefore, the output of the Embedding layer may not need to be consistent with the hidden layer size.

Therefore, ALBERT decomposes the parameter matrix of the Embedding layer, unbinding the output dimension of the Embedding layer and the hidden layer dimension, that is, adding a linear matrix to the rear of the Embedding layer for dimension transformation. ALBERT sets the output of the Embedding layer to 128, so a linear matrix of $128*768$ is added behind the Embedding layer to project the Embedding output back up to the hidden layer size. In other words, the parameters of the Embedding layer are reduced from $V*H$ to $V*E + E*H$. When the size of E is much smaller than H, the optimization of the Embedding layer parameters by this method will be obvious.

#### (2) Optimization 2: Parameter sharing across layers

By analyzing the parameters of BERT, ALBERT found that the parameters of each Encoder layer were highly consistent. Since the 24 Encoder layers bring huge model parameters, ALBERT proposes that each Encoder layer can share model parameters to reduce the number of parameters of the model.

In terms of specific implementation, ALBERT only initializes an Encoder layer. During the calculation process, 24 calculations will still be performed, but each calculation passes through this Encoder layer. Therefore, although it is a model calculated by 24 Encoders, there is only one layer of Encoder parameters, which greatly reduces the number of model parameters. In this case, the hidden layer dimension can be greatly expanded to achieve a wider model with smaller parameters. ALBERT's experiments show that, compared with the 334M BERT, it is also a 24-layer Encoder, but the ALBERT (xlarge version) with the hidden layer dimension set to 2048 has only 59M parameters, but it is better than BERT in terms of specific effects.

However, although the above optimization greatly reduces the amount of model parameters and also improves the model effect, there are also obvious shortcomings. Although the number of parameters of ALBERT is much smaller than BERT, the training efficiency is only slightly better than BERT, because in the model settings, although each layer shares weight, the calculation still needs to be calculated 24 times, which means that the speed of training and inference will be slower than BERT. This is also an important reason why ALBERT failed to replace BERT in the end.

#### (3) Optimization 3: Propose SOP pre-training task

Similar to RoBERTa, ALBERT also believes that NSP tasks are too simple and cannot have a significant impact on the improvement of model effectiveness in pre-training. However, unlike RoBERTa, which directly removes NSP, ALBERT chooses to improve NSP and increase its difficulty to optimize the pre-training of the model.

In traditional NSP tasks, positive examples are sentence pairs composed of two consecutive sentences, while negative examples are sentence pairs extracted from any two documents. The model can easily judge positive and negative examples and cannot learn deep semantics well. The improvement proposed by the SOP task is that the positive examples are also composed of two consecutive sentences, but the negative examples are to reverse the order of these two. In other words, the model not only needs to fit the relationship between two sentences, but also learn its sequential relationship, which greatly increases the difficulty of pre-training. For example, compared to the example of an NSP task we proposed above, an example of a SOP task is as follows:

    Input:
        Sentence A: I love you.
        Sentence B: Because you are wonderful.
    Output:
        1 (positive sample)

    Input:
        Sentence A: Because you are wonderful.
        Sentence B: I love you.
    Output:
        0 (negative sample)

ALBERT's experiments show that the SOP pre-training task significantly improves model performance. Models pre-trained with MLM + SOP outperform those pre-trained with MLM only, which in turn outperform those pre-trained with MLM + NSP.

Through the above three optimizations, ALBERT successfully achieved stronger performance with smaller parameters. Although the reduction in training and inference efficiency brought by its architecture limits the further development of the model, the idea of creating a wider model still provides reference value for many more powerful models.

As the king of NLP in the pre-training era, BERT and the BERT-family models play an extremely important role in multiple NLP tasks. In addition to RoBERTa and ALBERT mentioned above, there are many rising stars that optimize BERT from other angles, including ERNIE, which further improves the pre-training tasks; DistilBERT, a small model distilled from BERT; and XLM, which targets multilingual tasks. This article will not go into details one by one. The Encoder-Only architecture represented by BERT is not the only variant of Transformer. Next, we will introduce another mainstream architecture of Transformer, which is more similar to the original Transformer and represented by T5.

## 3.2 Encoder-Decoder PLM

In the previous section, we learned the model of Encoder-Only structure, which mainly introduced BERT's model architecture, pre-training tasks and downstream tasks fine-tuning. BERT is an Encoder-Only model based on Transformer. It learns bidirectional semantic relationships of text by pre-training tasks MLM and NSP, thus achieving excellent performance in downstream tasks. However, there are also problems with BERT, such as inconsistency between fine-tuning of MLM tasks and downstream tasks, and the inability to handle inputs that exceed the training length of the model. To solve these problems, researchers proposed the Encoder-Decoder model, which solved these problems by introducing the Decoder part, and also brought new ideas and methods to the NLP field.

In this section, we will learn the model of Encoder-Decoder structure, mainly introducing the model architecture and pre-training tasks of T5, as well as the NLP unified idea proposed by the T5 model for the first time.

### 3.2.1 T5 

T5 (Text-To-Text Transfer Transformer) is a pre-trained language model proposed by Google. It greatly simplifies model design and task processing by uniformly representing all NLP tasks as text-to-text conversion problems. T5 is based on the Transformer architecture, including two parts: encoder and decoder. It uses self-attention mechanism and multi-head attention to capture global dependencies, uses relative position encoding to process position information in long sequences, and includes feedforward neural networks in each layer to further process features.

T5's unified approach represents different NLP tasks, such as text classification, question answering and translation, uniformly as the conversion from input text to output text. This method simplifies the model design, parameter sharing and training process, and improves the generalization ability and efficiency of the model. Through this unified processing method, T5 not only reduces task-specific model debugging work, but also uses the same data processing and training framework, greatly improving the performance and application convenience of multi-task learning. Next, we will introduce the T5 model from three aspects: model structure, pre-training tasks and the unified text-to-text idea.

#### (1) Model structure: Encoder-Decoder

BERT adopts the Encoder-Only structure, which only contains the encoder part; while GPT adopts the Decoder-Only structure, which only contains the decoder part. T5 adopts the Encoder-Decoder structure, where both the encoder and decoder are designed based on the Transformer architecture. The encoder is used to process the input text, and the decoder is used to generate the output text. Information interaction between the encoder and the decoder is carried out through an attention mechanism, thereby realizing the conversion of input text to output text. Its main structure is shown in Figure 3.7:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/2-1.png" alt="Image Description" width="100%"/>
  <p>Figure 3.7 Detailed structure of T5 model</p>
</div>

As shown in Figure 3.8, the overall model structure of T5 includes the Tokenizer part and the Transformer part. The Tokenizer part is mainly responsible for converting the input text into an input format acceptable to the model, including word segmentation, encoding and other operations. The Transformer part is divided into two parts: EncoderLayers and DecoderLayers. They are composed of small blocks, each block containing a multi-head attention mechanism, a feedforward neural network and a Norm layer. Block design can make the model more flexible, like Lego, adjusting the number and layers of the Block according to the complexity of the task and the size of the dataset.

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/2-2.png" alt="Image Description" width="70%"/>
  <p>Figure 3.8 Overall structure of T5 model</p>
</div>

The Encoder and Decoder parts of the T5 model are designed based on the Transformer architecture, mainly including two structures: Self-Attention and feedforward neural network. Self-Attention is used to capture global dependencies in the input sequence, and feedforward neural networks are used to process nonlinear transformations of features.

Unlike Encoder, the Encoder-Decoder Attention structure is also included in the Decoder, which is used to capture the dependencies between input and output sequences. The two Attention structures are almost exactly the same, only in position coding and Mask mechanisms. As shown in Figure 3.9, the structure of Encoder and Decoder is as follows:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/2-3.png" alt="alt text" width="50%">
    <p>Figure 3.9 Encoder and Decoder</p>
</div>

The Self-Attention mechanism of T5 is the same as the Attention mechanism of BERT, and is designed based on the Self-Attention mechanism. The Self-Attention mechanism is a global dependency modeling method that captures global dependencies in the input sequence by calculating the similarity between Query, Key, and Value. Encoder-Decoder Attention is only different in position encoding and Mask mechanisms, mainly to distinguish input and output sequences. As shown in Figure 3.10, the Self-Attention structure is as follows:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/2-4.png" alt="alt text" width="50%">
    </p>Figure 3.10 Self-Attention Structure</p>
</div>

Unlike the original Transformer model, the LayerNorm of the T5 model uses RMSNorm, which normalizes the activation value of each hidden layer by calculating the root mean square of each neuron. The parameter setting of RMSNorm is simpler than Layer Normalization, with only one tunable parameter that can better adapt to different tasks and datasets. The RMSNorm function can be expressed by the following mathematical formula:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \cdot \gamma
$$

where:
- $x_i$ is the $i$-th element of the input vector
- $\gamma$ is a learnable scaling parameter
- $n$ is the number of dimensions of the input vector
- $\epsilon$ is a small constant for numerical stability (to avoid dividing by zero)

This normalization helps stabilize the learning process by ensuring that the scale of the weights does not become too large or too small, which is particularly useful in deep learning models with many layers.

#### (2) Pre-training tasks

The pre-training task of the T5 model is a key component, which enables the model to learn rich language representations, and language representation capabilities can be migrated to various downstream tasks during subsequent fine-tuning. The data set used for training is a large-scale text data set that contains a variety of text data, such as Wikipedia, news, books, etc. After careful processing of the data, a 750GB dataset C4 was generated for training, and it has been open sourced in TensorflowData.

We can briefly summarize the pre-training tasks of T5, which mainly include the following parts:

- Pre-training task: The pre-training task of the T5 model is MLM, also known as the BERT-style target. Specifically, it is to randomly mask 15% of tokens in the input text, and then let the model predict these masked tokens. This process does not require labels and can be performed on a large amount of unlabeled text.
- Input format: During pre-training, T5 converts the input text to "text-to-text" format. For a given text sequence, randomly select some tokens for occlusion and replace them with special placeholders (tokens). The obscured token sequence is then used as the output target of the model.
- Pre-trained dataset: T5 uses the large-scale dataset it created, the "Colossal Clean Crawled Corpus" (C4), which extracts a large amount of clean English text from Common Crawl. The C4 dataset has been cleaned to remove meaningless text, duplicate text, etc.
- Multitasking Pre-training: T5 also tried mixing multiple tasks together for pre-training, not just individual MLM tasks. This helps the model learn more general language representations.
- Pre-training to fine-tuning conversion: After pre-training is completed, the T5 model will be fine-tuned on downstream tasks. When fine-tuning, the model is trained on a task-specific dataset and adjusts the decoding strategy according to the task.

Through large-scale pre-training, the T5 model can learn rich language knowledge and obtain strong language representation capabilities, achieving excellent performance on multiple NLP tasks. Pre-training is one of the key factors in T5's success.

#### (3) The Unified Text-to-Text Idea

A core concept of the T5 model is the "unified idea", that is, all NLP tasks can be unified into text-to-text tasks, which has a profound impact on the field of natural language processing. Its design philosophy is to convert all different types of NLP tasks (such as text classification, translation, text generation, Q&A, etc.) into a unified format: both input and output are plain text.

For example:
- For text classification tasks, the input can be "classify: This is a good product" and the output is "positive";
- For translation tasks, the input can be "translate English to French: How are you?" and the output can be "Comment ça va?".

T5 is pre-trained through large-scale text data and then fine-tuned on specific tasks. This process is similar to models such as BERT and GPT, but T5 unifies tasks in the pre-training and fine-tuning stage into text-to-text form, making it more adaptable to various tasks.

We can understand the unified idea of T5 more intuitively through Figure 3.11:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/2-0.png" alt="alt text" width="90%">
    <p>Figure 3.11 T5's unified text-to-text idea</p>
</div>

For different NLP tasks, a task description prefix will be added before each input, which clearly specifies the type of the current task. This not only helps the model learn common features between different tasks during the pre-training stage, but also facilitates rapid adaptation to specific tasks during the fine-tuning stage. For example, the task prefix can be "summarize:" for summary tasks, or "translate English to German:" for translation tasks.

T5's unified idea simplifies the task processing process and enhances the universality and adaptability of the model by unifying all NLP tasks into text-to-text form. This idea not only promotes the development of natural language processing technology, but also provides more convenient and efficient solutions for practical applications.

## 3.3 Decoder-Only PLM

In the first two sections, we explain two model architectures developed by Transformer - the Encoder-Only model represented by BERT and the Encoder-Decoder model represented by T5. Then, it is natural to imagine that in addition to the above two architectures, there can be a model architecture - Decoder-Only, that is, a model made of only Decoder stacking.

In fact, Decoder-Only is the underlying architecture of today's popular LLMs. Currently, all LLMs are basically Decoder-Only models (except non-Transformer architectures such as RWKV and Mamba). ChatGPT, which triggered the LLM craze, is the culmination of the GPT series, the representative models of the Decoder-Only family. The LLaMA model, which is currently the basic architecture of open source LLM, is also optimized and developed based on the model architecture of GPT. Therefore, in this section, we will not only analyze the principles, architecture and characteristics of the Decoder-Only representative model GPT in detail, but also go deep into the current mainstream open source LLM to analyze their structure and characteristics, and combine the previous analysis of other models of the Transformer series to help everyone understand in-depth how LLM, which is currently highly anticipated and considered to be the only way for AGI, developed from traditional PLM step by step.

First, let’s learn a representative model that opens the door to the LLM world – GPT released by OpenAI.

### 3.3.1 GPT

GPT, or Generative Pre-Training Language Model, is a pre-trained language model released by the OpenAI team in 2018. Although the academic community generally recognizes BERT as a representative of the pre-training language model era, the model that first clearly proposes the pre-training-fine-tuning idea is actually GPT. GPT proposes the concept of general pre-training, that is, pre-training on a massive unsupervised corpus, and then fine-tuning on each specific task, thereby achieving the huge benefits of these tasks. Although at release its performance was slightly inferior to BERT, which came out shortly after, so it neither made a splash nor made its Decoder-Only architecture the mainstream of academic research, the OpenAI team firmly chose to continuously expand pre-training data, increase model parameters, and continuously optimize the GPT architecture. Finally, GPT-3, released in 2020, laid the foundation of the LLM era, and ChatGPT, which uses GPT-3 as its base model, successfully opened the door to the new era, making OpenAI the strongest competitor and so far the biggest winner of the LLM era.

This section will take GPT as an example to deeply analyze GPT and its representative Decoder-Only model from three aspects: model architecture, pre-training tasks, and the development history of GPT series models, and further introduce the current mainstream LLM architecture - LLaMA.

#### (1) Model architecture——Decoder Only

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/3-0.png" alt="alt text" width="100%">
    <p>Figure 3.12 GPT model structure</p>
</div>

As can be seen in Figure 3.12, the overall structure of GPT is somewhat similar to BERT, but compared with BERT's Encoder, Decoder is chosen to stack model structures. Since the Decoder-Only structure is also naturally suitable for text generation tasks, compared with BERT that is more in line with NLU task design, the model design of GPT and T5 is more in line with NLG tasks and Seq2Seq tasks. Similarly, for input to a natural language text, word segmentation is first performed through tokenizer and converted into input_ids corresponding to the dictionary serial number.

The input_ids input is first passed through the Embedding layer, and then through Positional Embedding for positional encoding. Unlike BERT that chooses a trainable fully connected layer as position encoding, GPT continues to use Transformer's classic Sinusoidal position encoding, that is, absolute position encoding is performed through trigonometric functions. I will not go into details here. Interested readers can refer to the analysis of the details of the Transformer model in Chapter 2.

After encoding the Embedding layer and Positional Embedding layer into hidden_states, you can enter the decoder (Decoder). The first generation GPT model is similar to the original Transformer model. The 12-layer decoder layer is selected. However, inside the decoder layer, compared with the dual attention layer design of the original Transformer Decoder layer, the Decoder layer of GPT is more like the Encoder layer. Since there is no longer an encoded input from the Encoder, the Decoder layer only retains one masked attention layer, and the LayerNorm layer is moved from after the attention layer (as in the Transformer) to before it. After hidden_states enters the Decoder layer, LayerNorm is applied first, then masked attention is computed, and then, after a residual connection and another LayerNorm, it enters the MLP to produce the final output.

Since there is no Encoder encoding result, the mask attention in the Decoder layer is also a self-attention calculation. That is, for hidden_states of an input, query, key and value will be generated through three parameter matrices, instead of being output as key and value by Encoder like Decoder in Transformer. The subsequent attention calculation process is similar to BERT. except that after the attention weights are calculated, the mask matrix is used to block the attention weight of the future token, thus limiting each token to only focus on the attention of the previous token to realize the calculation of mask self-attention.

Another structural difference is that the MLP layer of GPT does not select a linear matrix for feature extraction, but instead selects two one-dimensional convolution kernels for extraction. However, in terms of effect, there is not much difference between the two. The hidden_states after N Decoder layers are finally mapped to the vocabulary dimension through a linear matrix, and can be converted into natural language tokens, thereby generating our target sequence.

#### (2) Pre-training task-CLM

The model structure of Decoder-Only is often more suitable for text generation tasks. Therefore, the Decoder-Only model often chooses the most traditional and direct pre-training task - the causal language model (Causal Language Model, CLM).

CLM can be seen as a direct extension of the N-gram language model. The N-gram language model predicts the next token based on the first N tokens, while the CLM predicts the next token based on all the previous tokens of a natural language sequence. By constantly repeating this process, the target text sequence is generated. That is, CLM is a classic completion task. For example, the input and output of the CLM may be:

    input: The weather today
    output: The weather today is

    input: The weather today is
    output: The weather today is good

Therefore, for a task with a length of 256 input target sequence and expecting an output sequence length of 256, the model will continuously calculate 256 times based on the first 256 tokens, 257 tokens (the input + the first predicted token)..., and finally generate an output text with a sequence length of 512. The first 256 tokens of this output text are input, and the last 256 tokens are the model output we expect.

As we said earlier, the reason why BERT can use the pre-training + fine-tuning paradigm to achieve major breakthroughs is precisely because the MLM and NSP selected can be directly trained on a large number of unsupervised corpus - it is obvious that CLM is a more direct pre-training task. It is inherently consistent with human habits of writing natural language texts and is directly matched with downstream tasks. Compared with MLM tasks, it can be directly applied on any natural language text. Therefore, CLM can also use a large number of natural language corpus for large-scale pre-training.

#### (3) Development of GPT series models

Since the launch of GPT-1, OpenAI has always believed in the model structure of Decoder-Only and the optimization idea of "size is justice", constantly expanding the pre-training data set and model volume, and making some small optimizations and corrections to the model to continuously explore more powerful pre-training models. From GPT-1 suppressed by BERT, to GPT-2 that did not attract enough attention, to GPT-3 that sparked emergent abilities and brought about the LLM era, and finally brought about the cross-age ChatGPT, OpenAI has proved the correctness of its thinking through decades of efforts.

The following table summarizes the changes in model structure and pre-trained corpus size from GPT-1 to GPT-3:

Model | Decoder Layer | Hidden_size | Attention head count | Attention dimension | Total parameter quantity | Pre-training corpus
---- | --------------|------------|------------|----------|----------|----------
GPT-1|12|3072|12|768|0.12B|5GB
GPT-2|48|6400|25|1600|1.5B|40GB
GPT-3|96|49152|96|12288|175B|570GB

GPT-1 is the pioneering work of the GPT series and the first pre-trained model to use Decoder-Only. However, GPT-1 has less model size and pre-trained data. It follows the traditional Transformer model structure and uses 12-layer Decoder Block and 768 hidden layer dimensions. The model parameters are only 117 million (0.12B), which are pre-trained on the BooksCorpus dataset with a size of 5GB. It can be seen that GPT-1's parameter count and pre-training scale are roughly comparable to BERT-base, but its performance is worse than BERT-base, which is also the reason why the GPT series models failed to become representatives of the pre-trained language model era.

GPT-2 is the product of OpenAI's further exploration of the multi-task learning ability of pre-trained language models based on GPT-1. The model structure of GPT-2 is roughly the same as that of GPT-1, but it expands the scale of model parameters and changes Post-Norm to Pre-Norm (that is, first perform LayerNorm calculations, and then enters the attention layer calculation). The core reason for these changes is that due to the increase in the number of layers and the increase in volume of the model, the risk of gradient disappearing and explosion continues to increase. In order to make the model gradient more stable, the above structure was optimized.

The core improvement of GPT-2 is the substantial increase in the pre-trained dataset and model volume. The number of Decoder Block layers of GPT-2 has reached 48 (note that GPT-2 has released four specifications of models. Here we only refer to the largest specifications of GPT-2 model), the hidden layer dimension has reached 1,600, the overall parameter volume of the model has reached 1.5 billion (1.5B). It uses the 40GB size WebText dataset that it grabs for pre-training. Whether it is the model structure or the pre-training size, it is more than an order of magnitude larger than GPT-1.

Another major breakthrough in GPT-2 is to use zero-shot learning as its main goal, that is, not to fine-tune the model and directly require the model to solve the task. For example, in the traditional pre-training-fine-tuning paradigm, we want to solve a problem, which generally requires collecting hundreds or thousands of training samples, and fine-tuning the pre-trained language model on these training samples to achieve the solution to the problem. Zero-shot emphasizes that no training samples are used, and the problem is solved directly by describing the problem to the pre-trained language model. The zero-shot idea is naturally a step further than, and more efficient than, the pre-training-fine-tuning paradigm. However, in the era of GPT-2, model capabilities did not support the better zero-shot effect. In the LLM era, zero-shot and its extension, few-shot (learning from a few examples), began to gradually become mainstream.

GPT-3 further demonstrates the core idea of OpenAI's "brute force works wonders" (scale conquers all) and is also the pioneering work of LLM. Based on GPT-2, OpenAI further increased the model size and pre-trained data volume, with the overall parameter volume reaching 175B, making it a well-deserved "large language model". There is basically no major change in the model structure, except that, because of the huge model size, a sparse attention mechanism replaces the traditional attention mechanism. In terms of pre-training data, it is centrally sampled from large corpuses such as CC, WebText, and Wikipedia, and a total of 45TB of data was sampled, leaving 570GB after cleaning. According to the calculation, GPT-3 needs to be trained on a distributed training cluster of 1024 A100s (80GB of GPU memory each) for 1 month.

The reason why GPT-3 is the pioneering work of LLM is that besides the emergent abilities brought about by its huge size, it also put forward the important idea of few-shot. Few-shot is an improvement on zero-shot. Researchers found that even for the 175B size GPT-3, it is still a difficult task to achieve better performance on zero-shot. And few-shot is a compromise on zero-shot, aiming to provide a model with few examples to teach it to accomplish the task. few-shot generally adds 3 to 5 examples to prompt (that is, the input of the model) to help the model understand. For example, for sentiment classification tasks:

    zero-shot: Please tell me whether the sentiment of ‘this is really an excellent opportunity’ is positive or negative. If it is positive, output 1; otherwise output 0

    few-shot: Please judge whether the sentiment of ‘this is really an excellent opportunity’ is positive or negative. If it is positive, output 1; otherwise output 0. You can judge by referring to the following example: ‘You performed very well’ – 1; ‘Too bad’ – 0; ‘What a good idea’ – 1.

By providing a small number of examples to the model, the model can achieve good performance that is much better than zero-shot. few-shot is also called In-context Learning, which is a solution to the problem that allows the model to learn from the examples in the provided context. The powerful capabilities of GPT-3 on few-shot have brought important progress to NLP's breakthroughs. If most tasks can be solved by artificially constructing 3 to 5 examples, the efficiency will be much higher than the traditional pre-training-fine-tuning paradigm, which means that wider real-world application of NLP becomes possible - and this is the core advantage of LLM.

Based on the GPT series of models, OpenAI released the cross-age ChatGPT, triggering the LLM boom by introducing three-stage training: pre-training, instruction fine-tuning, and reinforcement learning from human feedback (RLHF). It is also based on GPT-3 and ChatGPT that the release of models such as LLaMA and ChatGLM further reveals the endless potential of LLM. In the next section, we will conduct an in-depth analysis of the current universal architecture of LLM - LLaMA.

### 3.3.2 LLaMA

The LLaMA model is a series of large pre-trained language models developed by Meta (formerly Facebook). From LLaMA-1 to LLaMA-3, the LLaMA series models demonstrate the evolution of large-scale pre-trained language models and their significant potential in practical applications.

#### (1) Model architecture——Decoder Only

Like the GPT series models, the LLaMA model is also a pre-trained language model based on the Decoder-Only architecture. The overall structure of the LLaMA model is similar to that of the GPT series models, except that it differs in model size and pre-trained dataset. As shown in Figure 3.13 is a schematic diagram of the architecture of the LLaMA model:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/3-1.png" alt="alt text" width="100%">
    <p>Figure 3.13 LLaMA-3 model structure</p>
</div>

Similar to GPT, the processing flow of the LLaMA model also begins with encoding the input text through a tokenizer and converting it into a series of input_ids. These input_ids are data formats that the model can understand and process. Next, these input_ids will be converted by the embedding layer, where each input_id will be mapped to a vector in a high-dimensional space, namely the word vector. At the same time, the positional information of the input text will also be encoded through the positional embedding layer to ensure that the model can understand the word order context information.

In this way, input_ids combines the embedding layer and the positional embedding layer to form hidden_states. hidden_states contains the semantics and position information of the input text, which is the basis for the model to perform subsequent processing. hidden_states is then input to the decoder layer of the model.

In the decoder layer, hidden_states undergoes a series of processes that consist of multiple decoder blocks. Each decoder block is a core component of the model, and they are responsible for in-depth analysis and transformation of hidden_states. Inside each decoder block, first there is a masked self-attention layer. In this layer, the model will calculate the three vectors: query, key and value respectively. These vectors are obtained by hidden_states linear transformation, which are the basis for calculating attention weights. Then use the softmax function to calculate the attention score, which reflects the correlation intensity between different positions. Through attention score, the model can determine how much attention should be given to hidden_states in different positions when generating the current word. Then, the model multiplies the value vector with the attention score to obtain the weighted value, which is the result of the attention.

After completing the masked self-attention layer, hidden_states will enter the MLP layer. In this multi-layer perceptron layer, the model performs further feature extraction of hidden_states through two fully connected layers. The first fully connected layer maps hidden_states to an intermediate dimension, and then performs nonlinear transformations through the activation function to increase the nonlinear capability of the model. The second fully connected layer maps the features back to the original hidden_states dimension again.

Finally, after multiple decoder blocks, hidden_states will be final mapped through a linear layer, and the output dimension of this linear layer is the same as the vocabulary dimension. In this way, the model can generate the probability distribution of the target sequence based on hidden_states, and then generate the final output sequence through sampling or greedy decoding. This process reflects the powerful sequence generation ability of the LLaMA model.

#### (2) The development history of the LLaMA model

**LLaMA-1 Series**:

- Meta released LLaMA-1 in February 2023, in four sizes: 7B, 13B, 30B and 65B parameters.
- These models were pre-trained on corpus over 1T tokens, with the largest 65B parameter model being trained on 2,048 A100 80G GPUs for nearly 21 days.
- LLaMA-1 quickly became one of the most popular large models in the open source community due to its open source nature and excellent performance.

**LLaMA-2 Series**:

- In July 2023, Meta released LLaMA-2, in four sizes: 7B, 13B, 34B and 70B parameters. Except for the 34B model, all others have been open sourced.
- LLaMA-2 expands the pre-trained corpus to 2T tokens and doubles the context length of the model from 2,048 to 4,096.
- Techniques such as Grouped-Query Attention (GQA) have been introduced.

**LLaMA-3 Series**:

- In April 2024, Meta released LLaMA-3, including two parameter versions of 8B and 70B, and also revealed that the 400B LLaMA-3 is still under training.
- LLaMA-3 supports 8K long text and adopts a tokenizer with higher encoding efficiency with a vocabulary size of 128K.
- More than 15T tokens of pre-trained corpus were used, more than 7 times that of LLaMA-2.

The LLaMA model is known for its technological innovation, multiple model sizes, large-scale pre-training and efficient architectural design. The model supports parameter volumes ranging from 700 million to tens of billions, adapting to application needs of different scales. LLaMA-1 is quickly welcomed by the community for its open source and excellent performance, while LLaMA-2 and LLaMA-3 further significantly improve model performance and application scope by introducing grouped-query attention and supporting longer text input. In particular, LLaMA-3 has achieved significant progress in multilingual and multitasking by using an efficient tokenizer with a 128K vocabulary and a huge 15T-token training dataset. Meta's continued focus on model safety and community support indicates that LLaMA will continue to serve as an important driving force for the development of AI technology and promote technology application and innovation around the world.

### 3.3.3 GLM

The GLM series model is one of the mainstream Chinese LLMs developed by Zhipu, including ChatGLM1, 2, 3 and GLM-4 series models. It covers various application scenarios such as instruction comprehension and code generation. It has achieved SOTA performance on a variety of Chinese evaluation sets.

ChatGLM-6B is the pioneering work of the GLM series, it is also the earliest open source Chinese LLM in China in 2023, and it is also the first LLM to propose a unique model architecture different from GPT and LLaMA. In the entire development process of Chinese LLM, GLM has unique and significant technical significance. This section will briefly describe the development of the GLM series and introduce its unique technical ideas that are different from the GPT and LLaMA series models.

#### (1) Model Architecture: Slight Modifications Relative to GPT

GLM was originally a general language model base released by the Department of Computer Science at Tsinghua University. Its core idea is to add MLM ideas based on traditional CLM pre-training tasks to build a unified model with good performance on both NLG and NLU tasks.

In terms of overall model structure, GLM and GPT are roughly similar, both are Decoder-Only structures, with only three subtle differences:

1. Use Post Norm instead of Pre Norm. Post Norm means that when performing residual connection calculation, the residual calculation is first completed and then LayerNorm calculation is performed; while models such as GPT and LLaMA use Pre Norm, that is, first perform LayerNorm calculation, and then perform residual calculation. Relatively speaking, because Post Norm normalizes after the residual, it has a stronger regularizing effect on the parameters, which makes the model more robust; in Pre Norm, part of the parameters is added directly to the output without being normalized, which helps prevent exploding or vanishing gradients. Therefore, for larger models, it is generally believed that Pre Norm will work better. However, the GLM paper proposes that using Post Norm can avoid numerical errors in LLM (although mainstream LLM still uses Pre Norm);

2. Use a single linear layer to achieve the prediction of the final token instead of using MLP; this structure is simpler and more robust, that is, it reduces the number of parameters of the final output and places a larger number of parameters on the model itself;

3. The activation function has been changed from ReLU to GeLUS. ReLU is a traditional activation function, and its core calculation logic is to remove propagation less than 0 and retain propagation greater than 0; the GeLUS core is to make a nonlinear mapping of forward propagation close to 0, ensuring the nonlinear output after the activation function and having a certain degree of continuity.

#### (2) Pre-training task-GLM

The core innovation of GLM mainly lies in the GLM (General Language Model) task it proposed, which is also the origin of the name of GLM. GLM is a pre-training method that combines autoencoding and autoregressive ideas. The so-called autoencoding idea is actually the task learning idea of MLM. It randomly deletes continuous tokens in the input text, requiring the model to learn deleted tokens; the so-called autoregressive idea is actually the traditional CLM task learning idea, which requires the model to reconstruct continuous tokens in order.

GLM achieves the combination of MLM and CLM ideas by optimizing an autoregressive blank filling task. The core idea is that for an input sequence, random masking will be performed similar to MLM, but the masking is not a single token like MLM, but a series of tokens are blocked each time. When learning the model, it is necessary to use the context of the masked part to predict the masked part, and within the masked part, it is necessary to complete the prediction of the masked tokens in the CLM mode. For example, the input and output might be:

    Input: I <MASK> because you <MASK>
    Output: <MASK> - love you; <MASK> - are a wonderful person

By combining MLM with CLM ideas, it not only suits generation tasks that produce output token by token, but also forces the model to learn the implicit relationships in the input text from both directions, so it also suits understanding tasks. The GLM model produced with the GLM pre-training task shows, to a certain extent, performance superior to BERT-family models of the same size:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/3-2.png" alt="alt text" width="90%">
    <p>Figure 3.14 alt text</p>
</div>

However, the advantages of the GLM pre-training task showed mainly in the pre-trained model era. After entering the LLM era, CLM has shown advantages far exceeding MLM for pre-training at very large scale and model size. By increasing the model size and expanding the pre-training scale, a generative model obtained through CLM pre-training can even surpass MLM-trained understanding models at text understanding. Therefore, the ChatGLM series of models only use the pre-training idea of GLM in the first generation of models. Starting from ChatGLM2, it still returns to traditional CLM modeling. Although the GLM pre-training task seems to be a failed attempt from the overall development path of LLM, the idea of integrating CLM with MLM through careful design, and its early release of a native Chinese open-source LLM, are still of great reference value.

#### (3) Development of GLM Family

Based on the GLM model (that is, early pre-trained models using native GLM architecture and pre-training tasks), and referring to ChatGPT's technical ideas for SFT and RLHF, Zhipu released the first Chinese open source LLM ChatGLM-6B in March 2023, becoming the starting point for many Chinese LLM researchers. ChatGLM-6B is pre-trained on 1T corpus, supporting a context length of 2K.

In June 2023, Zhipu open-sourced ChatGLM2-6B. Compared with the first generation, ChatGLM2 expands the context length to 32K, achieving a significant breakthrough in model performance through a larger pre-training scale. However, in ChatGLM2, the model architecture basically returns to the LLaMA architecture, introduces the attention mechanism of MQA, and the pre-training tasks also return to the classic CLM, giving up the failed attempt of GLM.

ChatGLM3-6B was released in October 2023. Compared with the second generation, it achieved the SOTA performance at that time in terms of semantics, mathematics, reasoning, code and knowledge. However, the official technical report shows that ChatGLM3 has not changed compared to the second generation in model architecture. The main optimization sources are more diverse training data sets, more sufficient training steps and more optimized training strategies. Another important improvement of ChatGLM3 is that it has begun to support function calling and a code interpreter. Developers can directly use the open source ChatGLM3 to implement Agent development, which has a wider application value.

In January 2024, Zhipu released the GLM-4 series, which includes several model types and supports a 128K context; evaluations show it reaches GPT-4 level on English benchmarks. However, Zhipu did not open-source GLM-4 itself, but instead open-sourced its lightweight version, the GLM-4-9B model, which is pre-trained on the multilingual corpus of 1T token, with a context length of 8K and post-trained using the same pipelines and data as GLM-4. With less training computation, it surpasses Llama-3-8B and supports the functionality of all tools in GLM-4.

Figure 3.15 shows the performance evolution of the GLM series models on the benchmark set:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/3-3.png" alt="alt text" width="90%">
    <p>Figure 3.15 alt text</p>
</div>

**References**

[1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv preprint arXiv:1810.04805.

[2] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv preprint arXiv:1907.11692.

[3] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. (2020). *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.* arXiv preprint arXiv:1909.11942.

[4] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. (2023). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* arXiv preprint arXiv:1910.10683.

[5] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* Journal of Machine Learning Research, 21(140), 1–67.

[6] Alec Radford, Karthik Narasimhan. (2018). *Improving Language Understanding by Generative Pre-Training*. Retrieved from https://api.semanticscholar.org/CorpusID:49313245

[7] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei. (2020). *Language Models are Few-Shot Learners.* arXiv preprint arXiv:2005.14165.

[8] Zhang Fan, Chen Andong's article "A long article of ten thousand words will take you to sort out the Llama open source family: from Llama-1 to Llama-3", source: https://mp.weixin.qq.com/s/5_VnzP3JmOB0D5geV5HRFg

[9] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Jingyu Sun, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao Liu, Minlie Huang, Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin Cao, Shuxun Yang, Weng Lam Tam, Wenyi Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai Zhang, Yifan An, Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, and Zihan Wang. (2024). *ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools.* arXiv preprint arXiv:2406.12793.

[10] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang and Jie Tang. (2022). *GLM: General Language Model Pretraining with Autoregressive Blank Infilling.* arXiv preprint arXiv:2103.10360.
