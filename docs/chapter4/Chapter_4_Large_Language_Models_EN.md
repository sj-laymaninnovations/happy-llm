# Chapter 4 Big Language Model

## 4.1 What is LLM

In the first three chapters, we introduce the core idea that triggers major changes in the NLP field - attention mechanism and Transformer architecture - based on the definition and main tasks of NLP. With the emergence of the Transformer architecture, the NLP field has gradually entered the pre-training-fine-tuning paradigm. Pre-trained language models based on Transformer and obtaining powerful text representation capabilities through pre-training emerge one after another, pushing various classic tasks of NLP to a new level.

As ChatGPT once again refreshes the upper limit of NLP's ability by the end of 2022, the Large Language Model (LLM) has begun to replace the traditional pre-trained Language Model (PLM) and become the mainstream direction of NLP. The new research paradigm based on LLM is also refreshing the pre-training-fine-tuning paradigm carried forward by BERT. NLP has thus ushered in another earth-shaking change. From the end of 2022 to the present, the upper limit of LLM capabilities has been constantly refreshed, and the number of general-purpose large models has increased exponentially. The concepts and applications based on LLM are also changing with each passing day, indicating the arrival of the big model era.

In Chapter 3, we analyze the classic models and training processes under the three architectures of Encoder-Only, Encoder-Decoder and Decoder-Only from the perspective of model architecture. Some of these models were milestones that were the protagonists of the era before the LLM era (such as BERT), while others were the protagonists of the LLM era and were strong competitors of Artificial General Intelligence (AGI). So, what exactly is LLM? What is the core difference between LLM and traditional PLM, and what makes researchers have such high enthusiasm and expectations for LLM?

In this chapter, we will combine the above model architecture explanation to deeply analyze the definition, characteristics and abilities of LLM, reveal the core differences between LLM and traditional deep learning models, and on this basis, demonstrate the actual three-stage training process of LLM, helping readers to conceptually sort out how LLM obtains such unique abilities, thereby providing a theoretical basis for further practicing complete LLM training.

### 4.1.1 Definition of LLM

LLM, or Large Language Model, is a language model with more parameters than traditional language models and pre-trained on larger-scale corpus.

In Chapter 1, we have introduced the concept of language model, namely, the NLP model trained by predicting the next token task. LLM uses architectures and pre-training tasks similar to traditional pre-training language models (such as Decoder-Only architecture and CLM pre-training tasks), but has larger parameters and pre-training on a larger amount of corpus, which also shows a completely different ability from traditional pre-training language models.

Generally speaking, LLM refers to language models containing tens of billions (or more) of parameters. They are often pre-trained on the number of T token corpus through multi-card distributed clusters, and have text understanding and generation capabilities far beyond traditional pre-trained models. However, with the continuous deepening of LLM research, LLMs of multiple parameter sizes have gradually become richer. The generalized LLM generally covers all large language models from **billion parameters** (such as Qwen-1.5B) to **billion parameters*** (such as Grok-314B). As long as the model shows the ability and potential that far exceeds traditional pre-trained models (such as BERT, T5) on a series of complex tasks, it can be called LLM.

It is generally believed that GPT-3 (175 billion parameters) is the beginning of LLM. Based on GPT-3, ChatGPT obtained through three-stage training of pre-training, supervised Fine-Tuning (SFT), and Reinforcement Learning with Human Feedback (RLHF) dominates the arrival of the LLM era. In less than two years since OpenAI released ChatGPT in November 2022, hundreds of LLMs with different characteristics and abilities have emerged. The following table lists some of the big models released at home and abroad from November 2022 to November 2023:

Time | Open Source LLM | Close Source LLM
-------- | -----                                         | --------
2022.11 | None | OpenAI-ChatGPT
2023.02 | Meta-LLaMA; Fudan-MOSS | None
2023.03 | Stanford-Alpaca, Vicuna; Zhipu-ChatGLM|OpenAI-GPT4; Baidu-Wenxin Yiyan; Anthropic-Claude; Google-Bard
2023.04 | Alibaba-Tongyi Qianwen; Stability AI-StableLM | SenseTime-Rising
2023.05 | Microsoft-Pi; Tll-Falcon | iFlytek-Spark Mockup; Google-PaLM2
2023.06 | Zhipu-ChatGLM2; Shanghai AI Lab-Scholar Puyu; Baichuan-BaiChuan; Hubo-TigerBot|360-intelligent brain model
2023.07 | Meta-LLaMA2 | Anthropic-Claude2; Huawei-Pangu Big Model 3
2023.08 | None | Bytes - Beanbao
2023.09 | Baichuan-BaiChuan2 | Google-Gemini; Tencent-Hunyuan Big Model
2023.11 | Zero One All Things - Yi; Magic Square - DeepSeek|xAI-Grok

At present, domestic and foreign companies and research institutes are constantly launching more powerful LLMs to explore the path to AGI.

### 4.1.2 LLM's capabilities

#### (1) Emergent Ability

The most prominent feature of distinguishing LLM from traditional PLM is that LLM has the ability to emerge. Emergency ability means that under the same model architecture and pre-training tasks, some capabilities are not obvious in small models, but are particularly prominent in large models. It can be compared to the phase change phenomenon in physics. The emergence ability is like the rapid increase in model performance as the scale increases, exceeding the random level, which is what we often call quantitative change causing qualitative change.

Specifically, emergence capability can be defined as capabilities related to certain complex tasks. But generally speaking, NLPs focus more on their common capabilities, that is, their ability to be applied to solve various NLP tasks. The emergence ability is the core of the industry and academic circles maintaining high enthusiasm and attention to LLM. That is, although there is still a big gap between LLM's current capabilities and tasks that can be solved and the general artificial intelligence that humans ultimately expect, under the influence of emergence ability, we believe that with the continuous deepening of research, the continuous emergence of high-quality data, and the emergence of more efficient model architectures and training frameworks, LLM will eventually have the capabilities that general artificial intelligence needs, thus bringing qualitative changes to human life.

#### (2) In-context Learning

Contextual learning capability was first introduced by GPT-3. Specifically, context learning refers to allowing a language model to perform tasks by understanding the context and generating corresponding output without additional training or parameter updates, in the case of providing natural language instructions or multiple task examples.

For traditional PLM, after high-cost pre-training, it is often necessary to supervise fine-tune the specified downstream tasks. Although traditional PLMs have a small size and require low computing power, for example, for BERT-like models (0.5B parameters), supervised fine-tuning generally requires more than 10G of video memory, which has a certain computing power cost. Meanwhile, supervised fine-tuning training data is more costly. In view of the different difficulty of downstream tasks, the number of training samples required often ranges from 1k to tens of k, and all require manual annotation, which is quite costly in data acquisition. LLMs with context learning often do not require high-cost additional training or fine-tuning, but can handle most tasks through a few examples or adjusting natural language instructions, thus greatly saving computing power and data costs.

Contextual learning abilities are also triggering changes in the NLP research paradigm. In the traditional PLM era, the general paradigm for solving downstream tasks of NLP is pre-training-fine-tuning, that is, selecting a suitable pre-training model and preparing supervised data for your downstream tasks to fine-tune. By using LLM with contextual learning capabilities, the general paradigm begins to transform LLM's capabilities toward Prompt Engineering, that is, adjust Prompt. For example, most NLP tasks can now achieve the effect of fine-tuning beyond traditional PLM by adjusting Prompt or providing 1 to 5 natural language examples.

#### (3) Instruction Following

By fine-tuning using multitasking data described in natural language, known as `instruction fine-tuning`, LLM has proven to perform well on unseen tasks that also formally describe using instructions. That is, an instruction fine-tuned LLM can understand and follow unseen instructions and execute tasks according to task instructions without seeing specific examples beforehand, demonstrating its powerful generalization ability.

The ability to follow instructions means we no longer need to teach the model first for everything before it can do it. We only need to mix multiple instructions during the instruction fine-tuning stage to train their generalization capabilities. LLM can handle most human instructions, that is, we can flexibly solve problems encountered by users. This is particularly evident in ChatGPT. The core reason why ChatGPT can be extremely popular is that it is no longer a theoretical model that can be used only for academic and industry research, but can also serve users from all walks of life. By entering instructions to ChatGPT, it can write essays, program programs, correct test papers, read newspapers, etc.

The ability to follow instructions allows LLM to truly combine with multiple industries, empower all aspects of human life through artificial intelligence technology, thus bringing qualitative changes to human beings. Whether it is the current popular Agent, WorkFlow, or the all-round assistant and super intelligence that may appear in the not-far future, they essentially rely on the LLM's command compliance ability.

#### (4) Step by Step Reasoning

Logical reasoning, especially complex reasoning tasks involving multiple inference steps, has always been a difficult point in NLP and an important reason why artificial intelligence is difficult to gain universal recognition. After all, if a model cannot answer the basic "chicken and rabbit in the same cage" problem, or cannot recognize logical traps in language, it is difficult for you to think it is "intelligent" rather than "intellectually retarded".

However, traditional NLP models often struggle to solve complex tasks involving multiple inference steps, such as mathematical problems. However, LLM can solve these tasks by adopting a Chain-of-Thought (CoT) reasoning strategy, using a prompt mechanism containing intermediate reasoning steps can be used to solve these tasks, thereby drawing the final answer. It is presumed that this ability may be obtained through training the code.

Step-by-step reasoning ability means that LLM can handle complex logical tasks, that is, it can solve most of the problems that require logical judgment in daily life, thus taking a solid step towards a "reliable" smart assistant.

These unique abilities are an important advantage that distinguishes LLM from traditional PLMs, and also make LLMs excellent in handling various tasks, making them a powerful tool for solving complex problems and applying them to multiple fields. It is precisely because of the emergence ability, context learning ability, instruction compliance ability and step-by-step reasoning ability that NLP researchers believe that LLM is an important way to move towards universal artificial intelligence and help human society achieve qualitative change in productivity. In fact, there are many LLM-based applications that aim to significantly increase productivity by leveraging the unique capabilities of LLM. For example, Microsoft's Copilot launched by GPT-4 is based on LLM's powerful instruction compliance and step-by-step reasoning capabilities. By providing various functions such as code completion, code prompts, and code writing, it helps programmers write programs more efficiently, conveniently and accurately, greatly improving programmers' productivity.

### 4.1.3 Features of LLM

In addition to the core capabilities of LLM discussed above, LLM also has some additional, interesting or dangerous characteristics, which are also important research directions for LLM. Here are some of them:

#### (1) Multilingual support

Multilingual and cross-language models were once an important research direction for NLP, but LLM needs to use a large number of corpuses for pre-training, and training corpuses are often multilingual in themselves. Therefore, LLM is born with multilingual and cross-language capabilities, but with the differences in training corpus and instructions fine-tuning, the abilities in different languages vary. Since high-quality English corpus still accounts for the majority, most models represented by GPT-4 have the ability to significantly surpass Chinese in English. Although they can all be processed in multiple languages, domestic models (such as Wen Xin Yi Yan, Tong Yi Qianwen, etc.) that are additionally trained and optimized for Chinese, can often show better results in the Chinese environment.

#### (2) Long text processing

Because of how long the context text can be processed, it determines to a certain extent part of the model's capabilities upper limit, and LLMs often value long text processing capabilities more than traditional PLMs. Compared with traditional PLMs with 512 tokens (such as BERT, T5 and other models with maximum context lengths are 512), LLM can be said to be a quick idea in broadening the maximum context length. Because of training on massive distributed training clusters, LLMs often support context lengths of 4k, 8k or even 32k during training. At the same time, most LLMs use Rotary Positional Encoding (RoPE) (or AliBi, which also has extrapolation capabilities) as position encoding, and have a certain length extrapolation capability, that is, it can process texts significantly longer than the training length when inference. For example, InternLM is pre-trained on a 32k-length context, but 200k-length context processing is achieved through RoPE. By continuously enhancing the ability to process long texts, LLM can often have stronger information reading and information summary capabilities, thereby solving the "century problems" such as requiring LLM to finish reading "Dream of the Red Chamber" and write a corresponding college entrance examination essay.

#### (3) Expand multimodal

LLM's powerful capabilities also bring it a powerful cross-modal performance. With the continuous improvement of LLM, it has been a successful method to use the powerful capabilities of LLM to create a model that supports text and image bimodality. By introducing Adapter layers and image encoder, and performing targeted supervised fine-tuning on the graphic and text data, the model can have good graphic and text Q&A and even generation capabilities. In the future, how to align the representation of text and images to create a more powerful multimodal model and radiate the capabilities of LLM to more modes is an important research direction.

#### (4) Lingering illusion

Hallucination refers to the manifestation of LLM fabricating false and misinformation generated based on Prompt. For example, when we ask LLM to generate an academic paper and its reference list, it often fabricates numerous papers and studies that seem "serious" but do not exist at all. The hallucination problem is an inherent flaw of LLM and is also a huge challenge in the current research and application of LLM. Especially in areas such as medicine and finance that emphasize precision and correctness, the existence of hallucinations may cause very serious consequences. At present, many studies have provided some methods to weaken hallucinations, such as restriction in Prompt, guiding generation through RAG (retrieval enhancement generation), etc., but they can only weaken hallucinations to a certain extent and cannot be completely eradicated.

In addition to the above points, LLM also has many characteristics for research, such as the three-stage LLM training process, the self-reflectiveness of LLM, etc., which we will discuss in detail in the next section. We will not list them all here.

## 4.2 How to train an LLM

In the previous section, we analyzed the definition of LLM and its unique powerful capabilities. Through larger parameters and massive training corpus, we have acquired the emergence ability far exceeding traditional pre-trained models, showing strong context learning, instruction compliance and gradual reasoning capabilities, bringing new changes in the NLP field. So, what steps can we train an LLM with emergent capabilities? What is the difference between training an LLM and training a traditional pre-trained model?

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/4-figures/2-0.jpg" alt="alt text" width="90%">
    <p>Figure 4.1 Three stages of training LLM</p>
</div>

Generally speaking, training a complete LLM requires going through three stages in Figure 1 - Pretrain, SFT, and RLHF. In this section, we will discuss in detail the three stages of training LLM, and analyze the process of each stage, its core difficulties and precautions, to help readers understand from the theoretical perspective what steps it takes to train an LLM.

### 4.2.1 Pretrain

Pretrain, or pre-training, is the most core and first step in training LLM. The pre-training of LLM is very similar to the traditional pre-trained model, and it also uses massive unsupervised text to train randomly initialized model parameters. As we can see in Chapter 3, almost all mainstream LLMs currently use Decoder-Only GPT-like architecture (LLaMA architecture), and their pre-training tasks also follow the classic pre-training task of GPT model - Causal Language Model (CLM).

Causal language modeling, that is, consistent with the original language model, is trained by giving the above-described model to predict the next token. We have discussed the process and principles of CLM in detail in Chapter 3, so I will not repeat it here. The core difference between LLM's pre-training and traditional pre-training models is the volume and resource consumption of pre-training.

By definition, the core feature of LLM is that it has a much greater number of parameters than traditional pre-trained models, and is pre-trained on a larger amount of corpus. Traditional pre-trained models such as BERT have two versions: base and large. The BERT-base model consists of 12 Encoder layers, with hidden_size of 768, and uses 12 heads as the multi-head attention layer, with an overall parameter volume of 100 million (110M); while the BERT-large model consists of 24 Encoder layers, with hidden_size of 1024, with 16 heads, with an overall parameter volume of 300 million (340M). At the same time, BERT pre-training used 3.3 billion (3B) token corpus and trained on 64 TPUs for 4 days. In fact, compared with traditional deep learning models, BERT with 300 million parameters and 3.3 billion training data is already a behemoth with superb capabilities and huge resource consumption.

However, as we mentioned earlier, LLMs generally have tens of billions or even hundreds of billions of parameters. Even the smallest LLM in a broad sense generally has more than one billion (1B). For example, taking the pioneering work GPT-3 as an example, it has 96 Decoder layers, 12288 hidden_size and 96 heads, with a total of 175 billion (175B) parameters, which is 3 orders of magnitude faster than BERT. Even the currently popular small LLMs such as Qwen-1.8B have 24 Decoder layers, hidden_size of 2048 and 16 attention heads, with an overall parameter volume of 1.8 billion (1.8B).

Model|hidden_layers|hidden_size|heads|overall parameter quantity|pretrained data quantity
----| -----------|-----------|------|---------|---------
BERT-base|12|768|12|0.1B|3B
BERT-large|24|1024|16|0.3B|3B
Qwen-1.8B|24|2048|16|1.8B|2.2T
LLaMA-7B|32|4096|32|7B|1T
GPT-3|96|12288|96|175B|300B


More importantly, LLMs often need to use larger-scale pre-trained corpus. According to the Scaling Law proposed by OpenAI: C ~ 6ND, where C is the calculation amount, N is the model parameter, and D is the number of trained tokens, it can be experimentally concluded that the number of trained tokens should be 1.7 times the model parameter, that is, for 175B GPT-3, 300B token is required for pre-training. LLaMA further proposed that using 20 times token to train the model can achieve the best results. Therefore, 175B's GPT-3 can be pre-trained with 3.5T token data to achieve optimal performance.

Such huge model parameters and pre-training data make the computing power resources required to pre-train an LLM extremely large. In fact, even if a 1B big model is pre-trained, at least a multi-card distributed GPU cluster is required to segment the model parameters, training intermediate parameters and training data through a distributed framework, so that it can be completed through long-term training in units of days. Generally speaking, a billion-level LLM requires more than one month to train 1024 A100s, while a billion-level LLM generally requires 256 A100s for two or three days to train, and the computing resource consumption is very high.

Because of this, distributed training frameworks have become an indispensable part of LLM training. The core idea of a distributed training framework is data parallelism and model parallelism. The so-called data parallelism means that the size of the training model can be contained in a single GPU memory, but because the enlargement of the training batch_size will increase the memory overhead, it is impossible to use a larger batch_size for training; at the same time, the amount of training data is very large, and the training time using a single GPU is difficult to accept.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/4-figures/2-1.jpg" alt="alt text" width="60%">
    <p>Figure 4.2 Model and data parallelism</p>
</div>

Therefore, as shown in Figure 4.2, the model instance can be allowed to run on different GPUs and different batches of data. After each forward pass is completed, the gradients of all instances are collected and the gradient updates are calculated, and the model parameters are updated and then passed to all instances. That is, in the case of data parallelism, the model parameters on each GPU are consistent, and the total batch size of training is equal to the sum of batch sizes on each card.

However, when LLM expands to tens of billions of parameters, a single GPU memory often cannot store complete model parameters. As shown in Figure 4.3, in this case, the model can be split on multiple GPUs, each of which stores different layers or different parts to achieve model parallelism.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/4-figures/2-2.jpg" alt="alt text" width="30%">
    <p>Figure 4.3 Model Parallel</p>
</div>


Based on the idea of data parallelism and model parallelism, many more efficient distributed methods have also evolved, such as tensor parallelism, 3D parallelism, ZeRO (Zero Redundancy Optimizer, zero redundancy optimizer), etc. At present, the mainstream distributed training frameworks include Deepspeed, Megatron-LM, ColossalAI, etc., among which Deepspeed is the most widely used.

The core strategies of Deepspeed are ZeRO and CPU-offload. ZeRO is a data parallel solution that optimizes video memory. Its core idea is to optimize the video memory usage of each card when data is parallelized, thereby achieving support for larger-scale models. ZeRO divides the video memory occupied by each card in the model training stage into two categories:

- Model States, including model parameters, model gradients, and state parameters of the optimizer Adam. Assuming the model parameter amount is 1M, generally speaking, in the case of mixed precision training, this part requires 16M of space for storage, where the Adam status parameter will occupy 12M of storage space.
- Residual States, video memory footprint except model state, including activation values, various caches and video memory fragments.

In response to the above video memory usage, ZeRO proposed three continuous optimization strategies:

1. ZeRO-1, slice the Adam status parameters in the model state, that is, each card only stores the Adam status parameters of $\frac{1}{N}$, and other parameters are still one copy per card.
2. ZeRO-2, continue to shard the model gradient. Each card only stores the model gradient and Adam status parameters of $\frac{1}{N}$, and only the model parameters are kept for one copy per card.
3. ZeRO-3, also shard the model parameters, and each card only stores the model gradient, model parameters, and Adam status parameters of $\frac{1}{N}$.

It can be seen that as the amount of parameters of the shard continues to increase, the video memory required for each card is also reduced. Of course, the increase in shards means an increase in communication overhead during training. Generally speaking, the GPU utilization rate of each card is the highest and the ZeRO-3 is the lowest. What strategy should be used specifically? It needs to be dynamically determined based on the computing resources situation and the model volume that needs to be trained.

In addition to the requirements of computing resources, the training data itself is also a major challenge in pre-training LLM. To train an LLM, it takes at least hundreds of pre-trained corpus of B or even T. According to research, most of the knowledge mastered by LLM is learned during pre-training. Therefore, in order for the trained LLM to cover the widest knowledge possible, the pre-training corpus needs to organize data from multiple sources and mix them in a certain proportion. Currently, the main open source pre-training corpus include CommonCrawl, C4, Github, Wikipedia, etc. Different LLMs often add some private high-quality corpus based on the open source pre-training corpus, and then construct the pre-training data set based on the best ratio obtained by their own experiments. In fact, data ratios have always been the "core secret" of pre-trained LLM, and different ratios often affect the performance trained by the final model to a considerable extent. For example, the following table shows the pre-training data and ratios for LLaMA:

Dataset | Disk size
-----|----|---------------------
CommonCrawl|67.0%|3.3 TB
C4|15.0%|783 GB
Github|4.5%|328 GB
Wikipedia|4.5%|83 GB
Books|4.5%|85 GB
ArXiv|2.5%|92 GB
StackExchange|2.0%|78 GB

Training a Chinese LLM will make the training data more difficult. At present, most high-quality corpus is concentrated in the English category. For example, Wikipedia, Arxiv, etc. in the above table are all English data sets; while in multilingual data sets such as C4, English corpus also occupies a major position. Currently, open source Chinese LLMs such as ChatGLM and Baichuan have not opened their pretrained data sets. Currently, only the open source Chinese pretrained data sets include Kunlun Tiangong's open source [SkyPile](https://huggingface.co/datasets/Skywork/SkyPile-150B) (150B), and Zhongke Wenge's open source [yayi2](https://huggingface.co/datasets/wenge-research/yayi2_pretrain_data) (100B), etc., which are significantly different from the English open source data sets.

The processing and cleaning of pre-training data is also an important part of LLM pre-training. Many studies have proved that the quality of pre-trained data is often more important than its size. Pre-training data processing generally includes the following processes:

1. Document preparation. Since massive pre-training corpus is often obtained from the Internet, it is generally necessary to obtain natural language documents from crawled websites. Document preparation mainly includes URL filtering (filtering out harmful content based on web URL), document extraction (extracting plain text from HTML), language selection (determining the language of the extracted text), etc.
2. Corpus filtering. The core purpose of corpus filtration is to remove low-quality, meaningless, toxic and harmful content, such as garbled code, advertising, etc. There are generally two methods for corpus filtering: a model-based method, that is, training a text classifier through a high-quality corpus for filtering; a heuristic method is generally filtered by manually defining the quality indicators of web content and calculating the index value of the corpus.
3. Corpus removal. Experiments show that a large number of repeated texts will significantly affect the generalization ability of the model. Therefore, deduplication of corpus means deduplication, which is also an indispensable step to delete documents with very high similarity in the training corpus. Deduplication is generally based on the hash algorithm to calculate document similarity within or across data sets, and remove documents with similarity greater than the specified threshold; it can also accurately match deduplication at the sequence level based on substrings.

At present, there are many processed high-quality pre-training corpus and frameworks dedicated to pre-training data processing. For example, there are pre-trained datasets [RedPajama-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) based on the LLaMA idea, and the [SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B/tree/main/train) datasets that are filtered and deduplicated based on RedPajama, experiments have proved that the high-quality 627B Slimpajama dataset can achieve better results than the 1T RedPajama dataset.

### 4.2.2 SFT

Pre-training is the fundamental source of LLM's powerful abilities. In fact, the massive amount of knowledge covered by LLM basically comes from pre-training corpus. The core of LLM's performance itself lies in pre-training work. However, pre-training gives LLM the ability, but it still needs a second step to motivate it. The pre-trained LLM is like a scholar who reads a lot of books but does not understand it very well. He can smoothly follow the following questions about any strange problems, but he doesn't know the meaning of the problem itself and can only "resolvely endorse". The essence of this phenomenon is that the pre-training task of LLM is the classic CLM, that is, the ability to predict the next token is trained. It cannot be adapted to other downstream tasks or user instructions without further fine-tuning.

Therefore, we also need a second step to teach this extensive book-reading student how to use it, that is, SFT-Supervisor Finetune, with supervised fine-tuning. The so-called supervised fine-tuning is actually the pre-training-fine-tuning we talked about in Chapter 3. The slight difference is that for traditional pre-training models with limited capabilities, we need to fine-tune them separately for each downstream task to train the model's performance on this task. For example, to solve the problem of text classification, it is necessary to fine-tune the text classification of BERT; to solve the problem of entity recognition, it is necessary to fine-tune the entity recognition task.

Faced with powerful LLMs, we often no longer construct supervised data for fine-tuning on designated downstream tasks, but instead choose the "general instruction compliance capability" of the training model, that is, SFT is generally performed through the `instruction fine-tuning` method.

The so-called instruction fine-tuning means that the input we train is various types of user instructions, and the output that requires model fit is the reply we want the model to make after receiving the instruction. For example, one of our training samples could be:

    input: Tell me about today's weather forecast?
    output: According to the weather forecast, today's weather is sunny to cloudy, with the highest temperature of 26 degrees Celsius and the lowest temperature of 9 degrees Celsius. The temperature difference between day and night is large, please pay attention to keeping warm

In other words, the main goal of SFT is to enable the model to obtain generalized instruction compliance capabilities from various types and styles of instructions, that is, to be able to understand and reply to user instructions. Therefore, similar to Pretrain, the data quality and data ratio of SFT are also important factors that determine the compliance of model instructions.

First of all, the instruction data volume and coverage range. In order for LLM to obtain generalized instruction compliance capabilities, that is, be able to perform well on untrained instructions, it is necessary to collect a large number of user instructions of different categories and corresponding replies to train the LLM. Generally speaking, a training sample of 500~1000 on a single task can achieve good fine-tuning effects. However, in order to enable LLM to obtain generalized instruction compliance capabilities and perform well on multiple task instructions, it is necessary to cover multiple types of task instructions in the training data set, and a relatively large amount of training data is also required. The amount of good open source LLM SFT data is generally around the number B token.

In order to improve the generalization ability of LLM, the larger the coverage of the instruction dataset, the better. However, the ratio of multiple different types of instruction data is also a major challenge in LLM training. OpenAI-trained InstructGPT (the predecessor of ChatGPT) uses ten instructions derived from users using their API:

Instruction type|proportion
-------|-----
Text Generation | 45.6%
Open Domain Q&A | 12.4%
Brainstorm | 11.2%
Chat | 8.4%
Text Translation|6.6%
Text Summary | 4.2%
Text Classification | 3.5%
Others |3.5%
Domain-specific Q&A | 2.6%
Text extraction | 1.9%

High-quality instruction datasets have high difficulty in obtaining them. Unlike the unsupervised corpus used in pre-training, the instruction dataset used by SFT is supervised corpus. In addition to extensive and reasonable design, instruction replies need to be manually marked and the high quality of the label is ensured. In fact, a large part of ChatGPT's success comes from its high-quality manual annotation data. However, manual annotation of data is extremely costly, and few companies open source manual annotation instruction data sets. To reduce data costs, some scholars have proposed methods to use ChatGPT or GPT-4 to generate instruction data sets. For example, the classic open source instruction dataset [Alpaca](https://github.com/yizhongw/self-instruct/blob/main/human_eval/user_oriented_instructions.jsonl) is built based on some seed Prompt, generating more instructions through ChatGPT and replying to the instructions.

The instruction dataset used by SFT generally includes the following three keys:

```json
{
    "instruction":"即输入的用户指令",
    "input":"执行该指令可能需要的补充输入，没有则置空",
    "output":"即模型应该给出的回复"
}
```

For example, if our command is to translate the target text "It's so good today" into English, then the sample can be constructed into the following form:

```json
{
    "instruction":"将下列文本翻译成英文：",
    "input":"今天天气真好",
    "output":"Today is a nice day！"
}
```

At the same time, in order to enable the model to learn different paradigms from pre-training, specific formats are often set in a targeted manner during the SFT process. For example, the SFT format of LLaMA is:

    ### Instruction:\n{{content}}\n\n### Response:\n

The content is a specific user instruction, that is, for each user instruction, it will be embedded in the content part above. The user instruction here not only refers to the "instruction" in the above example, but is a splicing of the instructions and input, that is, a complete instruction that the model can execute. For example, for the above example, the input obtained by LLaMA should be:

    ### Instruction:\nTranslation of the following text into English: The weather is so good today\n\n### Response:\n

The output that needs to be fitted is:

    ### Instruction:\nTranslate the following text into English: The weather is so good today\n### Response:\nToday is a nice day!

Note that because instruction fine-tuning is essentially still CLM training on the model, it only requires the model to understand and reply to the instructions rather than simply predicting the next token, the result of the model prediction is not only output, but should be input + output. However, the input part does not participate in the calculation of loss, but the reply instruction itself is still implemented in the form of predicting the next token.

However, with the continuous improvement of LLM capabilities, the model's multi-round dialogue capabilities have gradually gained attention. The so-called multi-round dialogue means that the model can refer to the historical records of the previous dialogue to make a reply during each conversation. For example, an LLM without multiple rounds of conversational capabilities may have the following conversation record:

    User: Hello, I am a member of the open source organization Datawhale.
    Model: Hello, is there anything that can help you?
    User: Do you know what Datawhale is?
    Model: Sorry, I don't know what Datawhale is.

In other words, the model cannot record historical information that users have mentioned or answered by themselves. If it is an LLM with multiple rounds of dialogue capabilities, its dialogue record should be like this:

    User: Hello, I am a member of the open source organization Datawhale.
    Model: Hello, is there anything that can help you?
    User: Do you know what Datawhale is?
    Model: Datawhale is an open source organization.

Whether the model supports multiple rounds of conversation has nothing to do with pre-training. In fact, the model's multi-round dialogue capability comes entirely from the SFT stage. If we want the model to support multiple rounds of conversations, we need to construct the training data in multiple rounds of conversation formats in SFT, so that the model can use previous knowledge to generate answers. Suppose that the multi-round conversation we need to construct at the moment is:

    <prompt_1><completion_1><prompt_2><completion_2><prompt_3><completion_3>

There are generally three ways to construct multi-round dialogue samples:

1. Directly use the last model reply as output, and all previous historical conversations as input, and directly fit the last reply:
        
        input=<prompt_1><completion_1><prompt_2><completion_2><prompt_3><completion_3>
        output=[MASK][MASK][MASK][MASK][MASK]<completion_3>

2. Construct N rounds of dialogue into N samples:

        input_1 = <prompt_1><completion_1>
        output_1 = [MASK]<completion_1>

        input_2 = <prompt_1><completion_1><prompt_2><completion_2>
        output_2 = [MASK][MASK][MASK]<completion_2>

        input_3=<prompt_1><completion_1><prompt_2><completion_2><prompt_3><completion_3>
        output_3=[MASK][MASK][MASK][MASK][MASK]<completion_3>

3. Directly ask the model to predict the output of each round of conversation:

        input=<prompt_1><completion_1><prompt_2><completion_2><prompt_3><completion_3>
        output=[MASK]<completion_1>[MASK]<completion_2>[MASK]<completion_3>

It is obvious that the first method will lose a large amount of intermediate information, the second method will cause a large amount of repeated calculations, and only the third method is the most reasonable multi-round dialogue structure. The reason why we can construct multi-round conversation samples in the third way is because LLM is essentially a CLM task, performing one-way attention calculations, so when prediction, the fit will be performed from left to right, and the output prediction of the front wheel will not affect the prediction of the rear wheel. Currently, most LLMs use multiple rounds of conversation to perform SFT.

### 4.2.3 RLHF

RLHF, the full name is Reinforcement Learning from Human Feedback, which is a key step in using reinforcement learning to train LLM. Compared with SFT, which has already begun to take shape in GPT-3, RLHF is often considered to be the most core breakthrough of ChatGPT compared to GPT-3. In fact, from a functional perspective, we can divide the training process of LLM into two stages: pre-training and alignment. The core function of pre-training is to give the model a huge amount of knowledge, and the so-called alignment is actually to make the model consistent with human values, thereby outputting what humans want to output. In this process, SFT aligns LLM with human instructions, thus having the ability to follow instructions; while RLHF aligns LLM with human values from a deeper level, enabling them to meet the core standards of safety, usefulness and harmlessness.

As shown in Figure 4.4, ChatGPT divides alignment into three stages in the technical report. The next two stages train RM and PPO training, which is the steps of RLHF:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/4-figures/2-3.png" alt="alt text" width="100%">
    <p>Figure 4.4 Three stages of ChatGPT training</p>
</div>


The idea of RLHF is to introduce reinforcement learning technology, and through real-time human feedback, LLM can give more human satisfactory responses. Reinforcement learning is another machine learning method that is different from supervised learning. The main issue is how an agent can maximize the rewards it can receive in complex and uncertain environments. Reinforcement learning mainly consists of two parts: the agent and the environment. During the reinforcement learning process, the agent constantly acts and obtains feedback from the environment, and adjusts his or her actions based on the feedback. Applying to LLM alignment is actually aimed at different issues. LLM will continuously generate corresponding replies, and the manual annotator will constantly provide feedback on LLM's replies, so that LLM can learn to reply that humans prefer and like.

RLHF is similar to LLM As a student, he constantly does his homework to improve his problem-solving ability. If LLM is regarded as a student with strong abilities, Pretrain teaches him all the basic knowledge, and SFT teaches him how to read and solve problems, then RLHF is similar to real practice. LLM will constantly answer exercises based on the basic knowledge learned by Pretrain and the problem-solving ability learned by SFT. Then, humans will use their teacher to correct LLM to make LLM reflect on the wrong problem-solving methods and continuously strengthen the correct problem-solving methods.

As shown in the figure above, RLHF is divided into two steps: training RM and PPO training.

RM, Reward Model, that is, the reward model. RM is used to fit human preferences to provide feedback to LLM. In reinforcement learning training, RM will score each reply from LLM, which reflects the degree to which the generated reply is in line with human preferences. Then LLM will perform optimization training based on RM scoring based on the principle of reinforcement learning. Therefore, RM is essentially a text classification model, with a scalar reward for outputting a text, which is very similar to the hidden layer output in the text classification task. In terms of specific implementation, RM is often the traditional LLM architecture (or BERT architecture) plus a classification layer, which is exactly the same as the LLM architecture used for text classification, except that it uses hidden layer output instead of the final classification output.

However, when training RM, we often do not directly use text and its corresponding scalar rewards to train RM. Because we want to align human preferences, the preference data for RM training is often marked manually. However, since there are often values differences between labelers, numerical scalar rewards often amplify these differences, resulting in inconsistent rewards for the same level of replies during training, making it difficult for the model to fit to the correct scalar rewards. Therefore, we tend to rank different replies under the same completion and then convert the ranking into rewards.

For example, our training data tend to be like:

```json
{
    "prompt":"如果你打算从商店偷东西，你觉得早上好还是晚上好？",
    "chosen":"这是违法的事情，我不能提供建议",
    "rejected":"考虑晚上的人口贩运和监控摄像头的差别是件好事。夜间时间可能更有利于避免监控摄像头,但晚上的商店雇员会更能看见你。另一方面,由于白天通常不太忙,因此更容易避免被其他人注意到。无论如何,必须密切注意商店雇员和所有顾客的行为。他们也许能够看见你,即使他们不是直接面对你。为了安全起见,重要的是要密切注意商店里的一切事情,而不是不小心。"
}
```

Among them, prompt is the user's question, chosen is the answer that should be aligned and meets human preferences, and rejected is the answer that does not meet human preferences. In training, the propt will be spliced with chosen and rejected respectively to form chosen_example and rejected_example, and then enter the model to output a scalar reward through forward propagation. The model then calculates the loss by maximizing the scalar differences between chosen_example and rejected_example and performs backpropagation to complete the training.

It is worth noting that the models used for RM training are often different in size from the last LLM. For example, OpenAI uses 175B LLM and 6B RM. At the same time, the model used by RM can be LM after SFT or RM trained from de novo based on preference data. Which one is better has not yet been concluded.

After completing RM training, you can use the PPO algorithm to perform reinforcement learning training. PPO, Proximal Policy Optimization, a near-end policy optimization algorithm, is a classic RL algorithm. In fact, other reinforcement learning algorithms can also be used during reinforcement learning training, but the current PPO algorithm is still the most suitable algorithm for RLHF because it is mature and has low cost.

During the specific PPO training process, there will be four models. As shown in Figure 4.5, two LLMs and two RMs. The two LLMs are the actor model that performs fine-tuning and parameter updates, and the ref model that does not perform parameter updates, both are initialized from the LLM after SFT. The two RMs are the critical model that performs parameter updates and the reward model that does not perform parameter updates. Both are initialized from the RM trained in the previous step.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/4-figures/2-4.jpg" alt="alt text" width="100%">
    <p>Figure 4.5 PPO training process</p>
</div>

As shown in the figure above, the reinforcement learning training process using the PPO algorithm is as follows:

1. Initialize the two models from the LLM after SFT as Actor Model and Ref Model respectively; initialize the two models from the trained RM as Reward Model and Critic Model respectively;
2. Enter a Prompt, and Actor Model and Ref Model generate replies to Prompt respectively;
3. Actor Response and Ref Response calculate KL divergence: $r_{KL} = -\theta_{KL}D_{KL}(\pi_{PPO}(y|x)||\pi_{base}(y|x))$ where $\pi_{PPO}(y|x)$ is the output of Actor Model, and $\pi_{base}(y|x)$ is the output of Ref Model, and $\theta_{KL}D_{KL}$ is the method of calculating KL divergence;
4. Actor Response is input to Reward Model and Critic Model for scoring, where Reward Model outputs the corresponding scalar reward for replying, and Critic Model will also output accumulative reward (i.e., the cumulative reward from the i position to the last);
5. The calculated KL divergence and the scoring of both models are input into the reward function, and the reward is calculated: $loss = -(kl_{ctl} \cdot r_{KL} + \gamma \cdot V_{t+1} - V_{t}) \log P(A_t|V_t)$ , here $kl_{ctl}$ is the weight parameter that controls the impact of KL divergence on the result, $\gamma$ is the weight parameter that controls the impact of the score on the result at the next time (that is, the sample) scoring on the result, $V_t$ is the scoring output of the Critic Model, and $A_t$ is the scoring output of the Reward Model;
6. Update the parameters of the Actor Model and the Critic Model parameters according to the reward function. Note that the parameter update methods of the Actor Model and the Critic Model are different, so I will not elaborate on them one by one here. Interested readers can conduct in-depth research on the relevant theories of reinforcement learning.

In the above process, since four models are used, the memory usage will be several times that of SFT. For example, if both RM and LLM use 7B volume, we need about 240G (4 80G A100 pieces, each card takes 60G) of video memory to load the model. So, why do we need four models? Actor Model and Critic Model are easier to understand, and the reason we still need to keep the original parameters of Ref Model and Reward Model without updating is to limit the update of the model not to deviate from the original model too much and lose the capabilities granted by Pretrain and SFT.

Of course, such a large resource occupation and complex training process make RLHF a very high-bar stage. Some scholars have also proposed DPO (Direct Preference Optimization) based on the idea of supervised learning, which can replace RLHF with low threshold. The core idea of DPO is to transform RLHF reinforcement learning problems into supervised learning to directly learn human preferences. DPO demonstrates that the constraint reward maximization problem can be optimized through single-stage strategy training by using the mapping between reward functions and optimal strategies. That is to say, by learning the optimization goals proposed by DPO, human preferences can be directly learned without the need to train RM and carry out reinforcement learning. Since training is directly used with supervised learning, DPO only needs two LLMs to complete the training, and the training process is much simpler than PPO, and it is a simpler alternative version of RLHF. Why the optimization goals proposed by DPO can directly learn human preferences? The author has completed the proof through a series of mathematical derivations. Interested readers can come down to read further. I will not go into details here.

Next, we will implement how to train an LLM from scratch in turn, including pre-training, SFT, and RLHF.

**References**

[1] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe. (2022). *Training language models to follow instructions with human feedback.* arXiv preprint arXiv:2203.02155.

[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv preprint arXiv:1810.04805.

[3] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei. (2020). *Scaling Laws for Neural Language Models.* arXiv preprint arXiv:2001.08361.

[4] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre. (2022). *Training Compute-Optimal Large Language Models.* arXiv preprint arXiv:2203.15556.

[5] Qi Wang, Yiyuan Yang, Ji Jiang. (2022). Easy RL: Reinforcement Learning Tutorial . Beijing: Posts & Telecom Press. ISBN: 9787115584700. https://github.com/datawhalechina/easy-rl

[6] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn. (2024). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* arXiv preprint arXiv:2305.18290.

[7] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen. (2025). *A Survey of Large Language Models.* arXiv preprint arXiv:2303.18223.
