# Chapter 1: Basic NLP Concept

Natural Language Processing (NLP) is an important branch of the field of artificial intelligence, aiming to enable computers to understand and process human language and realize natural communication between humans and machines. With the rapid development of information technology, text data has become an indispensable part of our daily life. The advancement of NLP technology provides us with a powerful tool for extracting useful information from massive texts and understanding the deep meaning of language. From the early rules-based methods, to the later statistical learning methods, to the widespread application of current deep learning technologies, the NLP field has undergone many technological innovations. As one of the core technologies of NLP, its research and progress have a decisive role in improving the performance of NLP systems.

Welcome to the study of the basic concepts of NLP. This chapter will introduce the basic concepts of NLP to you to help you better understand and review the relevant knowledge of NLP.

## 1.1 What is NLP

NLP is a technology that allows computers to understand, interpret and generate human language. It is an extremely active and important research direction in the field of artificial intelligence, and its core task is to simulate human cognition and use of language through computer programs. NLP combines knowledge and technologies from multiple disciplines such as computer science, artificial intelligence, linguistics and psychology, aims to break the barriers between human language and computer language and achieve seamless communication and interaction.

NLP technology enables computers to perform various complex language processing tasks, such as Chinese word segmentation, subword segmentation, part-of-speech annotation, text classification, entity recognition, relationship extraction, text summary, machine translation, automatic question and answer, etc. These tasks not only require computers to be able to recognize and process the superficial structure of language, but more importantly, they can understand the deep meaning behind language, including complex factors such as semantics, context, emotions and culture.

With the development of modern technologies such as deep learning, NLP has made significant progress. By training large amounts of data, deep learning models can learn the complex patterns and structures of the language, thus achieving near or even exceeding human-level performance on multiple NLP tasks. NLP, however, still faces many challenges, such as dealing with ambiguity, understanding abstract concepts, dealing with metaphors and irony. Researchers are working to solve these problems with more advanced algorithms, larger data sets and finer language models to drive the continuous evolution of NLP technology.

## 1.2 NLP development history

The development process of NLP is the evolution process from the early basic rules to statistical methods, and then to the current machine learning and deep learning methods. Each technological change has greatly promoted the development of NLP technology, allowing it to achieve remarkable achievements in tasks such as machine translation, sentiment analysis, entity recognition and text summary. With the continuous enhancement of computing power and the continuous optimization of algorithms, the future of NLP will be brighter and will play a more important role in more fields.

### Early Exploration (1940s - 1960s)

The early exploration of NLP began after World War II, when people realized the importance of automatically translating one language into another. In 1950, Alan Turing proposed the Turing Test.

> He said that if a machine could be part of a conversation by using a typewriter and could completely mimic humans without obvious differences, the machine could be considered to be able to think.

This is a test to determine whether a machine can exhibit intelligent behavior that is indistinguishable from humans. During this period, Nom Chomsky proposed the theory of generative grammar, which had an important influence on understanding how machine translation works. However, the machine translation system during this period was very simple, mainly relying on dictionary search and basic word order rules for translation, and the effect was not ideal.

### Symbolism and Statistical Methods (1970s - 1990s)

After the 1970s, NLP researchers began to explore new areas, including logically basic paradigms and natural language understanding. During this period, researchers were divided into two camps: symbolism (or rule basis) and statistical methods. Symbolist researchers focus on formal language and generative grammar, while statistical methods researchers focus more on statistical and probabilistic methods. In the 1980s, with the improvement of computing power and the introduction of machine learning algorithms, revolutionary changes in the field of NLP, and statistical models began to replace complex "handwriting" rules.

### Machine Learning and Deep Learning (2000s to Present)

After the 2000s, with the development of deep learning technology, the field of NLP has made significant progress. Technologies such as Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM) and attention mechanism have been widely used in NLP tasks and have achieved remarkable results. In 2013, the proposal of Word2Vec model pioneered a new era of word vector representation, providing a more effective text representation method for NLP tasks. In 2018, the advent of the BERT model led a new wave of pre-trained language models, bringing new opportunities and challenges to the development of NLP technology. In recent years, Transformer-based models such as GPT-3 have been able to generate high-quality text by training models with huge parameters, and in some cases can be comparable to human writing.


## 1.3 NLP Tasks

In the vast field of research in NLP, several core tasks form the basis of the NLP domain, covering all aspects from basic processing of text to complex semantic understanding and generation. These tasks include but are not limited to Chinese word segmentation, subword segmentation, part-of-speech annotation, text classification, entity recognition, relationship extraction, text summary, machine translation, and the development of automatic question and answer systems. Each task has its own specific challenges and application scenarios, and together they drive the development of language technology and provide powerful tools for processing and analyzing growing text data.

### 1.3.1 Chinese word segmentation

Chinese Word Segmentation (CWS) is a basic task in the field of NLP. When processing Chinese text, due to the characteristics of Chinese language, there is no obvious separation between words (such as spaces) like English, so the boundaries of words cannot be determined directly through spaces. Therefore, Chinese word segmentation has become the primary step in Chinese text processing, and its purpose is to divide continuous Chinese text into meaningful vocabulary sequences.

```
英文输入：The cat sits on the mat.
英文切割输出：[The | cat | sits | on | the | mat]
中文输入：今天天气真好，适合出去游玩.
中文切割输出：["今天", "天气", "真", "好", "，", "适合", "出去", "游玩", "。"]
```

Correct word participle results are crucial for subsequent tasks such as part of speech annotation, entity recognition, and syntactic analysis. If the word segmentation is inaccurate, it will directly affect the effect of the entire text processing process.

```
输入：雍和宫的荷花开的很好。

正确切割：雍和宫 | 的 | 荷花 | 开 | 的 | 很 | 好 | 。
错误切割 1：雍 | 和 | 宫的 | 荷花 | 开的 | 很好 | 。 （地名被拆散）
错误切割 2：雍和 | 宫 | 的荷 | 花开 | 的很 | 好。 （词汇边界混乱）
```

Correct word participle results are crucial for subsequent tasks such as part of speech annotation, entity recognition, and syntactic analysis. If the word segmentation is inaccurate, it will directly affect the effect of the entire text processing process.

### 1.3.2 Subword cleavage

Subword Segmentation is a common text preprocessing technique in the NLP field, aiming to further decompose vocabulary into smaller units, i.e. subwords. Subword segmentation is particularly suitable for dealing with the problem of sparse vocabulary, that is, when encountering rare words or new words that have not been seen, these words can be understood or generated through known subword units. Subword segmentation is particularly important in dealing with languages with complex spelling and many synthetic words (such as German) or in pre-trained language models (such as BERT, GPT series).

There are many ways to divide subwords, including Byte Pair Encoding (BPE), WordPiece, Unigram, SentencePiece, etc. The basic idea of these methods is to break down words into smaller, frequently occurring fragments that can be single characters, character combinations or roots and affixes.

```
输出：unhappiness

不使用子词切分：整个单词作为一个单位：“unhappiness”
使用子词切分（假设BPE算法）：单词被分割为：“un”、“happi”、“ness”
```

In this example, by dividing the subword, the word "unhappiness" is broken down into three parts: the prefix "un" means negation, "happi" is the root variant of "happy", which means happiness, and "ness" is the noun suffix, which means state. Even if the model has never seen the complete word “unhappiness”, it can be understood through these known subwords to roughly mean “unhappiness state.”

### 1.3.3 Part of speech annotation

Part-of-Speech Tagging (POS Tagging) is a basic task in the field of NLP. Its goal is to assign a part-of-speech tag to each word in the text, such as nouns, verbs, adjectives, etc. This process is usually based on predefined sets of part-of-speech tags, such as common tags in English include noun (Noun, N), verb (Verb, V), adjective (Adjective, Adj), etc. Part-of-speech annotation is crucial for advanced NLP tasks such as understanding sentence structure, performing syntactic analysis, and semantic role annotation. Through part-of-speech annotation, computers can better understand the meaning of text, and then perform more complex processing such as information extraction, sentiment analysis, and machine translation.

Suppose we have an English sentence: She is playing the guitar in the park.

The results of part-of-speech annotation are as follows:

- She (pronoun, Pronoun, PRP)
- is (verb, Verb, VBZ)
- playing (the present participle of verb, Verb, VBG)
- the (Quitword, Determiner, DT)
- guitar (noun, Noun, NN)
- in (preposition, Preposition, IN)
- the (Quitword, Determiner, DT)
- park (noun, Noun, NN)
- . (Punctuation, Punctuation,.)

Part-of-speech annotation usually depends on machine learning models, such as the Hidden Markov Model (HMM), Conditional Random Field (CRF), or the recurrent neural network RNN and the long and short-term memory network LSTM based on deep learning. These models predict the part of speech of each word in a new sentence by learning a large amount of labeled data.

### 1.3.4 Text classification

Text Classification is a core task in the NLP field that involves automatically assigning a given text to one or more predefined categories. This technology is widely used in various scenarios, including but not limited to sentiment analysis, spam detection, news classification, topic recognition, etc. The key to text classification is to understand the meaning and context of the text and map the text to a specific category based on this.

Suppose there is a text classification task that aims to classify news articles into one of the three categories: "sports", "politics", or "technology".

```
文本：“NBA季后赛将于下周开始，湖人和勇士将在首轮对决。”
类别：“体育”

文本：“美国总统宣布将提高关税，引发国际贸易争端。”
类别：“政治”

文本：“苹果公司发布了新款 Macbook，配备了最新的m3芯片。”
类别：“科技”
```

The key to the success of the text classification task is to select the appropriate feature representation and classification algorithm, and have high-quality training data. With the development of deep learning technologies, using neural networks for text classification has become a trend, which can capture complex patterns and semantic information in text data, resulting in significant performance improvements in many tasks.

### 1.3.5 Entity Identification

Named Entity Recognition (NER), also known as named entity recognition, is a key task in the field of NLP, aiming to automatically identify entities with specific meanings in texts and classify them into predefined categories such as names, places, organizations, dates, times, etc. Entity recognition tasks are very important for applications such as information extraction, knowledge graph construction, question-and-answer systems, and content recommendation. They can help the system understand the key elements and their attributes in the text.

Suppose there is an entity recognition task that aims to identify entities such as person names, place names and organization names from the text.

```
输入：李雷和韩梅梅是北京市海淀区的居民，他们计划在2024年4月7日去上海旅行。

输出：[("李雷", "人名"), ("韩梅梅", "人名"), ("北京市海淀区", "地名"), ("2024年4月7日", "日期"), ("上海", "地名")]
```

Through the entity recognition task, we can not only identify entities in text, but also understand their categories, providing important information for a deep understanding of the content and context of the text. With the development of NLP technology, the accuracy and efficiency of entity recognition have been continuously improved, which can provide powerful support for various NLP applications.

### 1.3.6 Relationship Extraction

Relation Extraction is a critical task in the NLP field, and its goal is to identify semantic relationships between entities from text. These relationships can be causal relationships, ownership relationships, kinship relationships, geographical location relationships, etc. Relationship extraction is of great significance to understanding text content, building knowledge graphs, and improving the ability of machines to understand language.

Suppose we have the following sentence:

```
输入：比尔·盖茨是微软公司的创始人。

输出：[("比尔·盖茨", "创始人", "微软公司")]
```

In this example, the goal of the relationship extraction task is to identify the "founder" relationship between "Bill Gates" and "Microsoft" from the text. Through relationship extraction, we can extract useful information from the text, helping the computer better understand the text content, and providing support for subsequent knowledge graph construction, question-and-answer system and other tasks.

### 1.3.7 Text Summary

Text Summarization is an important task in NLP, with the purpose of generating a concise and accurate summary to summarize the main content of the original text. Depending on the generation method, text summary can be divided into two categories: Extractive Summarization and Generative Summarization.

- Extracted abstract: Extracted abstract consists of an abstract by directly selecting key sentences or phrases from the original text. The advantage is that the information in the abstract comes entirely from the original text, so it is more accurate. However, since it is just a splicing of sentences in the original text, sometimes the generated summary may not be smooth enough.
- Generative summary: Unlike extracted summary, generative summary not only involves selecting text fragments, but also requires reorganization and rewritten these fragments and generating new content. Generative summary is more challenging because it requires understanding the deep meaning of the text and being able to express the same information in new ways. Generative summary often requires more complex models, such as the sequence-to-sequence model (Seq2Seq) based on attention mechanisms.

Suppose we have the following news reports:

```
2021年5月22日，国家航天局宣布，我国自主研发的火星探测器“天问一号”成功在火星表面着陆。此次任务的成功，标志着我国在深空探测领域迈出了重要一步。“天问一号”搭载了多种科学仪器，将在火星表面进行为期90个火星日的科学探测工作，旨在研究火星地质结构、气候条件以及寻找生命存在的可能性。
```

Extraction summary:

```
我国自主研发的火星探测器“天问一号”成功在火星表面着陆，标志着我国在深空探测领域迈出了重要一步。
```

Generative summary:

```
“天问一号”探测器成功实现火星着陆，代表我国在宇宙探索中取得重大进展。
```

Text summary tasks are widely used in the fields of information retrieval, news push, report generation, etc. Through automatic summary, users can quickly obtain core information of text, save reading time and improve information processing efficiency.

### 1.3.8 Machine Translation

Machine Translation (MT) is a core task in the field of NLP, which refers to the process of automatically translating one natural language (source language) into another natural language (target language) using computer programs. Machine translation not only involves direct conversion of vocabulary, but more importantly, it is to accurately convey the semantics, styles and cultural background of the source language text, so that the translation results are natural, accurate and fluent in the target language, so as to overcome language barriers and promote communication and understanding between users of different languages.

Suppose we have a Chinese sentence: "The weather is good today." We want to translate it into English.

```
源语言：今天天气很好。

目标语言：The weather is very nice today.
```

In this simple example, machine translation can accurately convert Chinese sentences into English, maintaining the meaning and structure of the original sentence. However, the challenges faced by machine translation increase accordingly when dealing with longer, more complex texts. In order to improve the quality of machine translation, researchers have been constantly exploring new methods and technologies, such as Seq2Seq model and Transformer model based on neural networks. These models can learn the complex mapping relationship between the source language and the target language, thereby achieving more accurate and smooth translation.

### 1.3.9 Automatic Q&A

Automatic Question Answering (QA) is an advanced task in the NLP field that aims to enable computers to understand questions raised by natural language and automatically provide accurate answers based on a given data source. The automatic question and answer task simulates human ability to understand and answer questions, covering from simple fact queries to complex reasoning and explanations. The construction of an automatic question-and-answer system involves multiple NLP subtasks, such as information retrieval, text understanding, knowledge representation and reasoning.

Automatic Q&A can be roughly divided into three categories: Retrieval-based QA, Knowledge-based QA and Community-based QA. Search-based Q&A searches answers from a large number of texts through search engines and other methods; knowledge base Q&A answers answer questions through structured knowledge base; community Q&A relies on user-generated Q&A data, such as Q&A communities, forums, etc.

The development and optimization of automatic question-and-answer systems is a continuous process. With the advancement of technology and the improvement of algorithms, these systems have significantly improved in accuracy, understanding ability and application scope. By combining different types of data sources and technical approaches, automatic question-and-answer systems are becoming increasingly intelligent and increasingly capable of dealing with complex and diverse problems.

## 1.4 The development history of text representation

The purpose of text representation is to transform the natural form of human language into a form that computers can process, that is, digitize text data, so that computers can effectively analyze and process text. Text representation is a fundamental and necessary task in the field of NLP, which directly affects and even determines the quality and performance of NLP systems.

In NLP, text representation involves converting language units (such as words, words, phrases, sentences, etc.) in text and their relationships and structural information into forms that computers can understand and operate, such as vectors, matrices, or other data structures. Such representations not only need to retain sufficient semantic information to facilitate subsequent NLP tasks, such as text classification, sentiment analysis, machine translation, etc., but also need to consider computing efficiency and storage efficiency.

The development process of text representation has gone through multiple stages, from early rule-based methods, to statistical learning methods, to current deep learning technologies, text representation technology has been continuously evolving, providing strong support for the development of NLP.

### 1.4.1 Word vector

Vector Space Model (VSM) is a basic and powerful text representation method in the field of NLP, first proposed by Salton, Harvard University. Vector space models enable mathematical representation of text by converting text (including words, sentences, paragraphs, or entire documents) into vectors in high-dimensional space. In this model, each dimension represents a feature item (e.g., a word, word, phrase, or phrase), and each element value in the vector represents the weight of the feature item in the text. This weight is determined by specific calculation formulas (e.g., word frequency TF, inverse document frequency TF-IDF, etc.), reflecting the importance of the feature item in the text.

The vector space model is extremely widely used, including but not limited to natural language processing tasks such as text similarity calculation, text classification, and information retrieval. It converts complex text data into mathematical forms that are easy to calculate and analyze, making text similarity calculations and pattern recognition possible. In addition, through matrix operations such as eigenvalue calculation and singular value decomposition (SVD), text vector representation can be optimized to further improve processing efficiency and effect.

However, vector space models also have many problems. The most important ones are data sparseness and dimension disaster problems, because the huge number of feature terms leads to extremely high vector dimensions, and most elements have zero values. Furthermore, because the model ignores structural information in the text, such as word order and context information, based on the assumption of independence between feature terms, limits the expressiveness of the model. The shortcomings of feature items selection and weight calculation methods are also problems that need to be solved by vector space models.

VSM method word vector:

```python
# "雍和宫的荷花很美"
# 词汇表大小：16384，句子包含词汇：["雍和宫", "的", "荷花", "很", "美"] = 5个词

vector = [0, 0, ..., 1, 0, ..., 1, 0, ..., 1, 0, ..., 1, 0, ..., 1, 0, ...]
#                    ↑          ↑          ↑          ↑          ↑
#      16384维中只有5个位置为1，其余16379个位置为0
# 实际有效维度：仅5维（非零维度）
# 稀疏率：(16384-5)/16384 ≈ 99.97%
```
> A glossary is a collection of all possible words. In the vector space model, each word corresponds to a position in the vocabulary, in which the words can be converted into vector representations. For example, if the vocabulary size is 16384 , then each word will be represented as a 16384-dimensional vector, where only the word corresponds to a position of 1 and the rest are 0.

In order to solve these problems, researchers' research on vector space models mainly focuses on two aspects: one is to improve feature representation methods, such as keyword extraction with the help of graph methods, theme methods, etc.; the other is to improve and optimize the calculation method of feature term weights, which can be integrated calculations or propose new calculation methods based on existing methods.

### 1.4.2 Language Model

The N-gram model is a statistical-based language model in the field of NLP, which is widely used in many tasks such as speech recognition, handwriting recognition, spelling error correction, machine translation and search engines. The core idea of the N-gram model is based on the Markov hypothesis that the probability of a word's appearance depends only on the N-1 words before it. Here N represents the number of consecutive words, which can be any positive integer. For example, when N=1, the model is called unigram, which only considers the probability of a single word; when N=2, it is called bigram, which considers the previous word to estimate the probability of the current word; when N=3, it is called trigram, which considers the first two words to estimate the probability of the third word, and so on.

The N-gram model estimates the probability of the entire sentence through the conditional probability chain rule. Specifically, for a given sentence, the model calculates the conditional probability of each N-gram appearing and multiplies these probabilities by obtaining the probability of the entire sentence. For example, for the sentence "The quick brown fox", as a trigram model, we calculate the probabilities of $P("brown" | "The", "quick")$, $P("fox" | "quick", "brown")$, and multiply them.

The advantage of N-gram is that it is simple to implement and easy to understand, and it works well in many tasks. However, when N is large, data sparsity problems will occur. The parameter space of the model will increase sharply, and the probability of the same N-gram sequence appearing becomes very low, resulting in the model being unable to learn effectively and the model generalization ability decreases. Furthermore, the N-gram model ignores the scope dependencies between words and cannot capture the complex structural and semantic information in the sentence.

Despite limitations, the N-gram model is still widely used in many NLP tasks due to its simplicity and practicality. In some applications, combining N-gram models and other technologies such as deep learning models can achieve better performance.

### 1.4.3 Word2Vec

Word2Vec is a popular word embedding technology proposed by Tomas Mikolov et al. in 2013. It is a language model based on neural network NNLM, aiming to generate dense vector representations of words by learning the contextual relationship between words. The core idea of Word2Vec is to use the context information of words in the text to capture the semantic relationship between words, so that words with similar or related semantics are closer to the vector space.

The Word2Vec model mainly has two architectures: the continuous bag of words model CBOW (Continuous Bag of Words) calculates and outputs the vector representation of the target word based on the word vector corresponding to the word in the target word context; the Skip-Gram model, contrary to the CBOW model, uses the vector representation of the target word to calculate the word vector in the context. Practice proves that CBOW is suitable for small data sets, while Skip-Gram performs better in large corpus.

Compared with traditional high-dimensional sparse representations (such as One-Hot encoding), Word2Vec generates dense vectors with low-dimensional (usually hundreds of dimensions), which helps reduce computational complexity and storage requirements. The Word2Vec model can capture the semantic relationship between words, such as the position of "king" and "queen" in vector space is closer, because in large amounts of text, they usually appear in similar contexts. The Word2Vec model can also be generalized well to unseen words because it is learned based on context information rather than dictionary. However, since the CBOW/Skip-Gram model is based on local context, cannot capture long-distance dependencies and lacks the overall word-to-word relationship, it performs poorly on some complex semantic tasks.

### 1.4.4 ELMo

ELMo (Embeddings from Language Models) realizes a leapfrog transformation from one word, static word vector to dynamic word vector. First, the language model is trained on a large corpus to obtain the word vector model, and then fine-tune the model on a specific task to obtain the word vector that is more suitable for the task. ELMo introduced the pre-training idea to the generation of word vectors for the first time. Using a bidirectional LSTM structure, it can capture the context information of the vocabulary and generate a richer and more accurate word vector representation.

ELMo adopts a typical two-stage process: The first stage is to pre-train using the language model; the second stage is to extract the word vectors of the corresponding words from the pre-trained network as new features to supplement them into the downstream tasks when doing a specific task. The LSTM model based on RNN takes a long training time, and feature extraction is the key to the optimization and improvement of the ELMo model.

The main advantage of the ELMo model is that it can capture the ambiguousness and context information of the vocabulary, and the generated word vectors are richer and more accurate, and are suitable for a variety of NLP tasks. However, there are also some problems with the ELMo model, such as high model complexity, long training time, and high computing resource consumption.

## References

[1] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* arXiv preprint arXiv:1310.4546.

[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.

[3] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. (2023). *Attention Is All You Need.* arXiv preprint arXiv:1706.03762.

[4] Malek Hajjem, Chiraz Latiri. (2017). *Combining IR and LDA Topic Modeling for Filtering Microblogs.* Procedia Computer Science, 112, 761–770. https://doi.org/10.1016/j.procs.2017.08.166.

[5] Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer. (2018). *Deep contextualized word representations.* arXiv preprint arXiv:1802.05365.

[6] Salton, G., Wong, A., Yang, C. S. (1975). *A vector space model for automatic indexing.* Communications of the ACM, 18(11), 613–620. https://doi.org/10.1145/361219.361220.

[7] Zhao Jingsheng, Song Mengxue, Gao Xiang, et al. Research on text representation in natural language processing [J]. Acta Software Sinica, 2022, 33(01):102-128.DOI:10.13328/j.cnki.jos.006304.

[8] Chinese Information Processing Development Report (2016) Preface [C]//Chinese Information Processing Development Report (2016). Chinese Society of Information Technology;, 2016:2-3.DOI:10.26914/c.cnkihy.2016.003326.

