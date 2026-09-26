# Chapter 1: NLP Basic Concepts

Natural Language Processing (NLP) is an important branch of artificial intelligence, aimed at enabling computers to understand and process human language, achieving natural communication between humans and machines. With the rapid development of information technology, text data has become an indispensable part of our daily lives. Advances in NLP technology provide powerful tools for extracting useful information from massive texts and understanding the deeper meanings of language. From early rule-based methods, to later statistical learning approaches, and now the widespread application of deep learning technologies, the NLP field has undergone multiple technological revolutions. Text representation, as one of the core technologies in NLP, plays a decisive role in improving the performance of NLP systems through its research and advancements.

Welcome to the study of NLP basic concepts. This chapter will introduce the fundamental concepts of NLP to help everyone better understand and review related knowledge in NLP.

## 1.1 What is NLP

NLP is a technology that allows computers to understand, interpret, and generate human language. It is an extremely active and important research direction in the field of artificial intelligence. Its core task is to simulate human cognition and use of language through computer programs. NLP combines knowledge and techniques from multiple disciplines such as computer science, artificial intelligence, linguistics, and psychology, aiming to break the barriers between human language and computer language to achieve seamless communication and interaction.

NLP technology enables computers to perform various complex language processing tasks, such as Chinese word segmentation, subword segmentation, part-of-speech tagging, text classification, entity recognition, relation extraction, text summarization, machine translation, and automatic question answering. These tasks not only require computers to recognize and process the surface structure of language but also, more importantly, to understand the deeper meanings behind the language, including complex factors such as semantics, context, emotions, and culture.

With the development of modern technologies like deep learning, NLP has made significant progress. By training on large amounts of data, deep learning models can learn complex patterns and structures in language, achieving performance close to or even surpassing human levels in multiple NLP tasks. However, NLP still faces many challenges, such as handling ambiguity, understanding abstract concepts, and dealing with metaphors and sarcasm. Researchers are committed to solving these problems through more advanced algorithms, larger-scale datasets, and more refined language models to continuously advance NLP technology.

## 1.2 NLP Development History

The development history of NLP is an evolutionary process from early rule-based methods, to statistical methods, and now to machine learning and deep learning approaches. Each technological revolution has greatly promoted the development of NLP technology, achieving significant accomplishments in tasks such as machine translation, sentiment analysis, entity recognition, and text summarization. With the continuous enhancement of computing power and optimization of algorithms, the future of NLP will be even brighter, playing a more important role in more fields.

### Early Exploration (1940s - 1960s)

The early exploration of NLP began after World War II, when people recognized the importance of automatically translating one language into another. In 1950, Alan Turing proposed the Turing Test.

> He said that if a machine could become part of a conversation using a typewriter and fully imitate a human without any noticeable difference, then the machine could be considered capable of thinking.

This is a test to determine whether a machine can exhibit intelligent behavior indistinguishable from that of a human. During this period, Noam Chomsky proposed the theory of generative grammar, which had a significant impact on understanding how machine translation works. However, machine translation systems in this period were very simple, mainly relying on dictionary lookups and basic word order rules for translation, with unsatisfactory results.

### Symbolism and Statistical Methods (1970s - 1990s)

After the 1970s, NLP researchers began exploring new fields, including logic-based paradigms and natural language understanding. During this period, researchers were divided into two camps: symbolism (or rule-based) and statistical methods. Symbolist researchers focused on formal languages and generative grammar, while statistical method researchers paid more attention to statistics and probability methods. In the 1980s, with the improvement of computing power and the introduction of machine learning algorithms, revolutionary changes occurred in the NLP field, and statistical models began to replace complex "handwritten" rules.

### Machine Learning and Deep Learning (2000s to Present)

After the 2000s, with the development of deep learning technology, the NLP field made significant progress. Deep learning models such as Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and attention mechanisms have been widely applied to NLP tasks, achieving remarkable results. In 2013, the proposal of the Word2Vec model ushered in a new era of word vector representations, providing more effective text representation methods for NLP tasks. In 2018, the advent of the BERT model led a new wave of pre-trained language models, bringing new opportunities and challenges to the development of NLP technology. In recent years, Transformer-based models like GPT-3, by training models with huge parameters, can generate high-quality text and even rival human writing in some cases.

## 1.3 NLP Tasks

In the broad research field of NLP, several core tasks form the foundation of the NLP domain, covering various aspects from basic text processing to complex semantic understanding and generation. These tasks include but are not limited to Chinese word segmentation, subword segmentation, part-of-speech tagging, text classification, entity recognition, relation extraction, text summarization, machine translation, and the development of automatic question answering systems. Each task has its specific challenges and application scenarios, collectively driving the development of language technology and providing powerful tools for processing and analyzing the growing volume of text data.

### 1.3.1 Chinese Word Segmentation

Chinese Word Segmentation (CWS) is a fundamental task in the NLP field. When processing Chinese text, due to the characteristics of the Chinese language, there are no obvious delimiters (such as spaces) between words like in English, so word boundaries cannot be determined directly through spaces. Therefore, Chinese word segmentation becomes the primary step in Chinese text processing, with the goal of dividing continuous Chinese text into meaningful word sequences.

```
English Input: The cat sits on the mat.
English Segmentation Output: [The | cat | sits | on | the | mat]
Chinese Input: 今天天气真好，适合出去游玩. ("The weather is really nice today, perfect for going out.")
Chinese Segmentation Output: ["今天", "天气", "真", "好", "，", "适合", "出去", "游玩", "。"] (today | weather | really | nice | , | suitable for | go out | sightseeing | .)
```

Accurate segmentation results are crucial for subsequent tasks such as part-of-speech tagging, entity recognition, and syntactic analysis. If segmentation is inaccurate, it will directly affect the effectiveness of the entire text processing workflow.

```
Input: 雍和宫的荷花开的很好。 ("The lotus flowers at the Yonghe Temple are blooming beautifully.")

Correct Segmentation: 雍和宫 | 的 | 荷花 | 开 | 的 | 很 | 好 | 。 (Yonghe Temple | 's | lotus flowers | bloom | [particle] | very | well | .)
Error Segmentation 1: 雍 | 和 | 宫的 | 荷花 | 开的 | 很好 | 。 (Place name broken apart)
Error Segmentation 2: 雍和 | 宫 | 的荷 | 花开 | 的很 | 好。 (Word boundaries confused)
```

Accurate segmentation results are crucial for subsequent tasks such as part-of-speech tagging, entity recognition, and syntactic analysis. If segmentation is inaccurate, it will directly affect the effectiveness of the entire text processing workflow.

### 1.3.2 Subword Segmentation

Subword Segmentation is a common text preprocessing technique in the NLP field, aimed at further decomposing words into smaller units, namely subwords. Subword segmentation is particularly suitable for handling vocabulary sparsity issues, meaning that when encountering rare words or unseen new words, they can be understood or generated through known subword units. Subword segmentation is especially important in processing languages with complex spellings and compound words (such as German) or in pre-trained language models (such as BERT, GPT series).

There are many methods for subword segmentation, common ones include Byte Pair Encoding (BPE), WordPiece, Unigram, SentencePiece, etc. The basic idea of these methods is to break down words into smaller, frequently occurring fragments, which can be individual characters, character combinations, or roots and affixes.

```
Output: unhappiness

Without Subword Segmentation: The entire word as a unit: “unhappiness”
With Subword Segmentation (assuming BPE algorithm): The word is segmented into: “un”, “happi”, “ness”
```

In this example, through subword segmentation, the word “unhappiness” is broken down into three parts: the prefix “un” indicating negation, “happi” as a variant of the root “happy” meaning happiness, and “ness” as a noun suffix indicating state. Even if the model has never seen the complete word “unhappiness,” it can understand its approximate meaning as “the state of unhappiness” through these known subwords.

### 1.3.3 Part-of-Speech Tagging

Part-of-Speech Tagging (POS Tagging) is a fundamental task in the NLP field, with the goal of assigning a part-of-speech tag to each word in the text, such as noun, verb, adjective, etc. This process is usually based on a predefined set of POS tags, such as common tags in English including Noun (N), Verb (V), Adjective (Adj), etc. POS tagging is crucial for understanding sentence structure, performing syntactic analysis, semantic role labeling, and other advanced NLP tasks. Through POS tagging, computers can better understand the meaning of text, thereby enabling more complex processing such as information extraction, sentiment analysis, and machine translation.

Suppose we have an English sentence: She is playing the guitar in the park.

The POS tagging result is as follows:

- She (Pronoun, PRP)
- is (Verb, VBZ)
- playing (Verb gerund or present participle, VBG)
- the (Determiner, DT)
- guitar (Noun, NN)
- in (Preposition, IN)
- the (Determiner, DT)
- park (Noun, NN)
- . (Punctuation, .)

POS tagging typically relies on machine learning models, such as Hidden Markov Models (HMM), Conditional Random Fields (CRF), or deep learning-based Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM). These models predict the POS of each word in new sentences by learning from large amounts of annotated data.

### 1.3.4 Text Classification

Text Classification is a core task in the NLP field, involving automatically assigning given text to one or more predefined categories. This technology is widely used in various scenarios, including but not limited to sentiment analysis, spam detection, news classification, topic identification, etc. The key to text classification lies in understanding the meaning and context of the text and mapping it to specific categories based on that.

Suppose there is a text classification task aimed at classifying news articles into one of three categories: "Sports," "Politics," or "Technology."

```
Text: “NBA playoffs will start next week, with the Lakers and Warriors facing off in the first round.”
Category: “Sports”

Text: “The US President announced an increase in tariffs, sparking international trade disputes.”
Category: “Politics”

Text: “Apple released the new MacBook, equipped with the latest M3 chip.”
Category: “Technology”
```

The success of text classification tasks depends on selecting appropriate feature representations and classification algorithms, as well as having high-quality training data. With the development of deep learning technology, using neural networks for text classification has become a trend, as they can capture complex patterns and semantic information in text data, achieving significant performance improvements in many tasks.

### 1.3.5 Entity Recognition

Named Entity Recognition (NER), also known as entity recognition, is a key task in the NLP field, aimed at automatically identifying entities with specific meanings in text and classifying them into predefined categories, such as person names, locations, organizations, dates, times, etc. Entity recognition tasks are important for applications such as information extraction, knowledge graph construction, question answering systems, and content recommendation, as they help systems understand key elements and their attributes in text.

Suppose there is an entity recognition task aimed at identifying entities such as person names, location names, and organization names from text.

```
Input: Li Lei and Han Meimei are residents of Haidian District, Beijing. They plan to travel to Shanghai on April 7, 2024.

Output: [("Li Lei", "Person"), ("Han Meimei", "Person"), ("Haidian District, Beijing", "Location"), ("April 7, 2024", "Date"), ("Shanghai", "Location")]
```

Through entity recognition tasks, we can not only identify entities in text but also understand their categories, providing important information for a deeper understanding of text content and context. With the development of NLP technology, the accuracy and efficiency of entity recognition continue to improve, offering strong support for various NLP applications.

### 1.3.6 Relation Extraction

Relation Extraction is a key task in the NLP field, with the goal of identifying semantic relationships between entities in text. These relationships can be causal, ownership, kinship, geographical location, etc. Relation extraction is significant for understanding text content, building knowledge graphs, and enhancing machines' ability to understand language.

Suppose we have the following sentence:

```
Input: Bill Gates is the founder of Microsoft Corporation.

Output: [("Bill Gates", "Founder", "Microsoft Corporation")]
```

In this example, the goal of the relation extraction task is to identify the "founder" relationship between "Bill Gates" and "Microsoft Corporation" from the text. Through relation extraction, we can extract useful information from text, helping computers better understand text content and providing support for subsequent tasks such as knowledge graph construction and question answering systems.

### 1.3.7 Text Summarization

Text Summarization is an important task in NLP, aimed at generating a concise and accurate summary to outline the main content of the original text. Depending on the generation method, text summarization can be divided into two major categories: Extractive Summarization and Abstractive Summarization.

- Extractive Summarization: Extractive summarization forms the summary by directly selecting key sentences or phrases from the original text. The advantage is that the information in the summary comes entirely from the original text, so accuracy is high. However, since it is merely a concatenation of sentences from the original text, the generated summary may sometimes lack fluency.
- Abstractive Summarization: Unlike extractive summarization, abstractive summarization not only involves selecting text fragments but also requires reorganizing and rewriting these fragments and generating new content. Abstractive summarization is more challenging because it needs to understand the deep meaning of the text and express the same information in new ways. Abstractive summarization usually requires more complex models, such as sequence-to-sequence models (Seq2Seq) based on attention mechanisms.

Suppose we have the following news report:

```
On May 22, 2021, the China National Space Administration announced that China's independently developed Mars probe "Tianwen-1" successfully landed on the surface of Mars. The success of this mission marks an important step forward for China in the field of deep space exploration. "Tianwen-1" carries multiple scientific instruments and will conduct scientific exploration work on the Martian surface for 90 Martian days, aiming to study Mars' geological structure, climate conditions, and the possibility of finding signs of life.
```

Extractive Summary:

```
China's independently developed Mars probe "Tianwen-1" successfully landed on the surface of Mars, marking an important step forward for China in the field of deep space exploration.
```

Abstractive Summary:

```
The "Tianwen-1" probe successfully achieved a Mars landing, representing a major breakthrough for China in space exploration.
```

Text summarization tasks have wide applications in fields such as information retrieval, news push, and report generation. Through automatic summarization, users can quickly obtain the core information of text, saving reading time and improving information processing efficiency.

### 1.3.8 Machine Translation

Machine Translation (MT) is a core task in the NLP field, referring to the process of using computer programs to automatically translate one natural language (source language) into another natural language (target language). Machine translation involves not only direct conversion of vocabulary but also, more importantly, accurately conveying the semantics, style, and cultural background of the source language text, making the translation result natural, accurate, and fluent in the target language to bridge language barriers and promote communication and understanding among users of different languages.

Suppose we have a Chinese sentence: “The weather is very good today.” and we want to translate it into English.

```
Source Language: 今天天气很好。 ("The weather is very good today.")

Target Language: The weather is very nice today.
```

In this simple example, machine translation can accurately convert the Chinese sentence into English, maintaining the original meaning and structure. However, when dealing with longer and more complex texts, the challenges faced by machine translation increase accordingly. To improve the quality of machine translation, researchers continue to explore new methods and technologies, such as neural network-based Seq2Seq models and Transformer models, which can learn complex mapping relationships between source and target languages, achieving more accurate and fluent translations.

### 1.3.9 Automatic Question Answering

Automatic Question Answering (QA) is an advanced task in the NLP field, aimed at enabling computers to understand questions posed in natural language and automatically provide accurate answers based on given data sources. Automatic question answering tasks simulate human abilities to understand and answer questions, covering everything from simple fact queries to complex reasoning and explanations. The construction of automatic question answering systems involves multiple NLP subtasks, such as information retrieval, text understanding, knowledge representation, and reasoning.

Automatic question answering can be roughly divided into three categories: Retrieval-based QA, Knowledge-based QA, and Community-based QA. Retrieval-based QA retrieves answers from large amounts of text through search engines and other methods; Knowledge-based QA answers questions through structured knowledge bases; Community-based QA relies on user-generated question-and-answer data, such as Q&A communities and forums.

The development and optimization of automatic question answering systems is an ongoing process. With technological advancements and algorithm improvements, these systems have seen significant enhancements in accuracy, understanding capabilities, and application scope. By combining different types of data sources and technical methods, automatic question answering systems are becoming increasingly intelligent and capable of handling complex and diverse questions.

## 1.4 Development History of Text Representation

The purpose of text representation is to convert the natural form of human language into a form that computers can process, that is, to digitize text data so that computers can effectively analyze and process text. Text representation is a foundational and essential task in the NLP field, directly influencing or even determining the quality and performance of NLP systems.

In NLP, text representation involves converting linguistic units in text (such as characters, words, phrases, sentences, etc.) and the relationships and structural information between them into forms that computers can understand and operate on, such as vectors, matrices, or other data structures. Such representations need to retain sufficient semantic information to facilitate subsequent NLP tasks, such as text classification, sentiment analysis, machine translation, etc., while also considering computational and storage efficiency.

The development history of text representation has gone through multiple stages, from early rule-based methods, to statistical learning methods, and now to current deep learning technologies. Text representation technology continues to evolve, providing strong support for the development of NLP.

### 1.4.1 Word Vectors

The Vector Space Model (VSM) is a foundational and powerful text representation method in the NLP field, first proposed by Salton at Harvard University. The vector space model achieves mathematical representation of text by converting text (including words, sentences, paragraphs, or entire documents) into vectors in a high-dimensional space. In this model, each dimension represents a feature item (e.g., character, word, word group, or phrase), and each element value in the vector represents the weight of that feature item in the text. This weight is determined through specific calculation formulas (such as Term Frequency TF, Term Frequency-Inverse Document Frequency TF-IDF, etc.), reflecting the importance of the feature item in the text.

The vector space model has extremely wide applications, including but not limited to text similarity calculation, text classification, information retrieval, and other natural language processing tasks. It transforms complex text data into a mathematical form that is easy to compute and analyze, making text similarity calculations and pattern recognition possible. Additionally, through matrix operations such as eigenvalue calculations and singular value decomposition (SVD), text vector representations can be optimized, further improving processing efficiency and effectiveness.

However, the vector space model also has many problems. The most prominent are data sparsity and the curse of dimensionality, as the large number of feature items leads to extremely high vector dimensions, with most elements being zero values. Additionally, since the model is based on the independence assumption between feature items, it ignores structural information in the text, such as word order and context information, which limits the model's expressive power. The selection of feature items and inadequate weight calculation methods are also problems that the vector space model needs to address.

VSM method word vector:

```python
# "雍和宫的荷花很美" ("The lotus flowers at the Yonghe Temple are beautiful")
# Vocabulary size: 16384, sentence contains words: ["雍和宫", "的", "荷花", "很", "美"] = 5 words (Yonghe Temple, 's, lotus flowers, very, beautiful)

vector = [0, 0, ..., 1, 0, ..., 1, 0, ..., 1, 0, ..., 1, 0, ..., 1, 0, ...]
#                    ↑          ↑          ↑          ↑          ↑
#      In 16384 dimensions, only 5 positions are 1, the remaining 16379 positions are 0
# Effective dimensions: only 5 dimensions (non-zero dimensions)
# Sparsity rate: (16384-5)/16384 ≈ 99.97%
```
> The vocabulary is a collection containing all possible words. In the vector space model, each word corresponds to a position in the vocabulary. Through this method, words can be converted into vector representations. For example, if the vocabulary size is 16384, then each word will be represented as a 16384-dimensional vector, where only the position corresponding to that word is 1, and all other positions are 0.

To solve these problems, researchers' research on the vector space model mainly focuses on two aspects: one is improving feature representation methods, such as using graph methods, topic methods, etc., for keyword extraction; the other is improving and optimizing the calculation methods of feature item weights, which can be fused or new calculation methods proposed based on existing methods.

### 1.4.2 Language Models

The N-gram model is a statistical-based language model in the NLP field, widely used in numerous tasks such as speech recognition, handwriting recognition, spelling correction, machine translation, and search engines. The core idea of the N-gram model is based on the Markov assumption, that the probability of a word appearing depends only on the previous N-1 words. Here, N represents the number of consecutive words and can be any positive integer. For example, when N=1, the model is called unigram, considering only the probability of a single word; when N=2, it is called bigram, considering the previous word to estimate the probability of the current word; when N=3, it is called trigram, considering the previous two words to estimate the probability of the third word, and so on for N-gram.

The N-gram model estimates the probability of the entire sentence through conditional probability chain rule. Specifically, for a given sentence, the model calculates the conditional probability of each N-gram occurrence and multiplies these probabilities to get the probability of the entire sentence. For example, for the sentence “The quick brown fox”, as a trigram model, we would calculate $P("brown" | "The", "quick")$, $P("fox" | "quick", "brown")$, etc., and multiply them.

The advantages of N-gram are simplicity and ease of understanding, and it performs well in many tasks. However, when N is large, data sparsity issues arise. The model's parameter space increases dramatically, and the probability of the same N-gram sequences appearing becomes very low, leading to ineffective learning and reduced model generalization ability. Additionally, the N-gram model ignores long-range dependencies between words, failing to capture complex structures and semantic information in sentences.

Despite its limitations, the N-gram model is still widely used in many NLP tasks due to its simplicity and practicality. In some applications, combining N-gram models with other techniques (such as deep learning models) can achieve better performance.

### 1.4.3 Word2Vec

Word2Vec is a popular word embedding method proposed by Tomas Mikolov and others in 2013. It is a language model based on neural network NNLM, aimed at generating dense vector representations of words by learning the contextual relationships between words. The core idea of Word2Vec is to use the contextual information of words in text to capture semantic relationships between words, so that semantically similar or related words are closer in vector space.

Word2Vec has two main architectures: the Continuous Bag of Words model (CBOW), which computes and outputs the vector representation of the target word based on the word vectors of the words in the context of the target word; the Skip-Gram model, which is the opposite of CBOW, using the vector representation of the target word to compute the vectors of the words in the context. Practice shows that CBOW is suitable for small datasets, while Skip-Gram performs better on large corpora.

Compared to traditional high-dimensional sparse representations (such as One-Hot encoding), Word2Vec generates low-dimensional (usually hundreds of dimensions) dense vectors, helping to reduce computational complexity and storage requirements. The Word2Vec model can capture semantic relationships between words, for example, "king" and "queen" will be positioned relatively close in vector space because they often appear in similar contexts in large texts. The Word2Vec model can also generalize well to unseen words because it is based on contextual learning rather than dictionaries. However, since the CBOW/Skip-Gram models are based on local contexts, they cannot capture long-distance dependencies and lack global relationships between words, so they perform poorly on some complex semantic tasks.

### 1.4.4 ELMo

ELMo (Embeddings from Language Models) achieves a leap forward: it handles polysemy (one word with multiple meanings) and moves from static word vectors to dynamic, context-dependent word vectors. First, it trains a language model on a large corpus to obtain a word vector model, then fine-tunes the model on specific tasks to get word vectors more suitable for that task. ELMo first introduces pre-training ideas into word vector generation, using a bidirectional LSTM structure to capture contextual information of words, generating more rich and accurate word vector representations.

ELMo adopts a typical two-stage process: the first stage is pre-training using language models; the second stage is extracting the corresponding word vectors from the pre-trained network as new features to supplement downstream tasks. The LSTM-based RNN model has long training times, and feature extraction is key to optimizing the ELMo model.

The main advantages of the ELMo model are its ability to capture polysemy and contextual information of words, generating more rich and accurate word vectors, suitable for various NLP tasks. However, the ELMo model also has some issues, such as high model complexity, long training times, and high computational resource consumption.

## References

[1] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* arXiv preprint arXiv:1310.4546.

[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.

[3] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. (2023). *Attention Is All You Need.* arXiv preprint arXiv:1706.03762.

[4] Malek Hajjem, Chiraz Latiri. (2017). *Combining IR and LDA Topic Modeling for Filtering Microblogs.* Procedia Computer Science, 112, 761–770. https://doi.org/10.1016/j.procs.2017.08.166.

[5] Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer. (2018). *Deep contextualized word representations.* arXiv preprint arXiv:1802.05365.

[6] Salton, G., Wong, A., Yang, C. S. (1975). *A vector space model for automatic indexing.* Communications of the ACM, 18(11), 613–620. https://doi.org/10.1145/361219.361220.

[7] Zhao Jingsheng, Song Mengxue, Gao Xiang, et al. Research on Text Representation in Natural Language Processing [J]. Journal of Software, 2022, 33(01):102-128. DOI:10.13328/j.cnki.jos.006304.

[8] Preface to the Development Report on Chinese Information Processing (2016) [C]//Development Report on Chinese Information Processing (2016). Chinese Information Processing Society of China; ,2016:2-3. DOI:10.26914/c.cnkihy.2016.003326.