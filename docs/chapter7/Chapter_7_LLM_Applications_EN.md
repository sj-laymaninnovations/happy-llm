# Chapter 7: LLM Applications

In the previous chapters, we systematically covered the fundamentals of large language models (LLMs), how they are trained, and how they are fine-tuned. This chapter focuses on the key techniques and frameworks used when applying LLMs in practice, covering core topics such as LLM evaluation, RAG (retrieval-augmented generation), and agents. The goal is to help readers gain a deeper understanding of real-world LLM use cases and how they are implemented.

## 7.1 Evaluating LLMs

In recent years, with the rapid development of artificial intelligence, large-scale pre-trained language models (large language models, or LLMs for short) have become a core driving force behind technological progress. These models have shown astonishing capabilities in natural language processing and related tasks. However, measuring an LLM's performance accurately requires scientific and well-designed evaluation.

What is LLM evaluation? LLM evaluation means quantifying and comparing how LLMs perform on different tasks using a variety of standardized methods and datasets. Evaluation covers not only a model's accuracy on specific tasks but also its generalization ability, inference speed, resource consumption, and more. Through evaluation, we can form a more complete picture of how LLMs actually perform and of their potential for real-world applications.

Developing an LLM is expensive and requires large amounts of compute and data, so evaluation is essential to make sure a model delivers real value. First, evaluation reveals how a model performs across a range of tasks, helping researchers and companies judge whether it is suitable and reliable. Second, evaluation can expose a model's potential weaknesses, such as bias or lack of robustness, which provides a basis for further optimization and improvement. In addition, fair and open evaluation gives academia and industry a common standard, which promotes the exchange of ideas and technical progress.

### 7.1.1 LLM Evaluation Datasets

Using standardized evaluation sets is crucial when evaluating LLMs. Today, mainstream LLM evaluation sets assess models along the following dimensions, and each evaluation set has its own purpose and typical use cases:

1. **General evaluation sets**:
   - **MMLU (Massive Multitask Language Understanding)**: MMLU evaluates a model's understanding across many tasks spanning a wide range of subjects and knowledge domains. It includes task types such as history, mathematics, physics, biology, and law, giving a comprehensive test of the model's knowledge and language understanding across disciplines.

2. **Tool-use evaluation sets**:
   - **BFCL V2**: Evaluates a model's performance on complex tool-use tasks, especially the correctness and efficiency with which it carries out multi-step operations. These tasks usually involve interacting with databases or executing specific instructions to simulate real tool-use scenarios.

3. **Math evaluation sets**:
   - **GSM8K**: GSM8K is a dataset of grade-school math problems used to test a model's mathematical reasoning and logical analysis. Tasks include arithmetic, solving simple equations, and numerical reasoning. Although the problems in GSM8K look simple, the model has to understand what the problem is asking and then carry out the correct calculations, which makes it a combined challenge of logical reasoning and language understanding.
   - **MATH**: The MATH dataset tests a model's performance on more complex math problems, including algebra and geometry.

4. **Reasoning evaluation sets**:
   - **ARC Challenge**: ARC Challenge evaluates a model's performance on scientific reasoning tasks, especially answering commonsense and science questions. Typical use cases include answering science exam questions and building encyclopedia-style question-answering systems.
   - **GPQA**: Evaluates a model's ability to answer open-ended questions in a zero-shot setting. It is commonly relevant to customer-service chatbots and knowledge question-answering systems, helping models give reasonable answers even when domain-specific data is lacking.
   - **HellaSwag**: Evaluates a model's ability to choose the most logically consistent answer in a complex context. It suits scenarios that demand a high level of understanding and reasoning, such as story continuation and dialogue generation.

5. **Long-context understanding evaluation sets**:
   - **InfiniteBench/En.MC**: Evaluates a model's reading comprehension on long texts, especially scientific literature. It is relevant to applications such as automatic summarization of academic papers and analysis of long reports.
   - **NIH/Multi-needle**: Tests a model's ability to understand and summarize in long-document settings with many items to find. It applies to scenarios that involve processing large amounts of information, such as interpreting government reports or analyzing long internal company documents.

6. **Multilingual evaluation sets**:
   - **MGSM**: Evaluates a model's ability to solve math problems in different languages, testing its multilingual adaptability. It is especially relevant to math education in international settings and to cross-lingual technical support.

The diversity of these evaluation sets helps us assess LLMs comprehensively across different tasks and application scenarios, ensuring that models remain efficient and accurate when handling varied tasks. For example, on MMLU, some LLMs perform very well on subjects such as history and physics, showing a deep understanding of knowledge across many domains. On the GSM8K math benchmark, the latest LLMs approach or even exceed some human baselines in arithmetic and equation solving, showing their potential for complex mathematical reasoning. These real evaluation results demonstrate the progress and application potential of models across many kinds of complex tasks.


### 7.1.2 Mainstream Leaderboards

LLM evaluation is not limited to running specific datasets. Many organizations also publish model leaderboards based on evaluation results. These leaderboards are an important reference for academia and industry, helping them keep track of the most cutting-edge techniques and models. Below are some of the mainstream leaderboards:

#### Open LLM Leaderboard

The Open LLM Leaderboard is an open leaderboard run by Hugging Face. It gathers evaluation results for many open-source LLMs to help users understand how different models perform across various tasks. The leaderboard evaluates model performance on several standardized test sets and is continuously updated to reflect the latest technical progress, providing researchers and developers with a valuable point of comparison, as shown in Figure 7.1.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/7-images/7-1-Open%20LLM%20Leaderboard.png" alt="alt text" width="90%">
    <p>Figure 7.1 Open LLM Leaderboard</p>
</div>

#### Lmsys Chatbot Arena Leaderboard

This chatbot leaderboard, provided by lmsys, uses multi-dimensional evaluation to show how well various LLMs perform on conversational tasks. It evaluates conversation quality through real users interacting with the models, focusing on the models' natural language generation, their understanding of context, and user satisfaction. It is currently an important tool for evaluating chatbot performance, as shown in Figure 7.2.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/7-images/7-1-lmsys%20Chatbot%20Arena%20Leaderboard.png" alt="alt text" width="90%">
    <p>Figure 7.2 Lmsys Chatbot Arena Leaderboard</p>
</div>

#### OpenCompass

OpenCompass is a leaderboard based in China. It evaluates LLMs across multiple languages and tasks and provides a reference for applications specific to the Chinese market. It combines tests of Chinese language understanding with tests of multilingual ability to meet localization needs, and it pays particular attention to the accuracy, robustness, and adaptability of LLMs in Chinese-language contexts. It is an important reference for Chinese companies and researchers choosing a suitable model.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/7-images/7-1-opencompass.png" alt="alt text" width="90%">
    <p>Figure 7.3 OpenCompass</p>
</div>

### 7.1.3 Domain-Specific Leaderboards

There are also LLM leaderboards targeting specific tasks in particular domains, as shown in Figure 7.4. These leaderboards focus on specific application areas and help users understand how capable LLMs are within a given vertical domain:

- Finance leaderboard: Based on the CFBenchmark evaluation set, it assesses LLMs on a range of fundamental tasks such as financial natural language processing, financial forecasting and calculation, and financial analysis and security checks. Provided by Tongji University together with Shanghai AI Laboratory and East Money (Dongfang Caijing).

- Safety leaderboard: Based on the Flames evaluation set, it assesses how resistant LLMs are across five dimensions including fairness, safety, data protection, and legality, giving an in-depth view of model safety. Provided by Shanghai AI Laboratory and Fudan University.

- General-knowledge leaderboard: Based on the BotChat evaluation set, it assesses how well LLMs can generate everyday multi-turn conversations, judging whether a model reaches a human-like level in dialogue. Provided by Shanghai AI Laboratory.

- Law leaderboard: Based on the LawBench evaluation set, it assesses a model's understanding, reasoning, and application abilities in the legal domain, covering tasks such as answering legal questions, generating text, and analyzing legal precedents. Provided by Nanjing University.

- Medical leaderboard: Based on the MedBench evaluation set, it assesses LLMs on medical knowledge question answering, understanding of safety and ethics, and related areas. Provided by Shanghai AI Laboratory.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/7-images/7-1-垂直领域榜单.png" alt="alt text" width="90%">
    <p>Figure 7.4 Vertical-domain leaderboards</p>
</div>


## 7.2 RAG 

### 7.2.1 How RAG Works

Although large language models (LLMs) have powerful language understanding and generation abilities, they also face some challenges when generating content. For example, LLMs sometimes produce inaccurate or misleading content, a phenomenon known as LLM "hallucination." In addition, the training data a model relies on may be outdated, so when it comes to the latest information, the accuracy and timeliness of its outputs cannot be guaranteed. LLMs are also relatively inefficient at handling specialized domain knowledge and cannot deeply understand complex domain-specific material. Improving the quality and efficiency of LLM generation has therefore become an important research direction.

Against this background, retrieval-augmented generation (RAG) emerged and has become a major innovative trend in AI. Before generating an answer, RAG first retrieves relevant information from a large external document database and incorporates that information into the generation process, thereby guiding and improving the language model's output. This process not only greatly improves the accuracy and relevance of the generated content but also helps the content stay up to date.

The core idea of RAG is to combine "retrieval" with "generation": when a user submits a query, the system first uses a retrieval module to find text passages relevant to the question, then passes those passages to the language model as additional information, and the model uses them to produce a more precise and reliable answer. In this way, RAG effectively mitigates the "hallucination" problem of LLMs, because the generated content is grounded in real documents, which makes answers more traceable and trustworthy. At the same time, because it brings in up-to-date information sources, RAG greatly speeds up knowledge updates, allowing the system to absorb and reflect the latest developments in a field in a timely way.

### 7.2.2 Building a RAG Framework

Next, I will walk you step by step through implementing a simple RAG model. This model is a simplified version of RAG that we call Tiny-RAG. Tiny-RAG keeps only RAG's core functions, namely retrieval and generation, and its purpose is to help you better understand how RAG works and how it is implemented.

#### Step 1: Overview of the RAG Pipeline

Before the language model generates an answer, RAG first retrieves relevant information from a broad document database and then uses that information to guide generation, which greatly improves the accuracy and relevance of the content. RAG effectively mitigates hallucination, speeds up knowledge updates, and makes generated content more traceable, making large language models more practical and trustworthy in real applications.

What are the basic components of RAG?

- Vectorization module: converts document chunks into vectors.
- Document loading and splitting module: loads documents and splits them into chunks.
- Database: stores the document chunks and their corresponding vector representations.
- Retrieval module: retrieves relevant document chunks based on the query (the question).
- LLM module: answers the user's question based on the retrieved documents.

These are all of TinyRAG's modules, as shown in Figure 7.5.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/7-images/7-2-tinyrag.png" alt="alt text" width="90%">
    <p>Figure 7.5 TinyRAG project structure</p>
</div>

Next, let's walk through what the RAG pipeline looks like.

- **Indexing**: Split the document collection into shorter chunks and build a vector index using an encoder.
- **Retrieval**: Retrieve relevant document chunks based on the similarity between the question and the chunks.
- **Generation**: Generate an answer to the question, conditioned on the retrieved context.

The flowchart is shown in Figure 7.6 below. Image source: ***[Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/pdf/2312.10997.pdf)***

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/7-images/7-2-rag.png" alt="alt text" width="90%">
    <p>Figure 7.6 RAG flowchart</p>
</div>

#### Step 2: Loading and Splitting Documents

Next, we'll implement a class for loading and splitting documents. Its main job is to load documents and split them into chunks.

Documents can be any kind of text content, such as articles, books, conversations, or code, stored for example as pdf, md, or txt files. The full code can be found in the ***[RAG/utils.py](./RAG/utils.py)*** file. The code supports loading pdf, md, txt, and other file types; you only need to write the corresponding reader function for each.

```python
def read_file_content(cls, file_path: str):
    # Choose the reading method based on the file extension
    if file_path.endswith('.pdf'):
        return cls.read_pdf(file_path)
    elif file_path.endswith('.md'):
        return cls.read_markdown(file_path)
    elif file_path.endswith('.txt'):
        return cls.read_text(file_path)
    else:
        raise ValueError("Unsupported file type")
```

After a document is read, it needs to be split. We can set a maximum token length and split the document according to that limit. When splitting, it is best to work at the sentence level (roughly splitting on `\n`) and to keep some overlapping content between chunks to improve retrieval accuracy.

```python
def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
    chunk_text = []

    curr_len = 0
    curr_chunk = ''

    token_len = max_token_len - cover_content
    lines = text.splitlines()  # Assume the text is split into lines by newline characters

    for line in lines:
        # Keep inner spaces; only strip leading and trailing whitespace
        line = line.strip()
        line_len = len(enc.encode(line))
        
        if line_len > max_token_len:
            # If a single line already exceeds the limit, split it into multiple chunks
            # First save the current chunk (if it has content)
            if curr_chunk:
                chunk_text.append(curr_chunk)
                curr_chunk = ''
                curr_len = 0
            
            # Split the long line by token length
            line_tokens = enc.encode(line)
            num_chunks = (len(line_tokens) + token_len - 1) // token_len
            
            for i in range(num_chunks):
                start_token = i * token_len
                end_token = min(start_token + token_len, len(line_tokens))
                
                # Decode the token slice back into text
                chunk_tokens = line_tokens[start_token:end_token]
                chunk_part = enc.decode(chunk_tokens)
                
                # Add overlap content (except for the first chunk)
                if i > 0 and chunk_text:
                    prev_chunk = chunk_text[-1]
                    cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                    chunk_part = cover_part + chunk_part
                
                chunk_text.append(chunk_part)
            
            # Reset the current chunk state
            curr_chunk = ''
            curr_len = 0
            
        elif curr_len + line_len + 1 <= token_len:  # +1 for newline
            # The current line fits into the current chunk
            if curr_chunk:
                curr_chunk += '\n'
                curr_len += 1
            curr_chunk += line
            curr_len += line_len
        else:
            # The current line does not fit into the current chunk; start a new chunk
            if curr_chunk:
                chunk_text.append(curr_chunk)
            
            # Start a new chunk and add overlap content
            if chunk_text:
                prev_chunk = chunk_text[-1]
                cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                curr_chunk = cover_part + '\n' + line
                curr_len = len(enc.encode(cover_part)) + 1 + line_len
            else:
                curr_chunk = line
                curr_len = line_len

    # Add the final chunk (if it has content)
    if curr_chunk:
        chunk_text.append(curr_chunk)

    return chunk_text

```

#### Step 3: Vectorization

First, let's implement a vectorization class, which is the foundation of the RAG architecture. The vectorization class is mainly used to turn document chunks into vectors, mapping a piece of text to a single vector.

We start by defining a `BaseEmbeddings` base class. That way, when we want to use a different model, we only need to inherit from this base class and modify it as needed, which makes the code easy to extend.

```python
class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        """
        Initialize the embedding base class
        Args:
            path (str): Path to the model or data
            is_api (bool): Whether to use an API. True means an online API service, False means a local model
        """
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        """
        Get the embedding vector representation of a text
        Args:
            text (str): Input text
            model (str): Name of the model to use
        Returns:
            List[float]: Embedding vector of the text
        Raises:
            NotImplementedError: This method must be implemented in a subclass
        """
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        Compute the cosine similarity between two vectors
        Args:
            vector1 (List[float]): The first vector
            vector2 (List[float]): The second vector
        Returns:
            float: Cosine similarity of the two vectors, in the range [-1,1]
        """
        # Convert the input lists to numpy arrays with dtype float32
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)

        # Check whether the vectors contain infinity or NaN values
        if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
            return 0.0

        # Compute the dot product of the vectors
        dot_product = np.dot(v1, v2)
        # Compute the norms (lengths) of the vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # Compute the denominator (the product of the two norms)
        magnitude = norm_v1 * norm_v2
        # Handle the special case where the denominator is 0
        if magnitude == 0:
            return 0.0
            
        # Return the cosine similarity
        return dot_product / magnitude
```

The `BaseEmbeddings` base class has two main methods: `get_embedding` and `cosine_similarity`. `get_embedding` gets the vector representation of a text, and `cosine_similarity` computes the cosine similarity between two vectors. When the class is initialized, we set the model path and whether it is an API model; for example, using OpenAI's Embedding API requires setting `self.is_api=True`.

A class that inherits from `BaseEmbeddings` only needs to implement the `get_embedding` method; the `cosine_similarity` method is inherited. This is the benefit of writing a base class.

```python
class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            self.client = OpenAI()
            # Get the SiliconFlow API key from an environment variable
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            # Get the SiliconFlow base URL from an environment variable
            self.client.base_url = os.getenv("OPENAI_BASE_URL")
    
    def get_embedding(self, text: str, model: str = "BAAI/bge-m3") -> List[float]:
        """
        By default this uses SiliconFlow's free embedding model BAAI/bge-m3
        """
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError
```

> Note: By default we use the [SiliconFlow LLM API service platform](https://cloud.siliconflow.cn/i/ybUFvmqK), which is accessible to users in mainland China.


#### Step 4: Database and Vector Retrieval

Once document splitting and the embedding model are in place, we need to design a vector database to store the document chunks and their vector representations, as well as a retrieval module that retrieves relevant document chunks for a given query.

The vector database provides the following functions:

- `persist`: persist the database to disk.
- `load_vector`: load the database from local storage.
- `get_vector`: get the vector representations of the documents.
- `query`: retrieve document chunks relevant to a question.

The full code can be found in the ***[/VectorBase.py](./RAG/VectorBase.py)*** file.

```python
class VectorStore:
    def __init__(self, document: List[str] = ['']) -> None:
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        # Get the vector representations of the documents
        pass

    def persist(self, path: str = 'storage'):
        # Persist the database to disk
        pass

    def load_vector(self, path: str = 'storage'):
        # Load the database from local storage
        pass

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        # Retrieve document chunks relevant to the question
        pass
```

The `query` method vectorizes the user's question, then retrieves relevant document chunks from the database and returns them.

```python
def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
    query_vector = EmbeddingModel.get_embedding(query)
    result = np.array([self.get_similarity(query_vector, vector) for vector in self.vectors])
    return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()
```

#### Step 5: The LLM Module

Next comes the LLM module, which answers the user's question based on the retrieved documents.

We first implement a base class so that other models can easily be added later.

```python
class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass
```

`BaseModel` has two methods: `chat` and `load_model`. Open-source models run locally need to implement `load_model`, while API models do not. Here we again use the SiliconFlow LLM API service platform, which is accessible to users in mainland China. The advantage of using an API service is that users don't need local compute resources, which greatly lowers the barrier to entry for learners.

```python
from openai import OpenAI

class OpenAIChat(BaseModel):
    def __init__(self, model: str = "Qwen/Qwen2.5-32B-Instruct") -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")   
        client.base_url = os.getenv("OPENAI_BASE_URL")
        history.append({'role': 'user', 'content': RAG_PROMPT_TEMPLATE.format(question=prompt, context=content)})
        response = client.chat.completions.create(
                model=self.model,
                messages=history,
                max_tokens=2048,
                temperature=0.1
            )
        return response.choices[0].message.content

```

We design a prompt dedicated to RAG, as follows:

```python
RAG_PROMPT_TEMPLATE="""
Use the context below to answer the user's question. If you don't know the answer, say that you don't know. Always answer in English.
Question: {question}
Context you may refer to:
···
{context}
···
If the given context does not allow you to answer, reply that the database does not contain this content and that you don't know.
Helpful answer:
"""
```

And with that, we can use the InternLM2 model to do RAG!

#### Step 6: Tiny-RAG Demo

Next, let's take a look at the Tiny-RAG demo!

```python
from VectorBase import VectorStore
from utils import ReadFiles
from LLM import OpenAIChat
from Embeddings import OpenAIEmbedding

# No database saved yet
docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150) # Get the contents of all files in the data directory and split them
vector = VectorStore(docs)
embedding = OpenAIEmbedding() # Create the EmbeddingModel
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='storage') # Save the vectors and document contents to the storage directory; next time you can load the local database directly

# vector.load_vector('./storage') # Load the local database

question = 'What is the principle behind RAG?'

content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
chat = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
print(chat.chat(question, [], content))
```

You can also load an already-processed database from local storage:

```python
from VectorBase import VectorStore
from utils import ReadFiles
from LLM import OpenAIChat
from Embeddings import OpenAIEmbedding

# After the database has been saved
vector = VectorStore()

vector.load_vector('./storage') # Load the local database

question = 'What is the principle behind RAG?'

embedding = ZhipuEmbedding() # Create the EmbeddingModel

content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
chat = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
print(chat.chat(question, [], content))
```

> Note: All the code for Section 7.2 can be found in [Happy-LLM Chapter7 RAG](https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter7/RAG).

## 7.3 Agent

### 7.3.1 What Is an LLM Agent?

Simply put, an LLM agent is a system that uses an LLM as its core "brain" and gives it the ability to plan autonomously, remember, and use tools. It no longer just passively responds to the user's prompt; instead, it can:

1. Goal Understanding: Take in a relatively complex or high-level goal (for example, "Help me plan a weekend trip to Beijing and book the flights and hotel").
2. Planning: Break the big goal down into a series of small, executable steps (for example, "search for Beijing attractions," "check the weather," "compare flight prices," "find a suitable hotel," "call the booking API," and so on).
3. Memory: Have short-term memory (remembering the context of the current task) and long-term memory (learning from and retrieving information from past interactions or external knowledge bases).
4. Tool Use: Call external APIs, plugins, or code execution environments to obtain information (e.g., search engines, databases), take actions (e.g., send emails, book services), or perform calculations.
5. Reflection & Iteration: (In more advanced agents) evaluate its own actions and results, learn from them, and adjust subsequent plans.

A traditional LLM is like a very knowledgeable librarian who can only offer theory, while an LLM agent is more like an all-round personal assistant: it not only knows a lot, but can also run errands and get things done, and can even proactively work out the best plan.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/7-images/7-3-Agent工作原理.png" alt="alt text" width="90%">
    <p>Figure 7.7 How an agent works</p>
</div>

By combining the powerful language understanding and generation abilities of large language models with key modules such as planning, memory, and tool use, LLM agents achieve a level of autonomy and complex-task handling that goes beyond traditional LLMs. This ability gives LLM agents broad application potential in many vertical domains (such as law, medicine, and finance). Figure 7.7 shows how an agent works.

### 7.3.2 Types of LLM Agents

Although the concept of the LLM agent is still evolving rapidly, we can roughly divide agents into several categories based on their design philosophy and capability focus:

Task-Oriented Agents:
- Characteristics: Focus on completing well-defined tasks in a specific domain, such as customer service, code generation, or data analysis.
- How they work: They usually have a predefined workflow and a specific set of tools they can call. The LLM is mainly responsible for understanding user intent, filling task slots, generating responses, or calling the appropriate tools.
- Examples: A chatbot dedicated to booking restaurants, or a coding assistant (some advanced features of GitHub Copilot, for example, show agent-like characteristics).

Planning & Reasoning Agents:
- Characteristics: Emphasize the ability to autonomously decompose complex tasks, make multi-step plans, and adjust based on feedback from the environment. They usually require stronger reasoning abilities.
- How they work: They often adopt a specific thinking framework such as ReAct (Reason+Act), in which the model first "thinks" (Reasoning) to analyze the current situation and the action required, then performs an "action" (Action) by calling a tool, and then starts the next round of thinking based on the result the tool returns. Prompt engineering techniques such as Chain-of-Thought (CoT) also underpin their reasoning.
- Examples: A research agent that must combine web search, a calculator, database queries, and other tools to answer complex questions, or an agent that can autonomously complete a task like "write a report on topic XX and include relevant data charts."

Multi-Agent Systems:
- Characteristics: Multiple agents with different roles or abilities work together to achieve a larger goal.
- How they work: Agents can communicate, collaborate, debate, or even compete with each other. For example, one agent is responsible for planning, one for execution, and one for review.
- Examples: Simulating a software development team (a product manager agent, a programmer agent, and a tester agent) to automatically generate and test code; simulating a company's organizational structure to produce a business plan. Frameworks such as AutoGen and ChatDev support building such systems.

Exploration & Learning Agents:
- Characteristics: These agents not only carry out tasks but also actively learn new knowledge and skills or optimize their own strategies while interacting with the environment, similar to the notion of an agent in reinforcement learning.
- How they work: They may include more sophisticated memory and reflection mechanisms, and can adjust future plans and actions based on successful or failed experiences.
- Examples: An agent that can autonomously explore an unfamiliar software environment and learn how to operate it, or an agent that keeps improving its strategy while playing a game.

### 7.3.3 Building a Tiny-Agent by Hand

Let's use the `openai` library and its `tool_calls` feature to build a Tiny-Agent by hand. This agent is a simple task-oriented agent that can answer some simple questions based on the user's input.

The final result is shown in Figure 7.8:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/7-images/7-3-tinyagent-example.png" style="width: 100%;">
    <p>Figure 7.8 Example of the result</p>
</div>

#### Step 1 : Initialize the Client and Model

First, we need a client that can call an LLM. Here we use the `openai` library and configure it to point to a service endpoint compatible with the OpenAI API, such as [SiliconFlow](https://cloud.siliconflow.cn/i/ybUFvmqK). We also specify the model to use, such as `Qwen/Qwen2.5-32B-Instruct`.

```python
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key="YOUR_API_KEY",  # Replace with your API Key
    base_url="https://api.siliconflow.cn/v1", # Use SiliconFlow's API address
)

# Specify the model name
model_name = "Qwen/Qwen2.5-32B-Instruct"
```

> **Note:** You need to replace `YOUR_API_KEY` with a valid API key obtained from [SiliconFlow](https://cloud.siliconflow.cn/i/ybUFvmqK) or another provider.

#### Step 2: Define the Tool Functions

We define the tool functions the agent can use in the `src/tools.py` file. Each function needs a clear docstring describing what it does and its parameters, because the docstring will be used to automatically generate the tool's JSON Schema.

```python
# src/tools.py
from datetime import datetime

# Get the current date and time
def get_current_datetime() -> str:
    """
    Get the current date and time.
    :return: A string representation of the current date and time.
    """
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime

def count_letter_in_string(a: str, b: str):
    """
    Count how many times a letter appears in a string.
    :param a: The string to search.
    :param b: The letter to count.
    :return: The number of times the letter appears in the string.
    """
    return str(a.count(b))

def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for the given query and return summaries of the top three pages.
    :param query: The query string to search for.
    :return: A string containing the summaries of the top three pages.
    """
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:  # Take the first three page titles
        try:
            # Use the wikipedia module's page function to get the Wikipedia page object for the given title.
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            # Get the page summary
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
                wikipedia.exceptions.PageError,
                wikipedia.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No suitable results were found on Wikipedia"
    return "\n\n".join(summaries)
# ... (there may be other tool functions)
```

For the OpenAI API to understand these tools, we need to convert them into a specific JSON Schema format. This can be done with the `function_to_json` helper function in `src/utils.py`.

```python
# src/utils.py (excerpt)
import inspect

def function_to_json(func) -> dict:
    # ... (implementation details)
    # Return a dictionary that conforms to the OpenAI tool schema
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
```

#### Step 3: Build the Agent Class

We define the `Agent` class in the `src/core.py` file. This class is responsible for managing the conversation history, calling the OpenAI API, handling tool-call requests, and executing the tool functions.

```python
# src/core.py (excerpt)
from openai import OpenAI
import json
from typing import List, Dict, Any
from utils import function_to_json
# Import the tool functions defined earlier
from tools import get_current_datetime, add, compare, count_letter_in_string

SYSTEM_PROMPT = """
You are an AI assistant called "No Scallions, Ginger or Garlic". Your output should be in the same language as the user's.
When the user's question requires a tool, you can call the appropriate tool function from the list of tools provided.
"""

class Agent:
    def __init__(self, client: OpenAI, model: str = "Qwen/Qwen2.5-32B-Instruct", tools: List=[], verbose : bool = True):
        self.client = client
        self.tools = tools
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSREM_PROMPT},
        ]
        self.verbose = verbose

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        # Get the JSON schemas of all tools
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        # Handle a tool call
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        function_id = tool_call.id

        function_call_content = eval(f"{function_name}(**{function_args})")

        return {
            "role": "tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt) -> str:

        self.messages.append({"role": "user", "content": prompt})

        # Get the model's completion response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )
        
        # Check whether the model called a tool        
        if response.choices[0].message.tool_calls:
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            # Handle the tool calls
            tool_list = []
            for tool_call in response.choices[0].message.tool_calls:
                # Handle the tool call and add the result to the message list
                self.messages.append(self.handle_tool_call(tool_call))
                tool_list.append([tool_call.function.name, tool_call.function.arguments])
            if self.verbose:
                print("Calling tools:", response.choices[0].message.content, tool_list)
            # Get the model's completion response again, this time including the tool call results
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )

        # Add the model's completion response to the message list
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
```

The agent's workflow is as follows:

1.  Receive the user's input.
2.  Call the LLM (such as Qwen) and tell it which tools are available, along with their schemas.
3.  If the model decides to call a tool, the agent parses the request and executes the corresponding Python function.
4.  The agent returns the tool's execution result to the model.
5.  The model generates the final reply based on the tool result.
6.  The agent returns the final reply to the user.

Figure 7.9 shows how the agent calls tools:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/7-images/7-3-Tiny_Agent.jpg" alt="alt text" width="80%">
    <p>Figure 7.9 Agent workflow</p>
</div>

#### Step 4: Run the Agent

Now we can instantiate and run the agent. The `if __name__ == "__main__":` section of `demo.py` provides a simple command-line interaction example.

```python
# demo.py (excerpt)
if __name__ == "__main__":
    client = OpenAI(
        api_key="YOUR_API_KEY", # Replace with your API Key
        base_url="https://api.siliconflow.cn/v1",
    )

    # Create an Agent instance, passing in the client, model name, and list of tool functions
    agent = Agent(
        client=client,
        model="Qwen/Qwen2.5-32B-Instruct",
        tools=[get_current_datetime, add, compare, count_letter_in_string],
        verbose=True # Set to True to see tool-call information
    )

    # Start the interactive conversation loop
    while True:
        # Use colored output to distinguish user input from the AI's answers
        prompt = input("\033[94mUser: \033[0m")  # Show the user input prompt in blue
        if prompt.lower() == "exit":
            break
        response = agent.get_completion(prompt)
        print("\033[92mAssistant: \033[0m", response)  # Show the AI assistant's answer in green
```

**Example interaction:**

```bash
User: Hello
Assistant:  Hello! How can I help you?
User: Which is bigger, 9.12 or 9 .2?
Calling tools: ['compare']
Assistant:  9.2 is bigger than 9.12.
User: Why?
Assistant:  When we compare 9.12 and 9.2, we can think of them as 9.12 and 9.20. Since 9.20 has a 0 in the second decimal place while 9.12 has a 2 in the second decimal place, and the first decimal places are equal, 9.20 (that is, 9.2) is greater than 9.12. Therefore, 9.2 is bigger than 9.12.
User: How many r's are in strawberry?
Calling tools: ['count_letter_in_string']
Assistant:  There are 3 letter 'r's in the word "strawberry".
User: Are you sure?
Calling tools: ['count_letter_in_string']
Assistant:  Yes, I'm sure. The word "strawberry" really does contain 3 letter 'r's. Let's confirm once more: the letter 'r' does appear 3 times in "strawberry".
User: OK, you're grate. What time is it now?
Calling tools: ['get_current_datetime']
Assistant:  The current time is 17:01:33 on April 26, 2025. By the way, I noticed you wrote "you're grate", which seems to be a typo. If you have any other questions or need further help, please let me know!
User: exit
```

In addition, we have prepared a Streamlit app that you can run locally to showcase the agent's capabilities. Run it with `streamlit run web_demo.py`. The agent in action is shown below.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/7-images/7-3-streamlit-demo.png" alt="alt text" width="80%">
    <p>Figure 7.10 Streamlit Demo</p>
</div>


**References**

[1] Hugging Face. (2023). *Open LLM Leaderboard: A benchmarking platform for open-source large language models*. https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard  

[2] awacke1. (2023). *LMSYS Chatbot Arena Leaderboard: An arena-style evaluation platform for large language models*. https://huggingface.co/spaces/awacke1/lmsys-chatbot-arena-leaderboard  

[3] OpenCompass Team. (2023). *OpenCompass: A unified evaluation platform for large language models*. https://rank.opencompass.org.cn/home  

[4] OpenCompass Finance Leaderboard Team. (2024). *CFBENCHMARK: An LLM leaderboard for the financial domain*. https://specialist.opencompass.org.cn/CFBenchmark  

[5] OpenCompass Safety Leaderboard Team. (2024). *Flames: An LLM safety leaderboard*. https://flames.opencompass.org.cn/leaderboard  

[6] OpenCompass General-Knowledge Leaderboard Team. (2024). *BotChat: Evaluating the general conversational ability of LLMs*. https://botchat.opencompass.org.cn/  

[7] OpenCompass Law Leaderboard Team. (2024). *LawBench: Evaluating LLMs in the legal domain*. https://lawbench.opencompass.org.cn/leaderboard  

[8] OpenCompass Medical Leaderboard Team. (2024). *MedBench: Evaluating LLMs in the medical domain*. https://medbench.opencompass.org.cn/leaderboard  

[9] Zhi Jing, Yongye Su, and Yikun Han. (2024). *When Large Language Models Meet Vector Databases: A Survey.* arXiv preprint arXiv:2402.01763.

[10] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey.* arXiv preprint arXiv:2312.10997.

[11] Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md Rizwan Parvez, and Graham Neubig. (2023). *Learning to Filter Context for Retrieval-Augmented Generation.* arXiv preprint arXiv:2311.08377.

[12] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. (2023). *In-Context Retrieval-Augmented Language Models.* arXiv preprint arXiv:2302.00083.
