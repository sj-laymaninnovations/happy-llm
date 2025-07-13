from VectorBase import VectorStore
from utils import ReadFiles
from LLM import OpenAIChat
from Embeddings import OpenAIEmbedding

# No database saved
docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150) # Obtain all file contents in the data directory and split
vector = VectorStore(docs)
embedding = OpenAIEmbedding() # Create EmbeddingModel
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='storage') # Save the vector and document content to the storage directory, and then use it next time you can directly load the local database

# vector.load_vector('./storage') # Load the local database

question = 'What is the principle of RAG?'

content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
chat = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
print(chat.chat(question, [], content))