import wikipedia
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np



# Steps
# Step 1: First we retrieve the external knowledge base ( Wikipedia ) with the user's choice topic

def get_wikipedia_content(topic):
    try:
        page = wikipedia.page(topic)
        return page.content
    except wikipedia.exceptions.PageError:
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        # handle cases where the topic is ambiguous
        print(f"Ambiguous topic. Please be more specific. Options: {e.options}")
        return None

# user input
topic = input("Enter a topic to learn about: ")
document = get_wikipedia_content(topic)

if not document:
    print("Could not retrieve information.")
    exit()

# Use the pretrained tokenizer model to break text into chunks using sentence transformer with context maintaining

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

def split_text(text, chunk_size=256, chunk_overlap=20):
    tokens = tokenizer.tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokenizer.convert_tokens_to_string(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - chunk_overlap
    return chunks

chunks = split_text(document)
print(f"Number of chunks: {len(chunks)}")

# Step 2: Storing and Retrieving Knowledge using Sentence Transformers to convert text into embeddings and store them in a FAISS index
# Sentence Transformer model (all-mpnet-base-v2), which captures their semantic meaning
# Creat a FAISS index with an L2 (Euclidean) distance metric and stored the embeddings in it

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embeddings = embedding_model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Step 3: Querying the RAG Pipeline
# Convert the query into an embedding. Retrieve the top-k most relevant chunks using FAISS

query = input("Ask a question about the topic: ")
query_embedding = embedding_model.encode([query])

k = 3
distances, indices = index.search(np.array(query_embedding), k)
retrieved_chunks = [chunks[i] for i in indices[0]]
print("Retrieved chunks:")
for chunk in retrieved_chunks:
    print("- " + chunk)

# Step 4: Answering the Question with an LLM
# We will use a pre-trained question-answering model to extract the final answer from the retrieved context

qa_model_name = "deepset/roberta-base-squad2"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

context = " ".join(retrieved_chunks)
answer = qa_pipeline(question=query, context=context)
print(f"Answer: {answer['answer']}")


## Summary
# We built a Retrieval-Augmented Generation (RAG) pipeline for LLMs using:

# 1. Wikipedia as an external knowledge base
# 2. Sentence Transformers for embedding generation
# 3. FAISS for fast and efficient retrieval
# 4. Hugging Face’s QA pipeline to extract final answers