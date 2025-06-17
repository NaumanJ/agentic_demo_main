import os
import pandas as pd
#import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
#import torch
import logging
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
from neo4j_knowledge_graph import Neo4jKnowledgeGraph
from milvus_client import MilvusClient
import torch
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

document_repo_path = 'data/document_repo/smart_hospitality/'
txt_directory = document_repo_path + 'administrator/txt'

model_name = "EleutherAI/gpt-neo-2.7B"
# model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



##############################################################
def ensure_embedding_dimension(embedding, target_dim=768):
    if len(embedding) > target_dim:
        return embedding[:target_dim]
    elif len(embedding) < target_dim:
        return embedding + [0] * (target_dim - len(embedding))
    return embedding

def truncate_string(string, max_length=65535):
    if len(string) > max_length:
        logging.warning(f"Truncating string of length {len(string)} to {max_length}")
        return string[:max_length-1]
    return string

def ensure_consistent_data_types(documents):
    for doc in documents:
        if not isinstance(doc["content"], str):
            raise TypeError(f"Content must be a string, but got {type(doc['content'])}")
        # if not isinstance(doc["embedding"], list) or not all(isinstance(e, float) for e in doc["embedding"]):
        #     raise TypeError("Embeddings must be a list of floats.")

def ensure_floats(embeddings):
    """
    Ensures that all values in the embeddings are floats.
    Args:
        embeddings (list of list): List of embeddings (each embedding is a list).
    Returns:
        List of embeddings with float values.
    """
    return [[float(value) for value in embedding] for embedding in embeddings]
#################################################################


# Load Sentence Transformer for embeddings
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Parse TXT documents
def parse_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return {'title': os.path.basename(file_path), 'content': text}

# Parse documents from a directory
def parse_documents(directory):
    documents = []
    for root, dirs, files in os.walk(directory):
        logging.info(f"Parsing directory: {root}")
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.txt'):
                documents.append(parse_text_file(file_path))
    return documents

####################################################################

# Insert documents into Milvus
def insert_documents_to_milvus_R(collection, documents):

    ensure_consistent_data_types(documents)

    truncated_contents = [truncate_string(doc["content"]) for doc in documents]

    embeddings = [tokenizer.encode(doc["content"], max_length=2048, truncation=True, return_tensors="pt").to(device).tolist()[0] for doc in documents]
    # embeddings = tokenizer.decode(embeddings)

    embeddings = [ensure_embedding_dimension(embedding) for embedding in embeddings]

    # embeddings = ensure_floats(embeddings)

    print(embeddings)

    for embedding in embeddings:
        if len(embedding) != 768:
            # print (f"Embedding dimension mismatch: {len(embedding)}")
            raise ValueError(f"Embedding dimension mismatch: {len(embedding)}")

    data = {
        "id": [i for i in range(len(documents))],
        "embedding": embeddings,
        # "content": [doc["content"] for doc in documents]
        "content": truncated_contents
    }

    print(data)

    collection.insert([ data["id"], data["embedding"], data["content"]])
    # collection.insert([ data["embedding"], data["content"]])
    # collection.flush()
    print(f"Inserted {len(documents)} documents into Milvus.")

####################################################################

# Define the Milvus collection schema
def init_milvus_R(collection_name="html_document_embeddings"): #html_document_embeddings

    connections.connect("default", host="localhost", port="19530")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535) # 2048
    ]
    schema = CollectionSchema(fields, description="Document Embeddings") # HTML document embeddings

    # Create the collection if it doesn't exist
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(
            field_name="embedding",
            index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}} # IVF_FLAT,  L2, 128
        )
    else:
        collection = Collection(name=collection_name)

    collection.load()
    return collection

####################################################################


# Main function
def main():
    # Initialize Neo4j
    graph = Neo4jKnowledgeGraph('bolt://localhost:7687', 'neo4j', 'password')

    # Parse TXT documents
    documents = parse_documents(txt_directory)

    local_docs = []

    # Index in Neo4j
    logging.info(f"Starting Neo4j indexing process for directory: {txt_directory}")
    for doc in documents:
        title = doc.get('title', 'Untitled')
        content = doc.get('content', '')
        graph.create_document_node(title, content)
        logging.info(f"Indexed document in Neo4j: {title}")
        local_docs.append(doc)

    # Initialize Milvus collection and embedding model
    milvus_client = MilvusClient()
    milvus_embedding_model = milvus_client.get_embedding_model()

    collection = init_milvus_R()
    batch_size = 1 # Number of documents to process at a time
    for i in range(0, len(local_docs), batch_size):
        # Get a batch of 10 documents
        batch = local_docs[i:i + batch_size]
        insert_documents_to_milvus_R(collection, batch)


    graph.close()
    logging.info("Indexing completed.")

if __name__ == "__main__":
    main()
