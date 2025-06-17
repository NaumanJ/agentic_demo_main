import os
import pandas as pd
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import logging
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from InstructorEmbedding import INSTRUCTOR  # For generating embeddings
import nltk
from nltk.corpus import stopwords
from nltk import ngrams, pos_tag, word_tokenize
import requests
from urllib.parse import urljoin, urlparse
from pymilvus import utility
import wikipedia
from transformers import LlamaTokenizer, LlamaForCausalLM


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load LLaMA-2 Model
def load_llama_model():
    # Re-download the model
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Use the specified LLaMA-2 model
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    return model, tokenizer

# Initialize Milvus for embedding storage
def init_milvus():
    connections.connect("default", host="localhost", port="19530")
    collection_name = "document_embeddings"

    # Define the schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1024)
    ]
    schema = CollectionSchema(fields=fields, description="Document Embeddings")

    # Create the collection if it does not exist
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
    else:
        collection = Collection(name=collection_name)

    # Create an index if one does not exist
    if not collection.has_index():
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
        )
    
    collection.load()
    return collection

# Initialize embedding model
from sentence_transformers import SentenceTransformer

def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Fetch HTML content from URL
def fetch_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        logging.info(f"Successfully fetched content from {url}")
        return response.text
    else:
        logging.error(f"Failed to fetch content from {url} - Status Code: {response.status_code}")
        return None

# Extract valid internal links from HTML content
def extract_links(html_content, base_url):
    soup = BeautifulSoup(html_content, "html.parser")
    links = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if href.startswith("/wiki/") and ":" not in href:
            full_url = urljoin(base_url, href)
            links.append(full_url)

    logging.info(f"Extracted {len(links)} links from the page.")
    return links

# Fetch and save content from each link
def fetch_corpus_from_links(links, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for link in links:
        try:
            response = requests.get(link)
            if response.status_code == 200:
                page_name = urlparse(link).path.split('/')[-1]
                file_path = os.path.join(save_dir, f"{page_name}.html")
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(response.text)
                logging.info(f"Saved content of {link} to {file_path}")
            else:
                logging.warning(f"Failed to fetch {link} - Status Code: {response.status_code}")
        except Exception as e:
            logging.error(f"Error fetching {link}: {e}")

# Parse different types of documents
def parse_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    for element in soup(['script', 'style', 'noscript', 'iframe', 'meta', 'link']):
        element.extract()
    text_content = soup.get_text(separator=' ', strip=True)
    return {'title': os.path.basename(file_path), 'content': text_content}

def parse_pdf(file_path):
    with fitz.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return {'title': os.path.basename(file_path), 'content': text}

def parse_xls(file_path):
    df = pd.read_excel(file_path)
    records = df.to_dict(orient='records')
    return {'title': os.path.basename(file_path), 'content': str(records)}

def parse_documents(directory):
    documents = []
    for root, dirs, files in os.walk(directory):
        logging.info(f"Parsing directory: {root}")
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.html'):
                documents.append(parse_html(file_path))
            elif file.endswith('.pdf'):
                documents.append(parse_pdf(file_path))
            elif file.endswith('.xls') or file.endswith('.xlsx'):
                documents.append(parse_xls(file_path))
    return documents

# Extract key phrases from text
def extract_key_phrases(text, num_words=2):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    proper_nouns = [word for word, tag in pos_tags if tag in ('NNP', 'NNPS') and word.lower() not in stop_words]
    phrases = [' '.join(gram) for gram in ngrams(proper_nouns, num_words)]
    return phrases

# Insert documents into Milvus
def insert_documents_to_milvus(collection, documents, embedding_model):
    try:
        # Generate embeddings
        embeddings = embedding_model.encode([doc['content'] for doc in documents])
        embeddings_list = [embedding.tolist() for embedding in embeddings]

        # Ensure content is a string, not a list
        content_list = []
        for doc in documents:
            if isinstance(doc['content'], str):
                content_list.append(doc['content'])
            else:
                logging.error(f"Content for {doc['title']} is not a string: {doc['content']}")

        # Prepare the data to match the Milvus schema
        if len(content_list) != len(embeddings_list):
            raise ValueError("Mismatch between content list and embeddings list length.")

        # Insert into Milvus
        collection.insert([{"content": content_list, "embedding": embeddings_list}])
        collection.flush()
        logging.info("Data inserted successfully into Milvus.")
    except Exception as e:
        logging.error(f"Error inserting data into Milvus: {e}")

# Main function
def main():
    # Initialize variables
    directory = '/app/document_repo/sports/soccer/txt'
    html_data = '/app/document_repo/sports/soccer/html'

    # Initialize Milvus collection and embedding model
    collection = init_milvus()
    embedding_model = get_embedding_model()

    # Load and parse local documents
    local_docs = []
    documents = parse_documents(html_data)
    local_docs.extend(documents)

    # Insert parsed documents into Milvus
    insert_documents_to_milvus(collection, local_docs, embedding_model)

    # Load LLaMA-2 model and tokenizer
    logging.info("Loading LLaMA-2 model and tokenizer.")
    try:
        llm_model, tokenizer = load_llama_model()
        logging.info("LLaMA-2 model and tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading LLaMA-2 model and tokenizer: {e}")
        return

if __name__ == "__main__":
    main()