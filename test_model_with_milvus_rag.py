import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from sentence_transformers import SentenceTransformer

# Load your language model and tokenizer
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_model.to(device)

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Load your embedding model (using Sentence-Transformers for embedding generation)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define the function to add documents to Milvus
def add_documents_to_milvus(filenames, documents, collection_name="my_collection"):
    # Create a schema for the Milvus collection if it does not exist
    fields = [
        FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255, is_primary=True, auto_id=False),  # Primary key
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=embedding_model.get_sentence_embedding_dimension())
    ]
    schema = CollectionSchema(fields, description="Collection for storing text documents and their embeddings")

    # Use utility to check if the collection exists
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(field_name="embeddings", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    else:
        collection = Collection(collection_name)

    # Generate embeddings for the documents
    embeddings = [embedding_model.encode(doc).tolist() for doc in documents]
    
    # Prepare data for insertion (filenames and embeddings)
    data = [
        filenames,  # Filenames as primary keys
        embeddings  # Embeddings
    ]

    # Insert data into the collection
    collection.insert(data)
    collection.load()
    print(f"Successfully added {len(documents)} documents to Milvus.")

# Reading multiple files and preparing the documents
documents_to_add = [
    "cristiano_ronaldo.txt",  
    "fashion_football_cristiano_lionel.txt",  
    "friendlyMatch.txt", 
    "lionel_messi.txt", 
    "soccerMatch.txt", 
    "summary.txt"
]

# Read each file and collect the content
filenames = []
documents_corpus = []
for each_document in documents_to_add:
    try:
        with open('document_repo/sports/soccer/txt/' + each_document, 'r', encoding='utf-8') as file:
            text_document_content = file.read()
            if text_document_content:  # Ensure content is not empty
                filenames.append(each_document)  # Use filename as primary key
                documents_corpus.append(text_document_content)
    except FileNotFoundError:
        print(f"File {each_document} not found. Skipping...")

# Add the documents to Milvus
add_documents_to_milvus(filenames, documents_corpus)

# Example input question
input_text = input("Enter a Question:")

# Function to search in Milvus
def search_in_milvus(question, collection_name="my_collection", top_k=5):
    # Generate the embedding for the question
    question_embedding = embedding_model.encode(question).tolist()
    collection = Collection(collection_name)
    
    # Define search parameters
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    
    # Perform the search
    results = collection.search(
        data=[question_embedding],
        anns_field="embeddings",
        param=search_params,
        limit=top_k,
        expr=None
    )
    
    # Extract and return the relevant text or filename as strings
    relevant_texts = [hit.entity.get("filename") for hit in results[0]]
    
    # Ensure that all items in relevant_texts are strings
    relevant_texts = [str(text) for text in relevant_texts if text is not None]
    
    return relevant_texts

# Search for relevant documents in Milvus
relevant_docs = search_in_milvus(input_text)

# Ensure retrieved_context is a string before joining
if isinstance(relevant_docs, list) and all(isinstance(doc, str) for doc in relevant_docs):
    retrieved_context = "\n".join(relevant_docs)
else:
    retrieved_context = "No relevant documents found."

# Construct the prompt for the language model
prompt = f"Question: {input_text}\n\nRetrieved Context:\n{retrieved_context}\n\nResponse:"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
outputs = llm_model.generate(inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)