import os
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from bs4 import BeautifulSoup

# from pymilvus import MilvusClient

# client = MilvusClient("milvus_demo.db")

# if client.has_collection(collection_name="demo_collection"):

#      client.drop_collection(collection_name="demo_collection")

# client.create_collection(

#      collection_name="demo_collection",

#      dimension=768,  # The vectors we will use in this demo has 768 dimensions

#  )

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")
print("conncted to milvius")

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_model.to(device)

# Define the Milvus collection schema
def init_milvus(collection_name="html_document_embeddings"):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2048)
    ]
    schema = CollectionSchema(fields, description="HTML document embeddings")

    # Create the collection if it doesn't exist
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(
            field_name="embedding",
            index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
        )
    else:
        collection = Collection(name=collection_name)

    collection.load()
    return collection

# Parse the content of an HTML file
def parse_html_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    for element in soup(['script', 'style', 'noscript', 'iframe', 'meta', 'link']):
        element.extract()  # Remove unnecessary elements
    return soup.get_text(separator=' ', strip=True)

# Insert documents into Milvus
def insert_documents_to_milvus(collection, folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".html"):
            file_path = os.path.join(folder_path, file_name)
            content = parse_html_file(file_path)
            documents.append({"title": file_name, "content": content})

    embeddings = [tokenizer.encode(doc["content"], return_tensors="pt").to(device).tolist()[0] for doc in documents]

    data = {
        "id": [i for i in range(len(documents))],
        "embedding": embeddings,
        "content": [doc["content"] for doc in documents]
    }
    collection.insert([data["id"], data["embedding"], data["content"]])
    collection.flush()
    print(f"Inserted {len(documents)} documents into Milvus.")

# Search for relevant documents
def search_in_milvus(collection, query, top_k=5):
    query_embedding = tokenizer.encode(query, return_tensors="pt").to(device).tolist()[0]
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k
    )
    return [result.entity.get("content") for result in results[0]]

# Generate response using GPT-Neo
def generate_response(query, context):
    prompt = f"Question: {query}\n\nRetrieved Context:\n{context}\n\nAnswer:"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main function
if __name__ == "__main__":
    html_folder_path = "./html_pages"
    collection = init_milvus()

    # Index documents
    insert_documents_to_milvus(collection, html_folder_path)

    # Accept input question
    input_text = input("Enter your question: ")

    # Search for relevant documents
    relevant_docs = search_in_milvus(collection, input_text)
    if not relevant_docs:
        print("No relevant documents found.")
        exit()

    # Generate a response based on retrieved context
    retrieved_context = "\n".join(relevant_docs)
    response = generate_response(input_text, retrieved_context)
    print("\nGenerated Response:")
    print(response)
