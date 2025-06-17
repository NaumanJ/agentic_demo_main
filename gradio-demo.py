import os
import logging
import gradio as gr
from sqlalchemy import create_engine, inspect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

# Sample transcript for context augmentation
transcript_sample = """[Sample Transcript Data from Above]"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize PostgreSQL engine
def init_pg_engine():
    return create_engine('postgresql://postgres:password@localhost:5433/postgres')

# Initialize Neo4j Knowledge Graph
class Neo4jKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def search_documents(self, keywords):
        with self.driver.session() as session:
            result = session.read_transaction(self._search_documents, keywords)
        return result

    @staticmethod
    def _search_documents(tx, keywords):
        query = """
        MATCH (d:Document)
        WHERE ANY(keyword IN $keywords WHERE toLower(d.content) CONTAINS toLower(keyword))
        RETURN d.title AS title, d.content AS content
        """
        result = tx.run(query, keywords=keywords)
        documents = [{"title": record["title"], "content": record["content"]} for record in result]
        return documents

# Load LLM
def load_llm():
    model_name = "EleutherAI/gpt-neo-2.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer


# def load_llm():
#     model_name = "meta-llama/Llama-2-7b-hf"  # Adjust this for the exact version of LLaMA 3.1 7B
#     tokenizer = LlamaTokenizer.from_pretrained(model_name)
#     model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)  # Use FP16 for memory efficiency
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     return model, tokenizer

# Extract keywords from text
def extract_keywords(documents):
    texts = [doc['content'] for doc in documents if 'content' in doc]
    if not texts:
        return []
    combined_text = ' '.join(texts)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    X = vectorizer.fit_transform([combined_text])
    return vectorizer.get_feature_names_out()

# Generate a word cloud
def generate_wordcloud(keywords):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Get schema information from PostgreSQL
def get_postgres_schema(engine):
    inspector = inspect(engine)
    schemas = {}
    for table_name in inspector.get_table_names(schema='public'):
        columns = inspector.get_columns(table_name)
        column_names = [col['name'] for col in columns]
        schemas[table_name] = column_names
    return schemas

# Generate LLM response using the RAG pipeline
def generate_llm_response_with_rag(model, tokenizer, question, schemas, pg_engine, neo4j_docs, max_length=500):
    neo4j_docs_retrieved = retrieve_documents_from_neo4j(neo4j_docs)
    retrieved_context = "\n".join(neo4j_docs_retrieved)
    prompt = (
        f"Question: {question}\n\n"
        f"Schema Information:\n\n"
        f"Retrieved Context:\n{retrieved_context}\n\n"
        f"Generate a detailed answer using the provided context in a humorous tone. Avoid profanity or discriminatory language and do not answer questions on race, illicit matters or illegal substances."
    )
    max_model_length = getattr(model.config, 'max_position_embeddings', 2048)
    tokenized_inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=min(512, max_model_length)
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=tokenized_inputs['input_ids'],
            attention_mask=tokenized_inputs['attention_mask'],
            max_length=min(max_length, max_model_length),
            temperature=0.7,
            max_new_tokens=100,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Retrieve relevant documents from Neo4j
def retrieve_documents_from_neo4j(neo4j_docs):
    return [f"Neo4j Document: {doc['content'][:200]}..." for doc in neo4j_docs]

# Gradio Interface Functions
def chatbot_pipeline(question):
    # Initialize connections and models
    pg_engine = init_pg_engine()
    schemas = get_postgres_schema(pg_engine)
    graph = Neo4jKnowledgeGraph('bolt://localhost:7687', 'neo4j', 'password')
    llm_model, tokenizer = load_llm()

    # Combine transcript and question for context
    combined_text = transcript_sample + '\n' + question
    keywords = extract_keywords([{"content": combined_text}])

    # Search Neo4j for documents
    neo4j_docs = graph.search_documents(keywords)

    # Generate LLM response using RAG
    response = generate_llm_response_with_rag(llm_model, tokenizer, question, schemas, pg_engine, neo4j_docs)

    # Generate word cloud for extracted keywords
    wordcloud_img = generate_wordcloud(keywords)

    return response, process_image(wordcloud_img)
#f"data:image/png;base64,{wordcloud_img}"

import gradio as gr
from PIL import Image
from io import BytesIO
import base64

def process_image(base64_str):
    # Process the base64 string here
    # Decode base64 string into an image
    image_data = base64.b64decode(base64_str.split(",")[1])
    image = Image.open(BytesIO(image_data))

    # Save the image
    image.save("wordcloud.png")

    # Return the saved image path
    return "wordcloud.png"

# Example Gradio Interface
iface = gr.Interface(fn=process_image, inputs=gr.Textbox(label="Base64 Image"), outputs=gr.Image())
iface.launch()

# Gradio UI Definition
def gradio_app():
    interface = gr.Interface(
        fn=chatbot_pipeline,
        inputs=["text"],
        outputs=["text", gr.Image()],
        title="Sports Chatbot using Knowledge Graphs",
        description="Ask a question about sports and receive responses generated using a Retrieval-Augmented Generation (RAG) pipeline, leveraging both PostgreSQL and Neo4j for contextual knowledge.",
    )
    interface.launch()

# Run the Gradio app
if __name__ == "__main__":
    gradio_app()