import os
import logging
import shutil
import tempfile
import traceback
import json
import time

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from starlette.status import HTTP_401_UNAUTHORIZED
from passlib.context import CryptContext

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from sqlalchemy import create_engine, inspect
from neo4j import GraphDatabase
import nltk
from nltk.corpus import stopwords
import re

import streamlit as st
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
import uvicorn
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import sounddevice as sd
#import speech_recognition as sr
from gtts import gTTS
from pyvis.network import Network
import streamlit.components.v1 as components
from pydub import AudioSegment

# Custom imports
from milvus_client import MilvusClient
import postgres_client
import neo4j_knowledge_graph
from voice_recognition import VoiceRecognition
from openai import OpenAI
from video_annotation_loader import AnnotationLoader
import video_annotation_manager
import bitsandbytes
import groq


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the FastAPI app
app = FastAPI(title="Sports Chatbot API")

# Authentication and password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBasic()

# Hardcoded credentials for simplicity, but you should use environment variables
USERNAME = os.getenv("API_USERNAME", "admin")
PASSWORD_HASH = os.getenv("API_PASSWORD_HASH", pwd_context.hash("password"))  # Hash of the default password
video_dir = "/data/document_repo/tourism/video"
prev_response_document=[]

# Initialize databases
# Sentence Transformer for embeddings
def get_embedding_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Initialize SQLAlchemy engine for PostgreSQL
def init_pg_engine():
    return create_engine('postgresql://postgres:password@localhost:5433/postgres')

# Get schema information from PostgreSQL
def get_postgres_schema(engine):
    inspector = inspect(engine)
    schemas = {}
    for table_name in inspector.get_table_names(schema='public'):
        columns = inspector.get_columns(table_name)
        column_names = [col['name'] for col in columns]
        schemas[table_name] = column_names
    return schemas

def initialize_databases():
    """
    Initialize connections to Neo4j, PostgreSQL, and Milvus.
    
    Returns:
    - graph: The Neo4j knowledge graph object.
    - pg_engine: The SQLAlchemy engine for PostgreSQL.
    - embedding_model: The embedding model for Milvus.
    - collection: The initialized Milvus collection.
    """
    # Initialize Neo4j Knowledge Graph
    logging.info("Initializing Neo4j Knowledge Graph.")
    try:
        graph = neo4j_knowledge_graph.Neo4jKnowledgeGraph('bolt://localhost:7687', 'neo4j', 'password')
        logging.info("Neo4j Knowledge Graph initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing Neo4j Knowledge Graph: {e}")
        stack_trace = traceback.format_exc()
        raise Exception(f"{e}\n{stack_trace}")

    # Initialize PostgreSQL connection and load schema
    logging.info("Initializing PostgreSQL engine and retrieving schema.")
    try:
        pg_engine = init_pg_engine()
        schemas = get_postgres_schema(pg_engine)
        logging.info("PostgreSQL schema retrieved successfully.")
    except Exception as e:
        logging.error(f"Error initializing PostgreSQL engine or retrieving schema: {e}")
        stack_trace = traceback.format_exc()
        raise Exception(f"{e}\n{stack_trace}")

    # Initialize Milvus and embedding model
    logging.info("Initializing Milvus and embedding model.")
    try:
        embedding_model = get_embedding_model()
        collection = MilvusClient.init_milvus()
        logging.info("Milvus and embedding model initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing Milvus or embedding model: {e}")
        stack_trace = traceback.format_exc()
        raise Exception(f"{e}\n{stack_trace}")
    
    # Your existing imports and database configuration
    db_config = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'password',
        'host': 'localhost',
        'port': '5433'
    }

    # Initialize AnnotationLoader
    logging.info("Initializing Postgres AnnotationLoader.")
    try:
        annotation_loader = AnnotationLoader(db_config)
    except Exception as e:
        logging.error(f"Error Initializing Postgres AnnotationLoader: {e}")
        stack_trace = traceback.format_exc()
        raise Exception(f"{e}\n{stack_trace}")
    return graph, pg_engine, embedding_model, collection, annotation_loader

try:
    graph, pg_engine, milvus_embedding_model, milvus_collection, annotation_loader = initialize_databases()
except Exception as e:
    logging.error(f"Error during database initialization: {e}")
    
# Helper function to verify password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)



# Authentication function
def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    if not (credentials.username == USERNAME and verify_password(credentials.password, PASSWORD_HASH)):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Data Models
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# Helper Functions

def extract_keywords_from_text(text):
    words = set(re.findall(r'\b\w+\b', text.lower()))
    common_words = set(stopwords.words('english'))
    keywords = [word for word in words if word not in common_words]
    return keywords

def parse_content_from_neo4j_docs(neo4j_search_results):
    """
    Retrieve relevant documents from Neo4j based on keywords or the provided document context.
    """
    retrieved_docs = []
    for doc in neo4j_search_results:
        retrieved_docs.append(f"Neo4j Document: {doc['content']}")  # Using a snippet of the document
    # Log all document names
    document_names = [doc['title'] for doc in neo4j_search_results]
    logging.info(f"Document names retrieved from Neo4j: {document_names}")
    return retrieved_docs

def extract_keywords(question, max_features=10):
    """
    Extracts keywords from the given question using TfidfVectorizer.
    
    Args:
    - question (str): The input string from which to extract keywords.
    - max_features (int): The maximum number of keywords to extract (default is 10).
    
    Returns:
    - List[str]: A list of extracted keywords.
    """
    if not question.strip():
        return []

    # Initialize TfidfVectorizer to extract keywords, ignoring common English stop words
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    
    # Fit and transform the question into a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform([question])
    
    # Get the feature names (i.e., keywords) and their corresponding scores
    feature_names = vectorizer.get_feature_names_out()
    return list(feature_names)


# LLM Integration Functions

def generate_sql_query(llm_model, tokenizer, question, schemas):
    """Generate a SQL query based on a natural language question and schemas."""
    # Prepare the prompt
    system_prompt = "You are an expert SQL assistant."
    schema_info = ""
    for table_name, columns in schemas.items():
        schema_info += f"\nTable {table_name} has columns: {', '.join(columns)}."
    user_prompt = f"{system_prompt}\n\nGenerate a SQL query to answer the following question: '{question}'.{schema_info}\nSQL Query:"
    
    # Tokenize the prompt
    inputs = tokenizer.encode(user_prompt, return_tensors="pt").to(llm_model.device)
    
    # Generate SQL Query
    outputs = llm_model.generate(
        inputs,
        max_length=inputs.shape[1] + 100,
        temperature=0.0,
        do_sample=False,
        num_return_sequences=1,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the SQL query from the generated text
    sql_query = generated_text[len(user_prompt):].strip()
    # Clean up the SQL query
    sql_query = sql_query.split('\n')[0]  # Take only the first line if multiple lines are generated
    return sql_query

def generate_answer(llm_model, tokenizer, question, sql_result=None, neo4j_result=None):
    """Generate an answer to the user's question based on the query results."""
    # Prepare the prompt
    result_str = ""
    if sql_result:
        result_str += "Relational Data Result:\n" + sql_result + "\n"
    if neo4j_result:
        result_str += "Graph Data Result:\n" + neo4j_result + "\n"
    user_prompt = f"Question: {question}\n\n{result_str}\nProvide a concise and informative answer based on the result.\nAnswer:"
    
    # Tokenize the prompt
    inputs = tokenizer.encode(user_prompt, return_tensors="pt").to(llm_model.device)
    
    # Generate Answer
    outputs = llm_model.generate(
        inputs,
        max_length=inputs.shape[1] + 150,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=1,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the answer from the generated text
    answer = generated_text[len(user_prompt):].strip()
    return answer

def get_llm_response(chat_input, schemas, llm_model, tokenizer, pg_engine, neo4j_graph):
    """Process the user's input and generate a response using LLM, PostgreSQL, and Neo4j."""
    question = chat_input

    # Extract keywords
    keywords = extract_keywords_from_text(question)
    logging.info(f"Extracted Keywords: {keywords}")

    # Generate and execute SQL query
    sql_query = generate_sql_query(llm_model, tokenizer, question, schemas)
    logging.info(f"Generated SQL Query: {sql_query}")

    sql_result = None
    neo4j_result_str = ""

    try:
        with pg_engine.connect() as conn:
            result_set = conn.execute(sql_query)
            rows = result_set.fetchall()
            if rows:
                sql_result = '\n'.join(str(row) for row in rows)
                logging.info(f"SQL Query Result:\n{sql_result}")
            else:
                logging.info("SQL Query returned no results.")
    except Exception as e:
        logging.error(f"Error executing SQL query: {e}")

    # Query Neo4j
    neo4j_result = neo4j_graph.search_documents(keywords)
    if neo4j_result:
        neo4j_result_str = '\n'.join(doc['content'] for doc in neo4j_result)
        logging.info(f"Neo4j Query Result: {neo4j_result_str}")
    else:
        logging.info("Neo4j Query returned no results.")

    # Generate the final answer using LLM
    answer = generate_answer(llm_model, tokenizer, question, sql_result, neo4j_result_str)
    return answer

def generate_groq_llm_response(client, question, combined_docs):
    model_name = "llama3-8b-8192"
    # Ensure the question and context are not empty
    if not question.strip():
        return "Please provide a valid question."
    
    if not combined_docs:
        return "No context available to answer the question."

    # Combine all the documents into a single context
    retrieved_context = "\n".join(combined_docs)
    context = ( f"You are a knowledgeable soccer chatbot application with a video player handy."
                f"The user picks a video from the drop-down list and plays it and display its annotation while gleeaning context."
                f"Your response is in a friendly, engaging, upbeat and enthusiastic manner, as if you are talking to a fellow sports fan."
                f"The user will likely ask a question phrase as text in relation to the video such as players in the video or country teams playing the game etc."
                f"The user may choose not to ask a question about the video and ask general questions the soccer context."
                f"The user may choose not to ask a question about the video and ask general questions the soccer context."
                f"Answer the question in a friendly, engaging, and enthusiastic manner, as if you are talking to a fellow sports fan."
                f"include interesting facts or statistics about the topic or football if relevant."
                f"For a short response or no context or no answer available, respond politely and " 
                f"ask a follow up question based on the context provided to check if the user would like more information."
                f"Avoid repeating the context or question, and keep the tone light and conversational."
                f"Avoid any advise or profanity or discussion about political, illegal or controversial issues"
                f"Remove any html tags or markdown tags from your response to make it easily speakble so it can be converted to speech"
                f"If the context contains users interest or action to a specific video played based on a specific criteria use the context in the response.\n\n"
                f"Context from Files provided is {retrieved_context}\n\n")
    prompt = (
        f"Context: {context}\n\n"
        f"Provide concise answers to the question in 2 sentences or less for the user."
        f"Quote information from this context to complete your answer in 2 sentences as in the  example question and example answer." 
        f"Example question: What luxury accessories does Ronaldo often wear?"
        f"Example answer: Ronaldo is often seen sporting expensive watches and designer sunglasses. His favorite watch brands include Rolex and TAG Heuer, and he owns several custom-made timepieces."
                
    )            
    # Prepare the message content
    messages = [
        {"role": "user", "content": question},
        {"role": "system", "content": prompt}
    ]

    # Check for empty content before making the API request
    if not any(msg["content"].strip() for msg in messages):
        return "Error: Both question and context are empty. Please try again with a valid input."
    try:
        client = groq.Groq(api_key='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')
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        # Collect the streamed response
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        logging.info(f"LLM reponse is {response}")
        return response
    except Exception as e:
        print(f"Error during API request: {e}")
        print("An error occurred while processing your Groq LLM request. Please try again.")


# Neo4j Knowledge Graph Class

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
        documents = []
        for record in result:
            documents.append({"title": record["title"], "content": record["content"]})
        return documents

# Audio Transcription Function

# def transcribe_audio(audio_file_path):
#     # Placeholder implementation
#     # You can integrate with an actual transcription service or library
#     transcription = "This is a placeholder transcription of the audio."
#     return transcription

# def audio_to_text(audio_data, fs=44100):
#     """Convert recorded audio data to text using SpeechRecognition."""
#     recognizer = sr.Recognizer()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
#         sd.write(temp_audio.name, audio_data, fs)
#         with sr.AudioFile(temp_audio.name) as source:
#             audio = recognizer.record(source)
#         try:
#             text = recognizer.recognize_google(audio)
#             return text
#         except sr.UnknownValueError:
#             return "Sorry, I could not understand the audio."
#         except sr.RequestError:
#             return "Error with the Speech Recognition service."

def text_to_speech(text, speed_up_factor=1.2):
    speed_up_factor_sent = speed_up_factor
    """Convert text to speech using gTTS and play it."""
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
        tts.save(temp_audio.name)
    return(temp_audio.name)

    
# Helper LLM functions

import logging
import os
from typing import List, Dict

def generate_llm_response(question: str, graph, annotation_loader, video_dir, milvus_collection, milvus_embedding_model, prev_response_document):
    """
    Generates a response using the LLM based on the input question.
    
    Args:
        question (str): The question to be answered.
        graph: Neo4j graph object for retrieving documents.
        annotation_loader: Object for searching video annotations.
        video_dir (str): Directory containing video files.
        milvus_collection: Milvus collection for document search.
        milvus_embedding_model: Embedding model for Milvus.
        prev_response_document (List[str]): Previous response documents for context.

    Returns:
        str: The generated response from the LLM.
    """
    try:
        # Extract keywords from the question
        logging.info(f"Input question: {question}")
        keywords = extract_keywords(question)
        logging.info(f"Keywords extracted: {keywords}")

        # Handle video search based on keywords
        video_search_results = None
        video_detail_doc = {}
        video_related_keywords = {
            'search', 'videos', 'play', 'playing', 'video', 'mp4', 'avi', 'mov',
            'players', 'soccer', 'match', 'about', 'this', 'explain', 'summarize'
        }
        question_words = set(question.split())

        if question_words.intersection(video_related_keywords):
            logging.info("Invoking video search for key words: {question_words.intersection(video_related_keywords)}")
            video_search_results = annotation_loader.full_text_search(question)
            if video_search_results:
                selected_video = video_search_results[0]['video_file_name']
                video_path = os.path.join(video_dir, selected_video)
                logging.info(f"Selected video: {selected_video}")

                video_detail_doc = {
                    'title': selected_video,
                    'content': (
                        f"Soccer video being played: {selected_video} "
                        f"{video_search_results[0]['annotation']} between "
                        f"{video_search_results[0]['start_duration']} and {video_search_results[0]['end_duration']}."
                    )
                }

        # Retrieve documents from various sources
        neo4j_search_results = graph.get_relevant_documents(keywords, 5)
        neo4j_docs = parse_content_from_neo4j_docs(neo4j_search_results) if neo4j_search_results else []
        milvus_docs = MilvusClient.search_documents_milvus(milvus_collection, question, milvus_embedding_model)
        postgres_docs = []  # Placeholder if you have PostgreSQL document retrieval logic

        # Combine all retrieved documents
        combined_docs = (milvus_docs or []) + (postgres_docs or []) + (neo4j_docs or []) + ([video_detail_doc] if video_detail_doc else [])
        if prev_response_document:
            combined_docs += prev_response_document

    except Exception as e:
        logging.error(f"Error during document retrieval: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving documents.")
    
    try:
        # Load the LLM model and tokenizer
        model, tokenizer = load_nemotron_llm()
        logging.info("Calling LLM to generate response.")

        # Generate the response using the LLM
        response = generate_nemotron_response(model, tokenizer, question, combined_docs)
        logging.info("LLM Response generated successfully.")
        return response
    except Exception as e:
        logging.error(f"Error during response generation: {e}")
        raise HTTPException(status_code=500, detail="Error generating response.")
    

def load_nemotron_llm():
    # Initialize the NVIDIA LLM
    HF_TOKEN="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"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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"  # Replace with your API key
    )
        # Load the model and tokenizer
    #model_name =  "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF" 
    #model_name =   "meta-llama/Llama-3.1-405B-Instruct-FP8" 
    #model_name =  "tiiuae/falcon-40b-instruct"
    model_name = "meta-llama/Llama-2-7b-chat-hf"

# Set up the quantization configuration for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=None
#        llm_int8_has_fp16_weights=False
    )
    # Define memory limits for each GPU
    max_memory = {
        0: "30GiB",  # Adjust to leave headroom on each GPU
        1: "30GiB",
        2: "30GiB",
        3: "30GiB"
    }

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    
    # Load the model with multi-GPU support
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        quantization_config=quantization_config,
        device_map="balanced",
        max_memory=max_memory,  # Specify memory limits per GPU
        offload_folder="offload",  # Optional folder for offloading
        offload_state_dict=True   # Enable offloading state dict to CPU if necessary
    )
    #return client, model, tokenizer
    return model, tokenizer


# Sample transcript to augment the context (if needed)
transcript_sample = """
"""
#model, tokenizer = load_nemotron_llm()
client = groq.Groq(api_key='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')
combined_docs=[]
video_path=""


def generate_nemotron_response(model, tokenizer, question, combined_docs):
    # Ensure the question and context are not empty
    if not question.strip():
        return "Please provide a valid question."
    
    if not combined_docs:
        return "No context available to answer the question."

    # Combine all the documents into a single context
    retrieved_context = "\n".join(combined_docs)
    context = ( f"You are a knowledgeable soccer chatbot application with a video player handy."
                f"The user picks a video from the drop-down list and plays it and display its annotation while gleeaning context."
                f"Your response is in a friendly, engaging, upbeat and enthusiastic manner, as if you are talking to a fellow sports fan."
                f"The user will likely ask a question phrase as text in relation to the video such as players in the video or country teams playing the game etc."
                f"The user may choose not to ask a question about the video and ask general questions the soccer context."
                f"The user may choose not to ask a question about the video and ask general questions the soccer context."
                f"Answer the question in a friendly, engaging, and enthusiastic manner, as if you are talking to a fellow sports fan."
                f"include interesting facts or statistics about the topic or football if relevant."
                f"For a short response or no context or no answer available, respond politely and " 
                f"ask a follow up question based on the context provided to check if the user would like more information."
                f"Avoid repeating the context or question, and keep the tone light and conversational."
                f"Avoid any advise or profanity or discussion about political, illegal or controversial issues"
                f"Remove any html tags or markdown tags from your response to make it easily speakble so it can be converted to speech"
                f"If the context contains users interest or action to a specific video played based on a specific criteria use the context in the response.\n\n"
                f"Context from Files provided is {retrieved_context}\n\n")
    prompt = (
        f"Context: {context}\n\n"
        f"Provide concise answers to the question in 2 sentences or less for the user."
        f"Quote information from this context to complete your answer in 2 sentences as in the  example question and example answer." 
        f"Example question: What luxury accessories does Ronaldo often wear?"
        f"Example answer: Ronaldo is often seen sporting expensive watches and designer sunglasses. His favorite watch brands include Rolex and TAG Heuer, and he owns several custom-made timepieces."
                
    )            
    # Prepare the message content
    messages = [
        {"role": "user", "content": question},
        {"role": "system", "content": prompt}
    ]

    # Check for empty content before making the API request
    if not any(msg["content"].strip() for msg in messages):
        return "Error: Both question and context are empty. Please try again with a valid input."

    try:
        # model.to(device)
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        tokenized_message = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )

        # Generate the response
        # output = model.generate(
        #     **inputs,
        #     max_length=256,
        #     temperature=0.5,
        #     top_p=1.0,
        #     do_sample=True
        # )
        response_token_ids = model.generate(
            tokenized_message['input_ids'].cuda(),
            attention_mask=tokenized_message['attention_mask'].cuda(),
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_tokens = response_token_ids[:, len(tokenized_message['input_ids'][0]):]

        # Decode the output tokens to text
        #response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        logging.info(f"LLM reponse is -->> {response}")
        # Truncate the response to the desired number of sentences
        max_sentences=2
        sentences = response.split(". ")
        response = ". ".join(sentences[:max_sentences]) + ("" if len(sentences) <= max_sentences else response)
        logging.info(f"Shorter truncated LLM reponse is -->> {response}")
        logging.info("Response from local LLM generated successfully")
        return response
    except Exception as e:
        logging.error(f"Error during API request: {e}")
        return "An error occurred while processing your request. Please try again."

# Initialize LLM and Database Connections at Startup

@app.on_event("startup")
def startup_event():
    # Initialize databases
    try:
        start_time = time.time()
        graph, pg_engine, milvus_embedding_model, milvus_collection, annotation_loader = initialize_databases()
        logging.info(f"Database initialization completed in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"Error during database initialization: {e}")
        return

    # Initialize Neo4j Knowledge Graph
    graph = neo4j_knowledge_graph.Neo4jKnowledgeGraph('bolt://localhost:7687', 'neo4j', 'password')

# Shutdown Event to Close Connections

@app.on_event("shutdown")
def shutdown_event():
    global graph
    graph.close()

# api.py

# API Endpoints with Basic Authentication

# Data Models
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest, username: str = Depends(authenticate_user)):
    """
    Accepts a text-based question and returns a generated answer using the LLM.
    Requires basic authentication.
    """
    logging.info("Starting the Sports Tourism Chatbot with PostgreSQL, Neo4j, and Milvus")
    question = request.question
    response_json = {}
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if question:
        try:
            logging.info(f"Input question: {question}")
            keywords = extract_keywords(question)
            logging.info(f"Keywords extracted: {keywords}")

            # Handle video search based on keywords
            video_search_results = None
            video_detail_doc = {}
            video_related_keywords = {
                'search', 'videos', 'play', 'video', 'mp4', 'avi', 'mov',
                'players', 'soccer', 'match', 'about', 'this', 'explain', 'summarize'
            }
            question_words = set(question.split())

            if question_words.intersection(video_related_keywords):
                video_search_results = annotation_loader.search_video_by_annotation(selected_video, question)
                if video_search_results:
                    selected_video = video_search_results[0]['video_file_name']
                    video_path = os.path.join(video_dir, selected_video)
                    st.video(video_path, format="video/mp4", start_time=0)
                    video_detail_doc[selected_video] = video_search_results[0]['annotation']
            
            # Retrieve documents from various sources
            start_doc_time = time.time()
            neo4j_search_results = graph.get_relevant_documents(keywords, 5)
            logging.info(f"Neo4j document retrieval took {time.time() - start_doc_time:.2f} seconds.")
            neo4j_docs = parse_content_from_neo4j_docs(neo4j_search_results) if neo4j_search_results else []
            milvus_docs = MilvusClient.search_documents_milvus(milvus_collection, question, milvus_embedding_model)
            postgres_docs = []  # If you have PostgreSQL retrieval logic
        except Exception as e:
            logging.error(f"Error during document retrieval: {e}")
            raise HTTPException(status_code=500, detail="Internal server error. An error occurred while retrieving documents.")
    # Combine documents
    try:
        combined_docs = (milvus_docs or []) + (postgres_docs or []) + (neo4j_docs or []) + (video_detail_doc or [])
        if prev_response_document:
            combined_docs.append(prev_response_document)
    except Exception as e:
        logging.error(f"Error combining documents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. An error occurred while combining documents.")

    # Generate response using LLM
    try:
        # Directly use the question variable with your LLM setup from the Streamlit app
       #response=generate_llm_response(question, graph, annotation_loader, video_dir, milvus_collection, milvus_embedding_model, prev_response_document)
        response=generate_groq_llm_response(client, question, combined_docs)
        response_text = "Generated response based on the question: " + question + "Response: " + response
        logging.info(f"LLM Response: {response_text}")
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. An error occurred while processing the question.")
    # Convert text to speech
    try:
        start_tts_time = time.time()
        audio_file_path = text_to_speech(response, speed_up_factor=1.2)
        logging.info(f"Text-to-speech conversion took {time.time() - start_tts_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"Error during text-to-speech conversion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.An error occurred while converting text to speech.")

    video_file_path = video_path
    response_json = {"response": response, 
                        "video": video_file_path, 
                        "text_to_speech": audio_file_path}
    return AnswerResponse(answer=response_json)

# Run API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)