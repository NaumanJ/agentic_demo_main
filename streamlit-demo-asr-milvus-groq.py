# Filtered and deduplicated list of imports
import os
import logging
import traceback
import tempfile
import streamlit as st
import json
import numpy as np
from PIL import Image
import time
from sqlalchemy import create_engine, inspect
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import sounddevice as sd
#import speech_recognition as sr
from gtts import gTTS
from milvus_client import MilvusClient
import postgres_client
import neo4j_knowledge_graph
from voice_recognition import VoiceRecognition
from pyvis.network import Network
import streamlit.components.v1 as components
from openai import OpenAI
from video_annotation_loader import AnnotationLoader
#import video_annotation_manager
import bitsandbytes
import torch
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import time
import datetime
import groq

def load_local_llm():
    # Initialize the NVIDIA LLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Clear CUDA cache
    torch.cuda.empty_cache()
    #torch.cuda.reset_max_memory_allocated()
    #torch.cuda.reset_max_memory_cached()
    # client = OpenAI(
    #     base_url="https://integrate.api.nvidia.com/v1",
    #     api_key=f"{api_key}"  # Replace with your API key
    # )
        # Load the model and tokenizer
    #model_name =  "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF" 
    #model_name =   "meta-llama/Llama-3.1-405B-Instruct-FP8" 
    #model_name =  "tiiuae/falcon-40b-instruct"
    #model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF" 
    model_name = "meta-llama/Llama-2-7b-hf"
    pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf")

# Set up the quantization configuration for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=None
#        llm_int8_has_fp16_weights=False
    )
    # Define memory limits for each GPU
    max_memory = {
        0: "32GiB",  # Adjust to leave headroom on each GPU
        1: "32GiB",
        2: "32GiB",
        3: "32GiB"
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

def generate_local_llm_response(model, tokenizer, question, combined_docs):
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
        # Generate the response
        response_token_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode the generated tokens
        response = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
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
        return "An error occurred while processing your LLM request. Please try again."

def load_nemotron_nim_llm():
    # Initialize the NVIDIA LLM
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=f"{api_key}"  # Replace with your API key
    )
    return client

def generate_nemotron_nim_response(client, question, combined_docs):
    # Ensure the question and context are not empty
    if not question.strip():
        return "Please provide a valid question."
    
    if not combined_docs:
        return "No context available to answer the question."

    # Combine all the documents into a single context
    retrieved_context = "\n".join(combined_docs)
    context = (f"You are a knowledgeable soccer fan. Answer the question in a friendly, engaging, and "
               f"enthusiastic manner, as if you are talking to a fellow sports fan. "
               f"Quote information from this context to complete your answer"
               f"{retrieved_context}")

    prompt = (
        f"Context: {context}\n\n"
        "Summarize the response to be under 100 words"
        "Make sure to answer the question with a complete answer with an upbeat enthusiastic mood" 
        "include interesting facts or statistics about the topic or football if relevant."
        "Ask a follow up question based on the context provided to check if the user would like more information."
        "Avoid repeating the context or question, and keep the tone light and conversational."
        "Avoid any advise or profanity or discussion about political, illegal or controversial issues"
        "Remove any html tags or markdown tags from your response to make it easily speakble so it can be converted to speech"
    )

    question = question + " Answer very concise in 1 or 2 sentences only."
    # Prepare the message content
    messages = [
        {"role": "user", "content": question},
        {"role": "system", "content": prompt}
    ]

    # Check for empty content before making the API request
    if not any(msg["content"].strip() for msg in messages):
        return "Error: Both question and context are empty. Please try again with a valid input."

    # Create a completion request
    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=messages,
            temperature=0.5,
            top_p=1,
            max_tokens=512,
            stream=True
        )

        # Collect the streamed response
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        logging.info(f"LLM reponse is {response}")
        return response

    except Exception as e:
        logging.error(f"Error during API request: {e}")
        return "An error occurred while processing your LLM request. Please try again."

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


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Initialize SQLAlchemy engine for PostgreSQL
def init_pg_engine():
    return create_engine('postgresql://postgres:password@localhost:5433/postgres')

# Function to record audio
def record_audio(duration=10, fs=44100, device_index=1):
    """Capture audio from the microphone with progress indicator."""
    try:
        st.info("Listening to microphone..")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64', device=device_index)
        
        # Progress bar for recording
        progress_bar = st.progress(0)
        for i in range(100):
            sd.sleep(int(duration * 10))  # Sleep for 10% of the duration
            progress_bar.progress(i + 1)

        sd.wait()  # Wait until recording is finished
        st.success("Audio captured!")
        return recording.flatten()
    except Exception as e:
        st.error(f"Audio capture failed: {e}")
        return None
    
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

# def audio_to_text(audio_data, fs=44100, speed_up_factor=1.2):
#     """Convert recorded audio data to text, speed up the playback, and use SpeechRecognition."""
#     recognizer = sr.Recognizer()

#     # Save the audio data to a temporary WAV file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
#         sd.write(temp_audio.name, audio_data, fs)

#     # Speed up the audio using pydub
#     try:
#         audio = AudioSegment.from_wav(temp_audio.name)
#         faster_audio = audio.speedup(playback_speed=speed_up_factor)
        
#         # Save the sped-up audio to a new temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_faster_audio:
#             faster_audio.export(temp_faster_audio.name, format="wav")
#             sped_up_audio_path = temp_faster_audio.name
#     finally:
#         # Clean up the original temporary file
#         os.remove(temp_audio.name)

#     # Use SpeechRecognition to convert sped-up audio to text
#     with sr.AudioFile(sped_up_audio_path) as source:
#         audio = recognizer.record(source)
#     try:
#         text = recognizer.recognize_google(audio)
#         return text
#     except sr.UnknownValueError:
#         return "Sorry, I could not understand the audio."
#     except sr.RequestError:
#         return "Error with the Speech Recognition service."
#     finally:
#         # Clean up the sped-up audio file
#         os.remove(sped_up_audio_path)

def text_to_speech(text, speed_up_factor=1.2):
    speed_up_factor_sent = speed_up_factor
    """Convert text to speech using gTTS and play it."""
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
        tts.save(temp_audio.name)
        st.audio(temp_audio.name, format="audio/mp3")
    return(temp_audio.name)
# def text_to_speech(text, speed_up_factor=1.2):
#     """Convert text to speech, speed up the audio, and save it as a WAV file."""
#     # Generate speech using gTTS
#     tts = gTTS(text)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
#         tts.save(temp_audio.name)
#         audio = AudioSegment.from_file(temp_audio.name)

#     # Speed up the audio
#     # Set the path to ffmpeg/ffprobe binaries
#     AudioSegment.converter = "/app/ffmpeg/audio_segment_converter/ffmpeg/"
#     AudioSegment.ffprobe = "/app/ffmpeg/audio_segment_converter/ffprobe/"
#     faster_audio = audio.speedup(playback_speed=speed_up_factor)

#     # Save the faster audio to a WAV file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_wav:
#         faster_audio.export(temp_audio_wav.name, format="wav")
#         return temp_audio_wav.name

# Load LLM
# def load_llm():
#     model_name = "EleutherAI/gpt-neo-2.7B"  # Adjust model name as needed for the 7B model
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     # Load model with FP16 precision, which saves memory and speeds up inference on GPUs
#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)  
#     # Set device to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     # If using a larger model, it’s also a good idea to enable gradient checkpointing to save memory
#     if device == "cuda":
#         model.gradient_checkpointing_enable()
#     return model, tokenizer

# Extract keywords from text
def extract_keywords_from_documents(documents):
    """
    Extract keywords from a list of documents where each document is a dictionary
    with a 'content' key holding the text.
    """
    # Extract text content from each document
    texts = [doc['content'] for doc in documents if 'content' in doc]

    if not texts:
        return []
    
    # Combine all texts into one string for keyword extraction
    combined_text = ' '.join(texts)

    # Create TfidfVectorizer instance
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    
    # Fit the vectorizer and get the feature names (keywords)
    X = vectorizer.fit_transform([combined_text])
    feature_names = vectorizer.get_feature_names_out()
    logging.info(f"Generating features{feature_names}")
    return feature_names

from sklearn.feature_extraction.text import TfidfVectorizer

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

# Display word cloud
def display_word_cloud(keywords):
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return

# Get schema information from PostgreSQL
def get_postgres_schema(engine):
    inspector = inspect(engine)
    schemas = {}
    for table_name in inspector.get_table_names(schema='public'):
        columns = inspector.get_columns(table_name)
        column_names = [col['name'] for col in columns]
        schemas[table_name] = column_names
    return schemas


# LLM response generator
def retrieve_documents_from_postgres(pg_engine, question):
    """
    Retrieve relevant documents from PostgreSQL based on the user's question.
    We assume that we can match keywords from the question to the database content.
    """
    # Extract keywords from the question
    keywords = extract_keywords(question)

    # Query the PostgreSQL database for relevant information (e.g., using SQL LIKE queries)
    retrieved_docs = []
    try:
        with pg_engine.connect() as conn:
            for keyword in keywords:
                query = f"SELECT * FROM your_table WHERE column_name LIKE '%{keyword}%' LIMIT 5"
                result_set = conn.execute(query)
                rows = result_set.fetchall()
                if rows:
                    for row in rows:
                        retrieved_docs.append(f"Postgres Document: {row}")
    except Exception as e:
        logging.error(f"Error retrieving documents from PostgreSQL: {e}")
    
    return retrieved_docs


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

import string

def remove_non_readable_punctuation(text):
    """
    Removes standard punctuation that makes the text non-readable
    while preserving apostrophes and hyphens.
    
    Args:
    text (str): The input string from which to remove punctuation.
    
    Returns:
    str: The text with non-readable punctuation removed.
    """
    # Define the punctuation characters to remove
    punctuation_to_remove = string.punctuation.replace("'", "").replace("-", "")
    # Create a translation table that maps each of these characters to None
    translator = str.maketrans('', '', punctuation_to_remove)
    return text.translate(translator)

def get_postgres_schemas_as_json(pg_engine):
    """
    Fetch PostgreSQL table schemas and their columns as a JSON document.
    
    Args:
    pg_engine (psycopg2 connection object): A connection to the PostgreSQL database.
    
    Returns:
    dict: A dictionary where each key is a table name and the value is a list of columns with details.
    """
    try:
        # Create a cursor object
        cursor = pg_engine.cursor()
        
        # Query to get table names and column information from information_schema
        query = """
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
        """
        
        # Execute the query
        cursor.execute(query)
        results = cursor.fetchall()

        # Structure the results into a dictionary (JSON-like format)
        schema_dict = {}
        
        for table_name, column_name, data_type, is_nullable in results:
            if table_name not in schema_dict:
                schema_dict[table_name] = []
            schema_dict[table_name].append({
                "column_name": column_name,
                "data_type": data_type,
                "is_nullable": is_nullable
            })
        
        # Convert the dictionary to a JSON string
        schema_json = json.dumps(schema_dict, indent=4)
        
        # Close the cursor and return the JSON string
        cursor.close()
        return schema_json
    
    except Exception as e:
        print(f"Error fetching schemas: {e}")
        return None


def generate_llm_response_with_rag(model, tokenizer, question, combined_docs, max_length=100):
    """
    Generate a response to the user's question using a Retrieval-Augmented Generation (RAG) pipeline.
    
    Args:
    - model: The language model used for generating the response.
    - tokenizer: The tokenizer associated with the language model.
    - question: The user's question or input.
    - combined_docs: A list of documents retrieved from PostgreSQL, Neo4j, and Milvus.
    - max_length: The maximum length of the generated response.
    
    Returns:
    - A string response generated by the LLM model.
    """
    # Combine all the documents into a single context
    retrieved_context = "\n".join(combined_docs)
    question = "Answer really concise in 1 sentence." + question

    # Prepare the prompt for the LLM
    prompt = (
            f"You are a knowledgeable sports chatbot application with a video player handy."
            f"The user picks a video from the drop-down list and plays it and display its annotation while gleeaning context."
            f"The user will likely ask a question phrase as text in relation to the video such as players in the video or country teams playing the game etc."
            f"The user may choose not to ask a question about the video and ask general questions the soccer context."
            f"Answer the question in a friendly, engaging, and enthusiastic manner, as if you are talking to a fellow sports fan.\n\n"
            f"Make sure to answer the question with an upbeat enthusiastic mood with a complete answer in maximum 2 sentences . Here is an example." 
            f"Example: What luxury accessories does Ronaldo often wear? Ronaldo is often seen sporting expensive watches and designer sunglasses. His favorite watch brands include Rolex and TAG Heuer, and he owns several custom-made timepieces."
            f"Include interesting facts or statistics about the topic or football if relevant within the sentence." 
            f"If the question does not have an answer, ask a follow up question based on the context provided to check if the user would like more information."
            f"Avoid repeating the context or question, and keep the tone light and conversational."
            f"Avoid any advise or profanity or discussion about political, illegal or controversial issues"
            f"If the context contains users interest or action to a specific video played based on a specific criteria use the context in the response."
            f"If the response contains punctuation marks or asterisks etc avoid them"
            f"Here is the context: {retrieved_context}\n\n"
            f"Question: {question}\n\n"
        )

    # Tokenize the input prompt
    max_model_length = getattr(model.config, 'max_position_embeddings', 2048)
    tokenized_inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=min(512, max_model_length)  # Truncate if necessary
    ).to(model.device)

    # Generate the response from the model
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=tokenized_inputs['input_ids'],
            attention_mask=tokenized_inputs['attention_mask'],
            max_length=min(max_length, max_model_length),
            temperature=0.6,  # Slightly lower temperature for more coherent responses
            top_p=0.9,  # Use nucleus sampling for more diverse outputs
            max_new_tokens=64,  # Increase the length to allow a more detailed response
            do_sample=True
        )

    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Post-process the response to ensure it is clean and concise
    response = response.replace("Context:", "").replace("Question:", "").strip()
    response = remove_non_readable_punctuation(response)
    return response

def display_graph(graph_data):
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    graph_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Knowledge Graph</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .graph-container { margin-left: 20px; }
        </style>
    </head>
    <body>
        <h1>Knowledge Graph</h1>
        <div class="graph-container">
            <ul>
    """
    for node in nodes:
        graph_html += f"<li>{node['label']}</li>"
    graph_html += """
            </ul>
        </div>
    </body>
    </html>
    """
    st.components.v1.html(graph_html, height=600, scrolling=True)

def display_clickable_keywords(keywords):
    for keyword in keywords:
        st.markdown(f"[{keyword}](#)", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

text_document_content=""
combined_text=""
combined_docs=[]

# Sentence Transformer for embeddings
def get_embedding_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

from pyvis.network import Network
import streamlit.components.v1 as components

def display_neo4j_linkage_widget(neo4j_docs):
    """
    Display a Neo4j document linkage graph using pyvis in a Streamlit app.

    Args:
    - neo4j_docs (List[Dict]): List of documents with 'name' and 'related_docs'.
    """
    # Create a new Pyvis network graph
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")

    # Keep track of all added nodes
    added_nodes = set()

    # Add nodes to the graph
    for doc in neo4j_docs:
        doc_name = doc.get('name')
        if doc_name and doc_name not in added_nodes:
            net.add_node(doc_name, label=doc_name)
            added_nodes.add(doc_name)

    # Add edges between nodes based on relationships or content similarity
    for doc in neo4j_docs:
        doc_name = doc.get('name')
        if 'related_docs' in doc:
            for related_doc in doc['related_docs']:
                related_doc_name = related_doc.get('title')
                if related_doc_name and related_doc_name in added_nodes:
                    net.add_edge(doc_name, related_doc_name)

    # Save the graph as HTML and load it in Streamlit
    try:
        temp_file_path = '/tmp/neo4j_graph.html'
        net.save_graph(temp_file_path)
        with open(temp_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        components.html(html_content, height=600)
    except Exception as e:
        logging.error(f"Error displaying Neo4j linkage widget: {e}")
        st.error("Failed to display the document linkage graph.")

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
        st.error("Failed to initialize Neo4j connection.")
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
        st.error("Failed to initialize PostgreSQL connection.")
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
        st.error("Failed to initialize Milvus or embedding model.")
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
        st.error("Failed to initialize Initializing Postgres AnnotationLoader.")
        stack_trace = traceback.format_exc()
        raise Exception(f"{e}\n{stack_trace}")
    return graph, pg_engine, embedding_model, collection, annotation_loader

def play_selected_video(selected_video):
    if selected_video:
        video_path = os.path.join(video_dir, selected_video)
        st.video(video_path, format="video/mp4", start_time=0, loop=True, autoplay=True)  # Auto-play the video


def play_video_loop(selected_video, video_dir):
    """
    Plays the selected video on a loop.
    """
    video_path = os.path.join(video_dir, selected_video)
    st.video(video_path, format="video/mp4", start_time=0, loop=True, autoplay=True)
    logging.info(f"Playing video in loop: {selected_video}")

def handle_video_query(question, selected_video, video_dir, annotation_loader):
    """
    Parses the question and handles video playback based on detected keywords or phrases.
    """
    # Extract keywords or specific phrases from the question
    video_related_phrases = ["scored", "score", "goal", "kick", "kicked", "play", "played", "display", "watch", "find", "show", "seen"]
    logging.info(f"Input question: {question}")
    keywords = extract_keywords(question)
    logging.info(f"Keywords extracted: {keywords}")

    # Handle video search based on keywords
    video_search_results = None
    video_detail_doc = {}
    video_related_keywords = {
        'search', 'videos', 'play', 'video', 'mp4', 'avi', 'mov',
        'players', 'soccer', 'match', 'about', 'this', 'explain', 'summarize', "watch", "scored", "goal", 
    }
    question_words = set(question.split())

    if question_words.intersection(video_related_keywords) and len(question_words) > 3:
        video_search_results = annotation_loader.search_video_by_annotation(selected_video, question)
        if video_search_results and len(video_search_results) > 0:
            selected_video = video_search_results[0]['video_file_name']
            video_path = os.path.join(video_dir, selected_video)
            st.video(video_path, format="video/mp4", start_time=0)
            video_detail_doc[selected_video] = video_search_results[0]['annotation']
    keywords = extract_keywords(question)
    logging.info(f"Keywords extracted for video query: {keywords}")

    for phrase in video_related_phrases and len(question_words) > 3:
        if phrase in question.lower():
            # Use the annotation loader to perform a search for the phrase in the video annotations
            video_search_results = annotation_loader.search_video_by_annotation(selected_video, phrase)
            if video_search_results and len(video_search_results) > 0:
                selected_video = video_search_results[0]['video_file_name']
                start_duration = video_search_results[0]['start_duration']
                end_duration = video_search_results[0]['end_duration']
                annotation = video_search_results[0]['annotation']
                video_path = os.path.join(video_dir, selected_video)
                st.video(video_path, format="video/mp4", start_time=0,autoplay=True)
                st.text(f"Video seeked and played based on your question: '{phrase}'")
                return True  # Video handled successfully
    return False  # Video not handled, proceed with LLM


# Initialize global configuration
document_repo_path = '/app/document_repo/'
# Video Selection from Sidebar
video_dir = document_repo_path +"sports/soccer/video/"
icons_dir = document_repo_path + "icons/"
backgrounds_dir = document_repo_path + "backgrounds/"
HF_TOKEN="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"
api_key="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" 
model_name = "meta-llama/Llama-2-7b-chat-hf"
chat_history = []
prev_response_document = []
## When False runs Nvidia NIMs to connect from Laptops 
## SET THIS TO True ONLY ON Physical Servers with NVIDIA GPUs (min A100 required) ##
llm_choice = 'groq'


if llm_choice == True:
    model, tokenizer = load_local_llm()

def main():
    logging.info("Starting the Sports Chatbot with PostgreSQL, Neo4j, and Milvus")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #000000; 
            background-image: url(f"file://{backgrounds_dir}abstract-soccer-background.jpg");
            background-size: cover;
            background-position: center;
            color: #FFFFFF;
        }
        .css-1d391kg {
            background-color: #F0F2F6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # Sidebar for video thumbnails and branding
    st.sidebar.image("/app/document_repo/icons/centific.png", use_column_width=True)
    st.sidebar.header("Football Footage")

    
    try:
        videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        selected_video = st.sidebar.selectbox("Choose a video to play", videos)
        if selected_video:
            video_path = os.path.join(video_dir, selected_video)
            st.video(video_path, format="video/mp4", start_time=0)
            logging.info(f"Selected video: {selected_video}")
    except Exception as e:
        logging.error(f"Error loading videos: {e}")
        st.sidebar.error("Failed to load videos.")

    # Initialize databases
    try:
        start_time = time.time()
        graph, pg_engine, milvus_embedding_model, milvus_collection, annotation_loader = initialize_databases()
        logging.info(f"Database initialization completed in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"Error during database initialization: {e}")
        return

    annotation_loader = annotation_loader
    # Chat input
    st.header("Interactive Soccerbot - AI Fun Experience")
    question = st.text_input("Soccer Fan Trivia, learn about your favorite player.")

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
            st.error("An error occurred while retrieving documents.")
            raise

        # Combine documents
        try:
            combined_docs = (milvus_docs or []) + (postgres_docs or []) + (neo4j_docs or []) + (video_detail_doc or [])
            if prev_response_document:
                combined_docs.append(prev_response_document)
        except Exception as e:
            logging.error(f"Error combining documents: {e}")
            st.error("An error occurred while combining documents.")
            raise

        # Generate response
        try:
            if llm_choice == 'local':
                #model, tokenizer = load_local_llm()
                start_response_time = time.time()
                response = generate_local_llm_response(model, tokenizer, question, combined_docs)
                logging.info(f"LLM response generated in {time.time() - start_response_time:.2f} seconds.")
            elif llm_choice == 'nvidia':
                client = load_nemotron_nim_llm()
                start_response_time = time.time()
                response = generate_nemotron_nim_response(client, question, combined_docs)
                logging.info(f"LLM response generated in {time.time() - start_response_time:.2f} seconds.")
            elif llm_choice == 'groq':
                start_response_time = time.time()
                client = groq.Groq(api_key='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')
                response = generate_groq_llm_response(client, question, combined_docs)
                logging.info(f"LLM response generated in {time.time() - start_response_time:.2f} seconds.")
        except Exception as e:
            logging.error(f"Error during response generation: {e}")
            st.error("An error occurred while generating the response.")
            raise

        # Convert text to speech
        try:
            start_tts_time = time.time()
            audio_file_path = text_to_speech(response, speed_up_factor=1.2)
            logging.info(f"Text-to-speech conversion took {time.time() - start_tts_time:.2f} seconds.")
        except Exception as e:
            logging.error(f"Error during text-to-speech conversion: {e}")
            st.error("An error occurred while converting text to speech.")
            raise

        # Save chat entry to history
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_history.append({"timestamp": timestamp, "question": question, "response": response, "audio_path": audio_file_path})
        prev_response_document.append(f"Question: {question}\nResponse: {response}\nSelected Video: {selected_video}")

        # Display the chat log
        st.markdown('<div class="chat-log" style="background-color: #F0F2F6; padding: 10px; border-radius: 5px;">', unsafe_allow_html=True)
        for entry in chat_history:
            #st.markdown(f"..", unsafe_allow_html=True)  # Your existing chat log rendering
            for entry in chat_history:
                st.markdown(
                    f"""
                    <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                        <p><b>Timestamp:</b> {entry['timestamp']}</p>
                        <p><b>Question:</b> {entry['question']}</p>
                        <p><b>Response:</b> {entry['response']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if entry['audio_path']:
                    st.audio(entry['audio_path'], format="audio/wav")
        st.markdown('</div>', unsafe_allow_html=True)

        # Display word cloud
        response_words = extract_keywords(response)
        if keywords and response_words:
            display_word_cloud(list(keywords) + response_words)

    # Footer branding
    st.sidebar.markdown("---")
    st.sidebar.write("Built by **Centific Solutions** using **Nvidia** Architecture")
    st.sidebar.write("Scaled by **Amisys** on **Lenovo** platform")
    st.sidebar.image(
    [
            "/app/document_repo/icons/centific.png",
            "/app/document_repo/icons/Lenovo.png",
            "/app/document_repo/icons/Nvidia-H.png",
            "/app/document_repo/icons/Pitaya1.png",
        ],
        width=100
    )

if __name__ == "__main__":
    main()   
