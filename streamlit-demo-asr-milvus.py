import os
import logging
import streamlit as st
from sqlalchemy import create_engine, inspect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Streamlit app
import logging
import streamlit as st
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import sounddevice as sd
import speech_recognition as sr
from gtts import gTTS
import os
import streamlit as st
import tempfile
import json

import numpy as np
import  milvus_client
import postgres_client
import neo4j_knowledge_graph
from voice_recognition import VoiceRecognition


# Sample transcript to augment the context (if needed)
transcript_sample = """


"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Initialize SQLAlchemy engine for PostgreSQL
def init_pg_engine():
    return create_engine('postgresql://postgres:password@localhost:5433/postgres')

def record_audio(duration=5, fs=44100, device_index=1):  # Replace `1` with the correct device index
    """Record audio from the microphone with progress indicator."""
    try:
        st.info("Recording... Speak now!")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64', device=device_index)
        
        # Progress bar for recording
        progress_bar = st.progress(0)
        for i in range(100):
            sd.sleep(int(duration * 10))  # Sleep for 10% of the duration
            progress_bar.progress(i + 1)

        sd.wait()  # Wait until recording is finished
        st.success("Recording complete!")
        return recording.flatten()
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None
    
def audio_to_text(audio_data, fs=44100):
    """Convert recorded audio data to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        sd.write(temp_audio.name, audio_data, fs)
        with sr.AudioFile(temp_audio.name) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Error with the Speech Recognition service."

def text_to_speech(text):
    """Convert text to speech using gTTS and play it."""
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
        tts.save(temp_audio.name)
        st.audio(temp_audio.name, format="audio/mp3")

# Load LLM
def load_llm():
    model_name = "EleutherAI/gpt-neo-2.7B"  # Adjust model name as needed for the 7B model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load model with FP16 precision, which saves memory and speeds up inference on GPUs
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)  
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # If using a larger model, it’s also a good idea to enable gradient checkpointing to save memory
    if device == "cuda":
        model.gradient_checkpointing_enable()
    return model, tokenizer


# Extract keywords from text
def extract_keywords(documents):
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
    
    return feature_names

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


def retrieve_documents_from_neo4j(neo4j_docs):
    """
    Retrieve relevant documents from Neo4j based on keywords or the provided document context.
    """
    retrieved_docs = []
    for doc in neo4j_docs:
        retrieved_docs.append(f"Neo4j Document: {doc['content'][:200]}...")  # Using a snippet of the document
    return retrieved_docs

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
    

def generate_llm_response_with_rag(model, tokenizer, question, combined_docs, max_length=500):
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

    # Prepare the prompt for the LLM
    prompt = (
            f"You are a knowledgeable sports assistant. Answer the question in a friendly, engaging, and "
            f"enthusiastic manner, as if you are talking to a fellow sports fan.\n\n"
            f"Question: {question}\n\n"
            f"Context: {retrieved_context}\n\n"
            "Make sure to answer the question with a complete answer with an upbeat enthusiastic mood" 
            "include interesting facts or statistics about the topic or "
            "football if relevant. Ask a follow up question based on the context provided to check if the user would like more information."
            "Avoid repeating the context or question, and keep the tone light and conversational."
            "Avoid any advise or profanity or discussion about political, illegal or controversial issues"
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
            max_new_tokens=200,  # Increase the length to allow a more detailed response
            do_sample=True
        )

    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Post-process the response to ensure it is clean and concise
    response = response.replace("Context:", "").replace("Question:", "").strip()
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

# Sentence Transformer for embeddings
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def main():
    # Log the start of the main function
    logging.info("Starting the Sports Chatbot with PostgreSQL, Neo4j, and Milvus")

    # Initialize PostgreSQL connection and load schema
    st.title("Centific Video Language Model Analytics using Voice Chat (Nvidia AI Solutions)")
    logging.info("Initializing PostgreSQL engine and retrieving schema.")
    
    try:
        pg_engine = init_pg_engine()
        schemas = get_postgres_schema(pg_engine)
        logging.info("PostgreSQL schema retrieved successfully.")
    except Exception as e:
        logging.error(f"Error initializing PostgreSQL engine or retrieving schema: {e}")
        st.error("Failed to initialize PostgreSQL connection.")
        return
  

    text_document_content = transcript_sample

    # Display video thumbnails from the directory
    st.sidebar.header("Soccer Player Video Repository")
    video_dir = "/app/document_repo/sports/soccer/video"
    videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    selected_video = st.sidebar.selectbox("Choose a video to play", videos)
    
    # Display the selected video
    if selected_video:
        video_path = os.path.join(video_dir, selected_video)
        st.video(video_path, format="video/mp4", start_time=0)  

    # Initialize Neo4j Knowledge Graph
    logging.info("Initializing Neo4j Knowledge Graph.")
    try:
        graph = neo4j_knowledge_graph.Neo4jKnowledgeGraph('bolt://localhost:7687', 'neo4j', 'password')
        logging.info("Neo4j Knowledge Graph initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing Neo4j Knowledge Graph: {e}")
        st.error("Failed to initialize Neo4j connection.")
        return

    # Text input and voice recording section
    st.header("Ask a Question by Voice Input")
    question = st.text_input("Enter your question")

    # Add "Record Voice" button for voice input
    if st.button("Ask a question"):
        audio_data = record_audio()  # Record audio from the microphone
        if audio_data is not None:
            question = audio_to_text(audio_data)  # Convert audio to text
            st.write(f"Transcribed Question: {question}")

            # Playback the recorded question
            st.subheader("Playback Recorded Question")
            st.audio(tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name, format="audio/wav")

    # Process query when user submits
    if st.button("Submit") and question:
        try:
            # Step 1: Retrieve documents from Milvus, PostgreSQL, and Neo4j
            logging.info("Retrieving documents from all sources.")

            embedding_model=get_embedding_model()
            collection = milvus_client.init_milvus()
            milvus_docs = milvus_client.search_documents_milvus(collection, question, embedding_model)
            # Retrieve documents from PostgreSQL
            postgres_docs = retrieve_documents_from_postgres(pg_engine, question)         
            # Retrieve documents from Neo4j
            keywords = extract_keywords(question)
            neo4j_docs = graph.search_documents(keywords)
            
            # Combine all retrieved documents into a single list
            combined_docs = milvus_docs + postgres_docs + [doc['content'] for doc in neo4j_docs]
            logging.info(f"Content retrieved: {combined_docs}")

            # Step 2: Generate response using the RAG pipeline
            logging.info("Generating response using LLM and RAG pipeline.")
            llm_model, tokenizer = load_llm()
            combined_docs = milvus_docs + postgres_docs + [doc['content'] for doc in neo4j_docs]
            response = generate_llm_response_with_rag(
                llm_model, tokenizer, question, combined_docs
            )
            logging.info("Response generated successfully.")

            # Step 3: Display and playback the LLM-generated response
            st.subheader("LLM Response")
            st.text_area("Response", response, height=300)
            text_to_speech(response)  # Play the response as audio

            # Step 4: Display a word cloud of keywords from the combined context
            logging.info("Generating and displaying word cloud.")
            display_word_cloud(keywords)

        except Exception as e:
            logging.error(f"Error during question processing: {e}")
            st.error("An error occurred while processing your question.")

    # Close Neo4j connection on app shutdown
    try:
        graph.close()
        logging.info("Neo4j connection closed.")
    except Exception as e:
        logging.error(f"Error closing Neo4j connection: {e}")

if __name__ == "__main__":
    main()
