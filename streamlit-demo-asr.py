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
from pyvis.network import Network
import streamlit.components.v1 as components
from openai import OpenAI
import traceback

def load_nemotron_llm():
    # Initialize the NVIDIA LLM
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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"  # Replace with your API key
    )
    return client

# Sample transcript to augment the context (if needed)
transcript_sample = """
"""

def generate_nemotron_response(client, question, combined_docs):
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
        "Summarize the response to be under 200 words"
        "Make sure to answer the question with a complete answer with an upbeat enthusiastic mood" 
        "include interesting facts or statistics about the topic or football if relevant."
        "Ask a follow up question based on the context provided to check if the user would like more information."
        "Avoid repeating the context or question, and keep the tone light and conversational."
        "Avoid any advise or profanity or discussion about political, illegal or controversial issues"
        "Remove any html tags or markdown tags from your response to make it easily speakble so it can be converted to speech"
    )
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
        return "An error occurred while processing your request. Please try again."

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Initialize SQLAlchemy engine for PostgreSQL
def init_pg_engine():
    return create_engine('postgresql://postgres:password@localhost:5433/postgres')

def record_audio(duration=5, fs=44100, device_index=1):  # Replace `1` with the correct device index
    """Capture audio from the microphone with progress indicator."""
    try:
        st.info("Listening to microphone...")
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
            f"You are a knowledgeable soccer commentator who responds to the queries on soocer, soccer players and soccer team statistics."
            "Answer the question in a friendly, engaging, enthusiastic tone, as if talking to a fellow sports fan.\n"
            f"Question: {question}\n"
            f"Context: {retrieved_context}\n"
            "Answer the question with a complete answer with an upbeat enthusiastic mood" 
            "Include interesting facts or statistics about the topic or soccer."
            "Ask one and only one follow up question on soccer or about the player based on the context of the prior responses provided and check if the user would like more information."
            "Avoid repeating the context or question, and keep the tone light and conversational."
            "Avoid any advise or profanity or discussion about political, illegal or controversial issues"
            "If the user asks to play video of a situation use the keyword VIDEOPLAY-situation-VIDEOPLAY in the response"
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
combined_docs=[]

# Sentence Transformer for embeddings
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

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

import logging
import traceback
import os
import tempfile
import streamlit as st
import numpy as np

def main():
    # Log the start of the main function
    logging.info("Starting the Sports Chatbot with PostgreSQL, Neo4j, and Milvus")

    # Set the title and description for the app
    st.title("Soccer AI Fun Experience")
    st.markdown("An interactive chatbot using voice and text input to answer questions about sports videos, powered by Centific Solutions, Nvidia AI, and advanced database solutions.")

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
        return

    # Sidebar: Video selection
    selected_video=""
    st.sidebar.header("Football Footage")
    video_dir = "/app/document_repo/sports/soccer/video"
    try:
        videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        selected_video = st.sidebar.selectbox("Choose a video to play", videos)
        if selected_video:
            video_path = os.path.join(video_dir, selected_video)
            # Play video
            st.video(video_path, format="video/mp4", start_time=0)
    except Exception as e:
        logging.error(f"Error loading videos: {e}")
        st.sidebar.error("Failed to load videos.")
        stack_trace = traceback.format_exc()
        raise Exception(f"{e}\n{stack_trace}")

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
        return

    # Voice recording and text-to-speech controls above the chat interface
    st.header("Interactive Soccerbot")
    if st.button("Why type, Lets talk 🎤 "):
        audio_data = record_audio()  # Record audio from the microphone
        if audio_data is not None:
            question = audio_to_text(audio_data)  # Convert audio to text
            st.write(f"Transcribed Question: {question}")
            # Automatically play the audio file
            st.audio(tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name, format="audio/wav")

    # Chat interface using st.chat
    st.chat_message("You can type or record your question below:")
    question = None

    with st.chat_message("user"):
        question = st.text_input("Type your question here:")

    if question:
        try:
            # Step 1: Retrieve documents from Milvus, PostgreSQL, and Neo4j
            logging.info(f"Input question is -> {question}")
            logging.info("Retrieving documents from all sources.")
            # Keywords from question
            keywords = extract_keywords(question)
            logging.info(f"Keywords in question -> {keywords}")
            # Retrieve documents from Neo4j
            neo4j_search_results = graph.get_relevant_documents(keywords, 2)
            neo4j_docs = parse_content_from_neo4j_docs(neo4j_search_results)
            # Retrieve documents from Milvus
            embedding_model = get_embedding_model()
            collection = milvus_client.init_milvus()
            milvus_docs = milvus_client.search_documents_milvus(collection, question, embedding_model)
            # Retrieve documents from PostgreSQL
            postgres_docs = retrieve_documents_from_postgres(pg_engine, question)
        except Exception as e:
            logging.error(f"Error during document retrieval: {e}")
            st.error("An error occurred while retrieving documents.")
            raise

        try:
            # Combine all documents into a single list
            milvus_docs = milvus_docs if milvus_docs is not None else []
            postgres_docs = postgres_docs if postgres_docs is not None else []
            neo4j_docs = neo4j_docs if neo4j_docs is not None else []
            combined_docs = milvus_docs + postgres_docs + neo4j_docs
            logging.info(f"Content of length {len(combined_docs)} retrieved successfully")
        except Exception as e:
            logging.error(f"Error during document combination: {e}")
            st.error("An error occurred while combining documents.")
            raise

        try:
            # Step 2: Generate response using the NVIDIA LLM
            client = load_nemotron_llm()
            response = generate_nemotron_response(client, question, combined_docs)
            logging.info("LLM Response generated successfully.")
        except Exception as e:
            logging.error(f"Error during response generation: {e}")
            st.error("An error occurred while generating the response.")
            raise

        try:
            text_to_speech(response)
            logging.info("text_to_speech response generated successfully.")
            # Display the chat response
            with st.chat_message("assistant"):
                st.write(response)
        except Exception as e:
            logging.error(f"Error during audio speech response generation: {e}")
            st.error("An error occurred while generating the audio speech response.")
            raise
            # Step 3: Display a word cloud of extracted keywords
            if keywords:
                display_word_cloud(keywords)
    
    # Display Neo4j linkage widget if requested
    if st.button("Knowledge Graph") and neo4j_search_results:
        try:
            st.subheader("Document Links from Neo4j")
            display_neo4j_linkage_widget(neo4j_search_results)
        except Exception as e:
            logging.error(f"Error displaying Neo4j linkage widget: {e}")
            st.error("An error occurred while displaying the knowledge graph.")
            raise

    # Footer with logos and credits
    st.markdown("---")
    st.write("Built by **Centific Solutions** using **Nvidia** Architecture")
    st.write("Scaled by **Amisys** on **Lenovo** platform")
    st.image(
        [
            "/app/document_repo/icons/centific.png",
            "/app/document_repo/icons/Lenovo.png",
            "/app/document_repo/icons/Nvidia-H.png",
            "/app/document_repo/icons/Pitaya1.png",
        ],
        width=100,
    )

if __name__ == "__main__":
    main()
