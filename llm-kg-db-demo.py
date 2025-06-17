import os
import pandas as pd
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from neo4j import GraphDatabase
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
import duckdb
from transformers import AutoTokenizer, AutoModelForCausalLM
import wikipedia
import nltk
from nltk.corpus import stopwords
from nltk import ngrams
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Parse Documents Functions

def parse_doc(file_path):
    with fitz.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return {'title': os.path.basename(file_path), 'content': text}

def parse_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    return {'title': os.path.basename(file_path), 'content': soup.get_text()}

def parse_pdf(file_path):
    with fitz.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return {'title': os.path.basename(file_path), 'content': text}

def parse_xls(file_path):
    df = pd.read_excel(file_path)
    return df.to_dict(orient='records')

def parse_documents(directory):
    documents = []
    for root, dirs, files in os.walk(directory):
        logging.info(f"Parsing directory: {root}")
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.csv'):
                # Skip CSV files
                continue
            elif file.endswith('.doc') or file.endswith('.docx'):
                documents.append(parse_doc(file_path))
            elif file.endswith('.html'):
                documents.append(parse_html(file_path))
            elif file.endswith('.pdf'):
                documents.append(parse_pdf(file_path))
            elif file.endswith('.xls') or file.endswith('.xlsx'):
                documents.extend(parse_xls(file_path))
    return documents

# Extract Keywords Functions

def extract_key_phrases(text, num_words=2):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]
    phrases = [' '.join(gram) for gram in ngrams(tokens, num_words)]
    return phrases

def extract_keywords_from_text(text):
    # Simple keyword extraction
    import re
    words = set(re.findall(r'\w+', text.lower()))
    common_words = set(stopwords.words('english'))
    keywords = list(words - common_words)
    return keywords

def extract_keywords(documents):
    texts = [doc['content'] for doc in documents if 'content' in doc and doc['content']]
    if not texts:
        return []
    text = ' '.join(texts)
    key_phrases = extract_key_phrases(text)
    return key_phrases

# Neo4j Knowledge Graph Class

class Neo4jKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_document_node(self, title, content):
        with self.driver.session() as session:
            session.write_transaction(self._create_document_node, title, content)

    @staticmethod
    def _create_document_node(tx, title, content):
        tx.run("MERGE (d:Document {title: $title}) SET d.content = $content",
               title=title, content=content)

    def create_relationship(self, title1, title2, relationship):
        with self.driver.session() as session:
            session.write_transaction(self._create_relationship, title1, title2, relationship)

    @staticmethod
    def _create_relationship(tx, title1, title2, relationship):
        tx.run("MATCH (d1:Document {title: $title1}), (d2:Document {title: $title2}) "
               "MERGE (d1)-[:RELATED_TO {type: $relationship}]->(d2)",
               title1=title1, title2=title2, relationship=relationship)

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

    def get_top_keywords(self, limit=5):
        with self.driver.session() as session:
            result = session.read_transaction(self._get_top_keywords, limit)
        return result

    @staticmethod
    def _get_top_keywords(tx, limit):
        query = """
        MATCH (d:Document)
        WITH split(d.content, ' ') AS words
        UNWIND words AS word
        WITH word, count(*) AS frequency
        WHERE size(word) > 3
        RETURN word, frequency
        ORDER BY frequency DESC
        LIMIT $limit
        """
        result = tx.run(query, limit=limit)
        keywords = [record["word"] for record in result]
        return keywords

# DuckDB Manager Class

class DuckDBManager:
    def __init__(self, database=':memory:'):
        self.conn = duckdb.connect(database=database)
        self.tables = []
        self.schemas = {}

    def load_csv_files(self, directory):
        import glob
        import os
        csv_files = glob.glob(os.path.join(directory, "**/*.csv"), recursive=True)
        for csv_file in csv_files:
            table_name = os.path.basename(csv_file)[:-4]
            sanitized_table_name = self.sanitize_table_name(table_name)
            self.load_data_into_duckdb(csv_file, sanitized_table_name)
            self.tables.append(sanitized_table_name)
            schema_df = self.conn.execute(f'DESCRIBE "{sanitized_table_name}"').fetchdf()
            self.schemas[sanitized_table_name] = schema_df.to_dict(orient='records')

    def load_data_into_duckdb(self, csv_file, table_name):
        """Load a CSV file into a DuckDB table using Pandas."""
        try:
            # Try reading the CSV with 'utf-8' encoding first
            df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            # If 'utf-8' fails, try with 'latin1' encoding
            df = pd.read_csv(csv_file, encoding='latin1', on_bad_lines='skip')
        except Exception as e:
            logging.error(f"Error reading CSV with Pandas: {e}")
            return
        # Register the DataFrame as a temporary table in DuckDB
        self.conn.register('temp_df', df)
        # Use CREATE OR REPLACE TABLE to handle existing tables
        self.conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM temp_df')
        # Unregister the temporary table
        self.conn.unregister('temp_df')

    def execute_query(self, query):
        return self.conn.execute(query).fetchdf()

    def close(self):
        self.conn.close()

    @staticmethod
    def sanitize_table_name(name):
        """Sanitize the table name by replacing unsafe characters."""
        import re
        # Replace spaces and hyphens with underscores
        name = re.sub(r'[\s\-]+', '_', name)
        # Remove any characters that are not alphanumeric or underscore
        name = re.sub(r'[^\w]', '', name)
        # Ensure the name doesn't start with a number
        if re.match(r'^\d', name):
            name = f"t_{name}"
        return name

# Display Functions

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

def display_word_cloud(keywords):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def display_clickable_keywords(keywords):
    for keyword in keywords:
        st.markdown(f"[{keyword}](#)", unsafe_allow_html=True)

# Video Transcription Functions

def transcribe_video(video_path):
    # Placeholder for NVIDIA API call
    # For the purpose of this example, we'll return a sample transcription
    transcription = "Sample transcription about soccer, players, and famous matches."
    return transcription

def save_transcription(transcription, file_path):
    with open(file_path, 'w') as file:
        file.write(transcription)

def fetch_wikipedia_articles(phrases, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for phrase in phrases:
        try:
            page = wikipedia.page(phrase)
            html_content = page.html()
            file_name = f"{phrase.replace(' ', '_')}.html"
            file_path = os.path.join(save_dir, file_name)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logging.info(f"Saved Wikipedia page for '{phrase}'")
        except Exception as e:
            logging.error(f"Error fetching Wikipedia page for '{phrase}': {e}")

# LLM Integration Functions

def generate_sql_query(llm_model, tokenizer, question, schemas):
    """Generate a SQL query based on a natural language question and schemas."""
    # Prepare the prompt
    system_prompt = "You are an expert SQL assistant."
    schema_info = ""
    for table_name, schema in schemas.items():
        schema_info += f"\nTable {table_name} has columns: {', '.join([col['column_name'] for col in schema])}."
    user_prompt = f"{system_prompt}\n\nGenerate a SQL query to answer the following question: '{question}'.{schema_info}\nSQL Query:"
    
    # Tokenize the prompt
    inputs = tokenizer.encode(user_prompt, return_tensors="pt").to(llm_model.device)

    # Generate SQL Query
    outputs = llm_model.generate(
        inputs,
        max_length=inputs.shape[1] + 100,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=1,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the SQL query from the generated text
    sql_query = generated_text[len(user_prompt):].strip()
    return sql_query

def generate_answer(llm_model, tokenizer, question, duckdb_result=None, neo4j_result=None):
    """Generate an answer to the user's question based on the query results."""
    # Prepare the prompt
    result_str = ""
    if duckdb_result is not None and not duckdb_result.empty:
        result_str += "Relational Data Result:\n" + duckdb_result.to_string(index=False) + "\n"
    if neo4j_result is not None:
        neo4j_info = "\n".join([doc['content'] for doc in neo4j_result])
        result_str += "Graph Data Result:\n" + neo4j_info + "\n"
    user_prompt = f"Question: {question}\n\n{result_str}\nProvide a concise answer based on the result.\nAnswer:"
    
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

def get_llm_response(chat_input, transcription, schemas, keywords, llm_model, tokenizer, duckdb_conn, neo4j_graph):
    """Process the user's input and generate a response using LLM, DuckDB, and Neo4j."""
    # Combine chat input and transcription
    question = chat_input + " " + transcription

    # Determine the type of question (SQL or graph-related)
    user_prompt = f"Determine whether the user's question is about relational data (SQL) or graph data.\nQuestion: {question}\nType of question (SQL or Graph):"
    inputs = tokenizer.encode(user_prompt, return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(
        inputs,
        max_length=inputs.shape[1] + 10,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=1,
    )
    determination = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    determination = determination[len(user_prompt):].strip()

    duckdb_result = None
    neo4j_result = None

    if 'sql' in determination or 'relational' in determination:
        # Generate and execute SQL query
        sql_query = generate_sql_query(llm_model, tokenizer, question, schemas)
        logging.info(f"Generated SQL Query: {sql_query}")
        try:
            duckdb_result = duckdb_conn.execute(sql_query).fetchdf()
            logging.info(f"DuckDB Query Result:\n{duckdb_result}")
        except Exception as e:
            logging.error(f"Error executing SQL query: {e}")
    elif 'graph' in determination:
        # Query Neo4j
        neo4j_result = neo4j_graph.search_documents(keywords)
        logging.info(f"Neo4j Query Result: {neo4j_result}")
    else:
        # If determination is unclear, handle accordingly
        logging.warning("Could not determine the type of question. Proceeding with both queries.")
        # Attempt both queries
        try:
            sql_query = generate_sql_query(llm_model, tokenizer, question, schemas)
            logging.info(f"Generated SQL Query: {sql_query}")
            duckdb_result = duckdb_conn.execute(sql_query).fetchdf()
        except Exception as e:
            logging.error(f"Error executing SQL query: {e}")
        neo4j_result = neo4j_graph.search_documents(keywords)

    # Generate the final answer using LLM
    answer = generate_answer(llm_model, tokenizer, question, duckdb_result, neo4j_result)
    return answer

# Main Function

def main():
    # Initialize variables
    transcription = ""
    documents = []
    directory = 'document_repo/sports/soccer'

    # Initialize Neo4j Knowledge Graph
    graph = Neo4jKnowledgeGraph('bolt://localhost:7687', 'neo4j', 'password')

    # Initialize DuckDB Manager
    db_manager = DuckDBManager()

    # Load CSV files into DuckDB
    db_manager.load_csv_files(directory)

    # Initialize LLM for SQL Generation and Answering
    model_name = "EleutherAI/gpt-neo-2.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_model.to(device)

    st.title("Sports Chat using Knowledge Graph Search")

    # Transcribe Video Section
    st.header("Video Transcription")
    video_path = st.text_input("Video Path", key='video_path')
    transcribe_button = st.button("Transcribe Video", key='transcribe_video')
    if transcribe_button:
        if video_path:
            logging.info(f"Starting transcription for video: {video_path}")
            # Play the video
            st.video(video_path)
            transcription = transcribe_video(video_path)
            if transcription:
                st.text_area("Transcription", transcription, height=300)
                save_transcription(transcription, 'transcription.txt')
                st.success("Transcription completed successfully!")

                # Extract key phrases of 2 words each
                key_phrases = extract_key_phrases(transcription, num_words=2)
                logging.info(f"Extracted Key Phrases: {key_phrases}")

                # Fetch Wikipedia articles for key phrases
                fetch_wikipedia_articles(key_phrases, os.path.join(directory, 'html'))

                # Parse and index the new HTML documents into Neo4j
                new_documents = parse_documents(directory)
                for doc in new_documents:
                    title = doc.get('title', 'Untitled')
                    content = doc.get('content', '')
                    graph.create_document_node(title, content)
                    logging.info(f"Indexed document: {title}")

                st.success("Wikipedia articles fetched and indexed successfully!")

        else:
            st.error("Please enter the path to the video.")

    # Automatic Indexing of Documents
    logging.info(f"Starting indexing process for directory: {directory}")
    documents = parse_documents(directory)
    for doc in documents:
        title = doc.get('title', 'Untitled')
        content = doc.get('content', '')
        graph.create_document_node(title, content)
        logging.info(f"Indexed document: {title}")

    # Get top 5 keywords from Neo4j
    top_keywords = graph.get_top_keywords(limit=5)
    logging.info(f"Top keywords: {top_keywords}")

    # Display top keywords as a word cloud
    if top_keywords:
        display_word_cloud(top_keywords)

    # Display top keywords as clickable links
    if top_keywords:
        display_clickable_keywords(top_keywords)

    # Ask a Question Section
    st.header("Ask a Question")
    chat_input = st.text_input("Your Question", key='chat_input')
    ask_button = st.button("Ask Question", key='ask_question')
    if ask_button:
        if chat_input:
            logging.info(f"Processing question: {chat_input}")
            if os.path.exists('transcription.txt'):
                with open('transcription.txt', 'r') as file:
                    transcription = file.read()

            # Combine chat input and transcription for keyword extraction
            combined_text = chat_input + " " + transcription
            keywords = extract_keywords_from_text(combined_text)
            logging.info(f"Extracted Keywords: {keywords}")

            # Get LLM Response
            if not db_manager.schemas:
                st.warning("No schemas available. Ensure CSV files are loaded.")
            else:
                response = get_llm_response(
                    chat_input,
                    transcription,
                    db_manager.schemas,
                    keywords,
                    llm_model,
                    tokenizer,
                    db_manager.conn,
                    graph
                )
                st.text_area("Response", response, height=300)

                # Query Neo4j based on question keywords
                neo4j_documents = graph.search_documents(keywords)
                if neo4j_documents:
                    st.subheader("Related Documents from Knowledge Graph:")
                    for doc in neo4j_documents:
                        st.write(f"Title: {doc['title']}")
                        st.write(f"Content: {doc['content'][:200]}...")  # Display a snippet

                    # Display graph of documents
                    # Prepare graph data
                    graph_data = {
                        "nodes": [{"id": doc['title'], "label": doc['title']} for doc in neo4j_documents],
                        "edges": []  # Add logic to populate edges based on relationships
                    }
                    display_graph(graph_data)
        else:
            st.error("Please enter your question.")

    # Close connections
    graph.close()
    db_manager.close()

if __name__ == "__main__":
    main()