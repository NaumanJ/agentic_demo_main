# batch_indexer.py

import os
import pandas as pd
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import psycopg2
from sqlalchemy import create_engine
import wikipedia
import nltk
from nltk.corpus import stopwords
from nltk import ngrams
from nltk import ngrams, pos_tag, word_tokenize
import requests
from bs4 import BeautifulSoup
import logging
import os
from urllib.parse import urljoin, urlparse
# PostgreSQL Manager Class
from postgres_manager import PostgresManager
from neo4j_knowledge_graph import Neo4jKnowledgeGraph
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
import milvus_client
from milvus_client import MilvusClient
from transformers import pipeline 
from sentence_transformers import SentenceTransformer
from video_annotation_loader import AnnotationLoader 
#import video_annotation_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


document_repo_path = '/app/document_repo/'

def fetch_html(url):
    """
    Fetch the HTML content from a given URL.
    
    Args:
    url (str): The URL to fetch.
    
    Returns:
    str: HTML content of the URL.
    """
    response = requests.get(url)
    if response.status_code == 200:
        logging.info(f"Successfully fetched content from {url}")
        return response.text
    else:
        logging.error(f"Failed to fetch content from {url} - Status Code: {response.status_code}")
        return None

def extract_links(html_content, base_url):
    """
    Extracts all valid internal links from the HTML content of a page.
    
    Args:
    html_content (str): The HTML content of the page.
    base_url (str): The base URL of the page to resolve relative URLs.
    
    Returns:
    list: A list of full URLs for internal Wikipedia links.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    links = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]

        # Check if the link is a relative link or an internal link to Wikipedia
        if href.startswith("/wiki/") and ":" not in href:
            full_url = urljoin(base_url, href)
            links.append(full_url)

    logging.info(f"Extracted {len(links)} links from the page.")
    return links

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

from bs4 import BeautifulSoup
import os

def parse_html(file_path):
    # Read the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    # Remove script, style, and other non-visible elements
    for element in soup(['script', 'style', 'noscript', 'iframe', 'meta', 'link']):
        element.extract()
    
    # Get the text and strip extra spaces and newlines
    text_content = soup.get_text(separator=' ', strip=True)
    
    # Return the title and the cleaned text content
    return {
        'title': os.path.basename(file_path),  # File name as title
        'content': text_content
    }
def parse_text_file(file_path):
    with fitz.open(file_path) as doc:
        page = doc.load_page(0)  # Assuming a single-page text file
        text = page.get_text()
        return {'title': os.path.basename(file_path), 'content': text}

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
            if file.endswith('.csv'):
                # Here csv is parsed as text 
                documents.append(parse_text_file(file_path))
            elif file.endswith('.txt') or file.endswith('.txt'):
                documents.append(parse_text_file(file_path))
            elif file.endswith('.doc') or file.endswith('.docx'):
                documents.append(parse_doc(file_path))
            elif file.endswith('.html'):
                documents.append(parse_html(file_path))
            elif file.endswith('.pdf'):
                documents.append(parse_pdf(file_path))
            elif file.endswith('.xls') or file.endswith('.xlsx'):
                documents.append(parse_xls(file_path))
    return documents

# Extract Keywords Functions

def extract_key_phrases(text, num_words=2):
    """
    Extracts key phrases that include proper nouns and excludes common nouns.
    It generates 'num_words'-word phrases from proper nouns.
    """
    stop_words = set(stopwords.words('english'))

    # Tokenize and apply part-of-speech (POS) tagging
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    # Filter for proper nouns (NNP: singular proper noun, NNPS: plural proper noun)
    proper_nouns = [word for word, tag in pos_tags if tag in ('NNP', 'NNPS') and word.lower() not in stop_words]

    # Generate n-grams (phrases) from the proper nouns
    phrases = [' '.join(gram) for gram in ngrams(proper_nouns, num_words)]

    return phrases

# Video Transcription Functions

def transcribe_video(video_path):
    # Placeholder for transcription function
    transcription = """00:00 -> 00:30	The video features a soccer match between two teams, Argentina and Spain. The players are seen standing on the field, dressed in their respective team uniforms. The match begins and both teams are seen playing aggressively. The players are seen running after the ball, trying to score goals. The goalkeeper is also seen diving to stop the ball. The video shows several goals being scored by both teams. The players are seen celebrating their goals. The video also shows the audience cheering for the players. The video ends with the final score being 4-3.
00:30 -> 01:00	The video shows a soccer match between two teams, one wearing red and the other wearing white. The red team scores a goal and celebrates, while the players from the white team look dejected. The goal is scored by a player named Messi, as indicated by the name on the screen. The video also shows a replay of the goal being scored. The players are seen running on the field, with the red team players moving in the same direction and the white team players moving in the opposite direction. The players are wearing jerseys with numbers and names on the back. The field is green with white boundary lines. The stands in the background are filled with spectators. The video is shot from a side angle, providing a clear view of the players and the goal.
01:00 -> 01:30	The video features a group of men playing soccer on a field. The main focus is on a player wearing a blue and white striped jersey, who is dribbling the ball towards the goal. As he approaches the goal, he is closely guarded by a defender in a red jersey. The goalkeeper, dressed in black, is ready to defend the goal. In the background, other players in red and white jerseys are watching the play unfold. The video captures the intensity and excitement of the game as the player in the blue and white striped jersey attempts to score a goal.
01:30 -> 02:00	The video appears to be a simple title screen with no discernible characters or actions taking place. The only information provided is the text "The Big Book of Near-Death Experiences" which could be the title of a book or documentary. There is no further information about the content of the video, the faces of the characters, or any actions taking place. The video seems to be a title screen or intro for a documentary or book about near-death experiences.
02:00 -> 02:30	The video shows a group of men playing soccer on a field. The team in red is trying to prevent the other team in white from scoring, and they are successful in blocking the goal. The players are running around the field, trying to score or defend as needed. One of the players from the red team scores a goal, and the audience cheers in response. The players are shown congratulating each other and celebrating their victory. The video also shows a replay of the goal that was scored.
02:30 -> 03:00	The video shows a group of men playing soccer on a field. The team in red is trying to prevent the other team in white from scoring, and they are successful in stopping the ball from entering their goal. The goalie is shown diving to his left to stop the ball. The players are running around the field trying to score and prevent the other team from scoring. The video is paused to show the action and the players' movements. The players are wearing jerseys with numbers printed on them, such as 7 and 21. The video is a close-up shot of the players and the action on the field.
03:00 -> 03:30	The video features a soccer match with a player named Cristiano Ronaldo, who is seen scoring a goal. The match is taking place in a stadium with a large crowd watching. After Ronaldo scores, he is seen celebrating his goal. The goalkeeper, who is also named Cristiano Ronaldo, is seen diving to the ground in an attempt to stop the ball. The video also shows other players on the field, including one named Lionel Messi, who is seen running towards the goal. The referee is also present on the field, signaling for offside. The video ends with the final whistle being blown.
03:30 -> 04:00	The video shows a group of men playing soccer on a field. The main focus is on a player wearing a white jersey who scores a goal and celebrates. Other players are also shown on the field, including one who scores a goal and celebrates with teammates. The players are seen running around the field, trying to score goals. The stadium is filled with spectators, cheering for their teams. The video captures the excitement and energy of a live soccer match.
04:00 -> 04:00	The video shows a group of people playing soccer on a field. The team wearing red is trying to score while the goalkeeper is guarding the goal. The soccer ball is in motion, and the players are actively engaged in the game. There are at least ten people visible on the field, including the goalkeeper. The players are spread out across the field, with some closer to the goal and others further away. The scene captures the excitement and action of a soccer match.

The World Cup Qatar 2022 was a thrilling event filled with surprises, historic moments, and exceptional talent, providing a captivating spectacle for football fans worldwide. The group stage kicked off with Group A, where the Netherlands, led by Virgil van Dijk, emerged as the obvious favorite and ultimately topped the group. Ecuador, with their young core, gave Senegal a scare but eventually finished third, while host nation Qatar fell flat, becoming the first host to finish the tournament without a point. In Group B, England showcased their strength with dominant wins over Iran (6-2) and Wales (3-0). The USA met expectations, with Christian Pulisic's historic strike against Iran uniting a nation and securing their progress to the knockout stages.
Group C brought one of the most shocking upsets in World Cup history, as Saudi Arabia pulled off a stunning 2-1 victory over eventual champions Argentina. Lionel Messi and his team fought back, securing wins against Poland and Mexico to top the group. Mexico, despite their efforts, failed to advance from the group stage for the first time in 40 years, falling to goal differential. In Group D, Denmark disappointed, failing to advance out of a seemingly easy group. France, the eventual runners-up, dominated with wins over Australia and Denmark, only losing to Tunisia while resting their starters. Australia, meanwhile, snuck through to the knockout stages, coming from down under to secure second place.
Group E saw Japan start off unexpectedly strong with a 2-1 upset over Germany, who, for the second time in history, did not make it past the group stage. Spain, despite a lackluster performance besides their 7-0 win over Costa Rica, continued on to the next round. In Group F, Morocco foreshadowed their unpredicted success early on, advancing at the top of their group with impressive performances against Belgium and Canada. Croatia, the 2018 World Cup runners-up, eked out a draw and edged past Belgium to secure their spot in the knockout stages.
Group G featured initial favorites Brazil advancing relatively easily, despite a 92nd-minute win for Cameroon. Switzerland showcased their resilience, getting the best of the Serbians in a hard-fought group. Group H saw Portugal advance through the group despite losing to South Korea and a temper tantrum from captain Cristiano Ronaldo. South Korea and Uruguay were tied on goal differential, but after a dramatic final goal against Portugal, the South Koreans flexed their resilience and advanced to the knockout stages.
The semi-finals featured back-to-back appearances for France and Croatia. Argentina, led by the indomitable Lionel Messi, was able to get past Croatia, brushing them aside in a 3-0 victory. The France vs. Morocco semi-final was a tense battle, with France scoring within the first five minutes and leaving Morocco chasing the game. Preying on late-game desperation from the underdogs, France slotted a second goal home in the 79th minute, dispatching the dreams of a nation.
The highly anticipated final between France and Argentina was a memorable game that showcased the exceptional talent of both teams. Argentina capitalized on a sloppy tackle, leading to Messi's first penalty of the game. Building on the momentum, Ángel Di María scored Argentina's thrilling second goal of the first half. Just as fans started losing hope for France, Kylian Mbappé scored France's first penalty in the 70th minute, following it with their second goal only a minute later. After a goal each during extra time, the final went to penalty shootouts for the first time since 2006. Argentina took the cup after an exciting 4-2 penalty kick win, securing Lionel Messi's legacy as one of the greatest football players of all time.
"""
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

def extract_corpus(source_urls, save_dir):
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Iterate over each URL in the source_url list
    for url in source_urls:
        # Fetch the HTML content of the main page
        html_content = fetch_html(url)
        if html_content is None:
            logging.error(f"Failed to retrieve content from {url}.")
            continue

        # Extract internal links from the main page
        links = extract_links(html_content, url)

        # Fetch and save content from each extracted link
        fetch_corpus_from_links(links, save_dir)

# Milvus setup
def init_milvus():
    # Connect to Milvus server
    connections.connect("default", host="localhost", port="19530")
    collection_name = "document_embeddings"


    # Define the schema for Milvus
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # Set dimension based on embedding model
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1024)
    ]
    schema = CollectionSchema(fields=fields, description="Document Embeddings")

    # Create collection if it doesn't exist
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
    else:
        collection = Collection(name=collection_name)

    # Check if the index exists; if not, create one
    if not collection.has_index():
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "IP",  # or "L2" depending on your retrieval needs
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
        )
    
    # Load the collection to prepare it for searching
    collection.load()
    return collection



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


# Main Function

def main():

    # Initialize variables
    transcription = ""
    documents = []
    directory = document_repo_path + 'sports/soccer/txt'
    video_annotation_data = document_repo_path + 'sports/soccer/csv/transcripts'
    html_data = document_repo_path + 'sports/soccer/html'
    save_dir = os.path.join(directory, 'soccer' ,'html')  

    # Initialize Neo4j Knowledge Graph
    graph = Neo4jKnowledgeGraph('bolt://localhost:7687', 'neo4j', 'password')

    # Initialize PostgreSQL Manager
    db_config = {
        'user': 'postgres',
        'password': 'password',
        'host': 'localhost',
        'port': '5433',
        'dbname': 'postgres'
    }
    pg_manager = PostgresManager(user='postgres', password='password', host='localhost', port=5433, database='postgres')

    # Create an instance of AnnotationLoader
    loader = AnnotationLoader(db_config)
    
    # Drop and recreate annotation table for full text indexing
    loader.drop_annotation_table()
    loader.create_annotation_table()

    # Load CSV files into PostgreSQL
    loader.load_annotations_from_csv(video_annotation_data + "/All_2022WC_Spain_Other_FIFA_FOX.csv")
    loader.load_annotations_from_csv(video_annotation_data + "/All_Ronaldo_IG.csv")
    loader.load_annotations_from_csv(video_annotation_data + "/soccerMatch.csv")

    # Perform a global full-text search
    global_search_results = loader.global_search_by_annotation("ronaldo")
    if global_search_results:
        logging.info("Filename, Start Duration, End Duration, Annotation")
        for result in global_search_results:
            logging.info(f"{result[0]}, {result[1]}, {result[2]}, {result[3]}")
    else:
        logging.info("No video annotations found for the string 'ronaldo'")

    # Perform a search by video duration
    search_results = loader.search_video_by_duration("soccerMatch.mp4", "01:03")
    if search_results:
        logging.info("Filename, Start Duration, End Duration, Annotation")
        for result in search_results:
            logging.info(f"{result[0]}, {result[1]}, {result[2]}, {result[3]}")
    else:
        logging.info("No video annotations found for search by duration '01:03' in 'soccerMatch.mp4'")

    # Perform a search by annotation for a specific video
    search_results = loader.search_video_by_annotation("soccerMatch.mp4", "ronaldo")
    if search_results:
        logging.info("Filename, Start Duration, End Duration, Annotation")
        for result in search_results:
            logging.info(f"{result[0]}, {result[1]}, {result[2]}, {result[3]}")
    else:
        logging.info("No video annotations found for search by annotation 'ronaldo' in 'soccerMatch.mp4'")

    # Close the connection
    loader.close()
    
    soccer_url = "https://en.wikipedia.org/wiki/The_Best_FIFA_Men%27s_Player"
    # olympics_url = "https://en.wikipedia.org/wiki/Olympic_Games"
    # cricket_url = "https://en.wikipedia.org/wiki/Cricket"
    # source_url = [soccer_url, olympics_url, cricket_url]
    source_url = [soccer_url]

    ##### Extract Web Text Corpus
    # for each_url in source_url:
    #     game = each_url.split("/")[-1]
    #     save_dir = os.path.join(html_data, game ,'html')
    #     extract_corpus(source_url, save_dir)

    local_docs = []
    
    # Automatic Indexing of Documents to neo4j
    logging.info(f"Starting Neo4j indexing process for directory: {directory}")
    documents = parse_documents(html_data)
    documents = parse_documents(directory)
    for doc in documents:
        title = doc.get('title', 'Untitled')
        content = doc.get('content', '')
        graph.create_document_node(title, content)
        logging.info(f"Indexed document: {title}")
        local_docs.append(content)
    
    documents = parse_documents(directory)
    for doc in documents:
        logging.info("Indexing in neo4j")
        title = doc.get('title', 'Untitled')
        content = doc.get('content', '')
        graph.create_document_node(title, content)
        logging.info(f"Indexed document: {title}")
        local_docs.append(content)

    # Initialize Milvus collection and embedding model
    milvus_client = MilvusClient()
    milvus_collection = init_milvus()
    milvus_embedding_model = milvus_client.get_embedding_model()


    # Load all local documents and insert them into Milvus
    for filename in os.listdir(document_repo_path + "sports/soccer/txt"):
        try:
            with open(os.path.join(document_repo_path + "sports/soccer/txt", filename), "r", encoding="utf-8") as file:
                local_docs.append({"content": file.read()})
        except Exception as e:
            logging.warning(f"Failed to load file {filename}: {e}")

    # milvus_client.insert_documents_to_milvus(milvus_collection, local_docs, milvus_embedding_model)
    # Close connections
    graph.close()
    pg_manager.close()

    # Close other resources if needed
    logging.info("Processing complete.")


if __name__ == "__main__":
    main()