import sounddevice as sd
import numpy as np
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from io import BytesIO
import streamlit as st
from neo4j import GraphDatabase
from langchain.agents import Agent, Tool
import ray
import riva

# Initialize Riva API
auth = riva.auth.Auth(uri="localhost:50051")

# Step 1: Capture Audio Input
def record_audio(duration=5, fs=44100):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Recording complete.")
    return audio

# Step 2: Speech-to-Text using NVIDIA Riva
def speech_to_text(audio):
    audio_file = "input.wav"
    sd.write(audio_file, audio, 44100)
    with open(audio_file, "rb") as f:
        audio_data = f.read()
    response = riva.audio.recognize(audio_data, uri="localhost:50051", auth=auth)
    return response.text

# Step 3: Web Search
def google_search(query, api_key, cse_id):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id).execute()
    return res['items']

# Step 4: Web Scraping
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    snippets = [p.get_text() for p in paragraphs[:3]]  # Get the first 3 paragraphs
    return snippets

# Step 5: Parse PDFs
def parse_pdf(pdf_url):
    response = requests.get(pdf_url)
    with fitz.open(stream=response.content, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# Step 6: OCR for Images
def ocr_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    text = pytesseract.image_to_string(image)
    return text

# Step 7: Text-to-Speech using NVIDIA Riva
def text_to_speech(text, output_file='output.wav'):
    response = riva.audio.synthesize(text, uri="localhost:50051", auth=auth)
    with open(output_file, "wb") as f:
        f.write(response.audio)
    return output_file

# Step 8: Document Classification using NVIDIA NeMo
def classify_document(text):
    model = nemo.collections.nlp.models.TokenClassificationModel.from_pretrained(model_name="nlp/bert-base-cased-finetuned-conll03-english")
    results = model.predict([text])
    return results

# Step 9: Build Knowledge Graph using Neo4j
class Neo4jKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_document_node(self, title, url, classification):
        with self.driver.session() as session:
            session.write_transaction(self._create_document_node, title, url, classification)

    @staticmethod
    def _create_document_node(tx, title, url, classification):
        tx.run("CREATE (d:Document {title: $title, url: $url, classification: $classification})",
               title=title, url=url, classification=classification)

    def create_relationship(self, title1, title2, relationship):
        with self.driver.session() as session:
            session.write_transaction(self._create_relationship, title1, title2, relationship)

    @staticmethod
    def _create_relationship(tx, title1, title2, relationship):
        tx.run("MATCH (d1:Document {title: $title1}), (d2:Document {title: $title2}) "
               "CREATE (d1)-[:RELATED_TO {type: $relationship}]->(d2)",
               title1=title1, title2=title2, relationship=relationship)

# Step 10: Generate HTML Report
def generate_html_report(results):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Search Results</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .document-tree { margin-left: 20px; }
        </style>
    </head>
    <body>
        <h1>Search Results</h1>
        <div class="document-tree">
    """
    for item in results:
        html += f"<h2>{item['title']}</h2>"
        html += f"<p>{item['snippet']}</p>"
        if 'pagemap' in item and 'cse_image' in item['pagemap']:
            html += f"<img src='{item['pagemap']['cse_image'][0]['src']}' alt='{item['title']}' width='200'>"
            image_text = ocr_image(item['pagemap']['cse_image'][0]['src'])
            html += f"<p>Image Text: {image_text}</p>"
        snippets = scrape_website(item['link'])
        for snippet in snippets:
            html += f"<p>{snippet}</p>"
        if item['link'].endswith('.pdf'):
            pdf_text = parse_pdf(item['link'])
            html += f"<p>PDF Text: {pdf_text}</p>"
    html += """
        </div>
    </body>
    </html>
    """
    return html

# Step 11: Display Results using Streamlit
def display_results(results, audio_file, html_report, knowledge_graph_html):
    st.title("Search Results")
    st.components.html(html_report, height=600, scrolling=True)
    st.audio(audio_file)
    st.components.html(knowledge_graph_html, height=600, scrolling=True)

# Step 12: Integrate LangChain for Multi-Agent Workflow
class VoiceAssistantAgent(Agent):
    def __init__(self, tools):
        super().__init__(tools)

    def run(self, query):
        results = self.tools[0].run(query)
        return results

class WebSearchTool(Tool):
    def __init__(self, api_key, cse_id):
        self.api_key = api_key
        self.cse_id = cse_id

    def run(self, query):
        return google_search(query, self.api_key, self.cse_id)

class TextToSpeechTool(Tool):
    def run(self, text):
        return text_to_speech(text)

class HTMLReportTool(Tool):
    def run(self, results):
        return generate_html_report(results)

class DocumentClassificationTool(Tool):
    def run(self, text):
        return classify_document(text)

class KnowledgeGraphTool(Tool):
    def __init__(self, uri, user, password):
        self.graph = Neo4jKnowledgeGraph(uri, user, password)

    def run(self, title, url, classification):
        self.graph.create_document_node(title, url, classification)

    def generate_knowledge_graph_html(self):
        with self.graph.driver.session() as session:
            result = session.read_transaction(self._get_knowledge_graph)
        return result

    @staticmethod
    def _get_knowledge_graph(tx):
        query = """
        MATCH (d1:Document)-[r:RELATED_TO]->(d2:Document)
        RETURN d1.title AS title1, d2.title AS title2, r.type AS relationship
        """
        result = tx.run(query)
        nodes = []
        edges = []
        for record in result:
            nodes.append({"id": record["title1"], "label": record["title1"]})
            nodes.append({"id": record["title2"], "label": record["title2"]})
            edges.append({"from": record["title1"], "to": record["title2"], "label": record["relationship"]})
        return {
            "nodes": nodes,
            "edges": edges
        }

# Step 13: Orchestrate the Pipeline Asynchronously using Ray
@ray.remote
def process_query(query, api_key, cse_id, knowledge_linkage):
    tools = [WebSearchTool(api_key, cse_id), TextToSpeechTool(), HTMLReportTool(), DocumentClassificationTool(), KnowledgeGraphTool('bolt://localhost:7687', 'neo4j', 'password')]
    agent = VoiceAssistantAgent(tools)
    results = agent.run(query)
    audio_file = tools[1].run(results[0]['snippet'] if results else "No results found.")
    html_report = tools[2].run(results[:3])  # Scrape the top 3 results

    if knowledge_linkage:
        for item in results[:3]:
            text = item['snippet']
            classification = tools[3].run(text)
            tools[4].run(item['title'], item['link'], classification)

    knowledge_graph_html = tools[4].generate_knowledge_graph_html()

    return results, audio_file, html_report, knowledge_graph_html

def main():
    # Step 1: Record Audio
    audio = record_audio()

    # Step 2: Convert Speech to Text
    query = speech_to_text(audio)

    # Step 3: Perform Google Search and Process Results Asynchronously
    api_key = 'YOUR_GOOGLE_API_KEY'
    cse_id = 'YOUR_CSE_ID'
    knowledge_linkage = st.sidebar.checkbox("Enable Knowledge Linkage", value=False)
    results, audio_file, html_report, knowledge_graph_html = ray.get(process_query.remote(query, api_key, cse_id, knowledge_linkage))

    # Step 4: Display Results using Streamlit
    display_results(results, audio_file, html_report, knowledge_graph_html)

if __name__ == "__main__":
    ray.init()
    main()
