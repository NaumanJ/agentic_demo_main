# 🧠 Multimodal Knowledge-RAG Sports Chatbot

A full-stack AI application integrating a Knowledge Graph (Neo4j), Relational DB (PostgreSQL), Vector Search (Milvus), and LLMs (Groq/Nemotron) to create a smart, voice-enabled chatbot that understands soccer videos and contextual queries from structured and unstructured data.

---

## 🚀 Project Features

### ✅ Key Capabilities
- **🎥 Video Annotation Indexing** – Full-text search on soccer match annotations using PostgreSQL GIN indexes.
- **🔍 Neo4j Knowledge Graph** – Captures relationships between documents, topics, and keywords.
- **📚 RAG with Milvus** – Vector similarity search using SentenceTransformers and Facebook/BART summarization.
- **🧠 LLM Integration** – Supports OpenAI/Groq/Nemotron LLMs for response generation and SQL query generation.
- **🗣️ Voice & TTS Support** – Record voice queries and respond with text-to-speech using `gTTS`.
- **🌍 Streamlit & FastAPI UI** – Dual front-end for interactive demos and secure API-based usage.

---

## 🛠️ Tech Stack

| Component      | Technology Used                          |
|----------------|-------------------------------------------|
| Backend API    | FastAPI                                   |
| Frontend       | Streamlit                                 |
| Graph DB       | Neo4j                                      |
| Relational DB  | PostgreSQL                                 |
| Vector DB      | Milvus                                     |
| Embeddings     | SentenceTransformers (MPNet, MiniLM)       |
| Summarization  | Facebook BART                             |
| LLMs           | LLaMA2 / GPT-Neo / Groq / Nemotron         |
| Audio          | `sounddevice`, `speech_recognition`, `gTTS`|
| Deployment     | Docker + docker-compose                    |

---

## 📂 Project Structure
.
├── api.py # FastAPI backend with RAG and voice response
├── streamlit-demo-asr-milvus-groq.py # UI for live interaction
├── neo4j_knowledge_graph.py # Neo4j graph management
├── postgres_manager.py # PostgreSQL data ingestion
├── postgres_client.py # SQL schema & document search
├── milvus_client.py # Milvus vector DB logic
├── video_annotation_manager.py # Annotation indexing logic
├── video_annotation_loader.py # CSV loading + search from annotations
├── voice_recognition.py # Audio input/output functions
├── batch_indexer.py # Data ingestion for KG and Milvus
├── test_model_with_milvus_rag.py# Script to validate RAG+LLM
├── docker-compose-postgres.yaml # Services: PostgreSQL + Neo4j
├── dockerfile.streamlit.yaml # Streamlit Docker build


---

## 🧪 How It Works

### 1. **Load Data**
- CSV/video annotations are indexed into PostgreSQL and Milvus.
- HTML/PDF/TXT files are parsed and inserted into Neo4j as `Document` nodes.

### 2. **Search Flow**
- Keywords are extracted from a user question.
- Parallel retrieval happens from:
  - Neo4j (graph search)
  - PostgreSQL (text pattern match)
  - Milvus (semantic vector search)

### 3. **Answer Generation**
- Context is passed to LLM (Groq/GPT/Nemotron).
- Answer is generated in a friendly, soccer-aware tone.
- Result is optionally converted to speech.

---

## 🐳 Run the Project with Docker

```bash
docker-compose -f docker-compose-postgres.yaml up -d

## 📌 Demo Use Cases
### “What is Ronaldo doing in the video?”

### “Show me annotations where Messi scores.”

### “Summarize the soccer match played in Spain.”

### “Generate SQL to find goals between 01:00 and 02:00.”

##✨ Future Enhancements
### ✅ Multilingual translation layer

### ✅ Avatar TTS response

### 🔜 Fine-tuned video captioning

### 🔜 Auto-indexing pipeline for new data
