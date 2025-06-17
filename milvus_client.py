from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from transformers import pipeline 
from sentence_transformers import SentenceTransformer
import torch

# Class for Milvus functionality
class MilvusClient:
    # Initialize the summarization pipeline
    # Specify a better model for summarization and use the GPU if available
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1 if torch.cuda.is_available() else -1)

    def __init__(self, host="localhost", port="19530", collection_name="document_embeddings"):
        self.collection_name = collection_name
        connections.connect("default", host=host, port=port)
        self.collection = self._initialize_collection()

    def _initialize_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1024)
        ]
        schema = CollectionSchema(fields=fields, description="Document Embeddings")

        if not utility.has_collection(self.collection_name):
            collection = Collection(name=self.collection_name, schema=schema)
        else:
            collection = Collection(name=self.collection_name)

        if not collection.has_index():
            collection.create_index(
                field_name="embedding",
                index_params={"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
            )

        collection.load()
        return collection

    def insert_document(self, document, embedding_model):
        embedding = embedding_model.encode(document)
        # Insert data into Milvus
        self.collection.insert([document, embedding])
        self.collection.flush()
    
    def summarize_content(content, max_length=150, min_length=50):
        """
            Summarizes content if it exceeds a specified length.

            Args:
                content (str): The content to potentially summarize.
                max_length (int): Maximum length of the summarized text.
                min_length (int): Minimum length of the summarized text.

            Returns:
                str: The original content if it's short enough, otherwise the summarized content.
        """
        from transformers import pipeline

        return content
    
    def summarize_content(self, content, max_length=1024, min_length=128):
        content = " ".join(content) if isinstance(content, list) else str(content)
        summarized_text=""
        if len(content) > 1024:
            try:
                summarized_text = MilvusClient.summarizer(content, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            except IndexError as e:
                summarized_text = ""
        return summarized_text
   
    # Function to add documents to Milvus
    def insert_documents_to_milvus(self, collection, documents, embedding_model):
        # Prepare content list with summarization
        content_list = [MilvusClient.summarize_content(self, doc) for doc in documents]

        # Generate embeddings for the (possibly summarized) content
        embeddings = embedding_model.encode(content_list)
        embeddings_list = [embedding.tolist() for embedding in embeddings]

        # Check if the length of content_list and embeddings_list match
        if len(content_list) != len(embeddings_list):
            raise ValueError("Mismatch between content list and embeddings list length.")

        # Prepare data to match Milvus schema
        entities = []
        for content, embedding in zip(content_list, embeddings_list):
            # Create a dictionary for each document
            entities.append({
                "content": content,  # Ensure this is a string, not a list
                "embedding": embedding  # Embedding should be a list of floats
            })
        # Insert each entity into Milvus
        collection.insert(entities)
        collection.flush()


    def search_documents(self, query, embedding_model, limit=5):
        query_embedding = embedding_model.encode([query])[0].tolist()
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["content"]
        )
        return [result.entity.get('content') for result in results[0]]
    
    # Sentence Transformer for embeddings
    def get_embedding_model(self):
        return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def split_text_into_chunks(text, max_tokens=1024):
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_tokens):
            chunks.append(" ".join(words[i:i + max_tokens]))
        return chunks

    # Milvus vector search
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

    def search_documents_milvus(collection, query, embedding_model, limit=5):
        query_embedding = embedding_model.encode([query])[0].tolist()
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["content"]
        )
        
        # Retrieve results
        retrieved_docs = [result.entity.get('content') for result in results[0]]
        return retrieved_docs