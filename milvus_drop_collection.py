from pymilvus import Collection
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# Drop the old collection (use the name of your existing collection)
connections.connect("default", host="localhost", port="19530")
collection = Collection("document_embeddings")
collection.drop()