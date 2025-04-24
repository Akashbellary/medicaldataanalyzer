import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.embeddings.nvidia import NVIDIAEmbedding

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index Configuration
index_name = "multimodal-rag"
dimension = 1024  # NVIDIA embedding model dimension

# Create/recreate index
if index_name in pc.list_indexes().names():
    print(f"Index {index_name} already exists. Deleting and recreating...")
    pc.delete_index(index_name)
    
pc.create_index(
    name=index_name,
    dimension=dimension,
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)

print(f"Successfully created index '{index_name}'!")

# Connect to Index
index = pc.Index(index_name)
print("Connected to Pinecone index successfully!") 