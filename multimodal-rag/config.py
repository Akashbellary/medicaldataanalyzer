import os

# Set your API keys here
NVIDIA_API_KEY = "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"
PINECONE_API_KEY = "pcsk_GH4x3_2W9qVRwe2VvoKN7J8Uv4nR3swfkg7Bk9BWd3EiQPA6t27gbLyq2ZEz5iormcYGp"

# Export to environment variables
os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY