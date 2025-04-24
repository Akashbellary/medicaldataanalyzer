import os
import streamlit as st
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from dotenv import load_dotenv

from document_processors import load_multimodal_data, load_data_from_directory
from utils import set_environment_variables

# Load environment variables
load_dotenv()

# Set up the page configuration
st.set_page_config(layout="wide")

# Initialize settings
def initialize_settings():
    Settings.embed_model = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5", truncate="END")
    Settings.llm = NVIDIA(model="meta/llama-3.1-70b-instruct")
    Settings.text_splitter = SentenceSplitter(chunk_size=600)

# Create index from documents
def create_index(documents):
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index("multimodal-rag")

    # Set up Pinecone vector store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        dimension=1024,  # Match the embedding model output dimension
        namespace="multimodal_data"  # Optional namespace for organization
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Main function to run the Streamlit app
def main():
    initialize_settings()

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.title("Multimodal RAG")
        
        input_method = st.radio("Choose input method:", ("Upload Files", "Enter Directory Path"))
        
        if input_method == "Upload Files":
            uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True)
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    documents = load_multimodal_data(uploaded_files)
                    if documents:
                        st.session_state['index'] = create_index(documents)
                        st.success("Documents processed and indexed successfully!")
        else:
            directory_path = st.text_input("Enter directory path containing documents")
            if directory_path and os.path.exists(directory_path):
                with st.spinner("Processing documents..."):
                    documents = load_data_from_directory(directory_path)
                    if documents:
                        st.session_state['index'] = create_index(documents)
                        st.success("Documents processed and indexed successfully!")

    with col2:
        if 'index' in st.session_state:
            st.title("Chat with your documents")
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message:
                        with st.expander("View Sources"):
                            st.write(message["sources"])

            # Chat input
            if prompt := st.chat_input("Ask a question about your documents"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get response from the index
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state['index'].as_query_engine().query(prompt)
                        
                        # Format the response
                        response_text = response.response
                        sources = []
                        for node in response.source_nodes:
                            if hasattr(node.node, 'metadata'):
                                source_info = f"Source: {node.node.metadata.get('source', 'Unknown')}"
                                if 'page_num' in node.node.metadata:
                                    source_info += f", Page: {node.node.metadata['page_num']}"
                                sources.append(source_info)
                        
                        # Display response
                        st.markdown(response_text)
                        
                        # Add sources in an expander
                        if sources:
                            with st.expander("View Sources"):
                                for source in sources:
                                    st.write(source)
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "sources": sources
                        })

            # Add clear chat button
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()

if __name__ == "__main__":
    main()