import streamlit as st
import os
from rag_core import TextProcessor, VectorStore

# Define directories for data storage
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
if not os.path.exists(VECTOR_STORE_DIR):
    os.makedirs(VECTOR_STORE_DIR)

# Initialize the vector store
vector_store = VectorStore(persist_directory=VECTOR_STORE_DIR)

st.title("RAG Writing Assistant")

st.sidebar.header("File Upload")
uploaded_file = st.sidebar.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success(f"File saved: {file_path}")

    # Process the uploaded file using TextProcessor
    tp = TextProcessor(chunk_size=750, chunk_overlap=150)
    chunks = tp.process_file(file_path)
    
    if chunks:
        st.write(f"Processed {len(chunks)} chunks from the file.")
        # Add chunks to the vector store
        vector_store.add_texts(chunks)
        st.write("Added text chunks to the vector store.")

st.sidebar.header("Query Collection")
query_text = st.sidebar.text_input("Enter your query")
if st.sidebar.button("Search"):
    if query_text:
        results = vector_store.similarity_search(query_text, n_results=5)
        st.write("### Search Results:")
        if results:
            for res in results:
                chunk_index = res['metadata'].get('chunk_index', 'N/A')
                st.markdown(f"**Chunk Index:** {chunk_index}")
                st.write(res['text'])
                st.markdown("---")
        else:
            st.write("No results found.")
    else:
        st.warning("Please enter a query.")
