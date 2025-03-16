import streamlit as st
import os
from rag_core import TextProcessor, VectorStore

st.title("RAG Chatbot (No More Errors)")

# Directories
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

VECTOR_STORE_DIR = "chroma_db"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Initialize
processor = TextProcessor()
vector_store = VectorStore(persist_directory=VECTOR_STORE_DIR)

# Upload section
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
if uploaded_file:
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Uploaded {uploaded_file.name}")

    # Process & add
    chunks = processor.process_file(file_path)
    vector_store.add_texts(chunks)
    st.info(f"Processed {len(chunks)} chunks and added to the VectorStore.")

# Query section
query = st.text_input("Enter a query for similarity search")
if st.button("Search"):
    if query.strip():
        results = vector_store.similarity_search(query)
        st.write("### Results:")
        if results:
            for r in results:
                st.write(f"**Chunk**: {r['metadata'].get('chunk_index')}")
                st.write(r["text"])
                st.write("---")
        else:
            st.write("No results found.")
    else:
        st.warning("Please enter a query.")

# Stats section
if st.button("Show Collection Stats"):
    stats = vector_store.get_collection_stats()
    st.write(stats)
