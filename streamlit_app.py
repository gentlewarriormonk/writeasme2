"""
Streamlit app for RAG Writing Assistant - Ultra-simplified version
This is a streamlined Streamlit app for the RAG Writing Assistant.
It's designed to be easily deployed to Streamlit Cloud with minimal dependencies.
"""

import os
import sys
import logging
import streamlit as st
from pathlib import Path
import tempfile
import time

# Import the RAG core module
from rag_core import RAGWritingAssistant

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CORPUS_DIR = "corpus"
DEFAULT_VECTOR_DB_DIR = "vector_db_data"

# Create directories if they don't exist
os.makedirs(DEFAULT_CORPUS_DIR, exist_ok=True)
os.makedirs(DEFAULT_VECTOR_DB_DIR, exist_ok=True)

# Initialize session state
if 'rag_assistant' not in st.session_state:
    st.session_state.rag_assistant = None
if 'initialization_status' not in st.session_state:
    st.session_state.initialization_status = "not_started"
if 'corpus_stats' not in st.session_state:
    st.session_state.corpus_stats = None
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False

def initialize_assistant():
    """Initialize the RAG assistant"""
    try:
        st.session_state.initialization_status = "in_progress"
        
        # Get API key from session state or secrets
        openai_api_key = st.session_state.get('openai_api_key', '')
        
        # Try to get from secrets if not in session state
        if not openai_api_key and 'api_keys' in st.secrets:
            openai_api_key = st.secrets.api_keys.get('openai', '')
        
        # Set environment variable
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
        else:
            st.error("Please provide your OpenAI API key")
            st.session_state.initialization_status = "error"
            return
        
        # Initialize RAG assistant
        st.session_state.rag_assistant = RAGWritingAssistant(
            corpus_directory=DEFAULT_CORPUS_DIR,
            vector_db_directory=DEFAULT_VECTOR_DB_DIR,
            embedding_model="text-embedding-3-small",
            llm_model="gpt-4o"
        )
        
        # Process corpus
        num_chunks = st.session_state.rag_assistant.process_corpus()
        
        # Update status
        if num_chunks > 0:
            st.session_state.initialization_status = "complete"
            st.session_state.corpus_stats = st.session_state.rag_assistant.get_corpus_stats()
            logger.info(f"RAG assistant initialized with {num_chunks} text chunks")
        else:
            st.session_state.initialization_status = "complete"
            st.session_state.corpus_stats = {"corpus_files": 0, "vector_documents": 0, "content_types": {}}
            logger.info("RAG assistant initialized. No text files found in corpus directory.")
        
    except Exception as e:
        error_msg = str(e)
        st.session_state.initialization_status = "error"
        logger.error(f"Error initializing RAG assistant: {error_msg}")
        st.error(f"Error initializing system: {error_msg}")

def upload_file(uploaded_file):
    """Upload and process a file"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        # Save to corpus directory
        file_path = os.path.join(DEFAULT_CORPUS_DIR, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Process the file
        num_chunks = st.session_state.rag_assistant.add_file(file_path)
        
        # Save the updated state
        st.session_state.rag_assistant.save_state()
        
        # Update corpus stats
        st.session_state.corpus_stats = st.session_state.rag_assistant.get_corpus_stats()
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return num_chunks
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise

def reprocess_corpus():
    """Reprocess the entire corpus"""
    try:
        # Reprocess corpus
        num_chunks = st.session_state.rag_assistant.process_corpus(reprocess=True)
        
        # Save the updated state
        st.session_state.rag_assistant.save_state()
        
        # Update corpus stats
        st.session_state.corpus_stats = st.session_state.rag_assistant.get_corpus_stats()
        
        return num_chunks
    except Exception as e:
        logger.error(f"Error reprocessing corpus: {str(e)}")
        raise

def generate_content(query, style_adjustments=None):
    """Generate content based on query"""
    try:
        # Generate content
        content = st.session_state.rag_assistant.generate_content(query, style_adjustments)
        return content
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        raise

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="RAG Writing Assistant",
        page_icon="✍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4a6fa5;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .status-complete {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-progress {
        color: #ffc107;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #4a6fa5;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #dee2e6;
        padding-bottom: 0.5rem;
    }
    .help-text {
        font-size: 0.9rem;
        color: #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">RAG Writing Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your personal writing assistant that captures your authentic voice</p>', unsafe_allow_html=True)
    
    # Sidebar - API Setup and Corpus Management
    with st.sidebar:
        st.markdown('<h2 class="section-header">API Setup</h2>', unsafe_allow_html=True)
        
        # API key input
        openai_api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        
        if st.button("Save API Key"):
            if openai_api_key:
                st.session_state.openai_api_key = openai_api_key
                st.session_state.api_key_set = True
                st.success("API key saved!")
                
                # Reset initialization if key changed
                if st.session_state.initialization_status == "complete":
                    st.session_state.initialization_status = "not_started"
                    st.experimental_rerun()
            else:
                st.error("Please provide your OpenAI API key")
        
        st.markdown('<h2 class="section-header">Corpus Management</h2>', unsafe_allow_html=True)
        
        # Corpus stats
        if st.session_state.corpus_stats:
            st.markdown("### Corpus Statistics")
            st.write(f"**Files:** {st.session_state.corpus_stats.get('corpus_files', 0)}")
            st.write(f"**Text Chunks:** {st.session_state.corpus_stats.get('vector_documents', 0)}")
            
            content_types = st.session_state.corpus_stats.get('content_types', {})
            if content_types:
                st.markdown("**Content Types:**")
                for content_type, count in content_types.items():
                    st.write(f"- {content_type}: {count}")
        
        # File upload
        st.markdown("### Upload Text File")
        uploaded_file = st.file_uploader("Choose a text file", type="txt", help="Upload a .txt file to add to your corpus")
        
        if uploaded_file is not None:
            if st.button("Process File"):
                if st.session_state.initialization_status == "complete":
                    with st.spinner("Processing file..."):
                        try:
                            num_chunks = upload_file(uploaded_file)
                            st.success(f"File processed into {num_chunks} chunks")
                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
                else:
                    st.error("System not initialized. Please initialize first.")
        
        # Reprocess corpus
        if st.button("Reprocess All Files"):
            if st.session_state.initialization_status == "complete":
                with st.spinner("Reprocessing corpus..."):
                    try:
                        num_chunks = reprocess_corpus()
                        st.success(f"Corpus reprocessed into {num_chunks} chunks")
                    except Exception as e:
                        st.error(f"Error reprocessing corpus: {str(e)}")
            else:
                st.error("System not initialized. Please initialize first.")
        
        # Help section
        st.markdown('<h2 class="section-header">Help</h2>', unsafe_allow_html=True)
        with st.expander("File Naming Tips"):
            st.markdown("""
            For best results, name your files with content type and tags:
            ```
            essay_1_title.txt
            podcast_episode_2.txt
            substack_3_topic.txt
            uni_reflection_4.txt
            ```
            
            The system will automatically detect content types from prefixes:
            - essay_
            - podcast_
            - substack_
            - uni_reflection_
            """)
        
        with st.expander("Style Adjustments"):
            st.markdown("""
            Add style instructions in your prompt:
            - "make this more humorous"
            - "make this more formal"
            - "make this more concise"
            
            Or use the style dropdown to add these automatically.
            """)
    
    # Main content area
    if not st.session_state.api_key_set:
        st.warning("Please enter your OpenAI API key in the sidebar to get started")
    else:
        # Initialize button
        if st.session_state.initialization_status == "not_started":
            if st.button("Initialize System"):
                with st.spinner("Initializing system..."):
                    initialize_assistant()
        
        # Status message
        if st.session_state.initialization_status == "in_progress":
            st.markdown('<p class="status-progress">⟳ System initialization in progress...</p>', unsafe_allow_html=True)
        elif st.session_state.initialization_status == "error":
            st.markdown('<p class="status-error">✗ Error initializing system</p>', unsafe_allow_html=True)
        elif st.session_state.initialization_status == "complete":
            st.markdown('<p class="status-complete">✓ System initialized and ready</p>', unsafe_allow_html=True)
            
            # Content generation section
            st.markdown('<h2 class="section-header">Generate Content in Your Style</h2>', unsafe_allow_html=True)
            
            # Query input
            query = st.text_area("Enter your request", height=150, 
                                placeholder="e.g., 'Write a short essay about artificial intelligence' or 'Draft a podcast intro about climate change'")
            
            # Style adjustments
            col1, col2 = st.columns([3, 1])
            with col1:
                style_options = [
                    "None",
                    "Make this more formal",
                    "Make this more conversational",
                    "Make this more humorous",
                    "Make this more technical",
                    "Make this more concise",
                    "Make this more detailed"
                ]
                style_adjustment = st.selectbox("Style Adjustment", style_options)
            
            with col2:
                generate_button = st.button("Generate", use_container_width=True)
            
            # Process style adjustment
            final_query = query
            final_style = None
            
            if style_adjustment != "None" and query:
                style_text = style_adjustment.lower()
                
                # Check if query already has style instructions
                import re
                style_regex = r'\[(make this .*?)\]|\((make this .*?)\)|make this (more|less) (\w+)'
                if re.search(style_regex, query, re.IGNORECASE):
                    # Replace existing style instruction
                    final_query = re.sub(style_regex, f"[{style_text}]", query, flags=re.IGNORECASE)
                else:
                    # Add style instruction at the end
                    final_query = f"{query} [{style_text}]"
                
                final_style = style_text
            
            # Generate content
            if generate_button and query:
                with st.spinner("Generating content..."):
                    try:
                        content = generate_content(final_query, final_style)
                        
                        # Display result
                        st.markdown('<h3>Generated Content</h3>', unsafe_allow_html=True)
                        st.markdown("""---""")
                        st.markdown(content)
                        st.markdown("""---""")
                        
                        # Copy button
                        if st.button("Copy to Clipboard"):
                            st.write("Content copied to clipboard! (Note: This works when running locally)")
                            try:
                                import pyperclip
                                pyperclip.copy(content)
                            except ImportError:
                                st.info("Pyperclip not available. Copy manually by selecting the text.")
                    except Exception as e:
                        st.error(f"Error generating content: {str(e)}")
            elif generate_button:
                st.error("Please enter a request")

if __name__ == "__main__":
    main()
