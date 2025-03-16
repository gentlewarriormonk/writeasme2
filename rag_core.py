"""
RAG Core - Ultra-simplified implementation of RAG Writing Assistant
This file contains the core functionality for the RAG Writing Assistant in a single file
to simplify deployment to Streamlit Cloud.
"""

import os
import logging
import json
from typing import List, Dict, Any
from pathlib import Path
import tempfile
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing required packages - Wrap each import in try/except
openai_imports_success = False
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    openai_imports_success = True
except ImportError as e:
    logger.error(f"Error importing OpenAI packages: {str(e)}")
    logger.info("Please install required packages: pip install langchain-openai")

langchain_imports_success = False
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import StrOutputParser
    from langchain.schema.runnable import RunnablePassthrough
    langchain_imports_success = True
except ImportError as e:
    logger.error(f"Error importing LangChain packages: {str(e)}")
    logger.info("Please install required packages: pip install langchain")

chromadb_import_success = False
try:
    import chromadb
    from chromadb.utils import embedding_functions
    chromadb_import_success = True
except ImportError as e:
    logger.error(f"Error importing ChromaDB: {str(e)}")
    logger.info("Please install chromadb: pip install chromadb==0.4.22")  # Using a specific version known to be stable

# Check if all required imports were successful
all_imports_success = openai_imports_success and langchain_imports_success and chromadb_import_success
if not all_imports_success:
    logger.error("Not all required packages could be imported.")
    logger.info("Please install required packages: pip install langchain langchain-openai openai chromadb==0.4.22")


class TextProcessor:
    """Process text files into chunks with metadata."""
    
    def __init__(self, chunk_size: int = 750, chunk_overlap: int = 150):
        """
        Initialize the text processor.
        
        Args:
            chunk_size: Target size of text chunks in words
            chunk_overlap: Overlap between chunks in words
        """
        if not langchain_imports_success:
            raise ImportError("Required LangChain packages are not available")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size * 4,  # Approximate characters per word
            chunk_overlap=chunk_overlap * 4,
            length_function=len
        )
        logger.info(f"Initialized TextProcessor with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """
        Extract metadata from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Dictionary containing metadata
        """
        # Remove extension and path
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Initialize metadata
        metadata = {
            "title": name_without_ext,
            "content_type": "unknown",
            "tags": []
        }
        
        # Extract content type from prefix
        content_type_patterns = {
            r'^essay[_-]': "essay",
            r'^podcast[_-]': "podcast",
            r'^substack[_-]': "newsletter",
            r'^uni_reflection[_-]': "reflection"
        }
        
        for pattern, content_type in content_type_patterns.items():
            if re.match(pattern, name_without_ext, re.IGNORECASE):
                metadata["content_type"] = content_type
                # Remove prefix from title
                title_parts = re.split(pattern, name_without_ext, 1, re.IGNORECASE)
                if len(title_parts) > 1:
                    metadata["title"] = title_parts[1]
                break
        
        # Extract tags if present in brackets
        tag_match = re.search(r'\[(.*?)\]', name_without_ext)
        if tag_match:
            tags_str = tag_match.group(1)
            metadata["tags"] = [tag.strip() for tag in tags_str.split(',')]
            # Remove tags from title
            metadata["title"] = re.sub(r'\[.*?\]', '', metadata["title"]).strip()
        
        return metadata
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single text file into chunks with metadata.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Extract metadata from filename
            metadata = self.extract_metadata_from_filename(file_path)
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create result with metadata
            result = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["source"] = file_path
                
                result.append({
                    "text": chunk,
                    "metadata": chunk_metadata
                })
            
            logger.info(f"Processed file {file_path} into {len(chunks)} chunks")
            return result
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all text files in a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        try:
            all_chunks = []
            
            # Check if directory exists
            if not os.path.exists(directory_path):
                logger.warning(f"Directory {directory_path} does not exist")
                return all_chunks
            
            # Process all text files
            for filename in os.listdir(directory_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(directory_path, filename)
                    chunks = self.process_file(file_path)
                    all_chunks.extend(chunks)
            
            logger.info(f"Processed {len(all_chunks)} chunks from directory {directory_path}")
            return all_chunks
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            return []


class VectorStore:
    """Vector database for storing and retrieving text embeddings."""
    
    def __init__(self, 
                 persist_directory: str,
                 embedding_model: str = "text-embedding-3-small",
                 collection_name: str = "user_writings"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the database
            embedding_model: Name of the embedding model to use
            collection_name: Name of the collection in the database
        """
        if not chromadb_import_success:
            raise ImportError("ChromaDB is not available")
            
        if not openai_imports_success:
            raise ImportError("OpenAI packages are not available")
            
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embedding function
        try:
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
            logger.info(f"Initialized OpenAI embeddings with model {embedding_model}")
        except Exception as e:
            logger.error(f"Error initializing OpenAI embeddings: {str(e)}")
            # Fall back to default embedding function
            self.embeddings = embedding_functions.DefaultEmbeddingFunction()
            logger.info("Falling back to default embedding function")
        
        # Initialize ChromaDB client
        try:
            try:
                self.client = chromadb.PersistentClient(path=persist_directory)
            except TypeError:
                self.client = chromadb.PersistentClient(persist_directory=persist_directory)
            logger.info(f"Initialized ChromaDB client with persist_directory={persist_directory}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            raise
        
        # Get or create collection using the available embedding function
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embeddings  # use OpenAI or default embedding function
            )
            logger.info(f"Using collection {collection_name}")
        except Exception as e:
            logger.error(f"Error getting or creating collection: {str(e)}")
            raise
    
    def add_texts(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add text chunks to the vector database.
        
        Args:
            chunks: List of dictionaries containing text chunks and metadata
        """
        try:
            if not chunks:
                logger.warning("No chunks to add")
                return
            
            ids = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"chunk_{i}"
                ids.append(chunk_id)
                documents.append(chunk["text"])
                metadatas.append(chunk["metadata"])
            
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(chunks)} chunks to collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error adding texts to collection: {str(e)}")
            raise
    
    def similarity_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in the vector database.
        
        Args:
            query: Query text
            n_results: Number of results to return
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            formatted_results = []
            if results["documents"] and len(results["documents"]) > 0:
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append({
                        "text": doc,
                        "metadata": results["metadatas"][0][i] if i < len(results["metadatas"][0]) else {}
                    })
            
            logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching collection: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary containing statistics
        """
        try:
            all_items = self.collection.get()
            
            total_documents = len(all_items["ids"]) if "ids" in all_items else 0
            
            content_types = {}
            if "metadatas" in all_items and all_items["metadatas"]:
                for metadata in all_items["metadatas"]:
                    if metadata and "content_type" in metadata:
                        content_type = metadata["content_type"]
                        content_types[content_type] = content_types.get(content_type, 0) + 1
            
            stats = {
                "total_documents": total_documents,
                "content_types": content_types
            }
            
            logger.info(f"Collection stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"total_documents": 0, "content_types": {}}
    
    def save_to_disk(self, output_file: str) -> None:
        """
        Save the current state to disk.
        
        Args:
            output_file: Path to save the state
        """
        try:
            all_items = self.collection.get()
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_items, f)
            logger.info(f"Saved collection state to {output_file}")
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            raise
    
    def load_from_disk(self, input_file: str) -> None:
        """
        Load the state from disk.
        
        Args:
            input_file: Path to load the state from
        """
        try:
            if not os.path.exists(input_file):
                logger.warning(f"File {input_file} does not exist")
                return
            
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Clear the existing collection
            self.collection.delete(where={})
            
            # Add items from the loaded state
            if "ids" in data and data["ids"]:
                self.collection.add(
                    ids=data["ids"],
                    documents=data["documents"],
                    metadatas=data["metadatas"]
                )
            logger.info(f"Loaded collection state from {input_file}")
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            raise
