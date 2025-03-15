"""
RAG Core - Ultra-simplified implementation of RAG Writing Assistant
This file contains the core functionality for the RAG Writing Assistant in a single file
to simplify deployment to Streamlit Cloud.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing required packages
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import StrOutputParser
    from langchain.schema.runnable import RunnablePassthrough
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError as e:
    logger.error(f"Error importing required packages: {str(e)}")
    logger.info("Please install required packages: pip install langchain langchain-openai openai chromadb")
    raise

class TextProcessor:
    """Process text files into chunks with metadata."""
    
    def __init__(self, chunk_size: int = 750, chunk_overlap: int = 150):
        """
        Initialize the text processor.
        
        Args:
            chunk_size: Target size of text chunks in words
            chunk_overlap: Overlap between chunks in words
        """
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
            self.client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"Initialized ChromaDB client with persist_directory={persist_directory}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            raise
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_functions.DefaultEmbeddingFunction()
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
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"chunk_{len(ids) + i}"
                ids.append(chunk_id)
                documents.append(chunk["text"])
                metadatas.append(chunk["metadata"])
            
            # Add to collection
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
            # Query collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
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
            # Get all items in collection
            all_items = self.collection.get()
            
            # Count documents
            total_documents = len(all_items["ids"]) if "ids" in all_items else 0
            
            # Count content types
            content_types = {}
            if "metadatas" in all_items and all_items["metadatas"]:
                for metadata in all_items["metadatas"]:
                    if metadata and "content_type" in metadata:
                        content_type = metadata["content_type"]
                        if content_type not in content_types:
                            content_types[content_type] = 0
                        content_types[content_type] += 1
            
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
            # Get all items in collection
            all_items = self.collection.get()
            
            # Save to file
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
            # Check if file exists
            if not os.path.exists(input_file):
                logger.warning(f"File {input_file} does not exist")
                return
            
            # Load from file
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Clear collection
            self.collection.delete(where={})
            
            # Add items to collection
            if "ids" in data and data["ids"]:
                self.collection.add(
                    ids=data["ids"],
                    documents=data["documents"],
                    metadatas=data["metadatas"]
                )
            
            logger.info(f"Loaded collection state from {input_file} with {len(data['ids'])} items")
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            raise


class LanguageModelIntegration:
    """Integration with language models for generating content."""
    
    def __init__(self, 
                 model_name: str = "gpt-4o",
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        Initialize the language model integration.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature for generation (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize language model
        self.llm = self._initialize_llm()
        
        logger.info(f"Initialized LanguageModelIntegration with model={model_name}")
    
    def _initialize_llm(self):
        """
        Initialize the language model.
        
        Returns:
            An initialized language model
        """
        try:
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            logger.info(f"Initialized OpenAI model: {self.model_name}")
            return llm
        except Exception as e:
            logger.error(f"Error initializing language model: {str(e)}")
            raise
    
    def _get_style_guidance(self, query):
        """
        Extract style guidance from the query if present.
        
        Args:
            query: The user query
            
        Returns:
            Style guidance string or empty string
        """
        # Look for style instructions in brackets or parentheses
        style_markers = [
            (r'\[make this (.*?)\]', r'Style adjustment: \1'),
            (r'\(make this (.*?)\)', r'Style adjustment: \1'),
            (r'make this (more|less) (\w+)', r'Style adjustment: \1 \2')
        ]
        
        guidance = ""
        
        for pattern, replacement in style_markers:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                guidance = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                break
        
        if guidance:
            return f"Style guidance: {guidance}"
        return ""
    
    def generate_with_style(self, query: str, context_docs: List[Dict[str, Any]], style_adjustments: Optional[str] = None) -> str:
        """
        Generate content based on query and context with optional style adjustments.
        
        Args:
            query: The user query
            context_docs: List of context documents
            style_adjustments: Optional style adjustment instructions
            
        Returns:
            Generated content
        """
        try:
            # Format context from documents
            context = self._format_context_from_docs(context_docs)
            
            # Add style adjustments if provided
            style_guidance = ""
            if style_adjustments:
                style_guidance = f"Style guidance: {style_adjustments}"
            
            # Create the prompt
            prompt = f"""
You are a writing assistant that mimics the style and voice of the user based on their previous writings.
Your goal is to generate new content that sounds authentically like the user wrote it.

Here are relevant examples of the user's writing style:

{context}

Based on these examples, please write a response to the following request in the user's authentic voice:

{query}

{style_guidance}

Remember to maintain the user's unique voice, vocabulary choices, sentence structures, and thematic preferences.
"""
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            # Extract content from response
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            logger.info(f"Generated content for query: {query[:50]}...")
            return content
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise
    
    def _format_context_from_docs(self, docs: List[Dict[str, Any]]) -> str:
        """
        Format context from retrieved documents.
        
        Args:
            docs: List of document dictionaries
            
        Returns:
            Formatted context string
        """
        formatted_context = ""
        
        for i, doc in enumerate(docs):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            content_type = metadata.get("content_type", "unknown")
            title = metadata.get("title", f"Document {i+1}")
            
            formatted_context += f"--- Example {i+1} (Content type: {content_type}) ---\n"
            formatted_context += f"Title: {title}\n\n"
            formatted_context += f"{text}\n\n"
        
        return formatted_context


class RAGWritingAssistant:
    """Main class that integrates all components of the RAG writing assistant."""
    
    def __init__(self, 
                 corpus_directory: str,
                 vector_db_directory: str,
                 embedding_model: str = "text-embedding-3-small",
                 llm_model: str = "gpt-4o",
                 collection_name: str = "user_writings"):
        """
        Initialize the RAG writing assistant with all components.
        
        Args:
            corpus_directory: Directory containing the user's text files
            vector_db_directory: Directory to store the vector database
            embedding_model: Name of the embedding model to use
            llm_model: Name of the language model to use
            collection_name: Name of the collection in the vector database
        """
        self.corpus_directory = corpus_directory
        self.vector_db_directory = vector_db_directory
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.collection_name = collection_name
        
        # Create directories if they don't exist
        os.makedirs(corpus_directory, exist_ok=True)
        os.makedirs(vector_db_directory, exist_ok=True)
        
        # Initialize components
        self.text_processor = TextProcessor(chunk_size=750, chunk_overlap=150)
        self.vector_store = VectorStore(
            persist_directory=vector_db_directory,
            embedding_model=embedding_model,
            collection_name=collection_name
        )
        self.language_model = LanguageModelIntegration(
            model_name=llm_model,
            temperature=0.7
        )
        
        logger.info(f"Initialized RAG Writing Assistant with corpus_directory={corpus_directory}, "
                   f"vector_db_directory={vector_db_directory}, embedding_model={embedding_model}, "
                   f"llm_model={llm_model}")
    
    def process_corpus(self, reprocess: bool = False) -> int:
        """
        Process the corpus directory and add to vector database.
        
        Args:
            reprocess: Whether to reprocess existing files
            
        Returns:
            Number of chunks processed
        """
        try:
            # Check if we need to process the corpus
            stats = self.vector_store.get_collection_stats()
            if stats["total_documents"] > 0 and not reprocess:
                logger.info(f"Using existing vector database with {stats['total_documents']} documents")
                return stats["total_documents"]
            
            # Process the corpus
            logger.info(f"Processing corpus directory: {self.corpus_directory}")
            chunks = self.text_processor.process_directory(self.corpus_directory)
            
            # Clear existing collection if reprocessing
            if reprocess and stats["total_documents"] > 0:
                logger.info("Clearing existing vector database for reprocessing")
                self.vector_store.collection.delete(where={})
            
            # Add to vector database
            if chunks:
                logger.info(f"Adding {len(chunks)} chunks to vector database")
                self.vector_store.add_texts(chunks)
                return len(chunks)
            else:
                logger.warning(f"No text files found in {self.corpus_directory}")
                return 0
        except Exception as e:
            logger.error(f"Error processing corpus: {str(e)}")
            raise
    
    def add_file(self, file_path: str) -> int:
        """
        Process a single file and add to vector database.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Number of chunks processed
        """
        try:
            logger.info(f"Processing file: {file_path}")
            chunks = self.text_processor.process_file(file_path)
            
            if chunks:
                logger.info(f"Adding {len(chunks)} chunks to vector database")
                self.vector_store.add_texts(chunks)
                return len(chunks)
            else:
                logger.warning(f"No chunks created from {file_path}")
                return 0
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise
    
    def generate_content(self, query: str, style_adjustments: Optional[str] = None, n_results: int = 5) -> str:
        """
        Generate content based on query with optional style adjustments.
        
        Args:
            query: The user query
            style_adjustments: Optional style adjustment instructions
            n_results: Number of similar documents to retrieve
            
        Returns:
            Generated content
        """
        try:
            # Get context documents
            context_docs = self.vector_store.similarity_search(query, n_results=n_results)
            
            if not context_docs:
                logger.warning("No relevant documents found in vector database")
                return "I don't have enough context to generate content in your style. Please add more text files to your corpus."
            
            # Generate content
            content = self.language_model.generate_with_style(query, context_docs, style_adjustments)
            
            logger.info(f"Generated content for query: {query[:50]}...")
            return content
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the corpus and vector database.
        
        Returns:
            Dictionary containing statistics
        """
        try:
            # Get vector database stats
            vector_stats = self.vector_store.get_collection_stats()
            
            # Get corpus file stats
            file_count = 0
            file_types = {}
            
            if os.path.exists(self.corpus_directory):
                files = [f for f in os.listdir(self.corpus_directory) if f.endswith('.txt')]
                file_count = len(files)
                
                for file in files:
                    # Try to determine file type from name
                    file_type = "unknown"
                    if "essay" in file.lower():
                        file_type = "essay"
                    elif "podcast" in file.lower():
                        file_type = "podcast"
                    elif "substack" in file.lower() or "newsletter" in file.lower():
                        file_type = "newsletter"
                    elif "reflection" in file.lower():
                        file_type = "reflection"
                    
                    if file_type not in file_types:
                        file_types[file_type] = 0
                    file_types[file_type] += 1
            
            stats = {
                "corpus_files": file_count,
                "file_types": file_types,
                "vector_documents": vector_stats["total_documents"],
                "content_types": vector_stats["content_types"]
            }
            
            logger.info(f"Corpus stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting corpus stats: {str(e)}")
            raise
    
    def save_state(self) -> None:
        """
        Save the current state of the vector database.
        """
        try:
            output_file = os.path.join(self.vector_db_directory, "vector_store_data.json")
            self.vector_store.save_to_disk(output_file)
            logger.info(f"Saved vector store state to {output_file}")
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            raise
    
    def load_state(self) -> None:
        """
        Load the saved state of the vector database.
        """
        try:
            input_file = os.path.join(self.vector_db_directory, "vector_store_data.json")
            if os.path.exists(input_file):
                self.vector_store.load_from_disk(input_file)
                logger.info(f"Loaded vector store state from {input_file}")
            else:
                logger.warning(f"No saved state found at {input_file}")
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            raise
