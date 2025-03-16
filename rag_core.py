"""
RAG Core - Modernized Implementation
Works with Python 3.10, chromadb==0.6.6, langchain==0.0.314
"""

import os
import logging
import json
import re
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LangChain imports
try:
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError as e:
    logger.error(f"Error importing LangChain packages: {str(e)}")

# Chroma imports
try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    logger.error(f"Error importing ChromaDB: {str(e)}")

# OpenAI is used for embeddings; ensure OPENAI_API_KEY is set as an env var or in .env
try:
    import openai
except ImportError as e:
    logger.error(f"Error importing OpenAI: {str(e)}")


class TextProcessor:
    """
    Processes text files into chunks with metadata.
    """

    def __init__(self, chunk_size: int = 750, chunk_overlap: int = 150):
        """
        Args:
            chunk_size: Target size of text chunks in words
            chunk_overlap: Overlap between chunks in words
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Approximate 4 chars per word
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size * 4,
            chunk_overlap=chunk_overlap * 4
        )
        logger.info(f"Initialized TextProcessor with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """
        Extract metadata from filename (e.g., type, tags, etc.).
        """
        base_name = os.path.basename(filename)
        name_no_ext = os.path.splitext(base_name)[0]

        metadata = {
            "title": name_no_ext,
            "content_type": "unknown",
            "tags": []
        }

        # Example patterns
        content_type_patterns = {
            r'^essay[_-]': "essay",
            r'^podcast[_-]': "podcast",
            r'^substack[_-]': "newsletter",
            r'^uni_reflection[_-]': "reflection"
        }

        for pattern, ctype in content_type_patterns.items():
            if re.match(pattern, name_no_ext, re.IGNORECASE):
                metadata["content_type"] = ctype
                # Remove the matched prefix from the title
                title_parts = re.split(pattern, name_no_ext, 1, re.IGNORECASE)
                if len(title_parts) > 1:
                    metadata["title"] = title_parts[1]
                break

        # Extract tags in [brackets]
        tag_match = re.search(r'\[(.*?)\]', name_no_ext)
        if tag_match:
            tags_str = tag_match.group(1)
            metadata["tags"] = [tag.strip() for tag in tags_str.split(',')]
            # Remove the bracketed text from the title
            metadata["title"] = re.sub(r'\[.*?\]', '', metadata["title"]).strip()

        return metadata

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Splits a text file into smaller chunks with metadata.
        """
        results = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            metadata = self.extract_metadata_from_filename(file_path)
            chunks = self.text_splitter.split_text(text)

            for i, chunk in enumerate(chunks):
                chunk_meta = dict(metadata)  # copy
                chunk_meta["chunk_index"] = i
                chunk_meta["source"] = file_path

                results.append({
                    "text": chunk,
                    "metadata": chunk_meta
                })

            logger.info(f"Processed {file_path} into {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
        return results

    def process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        Process all .txt files in a directory.
        """
        all_chunks = []
        if not os.path.isdir(dir_path):
            logger.warning(f"Directory {dir_path} does not exist.")
            return all_chunks

        for filename in os.listdir(dir_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(dir_path, filename)
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)

        logger.info(f"Processed {len(all_chunks)} total chunks from directory {dir_path}.")
        return all_chunks


class VectorStore:
    """
    A minimal vector database manager using ChromaDB.
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "user_writings"
    ):
        """
        Args:
            persist_directory: Where Chroma will store data
            collection_name: Name of the collection to store your chunks
        """
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize the Chroma client
        try:
            self.client = chromadb.Client(
                Settings(
                    persist_directory=self.persist_directory,
                    chroma_db_impl="duckdb+parquet"
                )
            )
            logger.info(f"Initialized ChromaDB client in {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {str(e)}")
            raise

        # Create or load the collection
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"Using collection '{collection_name}'.")

        # Set up the OpenAI embeddings
        try:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            logger.info("OpenAI embeddings (text-embedding-ada-002) initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAIEmbeddings: {str(e)}")
            raise

    def add_texts(self, chunks: List[Dict[str, Any]]):
        """
        Adds text chunks to the Chroma collection.
        """
        if not chunks:
            logger.warning("No chunks to add.")
            return

        ids = []
        docs = []
        metas = []
        for i, c in enumerate(chunks):
            ids.append(f"chunk_{i}")
            docs.append(c["text"])
            metas.append(c["metadata"])

        try:
            self.collection.add(
                ids=ids,
                documents=docs,
                metadatas=metas
            )
            logger.info(f"Added {len(chunks)} chunks to the collection.")
        except Exception as e:
            logger.error(f"Error adding texts to collection: {str(e)}")

    def similarity_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a similarity search on the stored documents.
        """
        try:
            result = self.collection.query(query_texts=[query], n_results=n_results)
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]

            # Pair up doc + metadata
            return [{"text": doc, "metadata": meta} for doc, meta in zip(docs, metas)]
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Returns basic info about the collection.
        """
        try:
            items = self.collection.get()
            total_docs = len(items["ids"]) if "ids" in items else 0

            # Count content types
            content_counts = {}
            for meta in items.get("metadatas", []):
                if meta and "content_type" in meta:
                    ctype = meta["content_type"]
                    content_counts[ctype] = content_counts.get(ctype, 0) + 1

            return {"total_documents": total_docs, "content_types": content_counts}
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"total_documents": 0, "content_types": {}}

    def save_to_disk(self, output_file: str):
        """
        Saves the entire collection to a JSON file.
        """
        try:
            items = self.collection.get()
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(items, f)
            logger.info(f"Collection saved to {output_file}.")
        except Exception as e:
            logger.error(f"Error saving to disk: {str(e)}")

    def load_from_disk(self, input_file: str):
        """
        Loads the entire collection from a JSON file, replacing the current data.
        """
        if not os.path.exists(input_file):
            logger.warning(f"{input_file} not found.")
            return

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Clear existing data
            self.collection.delete(where={})

            # Re-import
            self.collection.add(
                ids=data["ids"],
                documents=data["documents"],
                metadatas=data["metadatas"]
            )
            logger.info(f"Loaded collection from {input_file}.")
        except Exception as e:
            logger.error(f"Error loading from disk: {str(e)}")
