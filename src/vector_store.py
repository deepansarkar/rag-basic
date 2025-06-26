import os
import pickle
import shutil
from sentence_transformers import SentenceTransformer, util
from torch import cat
from src.pdf_loader import load_pdf, chunk_text_simple

class VectorStore:
    def __init__(self, cache_dir="data/cache"):
        """
        Initialize the VectorStore with:
        - A pre-trained sentence transformer model
        - A directory to store cached embeddings
        """
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.cache_dir = cache_dir
        # Ensure the cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_path(self, pdf_name):
        """
        Returns the full path to the cache file for a given PDF name.
        Removes the file extension and appends '.pkl'.

        Args:
            pdf_name (str): Filename of the PDF

        Returns:
            str: Full path to the cache file
        """
        base = os.path.splitext(pdf_name)[0]
        return os.path.join(self.cache_dir, base + ".pkl")

    def clear(self):
        """
        Clears the entire embedding cache by deleting the cache directory
        and recreating it. This resets the vector store.
        """
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            print(f"Cleared cache directory: {self.cache_dir}")
        os.makedirs(self.cache_dir, exist_ok=True)

    def create(self, pdf_name, chunks):
        """
        Creates embeddings for the provided chunks and saves them to cache.

        Args:
            pdf_name (str): Name of the source PDF file
            chunks (list of str): Text chunks to encode
        """
        # Encode the chunks
        embeddings = self.model.encode(chunks, convert_to_tensor=True)
        # Determine cache file path
        cache_file = self.get_cache_path(pdf_name)

        # Write the chunks and embeddings to cache using pickle
        with open(cache_file, "wb") as f:
            pickle.dump((chunks, embeddings), f)
        print(f"Created and cached embeddings for {pdf_name}")

    def load(self, pdf_name):
        """
        Loads cached chunks and embeddings for a given PDF.

        Args:
            pdf_name (str): Name of the PDF

        Returns:
            tuple: (chunks, embeddings)

        Raises:
            FileNotFoundError: If cache file is missing
        """
        cache_file = self.get_cache_path(pdf_name)
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"No cache found for {pdf_name}")

        # Load cached data from disk
        with open(cache_file, "rb") as f:
            print(f"Loaded cached data for {pdf_name}")
            return pickle.load(f)

    def fetch(self, pdf_name, chunks):
        """
        Attempts to load from cache. If not found, creates and then loads.

        Args:
            pdf_name (str): Name of the PDF
            chunks (list): Chunks to use for embedding if cache is missing

        Returns:
            tuple: (chunks, embeddings)
        """
        try:
            return self.load(pdf_name)
        except FileNotFoundError:
            self.create(pdf_name, chunks)
            return self.load(pdf_name)

    def fetch_all(self, pdf_folder="data/pdf"):
        """
        Processes all PDFs in the specified folder:
        - If cache exists, loads embeddings and chunks
        - Otherwise, creates and loads

        Returns:
            tuple: (all_chunks, combined_embeddings Tensor)
        """
        all_chunks = []
        all_embeddings = []

        for file in os.listdir(pdf_folder):
            if file.lower().endswith(".pdf"):
                path = os.path.join(pdf_folder, file)

                # Extract text and chunk it
                raw_text = load_pdf(path)
                chunks = chunk_text_simple(raw_text)

                # Fetch from cache or create+load
                chunks, embeddings = self.fetch(file, chunks)

                # Accumulate results
                all_chunks.extend(chunks)
                all_embeddings.append(embeddings)

        if not all_chunks:
            raise ValueError("No valid PDFs found in the specified folder.")

        # Merge all embeddings into a single tensor for querying
        combined_embeddings = cat(all_embeddings, dim=0)
        print(f"Fetched all: {len(all_chunks)} chunks from {pdf_folder}")
        return all_chunks, combined_embeddings

    def retrieve_top_k(self, chunks, embeddings, query, k=3):
        """
        Given a query, retrieves the top-k most relevant chunks using cosine similarity.

        Args:
            chunks (list): List of all text chunks
            embeddings (Tensor): Tensor of all chunk embeddings
            query (str): The user query
            k (int): Number of top results to return

        Returns:
            list: Top-k most relevant text chunks
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity to all chunk embeddings
        scores = util.cos_sim(query_embedding, embeddings)[0]

        # Get indices of top-k highest scores
        top_indices = scores.topk(k)[1]

        # Return the corresponding top-k chunks
        return [chunks[i] for i in top_indices]

    def reset(self, pdf_folder="data/pdf"):
        """
        Clears all cache and recreates embeddings/chunks for each PDF in the folder.
        Does not return anything — used to fully rebuild the vector store.

        Args:
            pdf_folder (str): Directory containing PDF files
        """
        self.clear()  # Delete existing cache

        for file in os.listdir(pdf_folder):
            if file.lower().endswith(".pdf"):
                path = os.path.join(pdf_folder, file)

                # Read and chunk the PDF
                raw_text = load_pdf(path)
                chunks = chunk_text_simple(raw_text)

                # Only create embeddings and save to cache
                self.create(file, chunks)

        print(f"Reset vector store: processed all PDFs in {pdf_folder}")