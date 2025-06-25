import os
import pickle
from sentence_transformers import SentenceTransformer, util

# Define a class to manage vector embeddings: caching, loading, and retrieval
class VectorStore:
    def __init__(self, cache_dir="data/cache"):
        """
        Initializes the VectorStore with a SentenceTransformer model and a cache directory.

        Parameters:
            cache_dir (str): Path where embedding cache files are stored. Defaults to 'data/cache'.
        """

        # Load a pre-trained sentence embedding model (small and fast)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Set the directory where cached embeddings will be stored
        self.cache_dir = cache_dir

        # Create the cache directory if it does not already exist
        os.makedirs(cache_dir, exist_ok=True)

    def load_or_create(self, pdf_name: str, chunks: list):
        """
        Loads cached embeddings for a given PDF if available; otherwise computes and caches them.

        Parameters:
            pdf_name (str): Name of the PDF file (used to derive the cache filename).
            chunks (list): List of text chunks to encode if cache is missing.

        Returns:
            tuple: A tuple containing the list of chunks and their corresponding embeddings.
        """

        # Derive the base name of the PDF (without extension)
        base = os.path.splitext(pdf_name)[0]

        # Create full path to the cache file using the base name
        cache_file = os.path.join(self.cache_dir, base + ".pkl")

        # Check if cached file exists
        if os.path.exists(cache_file):
            # Open the cache file and load the contents
            with open(cache_file, "rb") as f:
                print(f"Loaded cached data for {pdf_name}")
                return pickle.load(f)

        # Compute embeddings for each text chunk using the model
        embeddings = self.model.encode(chunks, convert_to_tensor=True)

        # Save both the chunks and their embeddings into a binary cache file
        with open(cache_file, "wb") as f:
            pickle.dump((chunks, embeddings), f)
            print(f"Cached embeddings for {pdf_name}")

        # Return the chunks and computed embeddings
        return chunks, embeddings

    def retrieve_top_k(self, chunks, embeddings, query, k=3):
        """
        Finds the top-k most relevant text chunks for a given query using cosine similarity.

        Parameters:
            chunks (list): List of text chunks corresponding to the document.
            embeddings (Tensor): Precomputed embeddings of the chunks.
            query (str): User's search query.
            k (int): Number of top similar chunks to retrieve. Defaults to 3.

        Returns:
            list: Top-k most relevant text chunks based on similarity to the query.
        """
        
        # Encode the query into an embedding vector
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity between query embedding and all chunk embeddings
        scores = util.cos_sim(query_embedding, embeddings)[0]

        # Retrieve the indices of the top-k most similar scores
        top_indices = scores.topk(k)[1]

        # Return the top-k matching chunks based on the retrieved indices
        return [chunks[i] for i in top_indices]