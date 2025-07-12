import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.pdf_loader import load_pdf, chunk_text_with_overlap

class VectorStoreFAISS:
    def __init__(self):
        """
        Initialize the FAISS-based vector store.

        Args:
            index_path (str): Base path (without extension) for saving/loading the FAISS index file.
            chunk_map_path (str): Path for saving/loading the mapping of FAISS indices to text chunks.
        """
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_path = "data/cache/faiss_index.index"
        self.chunk_map_path = "data/cache/chunk_map.pkl"
        self.index = None     # This will hold the FAISS index object
        self.chunks = []      # This list maps FAISS indices to text chunks

    def normalize_embeddings(self, embeddings):
        """
        Normalize vectors to unit length so that inner product becomes cosine similarity.

        Args:
            embeddings (np.ndarray): Array of shape (n_samples, dim)

        Returns:
            np.ndarray: Normalized array of the same shape
        """
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def build(self, pdf_folder="data/pdf"):
        """
        Processes all PDFs in a folder, creates embeddings, builds a FAISS index,
        and saves both the index and the chunk mapping to disk.

        Args:
            pdf_folder (str): Directory containing PDF files.
        """
        all_chunks = []
        all_embeddings = []

        # Iterate through each PDF file
        for file in os.listdir(pdf_folder):
            if file.lower().endswith(".pdf"):
                path = os.path.join(pdf_folder, file)

                # Load PDF text and chunk it into paragraphs/sentences
                raw_text = load_pdf(path)
                chunks = chunk_text_with_overlap(raw_text)

                # Encode the chunks into embeddings using SentenceTransformers
                embeddings = self.model.encode(chunks, convert_to_numpy=True)

                # Accumulate chunks and embeddings
                all_chunks.extend(chunks)
                all_embeddings.append(embeddings)

        if not all_chunks:
            raise ValueError("No valid PDFs found in the specified folder.")

        # Concatenate all embeddings into a single array
        all_embeddings = np.vstack(all_embeddings)

        # Normalize embeddings for cosine similarity
        all_embeddings = self.normalize_embeddings(all_embeddings)

        # Create a FAISS index for Inner Product (which acts like cosine after normalization)
        dim = all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # IP = Inner Product

        # Add all embeddings to the FAISS index
        self.index.add(all_embeddings)

        # Keep the chunks for later retrieval
        self.chunks = all_chunks

        # Save the index to disk
        faiss.write_index(self.index, self.index_path)

        # Save the chunk mapping to disk
        with open(self.chunk_map_path, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"Built and saved FAISS index with {len(self.chunks)} chunks.")

    def load(self):
        """
        Loads a previously saved FAISS index and chunk mapping from disk.
        """
        # Ensure the index and chunk map exist
        if not os.path.exists(self.index_path) or not os.path.exists(self.chunk_map_path):
            raise FileNotFoundError("Index or chunk mapping not found. Build the index first.")

        # Load the FAISS index
        self.index = faiss.read_index(self.index_path)

        # Load the chunk list mapping
        with open(self.chunk_map_path, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"Loaded FAISS index with {len(self.chunks)} chunks.")

    def retrieve_top_k(self, query, k=3):
        """
        Given a text query, finds the top-k most similar text chunks.

        Args:
            query (str): The user query string.
            k (int): Number of top results to retrieve.

        Returns:
            list: List of the top-k most similar text chunks.
        """
        if self.index is None or not self.chunks:
            raise RuntimeError("FAISS index not loaded. Call load() or build() first.")

        # Encode the query into a single embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Normalize the query embedding for cosine similarity
        query_embedding = self.normalize_embeddings(query_embedding)

        # Perform the search on the FAISS index
        # distances: similarity scores, indices: indices into self.chunks
        distances, indices = self.index.search(query_embedding, k)

        # Retrieve the corresponding text chunks
        top_chunks = [self.chunks[i] for i in indices[0]]

        return top_chunks

    def reset(self, pdf_folder="data/pdf"):
        """
        Clears the existing index and chunk mapping files, and rebuilds the index from PDFs.

        Args:
            pdf_folder (str): Directory containing PDF files to process.
        """
        # Delete existing files if they exist
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.chunk_map_path):
            os.remove(self.chunk_map_path)

        print("Cleared existing index and chunk map.")

        # Rebuild everything from scratch
        self.build(pdf_folder)
