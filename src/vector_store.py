import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.pdf_loader import load_pdf, chunk_text_with_overlap

class VectorStoreFAISS:
    def __init__(self, logger=None):
        """
        Initializes the FAISS-based vector store system.

        Args:
            logger: A logger object to record all operations and errors.

        Sets up:
            - The embedding model (SentenceTransformer).
            - Paths for saving/loading the FAISS index and chunk map.
            - Empty placeholders for index and chunks.
        """
        self.logger = logger
        
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.logger.info("Sentence Transformer all-MiniLM-L6-v2 loaded.")
        
        self.index_path = "data/cache/faiss_index.index"
        self.chunk_map_path = "data/cache/chunk_map.pkl"
        self.logger.debug(f"Index path: {self.index_path}, Chunk map path: {self.chunk_map_path}")

        self.index = None   # Will hold the FAISS index after building/loading
        self.chunks = []    # List mapping FAISS indices to text chunks

        self.logger.info("Initialized VectorStoreFAISS.")
        

    def normalize_embeddings(self, embeddings):
        """
        Normalizes vectors to unit length so that inner product can act as cosine similarity.

        Args:
            embeddings (np.ndarray): Embedding array of shape (n_samples, dim).

        Returns:
            np.ndarray: Normalized embedding array.
        """
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def build(self, pdf_folder="data/pdf"):
        """
        Builds the vector store from all PDFs in a specified folder.

        - Extracts text and splits into overlapping chunks.
        - Generates embeddings for these chunks.
        - Builds a FAISS index and saves it along with the chunk map.

        Args:
            pdf_folder (str): Path to folder containing PDF files.
        """
        all_chunks = []
        all_embeddings = []

        self.logger.info(f"Starting to build vector index from PDFs in folder: {pdf_folder}")

        # Iterate through PDF files and process them
        for file in os.listdir(pdf_folder):
            if file.lower().endswith(".pdf"):
                path = os.path.join(pdf_folder, file)
                try:
                    # Load and chunk PDF text
                    raw_text = load_pdf(path, self.logger)
                    chunks = chunk_text_with_overlap(raw_text, logger=self.logger)

                    # Encode chunks into embeddings
                    embeddings = self.model.encode(chunks, convert_to_numpy=True)

                    # Accumulate
                    all_chunks.extend(chunks)
                    all_embeddings.append(embeddings)

                    self.logger.info(f"Processed {file}: {len(chunks)} chunks extracted.")

                except Exception as e:
                    self.logger.error(f"Failed to process {file}: {e}", exc_info=True)

        if not all_chunks:
            err_msg = "No valid PDFs found or extracted text was empty."
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        # Stack embeddings into a single array
        all_embeddings = np.vstack(all_embeddings)
        all_embeddings = self.normalize_embeddings(all_embeddings)

        # Build FAISS index (inner product on normalized vectors approximates cosine)
        dim = all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(all_embeddings)

        # Keep chunk list for retrieval
        self.chunks = all_chunks

        # Persist index and chunk map to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.chunk_map_path, "wb") as f:
            pickle.dump(self.chunks, f)

        self.logger.info(f"Built and saved FAISS index with {len(self.chunks)} chunks.")

    def load(self):
        """
        Loads a previously built FAISS index and chunk mapping from disk.

        Raises:
            FileNotFoundError: If either the index or the chunk mapping is missing.
        """
        self.logger.info("Attempting to load existing FAISS index and chunk map from disk.")

        if not os.path.exists(self.index_path) or not os.path.exists(self.chunk_map_path):
            err_msg = "FAISS index or chunk mapping not found. Please build the index first."
            self.logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        # Load index and chunks
        self.index = faiss.read_index(self.index_path)
        with open(self.chunk_map_path, "rb") as f:
            self.chunks = pickle.load(f)

        self.logger.info(f"Loaded FAISS index with {len(self.chunks)} text chunks.")

    def retrieve_top_k(self, query, k=3):
        """
        Given a query string, retrieves the top-k most similar text chunks.

        Args:
            query (str): User query.
            k (int): Number of top results to return.

        Returns:
            list: Top-k text chunks most relevant to the query.
        """
        if self.index is None or not self.chunks:
            err_msg = "FAISS index not loaded. Call load() or build() first."
            if self.logger:
                self.logger.error(err_msg)
            raise RuntimeError(err_msg)

        self.logger.debug(f"Encoding query for retrieval: {query}")

        # Encode and normalize the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = self.normalize_embeddings(query_embedding)

        # Search for top-k nearest neighbors
        distances, indices = self.index.search(query_embedding, k)

        self.logger.debug(f"Retrieved indices: {indices[0]} with similarity scores: {distances[0]}")

        # Map indices back to chunks
        top_chunks = [self.chunks[i] for i in indices[0]]

        return top_chunks

    def reset(self, pdf_folder="data/pdf"):
        """
        Clears existing FAISS index and chunk mapping files, and rebuilds index from PDFs.

        Args:
            pdf_folder (str): Directory containing PDFs.
        """
        self.logger.warning("Resetting vector store: deleting existing index and chunk map.")

        # Delete existing files
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.chunk_map_path):
            os.remove(self.chunk_map_path)

        self.logger.info("Existing index and chunk map cleared. Rebuilding from scratch.")

        # Rebuild the index
        self.build(pdf_folder)
