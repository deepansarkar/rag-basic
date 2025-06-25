import os
from src.pdf_loader import load_pdf, chunk_text_simple
from src.vector_store import VectorStore
from src.openrouter_api import query_openrouter
from torch import cat

# Define a class to manage Retrieval-Augmented Generation (RAG) over PDF documents
class RAGChat:
    # Initialize the RAGChat system with a folder of PDFs
    def __init__(self, pdf_folder="data/pdf"):
        # Set the folder path where PDFs are located
        self.pdf_folder = pdf_folder

        # Initialize the vector store for embedding and retrieval
        self.vstore = VectorStore()

        # Load and process all PDFs in the folder into chunks and embeddings
        self.chunks, self.embeddings = self._load_all_pdfs()

    # Internal method to load all PDFs and prepare their vector embeddings
    def _load_all_pdfs(self):
        """
        Called once during initialization to process all PDFs.
        Caches chunks and their embeddings for all files.
        """

        # Initialize empty lists to hold all chunks and embeddings
        all_chunks = []
        all_embeddings = []

        # Loop through each file in the specified PDF folder
        for file in os.listdir(self.pdf_folder):
            # Check if the file has a .pdf extension
            if file.lower().endswith(".pdf"):
                # Construct the full file path
                path = os.path.join(self.pdf_folder, file)

                # Load and extract raw text from the PDF
                raw_text = load_pdf(path)

                # Split the raw text into smaller simple chunks
                chunks = chunk_text_simple(raw_text)

                # Load cached embeddings or compute and cache them if missing
                chunks, embeddings = self.vstore.load_or_create(file, chunks)

                # Add the current file's chunks to the global list
                all_chunks.extend(chunks)

                # Append the file's embeddings to the embedding list
                all_embeddings.append(embeddings)

        # If no valid PDF content was found, raise an error
        if not all_chunks:
            raise ValueError("No valid PDFs found in data/pdfs.")

        # Concatenate all individual tensors into one combined tensor for efficient querying
        combined_embeddings = cat(all_embeddings, dim=0)

        # Return the combined chunks and embeddings
        return all_chunks, combined_embeddings

    # Method to answer a user query using the PDF data
    def ask(self, question: str, top_k=3) -> str:
        """
        Answers a question using the cached chunks and embeddings.

        Parameters:
            question (str): The user’s input query.
            top_k (int): Number of top relevant chunks to retrieve.

        Returns:
            str: The model's answer based on the most relevant chunks.
        """
        
        # Retrieve the top-k most relevant chunks based on query similarity
        top_chunks = self.vstore.retrieve_top_k(self.chunks, self.embeddings, question, k=top_k)

        # Join the top chunks into a single context string for the model
        context = "\n\n".join(top_chunks)

        # Query the OpenRouter model with the question and extracted context
        return query_openrouter(question, context)