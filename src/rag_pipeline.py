import os
from src.pdf_loader import load_pdf, chunk_text_simple
from src.vector_store import VectorStore
from src.openrouter_api import query_openrouter


# Define a class to manage Retrieval-Augmented Generation (RAG) over PDF documents
class RAGChat:
    def __init__(self, pdf_folder="data/pdf"):
        """
        Initializes the RAGChat system.

        - Sets the PDF directory.
        - Instantiates the vector store.
        - Loads or creates embeddings from all PDFs using the vector store.
        """
        self.pdf_folder = pdf_folder
        self.vstore = VectorStore()

        # Use the fetch_all method to load all chunks and embeddings from cache or create them if missing
        self.chunks, self.embeddings = self.vstore.fetch_all(self.pdf_folder)

    def ask(self, question: str, top_k=3) -> str:
        """
        Answers a user query using the top-k most relevant chunks from the PDF corpus.

        Args:
            question (str): The input question from the user.
            top_k (int): Number of top relevant text chunks to retrieve.

        Returns:
            str: The model's answer generated using retrieved context.
        """
        # Retrieve the top-k most semantically relevant chunks for the input question
        top_chunks = self.vstore.retrieve_top_k(self.chunks, self.embeddings, question, k=top_k)

        # Join the top-k chunks into a single context string
        context = "\n\n".join(top_chunks)

        # Query the language model via OpenRouter with the context and question
        return query_openrouter(question, context)
