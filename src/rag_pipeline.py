from src.vector_store import VectorStoreFAISS
from src.openrouter_api import query_openrouter

class RAGChat:
    """
    RAGChat orchestrates Retrieval-Augmented Generation (RAG) over a corpus of PDFs.

    It uses a vector store (FAISS) to index text chunks, retrieve top-k relevant chunks
    for a given question, and then queries a language model with the context.

    All significant operations are logged using the provided logger.
    """

    def __init__(self, pdf_folder="data/pdf", logger=None):
        """
        Initializes the RAGChat system.

        Parameters:
            pdf_folder (str): Path to the folder containing PDF files.
            logger: A logger object to record info, debug, and error messages.

        Behavior:
            - Initializes the FAISS-based vector store.
            - Attempts to load prebuilt index & chunk map. If not found, rebuilds from PDFs.
            - Logs each step and any exceptions.
        """

        self.logger = logger
        self.logger.info("Initializing RAGChat pipeline.")

        self.pdf_folder = pdf_folder
        self.logger.debug(f"PDF folder set to: {self.pdf_folder}")
        
        self.logger.info("Initializing FAISS Vector Store.")
        self.vstore = VectorStoreFAISS(logger=logger)

        # Try loading an existing FAISS index; if missing, build from scratch
        try:
            self.logger.info("Attempting to load existing vector store index and chunk map.")
            self.vstore.load()
            self.logger.info("Successfully loaded existing index and chunk map.")
        except FileNotFoundError:
            # Likely the first time running, so build index from PDF corpus
            self.logger.warning("No existing index found. Building index from PDFs...")
            try:
                self.vstore.build(self.pdf_folder)
                self.logger.info("Successfully built index from PDFs.")
            except Exception as e:
                self.logger.error(f"Failed to build vector index: {e}", exc_info=True)
                raise


    def ask(self, question: str, top_k=3) -> str:
        """
        Processes a user query by retrieving relevant text chunks and generating an answer.

        Parameters:
            question (str): The user's input question.
            top_k (int): Number of most relevant chunks to retrieve for context.

        Returns:
            str: The generated answer from the language model.

        Behavior:
            - Retrieves top-k relevant chunks from the vector store.
            - Concatenates these into a single context string.
            - Calls the language model API with the question and context.
            - Logs each step and handles unexpected errors.
        """
        self.logger.info(f"Received question: {question}")
        self.logger.debug(f"Retrieving top {top_k} relevant chunks for the query.")

        try:
            # Retrieve the top-k most semantically similar text chunks
            top_chunks = self.vstore.retrieve_top_k(question, k=top_k)
            self.logger.debug(f"Retrieved {len(top_chunks)} chunks for context.")

            # Combine retrieved chunks into a single string to pass as context
            context = "\n\n".join(top_chunks)

            self.logger.info("Sending context and question to language model.")
            # Query the language model with the context-enhanced prompt
            answer = query_openrouter(question, context, self.logger)

            self.logger.debug("Received response from language model.")
            return answer

        except Exception as e:
            self.logger.error(f"Error while processing question '{question}': {e}", exc_info=True)
            raise
