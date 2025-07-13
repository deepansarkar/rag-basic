from src.vector_store import VectorStoreFAISS
from src.logger_manager import LoggerManager

def main():
    """
    Clears the existing vector cache and recreates it
    for all PDFs located in the default folder.
    """
    # Create the logger manager (automatically creates a new log file)
    logger_manager = LoggerManager()
    logger = logger_manager.get_logger()

    try:
        # Initialize vector store and reset (rebuild index from PDFs)
        logger.info("Initiating Vectore Store.")
        vstore = VectorStoreFAISS(logger)
        logger.info("Vectore Store initialized.")
        logger.info("Starting vector store reset process.")
        vstore.reset()
        logger.info("Vector store reset and rebuild completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during vector store reset: {e}", exc_info=True)
        print("Failed to reset Vector Store.")

if __name__ == "__main__":
    main()
