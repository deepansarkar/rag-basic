from src.rag_pipeline import RAGChat
from src.logger_manager import LoggerManager

def main():
    """
    Entry point for the RAG PDF chat application.
    Initializes the logger and RAG pipeline, then starts an interactive loop to answer user questions.
    """
    # Initialize the logger
    logger_manager = LoggerManager()
    logger = logger_manager.get_logger()
    logger.info("Starting RAG PDF Chat application.")

    # Display a title banner for the chat interface
    print("RAG Chatbot")

    try:
        # Initialize the RAGChat instance which loads and processes all PDFs at startup
        logger.info("Initializing RAGChat and loading vector store.")
        rag = RAGChat(logger=logger)
        logger.info("RAGChat initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RAGChat: {e}", exc_info=True)
        print("Could not initialize the RAG pipeline. Check the logs for details.")
        return

    # Prompt the user for input with usage instructions
    print("Type your question or type 'exit' or 'quit' to close the conversation.\n")

    # Start an infinite loop to interact with the user
    while True:
        try:
            # Prompt user for input and strip whitespace
            question = input("You: ").strip()

            # If user types "exit" or "quit", terminate the program gracefully
            if question.lower() in {"exit", "quit"}:
                print("RAG Bot: Goodbye!")
                logger.info("User exited the conversation.")
                break

            # If the user entered a non-empty question
            if question:
                logger.info(f"User question: {question}")

                # Pass the question to the RAG pipeline and get the answer
                answer = rag.ask(question)

                # Log the answer
                logger.info(f"RAG response: {answer}")

                # Print the answer to the console
                print("RAG Bot:")
                print(answer)
                print("")  # Blank line for spacing
            else:
                print("RAG Bot: Please enter a question.")

        # Handle Ctrl+C gracefully by exiting the loop
        except KeyboardInterrupt:
            print("\nGoodbye!")
            logger.info("Conversation interrupted by user (KeyboardInterrupt).")
            break

        # Catch and display any other unexpected errors
        except Exception as e:
            logger.error(f"Unexpected error during chat: {e}", exc_info=True)
            print(f"Error: {e}")

    logger.info("RAG PDF Chat application terminated.")

# Entry point check to run the main function if this file is executed directly
if __name__ == "__main__":
    main()
