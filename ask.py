from src.rag_pipeline import RAGChat

# Define the main function which serves as the entry point of the application
def main():
    # Display a title banner for the chat interface
    print("RAG PDF Chat")
    
    # Initialize the RAGChat instance which loads and processes all PDFs at startup
    rag = RAGChat()
    
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
                break

            # If the user entered a non-empty question
            if question:
                # Pass the question to the RAG pipeline and get the answer
                answer = rag.ask(question)

                # Print the answer as a raw multiline response
                print("RAG Bot:")
                print(answer)

                # Print a blank line for visual spacing
                print("")

            # If the user submitted an empty question
            else:
                print("RAG Bot: Please enter a question.")

        # Handle Ctrl+C gracefully by exiting the loop
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        # Catch and display any other unexpected errors
        except Exception as e:
            print(f"Error: {e}")

# Entry point check to run the main function if this file is executed directly
if __name__ == "__main__":
    main()