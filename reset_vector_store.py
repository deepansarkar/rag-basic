from src.vector_store import VectorStore

def main():
    """
    Clears the existing vector cache and recreates it
    for all PDFs located in the default folder.
    """
    vstore = VectorStore()
    vstore.reset(pdf_folder="data/pdf")

if __name__ == "__main__":
    main()