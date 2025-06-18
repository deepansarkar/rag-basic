from PyPDF2 import PdfReader

# Function to load and extract text from a PDF file
def load_pdf(file_path: str) -> str:
    """
    Loads a PDF file and extracts all text from each page.

    Parameters:
        file_path (str): Path to the PDF file.

    Returns:
        str: A single string containing all extracted text from the PDF.
    """

    # Create a PdfReader object to access the PDF
    reader = PdfReader(file_path)
    # Initialize an empty string to store the cumulative text
    text = ""

    # Loop through all pages in the PDF
    for page in reader.pages:
        # Extract text from the current page
        page_text = page.extract_text()

        # Append extracted text to the cumulative text if it exists
        if page_text:
            # Add a newline after each page's content
            text += page_text + "\n"  
    
    # Return the complete extracted text
    return text


# Function to split large text into smaller simple chunks
def chunk_text_simple(text: str, chunk_size: int = 500) -> list:
    """
    Splits a long string of text into smaller chunks of fixed size.

    Parameters:
        text (str): The full text to split.
        chunk_size (int): The number of characters per chunk (default is 500).

    Returns:
        list: A list of text chunks (each of length <= chunk_size).
    """
    
    # List comprehension that iterates through the text in steps of `chunk_size`,
    # extracting and trimming each chunk
    return [text[i:i+chunk_size].strip() for i in range(0, len(text), chunk_size)]