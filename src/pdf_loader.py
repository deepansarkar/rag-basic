from PyPDF2 import PdfReader


def load_pdf(file_path: str) -> str:
    """
    Loads a PDF file and extracts all text into a single long string.

    Parameters:
        file_path (str): Path to the PDF file.

    Returns:
        str: The concatenated text extracted from all pages of the PDF.

    Notes:
        - Each page's text is simply appended with a newline separator.
        - No special treatment of page or paragraph boundaries.
    """
    reader = PdfReader(file_path)
    text = ""

    # Iterate through each page in the PDF
    for page in reader.pages:
        # Extract text from the page (may be None if page is empty)
        page_text = page.extract_text()
        if page_text:
            # Append page text followed by a newline
            text += page_text + "\n"

    # Remove any trailing whitespace
    return text.strip()


def chunk_text_simple(text: str, chunk_size: int = 500) -> list:
    """
    Splits the input text into non-overlapping chunks of specified character size.

    Parameters:
        text (str): The full text to split into chunks.
        chunk_size (int): The maximum number of characters per chunk.

    Returns:
        list: A list of text chunks, each with up to `chunk_size` characters.

    Example:
        >>> chunk_text_simple("abcdef", chunk_size=2)
        ['ab', 'cd', 'ef']
    """
    # Use a list comprehension to slice the text into chunks
    return [text[i:i+chunk_size].strip() for i in range(0, len(text), chunk_size)]


def chunk_text_with_overlap(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Splits the input text into overlapping chunks.

    Parameters:
        text (str): The full text to split.
        chunk_size (int): The total size of each chunk (number of characters).
        overlap (int): Number of characters that overlap between consecutive chunks.

    Returns:
        list: A list of overlapping text chunks.

    Raises:
        ValueError: If `overlap` is greater than or equal to `chunk_size`.

    Example:
        >>> chunk_text_with_overlap("abcdefghi", chunk_size=4, overlap=2)
        ['abcd', 'cdef', 'efgh', 'ghi']
    """
    if overlap >= chunk_size:
        raise ValueError("`overlap` must be smaller than `chunk_size`.")

    chunks = []
    start = 0

    # Slide a window of size `chunk_size` across the text
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        # Only add non-empty chunks
        if chunk:
            chunks.append(chunk)

        # Move the start forward by (chunk_size - overlap) to achieve overlap
        start += chunk_size - overlap

    return chunks