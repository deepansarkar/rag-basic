from PyPDF2 import PdfReader


def load_pdf(file_path: str, logger) -> str:
    """
    Loads a PDF file and extracts all text into a single long string.

    Parameters:
        file_path (str): Path to the PDF file to be loaded.
        logger: A logger object to log all operations.

    Returns:
        str: The full text extracted from all pages of the PDF, concatenated.

    Behavior:
        - Logs the start of loading, extraction of each page, and final statistics.
        - Logs errors with stack trace if any issues occur.
    """
    text = ""
    logger.info(f"Loading PDF: {file_path}")

    try:
        # Open the PDF file for reading
        reader = PdfReader(file_path)

        # Iterate through each page in the PDF, keeping track of the page number
        for idx, page in enumerate(reader.pages, start=1):
            # Extract text content from the current page
            page_text = page.extract_text()
            if page_text:
                # If text was found, append it to the overall text
                text += page_text + "\n"
                logger.debug(f"Extracted text from page {idx}")
            else:
                # Warn if the page was empty (common in PDFs with only images)
                logger.warning(f"No text found on page {idx}")

        # Log summary statistics after processing the file
        logger.info(f"Finished extracting PDF: {file_path}. "
                    f"Total length: {len(text)} characters across {len(reader.pages)} pages.")

    except Exception as e:
        # Log detailed error information with stack trace to help debugging
        logger.error(f"Error loading PDF '{file_path}': {e}", exc_info=True)
        # Reraise to allow upstream handling (e.g. skip file or halt pipeline)
        raise

    # Remove trailing whitespace before returning the full text
    return text.strip()


def chunk_text_simple(text: str, chunk_size: int = 300, logger=None) -> list:
    """
    Splits the input text into non-overlapping chunks.

    Parameters:
        text (str): The text to split into chunks.
        chunk_size (int): Number of characters per chunk.
        logger: Logger to record chunking details.

    Returns:
        list: A list of text chunks.
    """
    # Use a list comprehension to create slices of text
    chunks = [text[i:i+chunk_size].strip() for i in range(0, len(text), chunk_size)]

    # Log the outcome of the chunking process
    logger.info(f"Chunked text into {len(chunks)} non-overlapping chunks "
                f"(chunk size = {chunk_size}).")
    return chunks


def chunk_text_with_overlap(text: str, chunk_size: int = 300, overlap: int = 50, logger=None) -> list:
    """
    Splits the input text into overlapping chunks.

    Parameters:
        text (str): The text to split.
        chunk_size (int): Number of characters per chunk.
        overlap (int): Number of characters to overlap between chunks.
        logger: Logger for recording the process.

    Returns:
        list: A list of overlapping text chunks.

    Raises:
        ValueError: If overlap is greater than or equal to chunk_size.
    """
    if overlap >= chunk_size:
        # Log the misconfiguration clearly before raising
        logger.error(f"Invalid parameters: overlap ({overlap}) >= chunk_size ({chunk_size}).")
        raise ValueError("`overlap` must be smaller than `chunk_size`.")

    chunks = []
    start = 0

    # Slide a window of size `chunk_size` over the text
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)
            logger.debug(f"Created chunk from position {start} to {end}.")

        # Move start forward by (chunk_size - overlap) to achieve overlap
        start += chunk_size - overlap

    logger.info(f"Chunked text into {len(chunks)} overlapping chunks "
                f"(chunk size = {chunk_size}, overlap = {overlap}).")
    return chunks
