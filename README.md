# 🧠 RAG PDF Chatbot with OpenRouter API

A simple Retrieval-Augmented Generation (RAG) system to answer questions from PDFs using OpenRouter and LLM Mistral: Devstral Small (free).

---

## Sample Data

Sample PDFs used for testing and development are located in the [`sample_data/`](./sample_data/) folder. These documents can be used as input for preprocessing, chunking, and embedding in the RAG pipeline.

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate it

**On Windows:**
```bash
venv\Scripts\activate
```

In case activation does not work try
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Upgrade pip

```bash
python.exe -m pip install --upgrade pip
```

### 4. Install required packages

```bash
pip install -r requirements.txt
```

---

## Run the chatbot

**1. Place your PDF files inside the data/pdfs/ folder.**
    a. Ensure each file has a .pdf extension.
    b. You can add multiple PDFs — the system will index them for retrieval.
    Note: The data/pdfs/.gitkeep file preserves folder structure in Git. It’s safe to leave it in place.

**2. Reset the Vector Store**
```bash
python reset_vector_store.py
```

**3. Start the Chatbot**
```bash
python ask.py
```

---

## Git Setup

### 1. Initialize Git repository

```bash
git init
git config --global user.name "Jane Doe"
git config --global user.email "jane.doe@example.com"
```

### 2. Add a `.gitignore` file

Create a `.gitignore` file in the project root with the following content:

```gitignore
# Ignore all files under the data/ directory, but keep folder structure
data/*
!data/**/
!data/.gitkeep
data/pdfs/*
!data/pdfs/.gitkeep
data/cache/*
!data/cache/.gitkeep

# Ignore __pycache__ folders only under src/
src/**/__pycache__/

# Ignore .env file
.env

# Ignore virtual environment folder
venv/
```

Then create a placeholder file to preserve the `data/` folder:

```bash
mkdir -p data/pdfs data/cache
touch data/.gitkeep data/pdfs/.gitkeep data/cache/.gitkeep
```

On Windows, if touch doesn't work, use echo.:

```bash
echo.> data\.gitkeep
```

### 3. Git Commands

#### a. Initial commit and push

```bash
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```

#### b. Later commits and pushes

```bash
git add .
git commit -m "Describe what changed"
git push
```

#### c. Pull updates

```bash
git pull origin main
```

### 4. Clone the repository

```bash
git clone https://github.com/deepansarkar/rag-basic.git
cd rag-basic
```

---

## Notes

- Make sure to keep your `.env` file private and **never commit secrets**.
- Use `requirements.txt` to manage dependencies reproducibly.
- Create an OpenRouter Key From **https://openrouter.ai/settings/keys**.


## Generating `requirements.txt`

To ensure that only actual third-party dependencies are listed (excluding built-in Python modules like `os` or `time`), this project uses [`pipreqs`](https://github.com/bndr/pipreqs) to generate the `requirements.txt` file.

### 🔧 Steps to Generate

1. **Install pipreqs** (if not already installed):

** Install in Global Python not Local**
```bash
pip install pipreqs
```

2. **Generate the requirements file from the project root**:

```bash
pipreqs . --force
(gc requirements.txt) -replace '==.*','' | Sort-Object -Unique | sc requirements.txt
```

. – Tells pipreqs to scan the current directory.
--force – Overwrites existing requirements.txt.

The resulting requirements.txt will include only the external packages used in the codebase, such as requests, PyPDF2, torch, and others.

3. **Modify the requirements file**:

Clean the requirements.txt file by removing the version number and duplicates.

Standard library modules (e.g., os, time, pickle) are automatically excluded.