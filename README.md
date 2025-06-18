# 🧠 RAG PDF Chatbot with OpenRouter API

A simple Retrieval-Augmented Generation (RAG) system to answer questions from PDFs using OpenRouter and Qwen-DeepSeek LLM.

---

## 🚀 Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate it

**On Windows:**
```bash
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install required packages

```bash
pip install -r requirements.txt
```

---

## 🧪 Run the chatbot

Place your PDF files inside the data/pdfs/ folder making sure the file has a .pdf extension. You can add multiple PDFs. The system will index them for retrieval.
Note: The data/pdfs/.gitkeep file is used to preserve the folder structure in Git. It’s safe to leave it in place.

```bash
python ask.py
```

---

## 🔧 Git Setup

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
touch data/.gitkeep
```

### 3 a. Commit and push - Initial

```bash
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/your-repo.git
#git remote add origin https://github.com/deepansarkar/rag-basic.git
git push -u origin main
```

### 3 b. Commit and push - Follow On

```bash
git add .
git commit -m "Describe what changed"
git push
```

### 3 c. Pull

```bash
git pull origin main
```

### 4. Clone

```bash
git clone https://github.com/deepansarkar/rag-basic.git
cd rag-basic
```

---

## ✅ Notes

- Make sure to keep your `.env` file private and **never commit secrets**.
- Use `requirements.txt` to manage dependencies reproducibly.
