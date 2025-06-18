# 🧠 RAG PDF Chatbot with OpenRouter API

A simple Retrieval-Augmented Generation (RAG) system to answer questions from PDFs using OpenRouter and Qwen-DeepSeek LLM.

---

## 🚀 Setup

### 1. Create a virtual environment
python -m venv venv

### 2. Activate it

**On Windows:**
venv\Scripts\activate

**On Mac/Linux:**
source venv/bin/activate

### 3. Upgrade pip
pip install --upgrade pip

### 4. Install required packages
pip install -r requirements.txt

## 🧪 Run the chatbot
python ask.py

## 🔧 Git Setup

### 1. Initialize Git repository
git init

### 2. Add a `.gitignore` file

Create a `.gitignore` file in the project root with the following content:

```gitignore
# Ignore all files under data/, but keep folder structure
data/**/* 
!data/.gitkeep

# Ignore __pycache__ folders only under src/
src/**/__pycache__/

# Ignore .env file
.env

# Ignore virtual environment
venv/
```

Then create a placeholder file to preserve the `data/` folder:
data/.gitkeep

### 3. Commit and push
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main

## ✅ Notes
- Make sure to keep your `.env` file private and **never commit secrets**.
- Use `requirements.txt` to manage dependencies reproducibly.