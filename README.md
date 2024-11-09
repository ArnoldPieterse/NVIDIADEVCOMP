# AGI with RAG (Retrieval-Augmented Generation)

An advanced AI-powered chat application that combines web search, document analysis, and emotional reasoning to provide comprehensive, context-aware responses.

## Features

- Dynamic web search and PDF conversion
- Multi-document processing and indexing
- Emotional and contextual analysis
- Concurrent external query processing
- Real-time response streaming
- File upload support
- Web page to PDF conversion

## Prerequisites

- Python 3.9+
- Chrome/Chromium browser
- wkhtmltopdf
- NVIDIA API key for embeddings and LLM

## Setup Guide

1. **Install System Dependencies**
```bash
# Install wkhtmltopdf (Windows)
choco install wkhtmltopdf

# Install Chrome WebDriver
choco install chromedriver


2. **Create Virtual Environment**
python -m venv venv
.\venv\Scripts\activate


3. **Install Python Dependencies**
pip install -r requirements.txt


4. **Environment Setup Create a .env file with:**
NVIDIA_API_KEY=your-api-key-here

5. **Install Required Python Packages**
pip install llama-index-core
pip install llama-index-embeddings-nvidia
pip install llama-index-llms-nvidia
pip install llama-index-vector-stores-chroma
pip install gradio
pip install pdfkit
pip install google
pip install selenium
pip install faiss-cpu
pip install chromadb


Usage
1. **Start the application:**
python app.py

Open browser at http://127.0.0.1:7860

Upload documents or ask questions directly

The system will:

Search relevant web content
Convert web pages to PDFs
Process and index documents
Generate contextual responses
Provide emotional analysis
Synthesize comprehensive answers
Note
Ensure your NVIDIA API key has access to:

NV-Embed-QA model
nvidia/llama-3.1-nemotron-70b-instruct model

The application combines RAG capabilities with emotional intelligence to provide more nuanced and context-aware responses. ```
