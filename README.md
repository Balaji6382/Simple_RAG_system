📚 PDF RAG System

This repository contains two RAG (Retrieval-Augmented Generation) implementations for answering questions from PDF documents:

CLI-based RAG System (Simple Python + Sentence Transformers)

Streamlit-based RAG System (LangChain + HuggingFace + FAISS + Groq LLMs)

Both projects allow you to upload or store PDFs, generate embeddings, and query documents intelligently.

🚀 Features

🔍 Extracts text from PDF documents (pdfplumber)

📑 Cleans and splits text into chunks for better retrieval

🧠 Generates embeddings with sentence-transformers / HuggingFace

📊 Vector search with FAISS

💬 Interactive Q&A with LangChain + Groq LLMs (Streamlit app)

⚡ Simple CLI version for lightweight usage

🛠 Installation

Clone the repository and install dependencies:

git clone https://github.com/Balaji6382/Simple_RAG_system.git
cd pdf-rag-system
pip install -r requirements.txt


✅ Works with Python 3.9+

📂 Project Structure
pdf-rag-system/
│── documents/              # PDF storage folder
│── sentence_transformers_rag.py              # CLI-based RAG system
│── langchain_rag.py        # Streamlit RAG system
│── requirements.txt        # Dependencies
│── README.md               # Documentation

⚡ CLI RAG System

Run the simple command-line RAG system:

python sentence_transformers_rag.py


Place your PDFs inside the documents/ folder

It will generate embeddings using sentence-transformers

Enter questions in the terminal and get answers with top sources

🎨 Streamlit RAG System

Run the interactive Streamlit app:

streamlit run langchain_rag.py 

Features:

Upload multiple PDFs

Embedding + Vector search with FAISS

LangChain pipeline for context retrieval

Uses Groq LLMs (llama-3.1-8b-instant) for fast responses

Displays top source documents for transparency

⚙️ Environment Setup

Create a .env file to store your API keys (if required by LangChain or Groq):

GROQ_API_KEY=your_groq_api_key_here

📊 Requirements

See requirements.txt
.

Key dependencies:

sentence-transformers – embeddings for CLI RAG

streamlit – UI for LangChain RAG

faiss-cpu – vector search

langchain, langchain-community, langchain-groq – RAG pipeline

torch, transformers – HuggingFace models

📌 Future Improvements

✅ Add support for multiple embedding models

✅ GPU acceleration with faiss-gpu

⏳ Chat history and conversation memory

⏳ Export Q&A sessions

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

📜 License

This project is licensed under the MIT License.

