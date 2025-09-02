import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pdfplumber  

def clean_text(text):
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'[^\w\s.,?!]', '', text) 
    return text.strip()

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks, i, chunk_idx = [], 0, 0

    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append({
            "chunk_index": chunk_idx,
            "text": " ".join(chunk)
        })
        i += chunk_size - overlap
        chunk_idx += 1
    return chunks

def load_documents(folder="documents"):
    docs = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += " " + page_text

            text = clean_text(text)
            chunks = chunk_text(text)
            for chunk in chunks:
                chunk["source"] = filename
            docs.extend(chunks)
    return docs

def generate_embeddings(docs):
    model = SentenceTransformer("all-MiniLM-L6-v2") 
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    for i, emb in enumerate(embeddings):
        docs[i]["embedding"] = emb.tolist()
    return docs, model

def similarity_search(query, docs, model, top_k=3):
    query_emb = model.encode([query], convert_to_numpy=True)
    doc_embeddings = np.array([d["embedding"] for d in docs])
    scores = cosine_similarity(query_emb, doc_embeddings)[0]
    
    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    return ranked

def main():
    print(" Loading & preprocessing documents...")
    docs = load_documents("documents")

    if not docs:
        print(" No PDF documents found in the 'documents' folder.")
        return

    print("🔎 Generating embeddings...")
    docs, model = generate_embeddings(docs)

    print(" RAG system ready! Enter queries (type 'exit' to quit).")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == "exit":
            break
        results = similarity_search(query, docs, model)
        print("\nTop Results:")
        for doc, score in results:
            print(f"Source: {doc['source']} | Score: {score:.4f}")
            print(f"Text: {doc['text']}\n")

if __name__ == "__main__":
    main()
