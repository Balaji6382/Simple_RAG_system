import os
import re
import streamlit as st
import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,?!]', '', text)
    return text.strip()

def load_pdfs(folder="documents"):
    all_texts = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text += " " + (page.extract_text() or "")
            cleaned_text = clean_text(text)
            all_texts.append({"source": filename, "text": cleaned_text})
    return all_texts

st.set_page_config(page_title="RAG system with LangChain", layout="wide")
st.title("PDF RAG System with LangChain")

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    os.makedirs("documents", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("documents", file.name), "wb") as f:
            f.write(file.getbuffer())

    st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully!")

docs = load_pdfs("documents")
if docs:
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?"]
    )
    texts = []
    metadatas = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        texts.extend(chunks)
        metadatas.extend([{"source": doc["source"]}] * len(chunks))

 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)


    system_prompt = """
You are a highly intelligent AI assistant. Only answer using the retrieved documents.
If the answer cannot be found, respond: "I’m sorry, the information is not available."
{documents}
Question: {question}
Answer:
"""

    prompt = PromptTemplate(template=system_prompt, input_variables=["documents", "question"])
    llm = ChatGroq(model="llama-3.1-8b-instant")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt, "document_variable_name": "documents"}, 
        return_source_documents=True
    )
    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Fetching answer..."):
            answer = qa.invoke({"query":query})
        st.markdown("**Answer:**")
        st.write(answer)
        st.markdown("---")
        st.markdown("**Top Sources:**")
        top_docs = retriever.get_relevant_documents(query)
        for doc in top_docs:
            st.write(f"Source: {doc.metadata['source']}")
            st.write(doc.page_content)
            st.markdown("---")
else:
    st.info("Upload PDF files to start the RAG system.")
