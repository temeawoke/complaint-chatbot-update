import streamlit as st
import sys
import os
sys.path.append("src")  # or '.' if rag_pipeline.py is in root

from rag_pipeline import generate_answer
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load RAG components
@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("vector_store/complaints_index.faiss")
    with open("vector_store/complaints_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, index, metadata

model, index, metadata = load_model_and_index()

# UI layout
st.set_page_config(page_title="CrediTrust Complaint Chatbot", layout="wide")
st.title("📋 CrediTrust Complaint Chatbot")
st.markdown("Ask me anything about customer complaints.")

question = st.text_input("Enter your question:")
submit = st.button("Ask")
clear = st.button("Clear")

if submit and question:
    with st.spinner("Generating response..."):
        answer, sources = generate_answer(question, model, index, metadata)
        st.markdown("### 🧠 Answer:")
        st.write(answer)
        st.markdown("---")
        st.markdown("### 📚 Sources:")
        for i, src in enumerate(sources[:2]):
            st.markdown(f"**Source {i+1}:** {src['chunk'][:500]}...")


