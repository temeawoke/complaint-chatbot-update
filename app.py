# app.py
import streamlit as st
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from rag_pipeline import generate_answer

st.title("💬 CrediTrust Complaint Chatbot")
question = st.text_input("Ask a question:")

if st.button("Submit") and question:
    with st.spinner("Thinking..."):
        answer, sources = generate_answer(question)
        st.subheader("Answer:")
        st.write(answer)

        st.subheader("Source Chunks:")
        for i, chunk in enumerate(sources[:2]):
            st.markdown(f"**Source {i+1}:**")
            st.info(chunk["chunks"])
