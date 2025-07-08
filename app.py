# app.py
import streamlit as st
import os
import sys

# Add src to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from rag_pipeline import generate_answer

st.set_page_config(page_title="CrediTrust Complaint Chatbot")
st.title("💬 CrediTrust Complaint Chatbot")
st.markdown("Ask a question based on customer complaints data.")

# User input
question = st.text_input("Type your question here:")

# Answer display
if st.button("Ask"):
    if question:
        with st.spinner("Retrieving answer..."):
            answer, sources = generate_answer(question)
            st.subheader("Answer:")
            st.write(answer)

            st.subheader("Retrieved Sources:")
            for i, src in enumerate(sources[:2]):
                st.markdown(f"**Source {i+1}:**")
                st.info(src['chunks'])
    else:
        st.warning("Please enter a question.")
