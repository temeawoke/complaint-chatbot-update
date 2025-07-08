import streamlit as st
import sys, os

# Add 'src' to path and import RAG function
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from rag_pipeline import generate_answer

st.set_page_config(page_title="CrediTrust Complaint Chatbot")
st.title("💬 Complaint Chatbot")
st.markdown("Ask me a question about consumer complaints!")

# Input box
question = st.text_input("Your question")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        answer, sources = generate_answer(question)
        st.markdown("### ✅ Answer:")
        st.write(answer)

        st.markdown("### 📚 Retrieved Context:")
        for i, chunk in enumerate(sources[:2]):
            st.markdown(f"**Source {i+1}:**")
            st.info(chunk['chunks'])
