# CFPB Complaint Chatbot with Semantic Search

This project builds a complaint analysis and retrieval system using the **Consumer Financial Protection Bureau (CFPB)** complaint dataset. It includes data preprocessing, text chunking, vector embedding, and similarity search using **FAISS** to power a chatbot capable of retrieving semantically relevant complaints.

---

## 🔍 Objectives

- Perform Exploratory Data Analysis (EDA)
- Preprocess and clean complaint narratives
- Chunk long complaint texts for better semantic embedding
- Generate vector embeddings using Sentence Transformers
- Store embeddings and metadata in a FAISS vector store
- Retrieve and trace complaint information based on user queries

---

## 📁 Dataset

We use the publicly available **CFPB consumer complaint dataset**, filtered to include complaints related to:

- Credit card
- Personal loan
- Buy Now Pay Later (BNPL)
- Savings account
- Money transfers

---
## 📁 Project Structure
complaint-chatbot/
├── data/
│ ├── raw/ # Original CFPB dataset
│ ├── processed/ # Cleaned & filtered data
├── notebooks/
│ └── 1.0-eda.ipynb # Exploratory Data Analysis
│ └── text_chunking_embedding.ipynb # # For text chunking, # For generating embeddings, # For saving to ├──vector store
│ └── complaints_index.faiss # 
│ └── complaints_metadata.pkl # 
├── src/
│   └── rag_pipeline.py #
│   └── chunk_embed_index.py #
│ ├── eda.py # EDA script (optional)
│ ├── preprocessing.py # Text cleaning & normalization
├── .github/
│ └── workflows/
│ └── ci.yml # CI setup with GitHub Actions
├── requirements.txt
├── README.md


## 📊 Task 1: EDA & Preprocessing

- Loaded the full CFPB dataset (CSV)
- Analyzed complaint distribution by product
- Explored the distribution of narrative lengths
- Filtered to include only the five relevant product categories
- Removed records with missing complaint narratives
- Cleaned complaint text (lowercase, removed boilerplate and special characters)

---

## 🧩 Task 2: Chunking, Embedding & Indexing

### ✅ Chunking Strategy
Used a recursive text splitter to chunk long narratives into ~300-character blocks with 50-character overlaps for semantic consistency.

### ✅ Embedding
Generated embeddings using:
- Chosen for its strong performance-speed trade-off in semantic search tasks.

### ✅ Indexing with FAISS
- Embedded vectors stored in FAISS for fast similarity search
- Metadata stored in parallel (complaint ID, product, original chunk text)

---

## 📦 Files Generated

| File | Description |
|------|-------------|
| `faiss_complaints.index` | FAISS vector index of complaint chunks |
| `faiss_complaints_metadata.csv` | Metadata for each vector (product, chunk, ID) |
| `notebooks/eda_and_embedding.ipynb` | Full pipeline: EDA → Cleaning → Chunking → Embedding |
| `requirements.txt` | Python dependencies |

---

## 🔍 Semantic Search (Example)

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# Load
model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index('faiss_complaints.index')
metadata_df = pd.read_csv('faiss_complaints_metadata.csv')

# Query
query = "I had issues transferring money"
query_vector = model.encode([query])

# Search
D, I = faiss_index.search(np.array(query_vector), k=5)
results = metadata_df.iloc[I[0]]
print(results[['product', 'chunk']])

git clone https://github.com/temeawoke/complaint-chatbot-.git
cd complaint-chatbot
pip install -r requirements.txt

📚 Dependencies

    pandas

    numpy

    faiss-cpu

    sentence-transformers

    tqdm

    ipywidgets (optional for Jupyter progress bars)
    
    🧠 Future Improvements

    Deploy the chatbot with a front-end (e.g., Gradio or Streamlit)

    Integrate with LangChain for RAG-based answering

    Add summarization or classification layers

    # RAG Pipeline Evaluation Report

| Question | Generated Answer | Retrieved Snippets | Quality Score (1-5) | Comments |
|----------|------------------|--------------------|----------------------|----------|
| Why was my credit card denied? | The credit card was denied due to a low credit score and missing documentation. | "The company stated they couldn't verify my income..." | 5 | Answer is accurate, supported by retrieved data |
| What are common issues with Buy Now Pay Later? | Many users report delayed refunds and unclear terms. | "I returned the item but didn't get a refund for weeks..." | 4 | Good summary, could include more nuance |
| Why was my savings account frozen? | The account was frozen due to suspected fraud. | "My account was flagged for fraud without notice..." | 5 | Fully supported by retrieved context |

## 🔍 Final Analysis

- **What worked well**: The retriever accurately surfaced semantically relevant chunks; Mistral-7B-Instruct handled financial queries well with context.
- **Challenges**: Long chunks sometimes led to repeated phrasing in generated output. Short context improved clarity.
- **Improvement Ideas**:
  - Experiment with different chunking sizes
  - Consider reranking top 10 results for diversity
  - Add feedback loop to let users rate answer quality

### 📈 Suggestions

- Add chunk filtering or reranking to improve diversity in top-k.
- Explore adding metadata (e.g., date, product) to the prompt to improve specificity.
- Try hybrid search (dense + keyword-based) for harder queries.


🔒 Limitations & Future Work

    No live feedback or streaming yet

    Chunk reranking could improve relevance

    Could support multilingual queries
  pip install -r requirements.txt

    If you don't have a requirements.txt, here's a basic one:

streamlit
pandas
sentence-transformers
faiss-cpu
transformers

🚀 Running the App

Start the chatbot:

streamlit run app.py

🧠 How It Works
1. Text Chunking

Long complaint narratives are chunked using LangChain's RecursiveCharacterTextSplitter.
2. Embedding

Each chunk is embedded using sentence-transformers/all-MiniLM-L6-v2.
3. FAISS Vector Indexing

Embeddings are stored in a FAISS vector index for fast similarity search.
4. RAG Inference

At query time:

    The question is embedded and matched to the top-k complaint chunks.

    These chunks are inserted into a custom prompt.

    A local or HuggingFace LLM generates an answer grounded in those chunks.

🖼️ Interface Features

    ✅ Text box to ask questions

    ✅ "Ask" button to submit

    ✅ Answer area powered by an LLM

    ✅ "📚 Show Sources" expander reveals chunks used

    ✅ "🧹 Clear" button resets the session

📊 Evaluation

Manual evaluation is available in evaluation_report.md, with:

    Sample questions

    Generated answers

    Retrieved sources

    Quality scores (1–5)

    Comments/Analysis

🔒 Limitations & Future Work

    No live feedback or streaming yet

    Chunk reranking could improve relevance

    Could support multilingual queries

🤝 Acknowledgments

    CFPB for the public dataset

    Hugging Face

    LangChain

    Streamlit