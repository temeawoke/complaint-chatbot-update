# complaint-chatbot-
this repo is mainly for a complaint-answering chatbot and you want clarity and ease of understanding for users or collaborators, go with:

# 🗣️ Complaint-Answering Chatbot using Retrieval-Augmented Generation (RAG)

This project implements a complaint-answering chatbot using **Retrieval-Augmented Generation (RAG)**. It is built on top of the **Consumer Financial Protection Bureau (CFPB)** complaint dataset, and involves cleaning, filtering, and preparing consumer complaint narratives for downstream NLP tasks.

---

## 📁 Project Structure
complaint-chatbot/
├── data/
│ ├── raw/ # Original CFPB dataset
│ ├── processed/ # Cleaned & filtered data
├── notebooks/
│ └── 1.0-eda.ipynb # Exploratory Data Analysis
├── src/
│ ├── eda.py # EDA script (optional)
│ ├── preprocessing.py # Text cleaning & normalization
├── .github/
│ └── workflows/
│ └── ci.yml # CI setup with GitHub Actions
├── requirements.txt
├── README.md

---

## ✅ Project Tasks Completed

### 1. 📥 Dataset Loading
- Loaded the full CFPB complaint dataset (~1M+ rows).
- Saved the original file to `data/raw/complaints.csv`.

### 2. 📊 Exploratory Data Analysis (EDA)
- Investigated missing values, top products/issues, complaint trends over time.
- Analyzed narrative lengths and distribution.
- Created bar plots and histograms using Seaborn/Matplotlib.

### 3. 🔍 Dataset Filtering
- Included only complaints with non-empty `Consumer complaint narrative`.
- Focused on **five product categories**:
  - Credit card
  - Personal loan
  - Buy Now, Pay Later (BNPL)
  - Savings account
  - Money transfers
- Removed duplicates and irrelevant rows.
- Saved filtered data to: `data/processed/filtered_five_products.csv`

### 4. 🧼 Text Cleaning & Normalization
Performed the following preprocessing steps:
- Lowercased all narratives.
- Removed digits, special characters, and boilerplate phrases (e.g., “I am writing to file a complaint”).
- Expanded contractions (e.g., “can’t” → “cannot”).
- Removed stopwords and performed lemmatization using `spaCy`.

Final cleaned data saved to:  
`data/processed/final_cleaned_complaints.csv`

---

## 🔄 CI/CD

Set up continuous integration using **GitHub Actions**:
- Workflow path: `.github/workflows/ci.yml`
- Performs linting with `flake8`, installs dependencies, and runs tests using `pytest`.

---

## 🧠 Next Steps (RAG Pipeline)
- Chunk and embed the cleaned narratives using a sentence transformer (e.g., `all-MiniLM-L6-v2`)
- Store vectors in a FAISS or Chroma vector database
- Use a retriever + generator setup (e.g., Hugging Face pipeline) to answer user queries based on similar complaints.

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt