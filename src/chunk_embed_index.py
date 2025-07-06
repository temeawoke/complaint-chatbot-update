import pandas as pd
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import pandas as pd


# Load your dataset (update filename if needed)
df = pd.read_csv('../data/processed/filtered_complaints.csv')

# Filter for specific products
selected_products = [
    'Credit card',
    'Personal loan',
    'Buy Now, Pay Later (BNPL)',
    'Savings account',
    'Money transfers'
]

# Filter relevant complaints and drop NaN narratives
filtered_df = df[df['Product'].isin(selected_products)]
filtered_df = filtered_df[filtered_df['Consumer complaint narrative'].notna()].copy()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\bi am writing.*?complaint\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

filtered_df['cleaned_narrative'] = filtered_df['Consumer complaint narrative'].apply(clean_text)

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

# Now apply chunking to cleaned text
filtered_df['chunks'] = filtered_df['cleaned_narrative'].apply(chunk_text)

# View first few chunked rows
filtered_df[['Product', 'cleaned_narrative', 'chunks']].head()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

filtered_df['chunks'] = filtered_df['cleaned_narrative'].apply(lambda x: text_splitter.split_text(x))

filtered_df[['cleaned_narrative', 'chunks']].head()

from langchain.text_splitter import RecursiveCharacterTextSplitter

settings = [
    {'chunk_size': 300, 'chunk_overlap': 50},
    {'chunk_size': 500, 'chunk_overlap': 100},
    {'chunk_size': 800, 'chunk_overlap': 200},
]

results = {}

for setting in settings:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=setting['chunk_size'],
        chunk_overlap=setting['chunk_overlap'],
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = filtered_df['cleaned_narrative'].apply(lambda x: splitter.split_text(x))
    chunk_counts = chunks.apply(len)
    avg_chunks = chunk_counts.mean()
    max_chunks = chunk_counts.max()
    results[f"{setting['chunk_size']}/{setting['chunk_overlap']}"] = {
        'avg_chunks': avg_chunks,
        'max_chunks': max_chunks
    }

# Show results
import pandas as pd
pd.DataFrame(results).T

from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Flatten all chunks into a list
all_chunks = [chunk for sublist in filtered_df['chunks'] for chunk in sublist]

# Generate embeddings
embeddings = model.encode(all_chunks, show_progress_bar=True)

filtered_df['chunks']  # each row is a list of strings
 
 
 # Flatten all chunked text into one list
all_chunks = [chunk for chunk_list in filtered_df['chunks'] for chunk in chunk_list]

model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings (each will be a 384-dimensional vector)
embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)


# Remove rows where chunks is empty
filtered_df = filtered_df[filtered_df['chunks'].map(len) > 0]

# Now safely explode
expanded_df = filtered_df.explode('chunks', ignore_index=True)

# Double-check lengths
print(f"Expanded rows: {len(expanded_df)}")
print(f"Number of embeddings: {len(embeddings)}")

expanded_df['embedding'] = list(embeddings)
filtered_df[filtered_df['chunks'].map(len) == 0]

embedding_dim = embeddings.shape[1]  # Should be 384 for all-MiniLM-L6-v2

# Convert to float32 if needed
embeddings = np.array(embeddings).astype('float32')

index = faiss.IndexFlatL2(embedding_dim)  # L2 = Euclidean similarity
#index.add(embeddings)  # Add vectors to the index

print(f"✅ FAISS index created with {index.ntotal} vectors.")


faiss.write_index(index, "../data/processed/complaints_index.faiss")
index = faiss.read_index("../data/processed/complaints_index.faiss")


vector_store_df = pd.DataFrame({
    'chunk': all_chunks,
    'embedding_index': range(len(all_chunks)),
    'product': expanded_df['Product'].values,  # or any metadata
})


vector_store_df.to_csv("../data/processed/vector_store_metadata.csv", index=False)



expanded_df = expanded_df.reset_index(drop=True)  # Ensure clean index
expanded_df['complaint_id'] = expanded_df.index   # Use index as ID


metadata_df = pd.DataFrame({
    'complaint_id': expanded_df['complaint_id'],
    'product': expanded_df['Product'],
    'chunk': expanded_df['chunks']
})


#faiss.write_index(faiss_index, "../data/processed/faiss_complaints.index")

metadata_df.to_csv("../data/processed/faiss_complaints_metadata.csv", index=False)
# Or for faster load times:
# metadata_df.to_pickle("faiss_complaints_metadata.pkl")
