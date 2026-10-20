import streamlit as st
import numpy as np
import faiss

st.write("1. FAISS import OK")

EMBEDDINGS_PATH = "image_embeddings_full.npy"

st.write("2. Loading embeddings...")

embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")

st.write("3. Embeddings loaded")
st.write(f"Shape: {embeddings.shape}")

st.write("4. Building FAISS index...")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(np.asarray(embeddings, dtype=np.float32))

st.write("5. FAISS index OK")

scores, indices = index.search(
    np.asarray(embeddings[:1], dtype=np.float32),
    5
)

st.write("6. Search OK")
st.write(scores)
st.write(indices)
