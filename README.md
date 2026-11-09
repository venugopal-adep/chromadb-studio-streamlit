# ChromaDB Studio 🔮✨

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=venugopal-adep/chromadb-studio-streamlit&branch=main&main_module=app.py)

An interactive, high-performance Vector Database Explorer & Embedding Playground built with **Streamlit** and **ChromaDB**.

## 🚀 One-Click Deploy to Streamlit Community Cloud

Click the button above or use this link:
👉 **[Deploy to Streamlit Cloud](https://share.streamlit.io/deploy?repository=venugopal-adep/chromadb-studio-streamlit&branch=main&main_module=app.py)**

---

## 🌟 Key Features

1. **🔍 Semantic Similarity Search & Distance Metrics**:
   - Query by natural language text.
   - Computes real-time relevance percentage and distance scores.
   - Supports all ChromaDB distance metrics: `cosine`, `l2` (Euclidean), and `ip` (Inner Product).

2. **🎯 Advanced Query Filters**:
   - **Metadata Filtering (`where`)**: Filter with operators (`$eq`, `$gt`, `$lt`, `$in`, `$and`, `$or`).
   - **Document Content Filtering (`where_document`)**: Substring matching with `{"$contains": "..."}`.

3. **🗺️ 2D Embedding Space Map (PCA)**:
   - Projects 384-dimensional dense vectors onto an interactive 2D scatter plot using Principal Component Analysis.
   - Plots search queries as a distinct red star (`⭐`) to visualize geometric nearest-neighbor clustering in vector space!

4. **📂 Complete Collection & Document Management (CRUD)**:
   - Create, list, switch, and delete collections.
   - Add, edit, upsert, and delete individual documents with custom metadata.
   - Pre-loaded with rich datasets (AI Glossary, E-Commerce Products, World Geography).

5. **💾 Data Export**:
   - Export collection records directly to JSON or CSV.

6. **📚 ChromaDB Cheatsheet & Guide**:
   - Mathematical formulas for distance metrics, query filtering syntax reference, and Python code snippets.

---

## 💻 Local Execution

```bash
git clone https://github.com/venugopal-adep/chromadb-studio-streamlit.git
cd chromadb-studio-streamlit
pip install -r requirements.txt
streamlit run app.py
```

