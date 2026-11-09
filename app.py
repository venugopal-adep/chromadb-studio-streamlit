# Streamlit Cloud SQLite Compatibility Fix for ChromaDB
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import html
import json
import os
import chromadb
from chromadb.config import Settings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="ChromaDB Studio 🔮✨",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
}

/* Aurora Hero Banner */
.aurora-hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 30%, #d946ef 65%, #f43f5e 100%);
    background-size: 200% 200%;
    animation: gradientShift 12s ease infinite;
    border-radius: 20px;
    padding: 28px 34px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 16px 36px -8px rgba(139, 92, 246, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.25);
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.aurora-title {
    font-size: 34px;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: -0.6px;
    display: flex;
    align-items: center;
    gap: 12px;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

.aurora-sub {
    font-size: 15px;
    opacity: 0.96;
    max-width: 860px;
    line-height: 1.6;
    font-weight: 500;
}

.hero-pill-bar {
    display: flex;
    gap: 10px;
    margin-top: 14px;
    flex-wrap: wrap;
}

.hero-pill {
    background: rgba(255, 255, 255, 0.22);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.35);
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.3px;
}

/* Metric Cards */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 22px;
}

.metric-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 14px 16px;
    text-align: center;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05);
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(99, 102, 241, 0.15);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
}

.card-coll::before { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.card-docs::before { background: linear-gradient(90deg, #ec4899, #f43f5e); }
.card-dim::before { background: linear-gradient(90deg, #10b981, #06b6d4); }
.card-dist::before { background: linear-gradient(90deg, #f59e0b, #f97316); }
.card-mode::before { background: linear-gradient(90deg, #8b5cf6, #d946ef); }

.m-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #64748b;
    margin-bottom: 4px;
}

.m-val {
    font-size: 22px;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #0f172a;
    line-height: 1.2;
    word-break: break-all;
}

.m-sub {
    font-size: 10.5px;
    color: #94a3b8;
    margin-top: 3px;
    font-weight: 500;
}

/* Query Result Card */
.result-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.result-card:hover {
    border-color: #8b5cf6;
    box-shadow: 0 6px 16px rgba(139, 92, 246, 0.12);
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    flex-wrap: wrap;
    gap: 8px;
}

.rank-badge {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-weight: 800;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
}

.distance-pill {
    background: #f1f5f9;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
}

.sim-pill {
    background: #dcfce7;
    color: #15803d;
    font-size: 12px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 6px;
}

.meta-pill {
    display: inline-block;
    background: #eff6ff;
    color: #1e40af;
    border: 1px solid #bfdbfe;
    font-size: 11.5px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 6px;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# Preset Datasets
PRESET_DATASETS = {
    "🧠 AI & Machine Learning Glossary": [
        {"id": "ai_001", "text": "Deep Learning utilizes multi-layered artificial neural networks with backpropagation to learn hierarchical feature representations.", "metadata": {"domain": "AI", "topic": "Deep Learning", "level": "Advanced", "year": 2024}},
        {"id": "ai_002", "text": "Decision Trees split data into subsets based on feature values, forming an interpretable flowchart-like tree structure for classification and regression.", "metadata": {"domain": "AI", "topic": "Classical ML", "level": "Beginner", "year": 2020}},
        {"id": "ai_003", "text": "Large Language Models like GPT-4 and Claude are transformer-based foundation models trained on massive text corpora for generative language understanding.", "metadata": {"domain": "AI", "topic": "NLP", "level": "Advanced", "year": 2024}},
        {"id": "ai_004", "text": "Convolutional Neural Networks (CNNs) use spatial kernel filters to extract visual patterns such as edges and textures from images.", "metadata": {"domain": "AI", "topic": "Computer Vision", "level": "Intermediate", "year": 2021}},
        {"id": "ai_005", "text": "Reinforcement Learning trains autonomous agents to maximize cumulative rewards through trial-and-error interaction with a dynamic environment.", "metadata": {"domain": "AI", "topic": "Reinforcement Learning", "level": "Advanced", "year": 2023}},
        {"id": "ai_006", "text": "Support Vector Machines (SVM) find the optimal maximum-margin hyperplane that separates classes in high-dimensional feature space.", "metadata": {"domain": "AI", "topic": "Classical ML", "level": "Intermediate", "year": 2019}},
        {"id": "ai_007", "text": "Retrieval-Augmented Generation (RAG) grounds language model outputs by retrieving relevant context chunks from vector databases.", "metadata": {"domain": "AI", "topic": "GenAI", "level": "Intermediate", "year": 2024}},
        {"id": "ai_008", "text": "K-Means is an unsupervised clustering algorithm that groups unlabeled data points into k distinct clusters based on Euclidean distance to centroids.", "metadata": {"domain": "AI", "topic": "Unsupervised", "level": "Beginner", "year": 2018}},
        {"id": "ai_009", "text": "Attention Mechanisms and Transformers compute self-attention matrix scores to model long-range contextual token dependencies simultaneously.", "metadata": {"domain": "AI", "topic": "NLP", "level": "Advanced", "year": 2023}},
        {"id": "ai_010", "text": "Gradient Descent is an iterative first-order optimization algorithm that updates model parameters in the direction of the negative gradient of the loss function.", "metadata": {"domain": "AI", "topic": "Optimization", "level": "Beginner", "year": 2020}}
    ],
    "🛍️ E-Commerce Products & Reviews": [
        {"id": "prod_001", "text": "Active Noise Cancelling Wireless Headphones with 40-hour battery life, plush memory foam earcups, and crystal clear microphone.", "metadata": {"category": "Electronics", "price": 199.99, "rating": 4.8, "in_stock": True}},
        {"id": "prod_002", "text": "Ergonomic Lumbar Support Mesh Office Chair with 3D adjustable armrests, reclining tilt mechanism, and breathable fabric.", "metadata": {"category": "Furniture", "price": 289.00, "rating": 4.6, "in_stock": True}},
        {"id": "prod_003", "text": "Organic Ceremonial Grade Japanese Matcha Green Tea Powder packed with rich antioxidants, l-theanine, and natural energy.", "metadata": {"category": "Groceries", "price": 24.50, "rating": 4.9, "in_stock": True}},
        {"id": "prod_004", "text": "Custom RGB Mechanical Gaming Keyboard featuring hot-swappable linear red switches, PBT double-shot keycaps, and aluminum casing.", "metadata": {"category": "Electronics", "price": 129.99, "rating": 4.7, "in_stock": False}},
        {"id": "prod_005", "text": "Double Wall Vacuum Insulated Stainless Steel Water Bottle that keeps drinks ice cold for 24 hours and hot for 12 hours.", "metadata": {"category": "Fitness", "price": 32.00, "rating": 4.5, "in_stock": True}},
        {"id": "prod_006", "text": "Advanced Waterproof GPS Smartwatch with optical heart rate monitor, sleep tracking, and multi-sport workout modes.", "metadata": {"category": "Wearables", "price": 179.00, "rating": 4.4, "in_stock": True}},
        {"id": "prod_007", "text": "Rapid Adjustable Dumbbell Set from 5 to 52.5 lbs with smooth turn selector dials for compact home gym strength training.", "metadata": {"category": "Fitness", "price": 349.00, "rating": 4.8, "in_stock": False}},
        {"id": "prod_008", "text": "65-Inch 4K OLED Ultra HD Smart Television with infinite contrast ratio, Dolby Vision Atmos, and 120Hz gaming refresh rate.", "metadata": {"category": "Electronics", "price": 1399.00, "rating": 4.9, "in_stock": True}}
    ],
    "🌍 World Geography & Travel Destinations": [
        {"id": "geo_001", "text": "Paris is the historic capital of France, globally celebrated for art museums like the Louvre, haute cuisine, and the Eiffel Tower.", "metadata": {"continent": "Europe", "country": "France", "population_m": 2.1, "climate": "Temperate"}},
        {"id": "geo_002", "text": "Tokyo is the bustling capital of Japan, blending futuristic neon skyscrapers with historic Shinto shrines and world-class ramen.", "metadata": {"continent": "Asia", "country": "Japan", "population_m": 14.0, "climate": "Subtropical"}},
        {"id": "geo_003", "text": "Cairo is the vibrant capital of Egypt located on the Nile river, home to the ancient Giza Pyramids and the Sphinx.", "metadata": {"continent": "Africa", "country": "Egypt", "population_m": 9.5, "climate": "Desert"}},
        {"id": "geo_004", "text": "Rio de Janeiro is known for Copacabana and Ipanema beaches, Christ the Redeemer statue atop Mount Corcovado, and annual Carnival.", "metadata": {"continent": "South America", "country": "Brazil", "population_m": 6.7, "climate": "Tropical"}},
        {"id": "geo_005", "text": "Sydney is Australia's largest coastal metropolis, world-famous for the iconic Sydney Opera House and Bondi Beach surfing.", "metadata": {"continent": "Oceania", "country": "Australia", "population_m": 5.3, "climate": "Oceanic"}},
        {"id": "geo_006", "text": "Reykjavik is Iceland's coastal capital, known for the Northern Lights, geothermal hot springs, and dramatic volcanic landscapes.", "metadata": {"continent": "Europe", "country": "Iceland", "population_m": 0.14, "climate": "Subpolar"}}
    ]
}

# ChromaDB Client Management in Streamlit Session State
@st.cache_resource
def get_chroma_client():
    # Ephemeral in-memory client
    return chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))

client = get_chroma_client()

# Ensure at least one default collection exists
existing_colls = [c.name for c in client.list_collections()]
if not existing_colls:
    default_col = client.create_collection(
        name="ai_knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    # Pre-populate with AI glossary
    sample_docs = PRESET_DATASETS["🧠 AI & Machine Learning Glossary"]
    default_col.add(
        ids=[d["id"] for d in sample_docs],
        documents=[d["text"] for d in sample_docs],
        metadatas=[d["metadata"] for d in sample_docs]
    )
    existing_colls = [c.name for c in client.list_collections()]

# Aurora Hero Header
st.markdown("""
<div class="aurora-hero">
    <div class="aurora-title">ChromaDB Studio 🔮✨</div>
    <div class="aurora-sub">
        Explore and understand the full capabilities of <strong>ChromaDB</strong>. Test Semantic Similarity Search, Metadata Filtering (<code>$eq</code>, <code>$gt</code>, <code>$in</code>), Vector Projections in 2D Space (PCA), Distance Metrics, and Real-Time Collection Management.
    </div>
    <div class="hero-pill-bar">
        <span class="hero-pill">⚡ Vector Embeddings (384-dim)</span>
        <span class="hero-pill">📐 Cosine / L2 / IP Distance</span>
        <span class="hero-pill">🔍 Metadata Filtering</span>
        <span class="hero-pill">🗺️ 2D Interactive PCA Map</span>
        <span class="hero-pill">💾 In-Memory ChromaDB</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar: Collections & Settings
st.sidebar.markdown("### 📂 Collection Selector")
selected_coll_name = st.sidebar.selectbox("Active Collection:", options=existing_colls, index=0)
active_collection = client.get_collection(selected_coll_name)

# Collection Action Modals in Sidebar Expander
with st.sidebar.expander("➕ Create New Collection", expanded=False):
    new_coll_name = st.text_input("Collection Name:", placeholder="my_new_collection").strip().lower().replace(" ", "_")
    space_metric = st.selectbox("Distance Function (`hnsw:space`):", options=["cosine", "l2", "ip"], index=0, help="cosine: Cosine Distance (default for text). l2: Squared Euclidean. ip: Inner Product.")
    if st.button("Create Collection 🚀", use_container_width=True):
        if new_coll_name and new_coll_name not in existing_colls:
            try:
                client.create_collection(name=new_coll_name, metadata={"hnsw:space": space_metric})
                st.success(f"Created '{new_coll_name}' with {space_metric} distance!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Collection name is empty or already exists.")

# One-Click Preset Datasets
with st.sidebar.expander("📥 Load Preset Dataset", expanded=False):
    st.caption("Instantly inject rich sample data into the active collection:")
    preset_choice = st.selectbox("Choose Sample Dataset:", options=list(PRESET_DATASETS.keys()))
    if st.button("Inject Dataset to Active Collection ⚡", use_container_width=True):
        dataset_items = PRESET_DATASETS[preset_choice]
        try:
            active_collection.add(
                ids=[d["id"] for d in dataset_items],
                documents=[d["text"] for d in dataset_items],
                metadatas=[d["metadata"] for d in dataset_items]
            )
            st.success(f"Added {len(dataset_items)} documents from '{preset_choice}'!")
            st.rerun()
        except Exception as e:
            st.error(f"Error adding documents: {e}")

# Delete Collection
with st.sidebar.expander("🗑️ Delete Collection", expanded=False):
    st.caption(f"Permanently delete active collection '{selected_coll_name}'.")
    if st.button("⚠️ Delete Current Collection", type="primary", use_container_width=True):
        if len(existing_colls) > 1:
            client.delete_collection(selected_coll_name)
            st.success(f"Deleted '{selected_coll_name}'!")
            st.rerun()
        else:
            st.warning("Cannot delete the only existing collection.")

# Collect Active Collection Metadata
coll_count = active_collection.count()
coll_meta = active_collection.metadata or {}
dist_metric = coll_meta.get("hnsw:space", "l2")

# Peek at vector dimension
emb_dim = "384"
if coll_count > 0:
    try:
        peek_data = active_collection.peek(limit=1)
        if peek_data and peek_data.get("embeddings") is not None and len(peek_data["embeddings"]) > 0:
            emb_dim = str(len(peek_data["embeddings"][0]))
    except Exception:
        emb_dim = "384"

# Top Metrics Row
st.markdown(f"""
<div class="metrics-grid">
    <div class="metric-card card-coll">
        <div class="m-label">Active Collection</div>
        <div class="m-val" style="font-size: 18px;">{html.escape(selected_coll_name)}</div>
        <div class="m-sub">Target storage unit</div>
    </div>
    <div class="metric-card card-docs">
        <div class="m-label">Documents</div>
        <div class="m-val">{coll_count:,}</div>
        <div class="m-sub">Total stored vectors</div>
    </div>
    <div class="metric-card card-dim">
        <div class="m-label">Embedding Dim</div>
        <div class="m-val">{emb_dim}</div>
        <div class="m-sub">all-MiniLM-L6-v2</div>
    </div>
    <div class="metric-card card-dist">
        <div class="m-label">Distance Metric</div>
        <div class="m-val" style="font-size: 20px;">{dist_metric.upper()}</div>
        <div class="m-sub">HNSW space metric</div>
    </div>
    <div class="metric-card card-mode">
        <div class="m-label">Total Collections</div>
        <div class="m-val">{len(existing_colls)}</div>
        <div class="m-sub">Active in ChromaDB</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_search, tab_map, tab_crud, tab_inspect, tab_guide = st.tabs([
    "🔍 Semantic Search & Filters",
    "🗺️ 2D Vector Map (PCA)",
    "➕ Add & Manage Documents (CRUD)",
    "📂 Collection Inspector",
    "📚 ChromaDB Cheatsheet & Guide"
])

# TAB 1: SEMANTIC SEARCH & FILTERS
with tab_search:
    st.markdown("### 🔍 Semantic Similarity Search")
    st.markdown("ChromaDB automatically computes vector embeddings for your query text and finds the **$k$ nearest neighbors** using HNSW approximate nearest neighbors.")

    col_q, col_k = st.columns([4, 1])
    with col_q:
        query_text = st.text_input(
            "Enter natural language query:",
            value="machine learning and neural networks",
            placeholder="e.g. Find products with high ratings, or machine learning algorithms"
        )
    with col_k:
        top_k = st.slider("Top Results ($k$):", min_value=1, max_value=min(20, max(1, coll_count)), value=min(4, max(1, coll_count)))

    # Filtering Options Accordion
    with st.expander("🎯 Advanced Query Filters (`where` & `where_document`)", expanded=False):
        c_filter1, c_filter2 = st.columns(2)
        with c_filter1:
            st.markdown("**Metadata Filtering (`where`):**")
            st.caption("Filter by metadata attributes (exact match or operators like `$eq`, `$gt`, `$in`):")
            where_raw = st.text_input("Metadata Filter JSON:", value="", placeholder='e.g. {"domain": "AI"} or {"price": {"$lt": 150}}')
        with c_filter2:
            st.markdown("**Document Substring Filtering (`where_document`):**")
            st.caption("Filter by text content inside the document:")
            where_doc_raw = st.text_input("Document Filter JSON:", value="", placeholder='e.g. {"$contains": "network"}')

    # Execute Search
    where_clause = None
    if where_raw.strip():
        try:
            where_clause = json.loads(where_raw.strip())
        except Exception as e:
            st.warning(f"Invalid JSON in where filter: {e}")

    where_doc_clause = None
    if where_doc_raw.strip():
        try:
            where_doc_clause = json.loads(where_doc_raw.strip())
        except Exception as e:
            st.warning(f"Invalid JSON in where_document filter: {e}")

    if query_text.strip() and coll_count > 0:
        try:
            query_args = {
                "query_texts": [query_text],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances", "embeddings"]
            }
            if where_clause:
                query_args["where"] = where_clause
            if where_doc_clause:
                query_args["where_document"] = where_doc_clause

            results = active_collection.query(**query_args)

            matched_ids = results["ids"][0] if results["ids"] else []
            matched_docs = results["documents"][0] if results["documents"] else []
            matched_metas = results["metadatas"][0] if results["metadatas"] else []
            matched_dists = results["distances"][0] if results["distances"] else []

            if matched_ids:
                st.markdown(f"**Found {len(matched_ids)} nearest neighbor matches:**")
                for rank, (doc_id, text, meta, dist) in enumerate(zip(matched_ids, matched_docs, matched_metas, matched_dists), start=1):
                    # Compute similarity percentage for display
                    if dist_metric == "cosine":
                        # distance = 1 - cosine_similarity => sim = 1 - dist
                        sim_pct = max(0.0, (1.0 - dist) * 100.0)
                    else:
                        sim_pct = max(0.0, (1.0 / (1.0 + dist)) * 100.0)

                    # Build metadata badges
                    meta_pills = "".join(f'<span class="meta-pill">{html.escape(k)}: {html.escape(str(v))}</span>' for k, v in (meta or {}).items())

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-header">
                            <div>
                                <span class="rank-badge">#{rank} Match</span>
                                <span style="font-weight: 700; margin-left: 8px; font-family: monospace;">ID: {html.escape(doc_id)}</span>
                            </div>
                            <div style="display: flex; gap: 8px;">
                                <span class="distance-pill">Distance: {dist:.4f}</span>
                                <span class="sim-pill">Relevance: {sim_pct:.1f}%</span>
                            </div>
                        </div>
                        <div style="font-size: 14.5px; line-height: 1.6; margin-bottom: 10px; color: #1e293b;">
                            {html.escape(text)}
                        </div>
                        <div>
                            {meta_pills if meta_pills else '<span style="color: #94a3b8; font-size: 12px;">No metadata</span>'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No matching documents found with the given query and filter constraints.")
        except Exception as e:
            st.error(f"Search query error: {e}")
    elif coll_count == 0:
        st.info("The active collection is empty. Go to the 'Add & Manage Documents' tab or load a preset dataset from the sidebar to populate documents.")

# TAB 2: 2D VECTOR MAP (PCA)
with tab_map:
    st.markdown("### 🗺️ 2D Embedding Space Map (Principal Component Analysis)")
    st.markdown("Vector embeddings have 384 dimensions. Using **PCA (Principal Component Analysis)**, we project the high-dimensional vectors down to 2D coordinates so you can geometrically see which documents cluster together!")

    if coll_count >= 2:
        # Fetch all embeddings
        all_data = active_collection.get(include=["documents", "metadatas", "embeddings"])
        docs_list = all_data.get("documents") or []
        ids_list = all_data.get("ids") or []
        metas_list = all_data.get("metadatas") or []
        embs_raw = all_data.get("embeddings")

        if embs_raw is not None and len(embs_raw) >= 2:
            embs_list = np.array(embs_raw)
            # Perform PCA projection to 2D
            pca = PCA(n_components=2)
            coords = pca.fit_transform(embs_list)

            # Determine category or domain for coloring
            categories = []
            for m in metas_list:
                if m:
                    cat = m.get("domain") or m.get("category") or m.get("continent") or m.get("topic") or "General"
                else:
                    cat = "General"
                categories.append(str(cat))

            # Build DataFrame for Plotly
            df_plot = pd.DataFrame({
                "x": coords[:, 0],
                "y": coords[:, 1],
                "id": ids_list,
                "category": categories,
                "document": [d[:90] + ("..." if len(d) > 90 else "") for d in docs_list],
            })

            fig = px.scatter(
                df_plot,
                x="x",
                y="y",
                color="category",
                hover_data={"id": True, "category": True, "document": True, "x": False, "y": False},
                title=f"2D Projection of {coll_count} Stored Documents (Collection: {selected_coll_name})",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )

            fig.update_traces(marker=dict(size=14, opacity=0.85, line=dict(width=1.5, color="DarkSlateGrey")))

            # If there's an active query from Tab 1, project the query point as a Star!
            if query_text.strip():
                try:
                    # Embed the query
                    temp_res = active_collection.query(query_texts=[query_text], n_results=1, include=["embeddings"])
                    if temp_res and temp_res.get("embeddings") is not None and len(temp_res["embeddings"]) > 0:
                        q_emb = np.array(temp_res["embeddings"][0])
                        q_coords = pca.transform(q_emb)

                        fig.add_trace(go.Scatter(
                            x=q_coords[:, 0],
                            y=q_coords[:, 1],
                            mode="markers+text",
                            name="Query: " + (query_text[:20] + "..."),
                            text=["⭐ QUERY"],
                            textposition="top center",
                            marker=dict(size=18, symbol="star", color="#ef4444", line=dict(width=2, color="black")),
                            hoverinfo="text",
                            hovertext=f"Search Query: '{query_text}'"
                        ))
                except Exception:
                    pass
                except Exception:
                    pass

            fig.update_layout(
                height=520,
                legend_title_text="Category / Domain",
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(showgrid=True, zeroline=True, gridcolor="#f1f5f9"),
                yaxis=dict(showgrid=True, zeroline=True, gridcolor="#f1f5f9"),
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 **Tip:** Documents closer together on this plot share stronger semantic similarity in ChromaDB's embedding space.")
        else:
            st.info("Need at least 2 documents with embeddings to plot PCA projections.")
    else:
        st.info("Add at least 2 documents to view the 2D PCA vector scatter plot.")

# TAB 3: ADD & MANAGE DOCUMENTS (CRUD)
with tab_crud:
    st.markdown("### ➕ Create, Read, Update & Delete Documents")
    
    col_add, col_del = st.columns([3, 2])

    with col_add:
        st.markdown("#### Add or Upsert a Document")
        new_doc_id = st.text_input("Document ID:", value=f"doc_{coll_count + 1}", placeholder="e.g. doc_101")
        new_doc_text = st.text_area("Document Content (Text):", placeholder="Enter the text content that will be embedded and searched...", height=110)
        new_doc_meta_raw = st.text_area("Metadata JSON (Optional):", value='{"category": "notes", "priority": "high"}', height=75)

        if st.button("➕ Add Document to Collection", variant="primary"):
            if new_doc_id.strip() and new_doc_text.strip():
                try:
                    meta_dict = json.loads(new_doc_meta_raw.strip()) if new_doc_meta_raw.strip() else {}
                    active_collection.upsert(
                        ids=[new_doc_id.strip()],
                        documents=[new_doc_text.strip()],
                        metadatas=[meta_dict] if meta_dict else None
                    )
                    st.success(f"Document '{new_doc_id}' successfully added/upserted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to add document: {e}")
            else:
                st.warning("Both Document ID and Content must not be empty.")

    with col_del:
        st.markdown("#### Delete Document by ID")
        if coll_count > 0:
            doc_to_delete = st.selectbox("Select Document ID to Delete:", options=active_collection.get()["ids"])
            if st.button("🗑️ Delete Selected Document", type="primary"):
                try:
                    active_collection.delete(ids=[doc_to_delete])
                    st.success(f"Deleted document '{doc_to_delete}'!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("No documents to delete.")

# TAB 4: COLLECTION INSPECTOR
with tab_inspect:
    st.markdown("### 📂 Complete Collection Inspector")
    st.markdown(f"Browsing all stored records inside collection: `{selected_coll_name}`")

    if coll_count > 0:
        all_records = active_collection.get(include=["documents", "metadatas"])
        df_display = pd.DataFrame({
            "Document ID": all_records["ids"],
            "Content (Text)": all_records["documents"],
            "Metadata": [json.dumps(m) if m else "{}" for m in all_records["metadatas"]],
            "Characters": [len(d) for d in all_records["documents"]],
            "Words": [len(d.split()) for d in all_records["documents"]]
        })
        st.dataframe(df_display, use_container_width=True)

        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.download_button(
                "💾 Export Collection as JSON",
                data=json.dumps(all_records, indent=2),
                file_name=f"{selected_coll_name}_chromadb_export.json",
                mime="application/json",
                use_container_width=True
            )
        with c_exp2:
            st.download_button(
                "💾 Export Collection as CSV",
                data=df_display.to_csv(index=False),
                file_name=f"{selected_coll_name}_chromadb_export.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("This collection is currently empty.")

# TAB 5: GUIDE & CHEATSHEET
with tab_guide:
    st.markdown("""
    ## 📚 ChromaDB Comprehensive Cheatsheet & Guide

    ### 1. What is ChromaDB?
    ChromaDB is an open-source, AI-native vector database designed to make it simple to build LLM apps by making knowledge, facts, and skills pluggable for language models.

    ### 2. How Does It Work?
    ```
    Text Documents ➔ Embedding Function (384-dim vector) ➔ HNSW Index ➔ Fast k-NN Search
    ```
    When you store text, ChromaDB automatically passes it through an embedding function (by default, `all-MiniLM-L6-v2`) which maps semantic meaning into a continuous 384-dimensional vector space.

    ### 3. Distance Metrics (`hnsw:space`)
    ChromaDB supports 3 distance metrics configured during collection creation:
    * **`cosine` (Cosine Distance)**: $d = 1 - \\frac{A \\cdot B}{\\|A\\| \\|B\\|}$
      * Measures the angle between two vectors, regardless of their magnitude. Standard choice for text embeddings.
    * **`l2` (Squared L2 / Euclidean Distance)**: $d = \\sum (A_i - B_i)^2$
      * Measures straight-line geometric distance in space. Default in ChromaDB.
    * **`ip` (Inner Product / Dot Product)**: $d = 1 - A \\cdot B$
      * Useful when embeddings are already normalized to unit length (where Cosine = Dot Product).

    ### 4. Querying with Metadata Filters (`where`)
    ChromaDB supports rich MongoDB-style filter syntax:
    ```python
    # Exact match
    collection.query(query_texts=["AI"], where={"category": "tech"})

    # Comparison operators ($gt, $gte, $lt, $lte, $ne, $in, $nin)
    collection.query(query_texts=["chair"], where={"price": {"$lt": 200}})

    # Logical operators ($and, $or)
    collection.query(
        query_texts=["laptop"],
        where={"$and": [{"rating": {"$gte": 4.5}}, {"in_stock": True}]}
    )

    # Document content filter
    collection.query(query_texts=["math"], where_document={"$contains": "integral"})
    ```

    ### 5. Python Quickstart Code
    ```python
    import chromadb

    # Initialize client
    client = chromadb.Client()

    # Create collection
    collection = client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})

    # Add items
    collection.add(
        documents=["Machine learning models learn patterns from data"],
        metadatas=[{"topic": "AI"}],
        ids=["id1"]
    )

    # Semantic search
    results = collection.query(query_texts=["artificial intelligence algorithms"], n_results=1)
    print(results["documents"])
    ```
    """)

