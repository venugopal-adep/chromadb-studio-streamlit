# Streamlit Cloud SQLite Compatibility Fix for ChromaDB
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import html
import json
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
    page_title="ChromaDB Query Studio 🔮✨",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Colorful Styling
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
    padding: 26px 32px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 16px 36px -8px rgba(139, 92, 246, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.25);
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.aurora-title {
    font-size: 32px;
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
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    transition: all 0.2s ease;
}

.result-card:hover {
    border-color: #8b5cf6;
    box-shadow: 0 8px 20px rgba(139, 92, 246, 0.15);
    transform: translateY(-1px);
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    flex-wrap: wrap;
    gap: 8px;
}

.rank-badge {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-weight: 800;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12.5px;
    letter-spacing: 0.3px;
}

.distance-pill {
    background: #f1f5f9;
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
}

.sim-pill {
    background: #dcfce7;
    color: #15803d;
    font-size: 12px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 6px;
}

.meta-pill {
    display: inline-block;
    background: #eff6ff;
    color: #1e40af;
    border: 1px solid #bfdbfe;
    font-size: 11.5px;
    font-weight: 600;
    padding: 2.5px 9px;
    border-radius: 6px;
    margin-right: 6px;
    margin-top: 5px;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# Curated Collections Data
COLLECTIONS_DATA = {
    "🧠 AI & Machine Learning Knowledge": {
        "description": "Foundational concepts across Deep Learning, NLP, Classical ML, and GenAI.",
        "space": "cosine",
        "sample_queries": [
            "What is a transformer and how does self-attention work?",
            "Neural networks with backpropagation and gradient descent",
            "Supervised classification algorithms and decision trees",
            "Retrieval-Augmented Generation for grounding LLMs",
        ],
        "documents": [
            {"id": "ai_001", "text": "Deep Learning utilizes multi-layered artificial neural networks with backpropagation to learn hierarchical feature representations from raw input data.", "metadata": {"domain": "AI", "topic": "Deep Learning", "level": "Advanced", "year": 2024}},
            {"id": "ai_002", "text": "Decision Trees split data into subsets based on feature values, forming an interpretable flowchart-like tree structure for classification and regression tasks.", "metadata": {"domain": "AI", "topic": "Classical ML", "level": "Beginner", "year": 2020}},
            {"id": "ai_003", "text": "Large Language Models like GPT-4 and Claude are transformer-based foundation models trained on massive text corpora for generative natural language processing.", "metadata": {"domain": "AI", "topic": "NLP", "level": "Advanced", "year": 2024}},
            {"id": "ai_004", "text": "Convolutional Neural Networks (CNNs) use spatial kernel filters to extract visual patterns such as edges, textures, and object shapes from images.", "metadata": {"domain": "AI", "topic": "Computer Vision", "level": "Intermediate", "year": 2021}},
            {"id": "ai_005", "text": "Reinforcement Learning trains autonomous agents to maximize cumulative rewards through trial-and-error interaction with a dynamic Markov Decision Process environment.", "metadata": {"domain": "AI", "topic": "Reinforcement Learning", "level": "Advanced", "year": 2023}},
            {"id": "ai_006", "text": "Support Vector Machines (SVM) find the optimal maximum-margin hyperplane that separates distinct classes in high-dimensional feature space using kernel tricks.", "metadata": {"domain": "AI", "topic": "Classical ML", "level": "Intermediate", "year": 2019}},
            {"id": "ai_007", "text": "Retrieval-Augmented Generation (RAG) grounds language model outputs by retrieving relevant context chunks from vector databases like ChromaDB.", "metadata": {"domain": "AI", "topic": "GenAI", "level": "Intermediate", "year": 2024}},
            {"id": "ai_008", "text": "K-Means is an unsupervised clustering algorithm that groups unlabeled data points into k distinct clusters based on Euclidean distance to centroids.", "metadata": {"domain": "AI", "topic": "Unsupervised", "level": "Beginner", "year": 2018}},
            {"id": "ai_009", "text": "Attention Mechanisms and Transformers compute self-attention matrix scores to model long-range contextual token dependencies simultaneously in parallel.", "metadata": {"domain": "AI", "topic": "NLP", "level": "Advanced", "year": 2023}},
            {"id": "ai_010", "text": "Gradient Descent is an iterative first-order optimization algorithm that updates model parameters in the direction of the negative gradient of the loss function.", "metadata": {"domain": "AI", "topic": "Optimization", "level": "Beginner", "year": 2020}},
        ]
    },
    "🛍️ E-Commerce Products & Reviews": {
        "description": "Product catalog with prices, categories, ratings, and in-stock inventory.",
        "space": "cosine",
        "sample_queries": [
            "Comfortable chair for home office ergonomics",
            "Wireless noise cancelling audio headphones",
            "Fitness equipment for strength training at home",
            "Healthy organic tea and antioxidant drinks",
        ],
        "documents": [
            {"id": "prod_001", "text": "Active Noise Cancelling Wireless Headphones with 40-hour battery life, plush memory foam earcups, and crystal clear microphone.", "metadata": {"category": "Electronics", "price": 199.99, "rating": 4.8, "in_stock": True}},
            {"id": "prod_002", "text": "Ergonomic Lumbar Support Mesh Office Chair with 3D adjustable armrests, reclining tilt mechanism, and breathable fabric.", "metadata": {"category": "Furniture", "price": 289.00, "rating": 4.6, "in_stock": True}},
            {"id": "prod_003", "text": "Organic Ceremonial Grade Japanese Matcha Green Tea Powder packed with rich antioxidants, l-theanine, and natural clean energy.", "metadata": {"category": "Groceries", "price": 24.50, "rating": 4.9, "in_stock": True}},
            {"id": "prod_004", "text": "Custom RGB Mechanical Gaming Keyboard featuring hot-swappable linear red switches, PBT double-shot keycaps, and aluminum casing.", "metadata": {"category": "Electronics", "price": 129.99, "rating": 4.7, "in_stock": False}},
            {"id": "prod_005", "text": "Double Wall Vacuum Insulated Stainless Steel Water Bottle that keeps drinks ice cold for 24 hours and hot for 12 hours.", "metadata": {"category": "Fitness", "price": 32.00, "rating": 4.5, "in_stock": True}},
            {"id": "prod_006", "text": "Advanced Waterproof GPS Smartwatch with optical heart rate monitor, sleep tracking, and multi-sport workout modes.", "metadata": {"category": "Wearables", "price": 179.00, "rating": 4.4, "in_stock": True}},
            {"id": "prod_007", "text": "Rapid Adjustable Dumbbell Set from 5 to 52.5 lbs with smooth turn selector dials for compact home gym strength training.", "metadata": {"category": "Fitness", "price": 349.00, "rating": 4.8, "in_stock": False}},
            {"id": "prod_008", "text": "65-Inch 4K OLED Ultra HD Smart Television with infinite contrast ratio, Dolby Vision Atmos, and 120Hz gaming refresh rate.", "metadata": {"category": "Electronics", "price": 1399.00, "rating": 4.9, "in_stock": True}},
        ]
    },
    "🌍 World Geography & Travel": {
        "description": "Global cities and destinations with continents, populations, and climates.",
        "space": "cosine",
        "sample_queries": [
            "Romantic European cities with art museums and monuments",
            "Futuristic Asian mega-city with shrines and cuisine",
            "Ancient historical wonders and pyramids along a river",
            "Warm tropical beach destinations with outdoor festivals",
        ],
        "documents": [
            {"id": "geo_001", "text": "Paris is the historic capital of France, globally celebrated for art museums like the Louvre, haute cuisine, and the Eiffel Tower.", "metadata": {"continent": "Europe", "country": "France", "population_m": 2.1, "climate": "Temperate"}},
            {"id": "geo_002", "text": "Tokyo is the bustling capital of Japan, blending futuristic neon skyscrapers with historic Shinto shrines and world-class ramen.", "metadata": {"continent": "Asia", "country": "Japan", "population_m": 14.0, "climate": "Subtropical"}},
            {"id": "geo_003", "text": "Cairo is the vibrant capital of Egypt located on the Nile river, home to the ancient Giza Pyramids and the Sphinx.", "metadata": {"continent": "Africa", "country": "Egypt", "population_m": 9.5, "climate": "Desert"}},
            {"id": "geo_004", "text": "Rio de Janeiro is known for Copacabana and Ipanema beaches, Christ the Redeemer statue atop Mount Corcovado, and annual Carnival.", "metadata": {"continent": "South America", "country": "Brazil", "population_m": 6.7, "climate": "Tropical"}},
            {"id": "geo_005", "text": "Sydney is Australia's largest coastal metropolis, world-famous for the iconic Sydney Opera House and Bondi Beach surfing.", "metadata": {"continent": "Oceania", "country": "Australia", "population_m": 5.3, "climate": "Oceanic"}},
            {"id": "geo_006", "text": "Reykjavik is Iceland's coastal capital, known for the Northern Lights, geothermal hot springs, and dramatic volcanic landscapes.", "metadata": {"continent": "Europe", "country": "Iceland", "population_m": 0.14, "climate": "Subpolar"}},
        ]
    }
}

# ChromaDB Client & Collections Cache
@st.cache_resource
def init_chroma():
    client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
    
    # Pre-populate the curated collections
    for coll_name, data in COLLECTIONS_DATA.items():
        safe_name = coll_name.split()[1].lower().replace("&", "and") + "_collection"
        try:
            coll = client.get_or_create_collection(
                name=safe_name,
                metadata={"hnsw:space": data["space"], "display_name": coll_name}
            )
            # Add documents if empty
            if coll.count() == 0:
                coll.add(
                    ids=[d["id"] for d in data["documents"]],
                    documents=[d["text"] for d in data["documents"]],
                    metadatas=[d["metadata"] for d in data["documents"]]
                )
        except Exception:
            pass
    return client

client = init_chroma()

# Hero Banner
st.markdown("""
<div class="aurora-hero">
    <div class="aurora-title">ChromaDB Query Studio 🔮✨</div>
    <div class="aurora-sub">
        Explore vector database querying with <strong>ChromaDB</strong>. Select a collection, enter semantic natural language queries, inspect similarity scores & distances, test metadata filters, and visualize nearest neighbors in 2D embedding space.
    </div>
    <div class="hero-pill-bar">
        <span class="hero-pill">⚡ 384-dim Dense Vectors</span>
        <span class="hero-pill">📐 Cosine Distance Matching</span>
        <span class="hero-pill">🔍 Metadata Filtering</span>
        <span class="hero-pill">🗺️ 2D PCA Vector Map</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar: Simple Collection Selection
st.sidebar.markdown("### 📂 Select Collection")
coll_options = list(COLLECTIONS_DATA.keys())
selected_coll_display = st.sidebar.selectbox("Active Vector Collection:", options=coll_options, index=0)

active_info = COLLECTIONS_DATA[selected_coll_display]
safe_coll_name = selected_coll_display.split()[1].lower().replace("&", "and") + "_collection"
collection = client.get_collection(safe_coll_name)

st.sidebar.info(f"**About this collection:**\n\n{active_info['description']}")

# Sample Query Shortcuts
st.sidebar.markdown("### 💡 Quick Sample Queries")
selected_sample = st.sidebar.radio("Click to load a sample query:", options=["-- Custom Query --"] + active_info["sample_queries"])

# Metric Cards
coll_count = collection.count()
st.markdown(f"""
<div class="metrics-grid">
    <div class="metric-card card-coll">
        <div class="m-label">Selected Collection</div>
        <div class="m-val" style="font-size: 16px;">{html.escape(selected_coll_display.split()[1])}</div>
        <div class="m-sub">{selected_coll_display}</div>
    </div>
    <div class="metric-card card-docs">
        <div class="m-label">Documents Indexed</div>
        <div class="m-val">{coll_count:,}</div>
        <div class="m-sub">Dense vector records</div>
    </div>
    <div class="metric-card card-dim">
        <div class="m-label">Vector Dimension</div>
        <div class="m-val">384</div>
        <div class="m-sub">all-MiniLM-L6-v2</div>
    </div>
    <div class="metric-card card-dist">
        <div class="m-label">Distance Metric</div>
        <div class="m-val">COSINE</div>
        <div class="m-sub">1 - Cosine Similarity</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Primary Navigation Tabs
tab_query, tab_map, tab_browse, tab_about = st.tabs([
    "🔍 Query Vector Database",
    "🗺️ 2D Vector Map (PCA)",
    "📂 Browse Collection Records",
    "📚 How ChromaDB Works"
])

# Default Query Value
default_q = active_info["sample_queries"][0]
if selected_sample != "-- Custom Query --":
    default_q = selected_sample

# TAB 1: QUERY VECTOR DATABASE
with tab_query:
    st.markdown("### 🔍 Semantic Similarity Search")
    st.markdown("Type a natural language concept or phrase. ChromaDB computes its vector embedding and finds the closest matching documents by angle in 384-dimensional space.")

    col_input, col_k = st.columns([4, 1])
    with col_input:
        query_text = st.text_input(
            "Natural Language Query:",
            value=default_q,
            placeholder="Type anything to search by meaning, not just exact keywords...",
        )
    with col_k:
        top_k = st.slider("Top Results ($k$):", min_value=1, max_value=min(10, coll_count), value=min(4, coll_count))

    # Advanced Filters Expander
    with st.expander("🎯 Metadata & Content Filters (`where` & `where_document`)", expanded=False):
        f1, f2 = st.columns(2)
        with f1:
            st.markdown("**Metadata Filter (`where`):**")
            st.caption("Filter by structured fields (e.g. `{\"domain\": \"AI\"}` or `{\"price\": {\"$lt\": 200}}`):")
            where_raw = st.text_input("Filter JSON (`where`):", value="", placeholder='e.g. {"domain": "AI"}')
        with f2:
            st.markdown("**Document Substring Filter (`where_document`):**")
            st.caption("Filter text inside the document:")
            where_doc_raw = st.text_input("Document Filter JSON:", value="", placeholder='e.g. {"$contains": "network"}')

    where_filter = None
    if where_raw.strip():
        try:
            where_filter = json.loads(where_raw.strip())
        except Exception as e:
            st.warning(f"Invalid JSON in where filter: {e}")

    where_doc_filter = None
    if where_doc_raw.strip():
        try:
            where_doc_filter = json.loads(where_doc_raw.strip())
        except Exception as e:
            st.warning(f"Invalid JSON in where_document filter: {e}")

    # Run Query
    if query_text.strip():
        try:
            query_params = {
                "query_texts": [query_text],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances", "embeddings"]
            }
            if where_filter:
                query_params["where"] = where_filter
            if where_doc_filter:
                query_params["where_document"] = where_doc_filter

            results = collection.query(**query_params)

            matched_ids = results["ids"][0] if results["ids"] else []
            matched_docs = results["documents"][0] if results["documents"] else []
            matched_metas = results["metadatas"][0] if results["metadatas"] else []
            matched_dists = results["distances"][0] if results["distances"] else []

            if matched_ids:
                st.markdown(f"**Found {len(matched_ids)} nearest neighbor matches in `{selected_coll_display}`:**")
                for rank, (doc_id, text, meta, dist) in enumerate(zip(matched_ids, matched_docs, matched_metas, matched_dists), start=1):
                    # In cosine distance: distance = 1 - cosine_similarity
                    relevance_pct = max(0.0, min(100.0, (1.0 - dist) * 100.0))

                    # Metadata badges
                    meta_pills = "".join(f'<span class="meta-pill">{html.escape(k)}: {html.escape(str(v))}</span>' for k, v in (meta or {}).items())

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-header">
                            <div>
                                <span class="rank-badge">Rank #{rank}</span>
                                <span style="font-weight: 700; margin-left: 8px; font-family: monospace; font-size: 13px;">ID: {html.escape(doc_id)}</span>
                            </div>
                            <div style="display: flex; gap: 8px; align-items: center;">
                                <span class="distance-pill">Cosine Distance: {dist:.4f}</span>
                                <span class="sim-pill">Relevance: {relevance_pct:.1f}%</span>
                            </div>
                        </div>
                        <div style="font-size: 15px; line-height: 1.6; color: #0f172a; margin-bottom: 12px; font-weight: 500;">
                            {html.escape(text)}
                        </div>
                        <div>
                            {meta_pills if meta_pills else '<span style="color: #94a3b8; font-size: 12px;">No metadata</span>'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No documents matched the query with the current filters.")
        except Exception as e:
            st.error(f"Search query error: {e}")

# TAB 2: 2D VECTOR MAP (PCA)
with tab_map:
    st.markdown("### 🗺️ 2D Embedding Space Map (Principal Component Analysis)")
    st.markdown("ChromaDB vector embeddings reside in **384-dimensional space**. Using **PCA**, we reduce those 384 dimensions to 2D coordinates so you can geometrically see how documents cluster by meaning!")

    all_data = collection.get(include=["documents", "metadatas", "embeddings"])
    docs_list = all_data.get("documents") or []
    ids_list = all_data.get("ids") or []
    metas_list = all_data.get("metadatas") or []
    embs_raw = all_data.get("embeddings")

    if embs_raw is not None and len(embs_raw) >= 2:
        embs_list = np.array(embs_raw)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embs_list)

        categories = []
        for m in metas_list:
            if m:
                cat = m.get("domain") or m.get("category") or m.get("continent") or m.get("topic") or "General"
            else:
                cat = "General"
            categories.append(str(cat))

        df_plot = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "id": ids_list,
            "category": categories,
            "document": [d[:85] + ("..." if len(d) > 85 else "") for d in docs_list],
        })

        fig = px.scatter(
            df_plot,
            x="x",
            y="y",
            color="category",
            hover_data={"id": True, "category": True, "document": True, "x": False, "y": False},
            title=f"2D Projection of {len(docs_list)} Stored Documents ({selected_coll_display})",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )

        fig.update_traces(marker=dict(size=15, opacity=0.88, line=dict(width=1.5, color="DarkSlateGrey")))

        # Project active query point as a Star!
        if query_text.strip():
            try:
                temp_res = collection.query(query_texts=[query_text], n_results=1, include=["embeddings"])
                if temp_res and temp_res.get("embeddings") is not None and len(temp_res["embeddings"]) > 0:
                    q_emb = np.array(temp_res["embeddings"][0])
                    q_coords = pca.transform(q_emb)

                    fig.add_trace(go.Scatter(
                        x=q_coords[:, 0],
                        y=q_coords[:, 1],
                        mode="markers+text",
                        name="Query: " + (query_text[:18] + "..."),
                        text=["⭐ QUERY"],
                        textposition="top center",
                        marker=dict(size=20, symbol="star", color="#ef4444", line=dict(width=2, color="black")),
                        hoverinfo="text",
                        hovertext=f"Current Query: '{query_text}'"
                    ))
            except Exception:
                pass

        fig.update_layout(
            height=540,
            legend_title_text="Category / Domain",
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(showgrid=True, zeroline=True, gridcolor="#f1f5f9"),
            yaxis=dict(showgrid=True, zeroline=True, gridcolor="#f1f5f9"),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 **Tip:** Points closer together in this 2D scatter plot share stronger semantic similarity in ChromaDB's vector space.")
    else:
        st.info("At least 2 documents with embeddings are required to plot PCA projection.")

# TAB 3: BROWSE COLLECTION RECORDS
with tab_browse:
    st.markdown(f"### 📂 All Records in `{selected_coll_display}`")
    st.markdown("Inspect all stored documents, unique IDs, and metadata tags in this collection:")

    all_records = collection.get(include=["documents", "metadatas"])
    if all_records and all_records.get("ids"):
        df_browse = pd.DataFrame({
            "Document ID": all_records["ids"],
            "Content (Text)": all_records["documents"],
            "Metadata": [json.dumps(m) if m else "{}" for m in all_records["metadatas"]],
            "Characters": [len(d) for d in all_records["documents"]],
            "Words": [len(d.split()) for d in all_records["documents"]]
        })
        st.dataframe(df_browse, use_container_width=True)

        # Export buttons
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "💾 Export Collection as JSON",
                data=json.dumps(all_records, indent=2),
                file_name=f"{safe_coll_name}_export.json",
                mime="application/json",
                use_container_width=True
            )
        with c2:
            st.download_button(
                "💾 Export Collection as CSV",
                data=df_browse.to_csv(index=False),
                file_name=f"{safe_coll_name}_export.csv",
                mime="text/csv",
                use_container_width=True
            )

# TAB 4: HOW CHROMADB WORKS
with tab_about:
    st.markdown("""
    ## 📚 Understanding Vector Search with ChromaDB

    ### 1. What Happens During a Query?
    1. **Embedding**: ChromaDB takes your search query text and converts it into a **384-dimensional vector** of floating-point numbers using an embedding model.
    2. **HNSW Index Lookup**: ChromaDB queries its Hierarchical Navigable Small World (HNSW) graph index to find the $k$ nearest vectors in milliseconds.
    3. **Cosine Distance**: For each document, it computes the Cosine distance:
       $$\\text{Distance} = 1 - \\frac{\\mathbf{A} \\cdot \\mathbf{B}}{\\|\\mathbf{A}\\| \\|\\mathbf{B}\\|}$$
       * **Distance = 0.0**: Identical semantic meaning (100% match).
       * **Distance > 0.5**: Weak or unrelated meaning.

    ### 2. Python Code to Run this Query
    ```python
    import chromadb

    # Connect to ChromaDB
    client = chromadb.Client()
    collection = client.get_collection("ai_knowledge_base")

    # Semantic search with metadata filter
    results = collection.query(
        query_texts=["neural networks and transformers"],
        n_results=3,
        where={"domain": "AI"}
    )

    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        print(f"Distance: {dist:.4f} | Content: {doc}")
    ```
    """)
