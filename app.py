import os
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from rag import (
    load_model,
    load_documents,
    build_vector_index,
    DEFAULT_DOCS_DIR
)
from agents import (
    query_agent,
    retrieval_agent,
    research_agent,
    document_analysis_agent,
    get_groq_client,
    DEFAULT_MODEL
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Personal Knowledge AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling & Subtle AI Pipeline Animation (Pure CSS, No External Libraries)
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Header Card */
    .header-box {
        padding: 20px 24px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-tagline {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 4px;
        margin-bottom: 12px;
    }
    .badge-kb {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Pipeline Visualization Container */
    .pipeline-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 18px 12px;
        margin: 15px 0 25px 0;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-radius: 14px;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.4);
    }
    
    /* Pipeline Nodes */
    .pipeline-node {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 9px 15px;
        border-radius: 10px;
        font-size: 0.88rem;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(71, 85, 105, 0.5);
        background: rgba(30, 41, 59, 0.7);
        color: #94a3b8;
    }
    
    /* Active Node with Subtle Glow and Pulsing */
    .node-active {
        background: rgba(14, 165, 233, 0.22) !important;
        border: 1.5px solid #38bdf8 !important;
        color: #38bdf8 !important;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.6);
        animation: pulseEffect 1.3s infinite alternate;
        transform: translateY(-2px);
    }
    
    /* Completed Node */
    .node-completed {
        background: rgba(16, 185, 129, 0.16) !important;
        border: 1.5px solid #10b981 !important;
        color: #10b981 !important;
    }
    
    .pipeline-arrow {
        color: #475569;
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    @keyframes pulseEffect {
        0% {
            box-shadow: 0 0 6px rgba(56, 189, 248, 0.4);
            transform: scale(1.0);
        }
        100% {
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.85);
            transform: scale(1.04);
        }
    }
    
    /* Compact Agent Activity Box */
    .activity-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.7);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 10px 0 16px 0;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    .activity-step {
        margin: 3px 0;
        display: flex;
        align-items: center;
        gap: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate animated pipeline HTML
def get_pipeline_html(active_agent="idle"):
    nodes = [
        {"id": "docs", "label": "Documents", "icon": "📄"},
        {"id": "query", "label": "Query Agent", "icon": "🧠"},
        {"id": "retrieval", "label": "Retrieval Agent", "icon": "🔎"},
        {"id": "research", "label": "Research Agent", "icon": "🌐"},
        {"id": "analysis", "label": "Document Analysis", "icon": "🧩"},
        {"id": "final", "label": "Final Answer", "icon": "🤖"}
    ]

    order = ["docs", "query", "retrieval", "research", "analysis", "final"]
    active_idx = order.index(active_agent) if active_agent in order else -1

    html_parts = ['<div class="pipeline-container">']
    for i, node in enumerate(nodes):
        node_class = "pipeline-node"
        if active_agent == "done":
            node_class += " node-completed"
        elif active_agent == node["id"]:
            node_class += " node-active"
        elif active_idx > i:
            node_class += " node-completed"

        html_parts.append(
            f'<div class="{node_class}">'
            f'<span>{node["icon"]}</span>'
            f'<span>{node["label"]}</span>'
            f'</div>'
        )
        if i < len(nodes) - 1:
            html_parts.append('<div class="pipeline-arrow">➔</div>')

    html_parts.append('</div>')
    return "".join(html_parts)

# Initialize Session State
if "memory" not in st.session_state:
    st.session_state.memory = []
if "kb_rebuild_count" not in st.session_state:
    st.session_state.kb_rebuild_count = 0
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Model and Knowledge Base Caching
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return load_model("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_knowledge_base(_model, rebuild_count):
    chunks, doc_stats = load_documents(DEFAULT_DOCS_DIR)
    index, dim = build_vector_index(chunks, _model)
    return index, chunks, doc_stats, dim

# Load embedding model and knowledge base
try:
    with st.spinner("Loading embedding model & knowledge base..."):
        embed_model = get_embedding_model()
        index, chunks, doc_stats, dim = get_knowledge_base(embed_model, st.session_state.kb_rebuild_count)
except Exception as e:
    st.error(f"Error initializing Knowledge Base: {e}")
    st.stop()

# Sidebar: Knowledge Base & Configuration
with st.sidebar:
    st.header("🗂️ Knowledge Base")
    st.markdown(f"**Documents:** `{doc_stats['doc_count']}`")
    st.markdown(f"**Chunks:** `{doc_stats['chunk_count']}`")
    st.markdown(f"**Vector Dimension:** `{dim}`")
    st.success("Status: Ready", icon="✅")

    # Document files list expander
    with st.expander("📁 Indexed Documents"):
        for f in doc_stats["files"]:
            st.markdown(f"- `{f}`")

    # Rebuild Knowledge Base
    if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
        st.session_state.kb_rebuild_count += 1
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # Upload Additional Document
    st.subheader("📤 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload additional document",
        type=["txt", "pdf", "docx", "pptx"],
        help="Supported formats: TXT, PDF, DOCX and PPTX."
    )

    if uploaded_file is not None:
        target_path = DEFAULT_DOCS_DIR / uploaded_file.name

        if target_path.exists() and uploaded_file.name in [
            "ai_ml_reference.txt",
            "career_roadmap.txt",
            "goals_and_tasks.txt",
            "personal_profile.txt",
            "projects.txt",
            "study_notes.txt"
        ]:
            st.warning("Cannot overwrite core knowledge documents.")

        else:
            try:
                target_path.write_bytes(uploaded_file.getbuffer())

                st.success(f"Saved `{uploaded_file.name}`!")

                if st.button("Rebuild to include new file"):
                    st.session_state.kb_rebuild_count += 1
                    st.cache_resource.clear()
                    st.rerun()

            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.divider()

    # Groq API Configuration
    st.subheader("⚙️ Settings")
    env_api_key = os.getenv("GROQ_API_KEY", "")
    api_key_input = st.text_input("Groq API Key", value=env_api_key, type="password", help="Reads from .env or allows manual entry.")

    model_choice = st.selectbox(
        "Groq Model",
        options=["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
        index=0,
        help="openai/gpt-oss-20b matches your previous workshop projects; 120b provides expanded reasoning capacity."
    )
    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.memory = []
        st.rerun()

# Validate API Key
api_key = api_key_input.strip() if api_key_input else env_api_key
if not api_key:
    st.error("⚠️ GROQ_API_KEY is missing. Please add it to `.env` or paste it in the sidebar settings.")
    st.stop()

# Initialize Groq client
try:
    groq_client = get_groq_client(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

# Main Application Header
st.markdown(f"""
<div class="header-box">
    <div class="header-title">🧠 Personal Knowledge Multi-Agent Document Assistant</div>
    <div class="header-tagline">Your documents. Your knowledge. Your AI.</div>
    <div class="badge-kb">Knowledge Base: {doc_stats['doc_count']} documents | {doc_stats['chunk_count']} chunks indexed</div>
</div>
""", unsafe_allow_html=True)

# Pipeline Visualizer Placeholder
pipeline_placeholder = st.empty()
pipeline_placeholder.markdown(get_pipeline_html("idle"), unsafe_allow_html=True)

# Sample Questions Suggestions
st.markdown("**Suggested questions for Aditya's documents:**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("💼 What skills do I currently have?", use_container_width=True):
        st.session_state.pending_query = "What skills do I currently have?"
with col2:
    if st.button("🎯 What should I learn before AI agents?", use_container_width=True):
        st.session_state.pending_query = "What should I learn before AI agents?"
with col3:
    if st.button("🚀 Skills to become an AI/ML engineer?", use_container_width=True):
        st.session_state.pending_query = "What skills should I learn to become an AI/ML engineer?"
with col4:
    if st.button("🔍 What are my current skill gaps?", use_container_width=True):
        st.session_state.pending_query = "Based on my documents, what are my current skill gaps?"

# Render Chat History
for message in st.session_state.memory:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If assistant message has stored activity metadata, render it
        if message["role"] == "assistant" and "activity" in message:
            act = message["activity"]
            st.markdown(f"""
            <div class="activity-box">
                <div class="activity-step">✅ <b>Query Agent:</b> {act.get('intent', 'Processed')} | Search: <i>"{act.get('query', '')}"</i></div>
                <div class="activity-step">✅ <b>Retrieval Agent:</b> Retrieved {act.get('chunk_count', 0)} chunks from {len(act.get('sources', []))} documents (Top score: {act.get('top_score', 0)})</div>
                <div class="activity-step">✅ <b>Document Analysis Agent:</b> Grounded personalized answer synthesized</div>
                <div class="activity-step">{'✅' if act.get('research_status') == 'completed' else '⚪'} <b>Research Agent:</b> {act.get('research_summary', 'Skipped')}</div>
            </div>
            """, unsafe_allow_html=True)

        # If assistant message has stored retrieved chunks, render expander
        if message["role"] == "assistant" and "retrieved_chunks" in message and message["retrieved_chunks"]:
            with st.expander("🔎 Retrieved Personal Context", expanded=False):
                for item in message["retrieved_chunks"]:
                    st.markdown(f"**Source:** `{item['source']}` | **Chunk ID:** `{item['chunk_id']}` | **Similarity Score:** `{item['score']}`")
                    st.code(item["text"], language=None)
                    st.markdown("---")

# Handle Input (Either from Chat Input or Sample Prompt Click)
user_prompt = st.chat_input("Ask a question about your documents...")
active_query = user_prompt or st.session_state.pending_query

if active_query:
    st.session_state.pending_query = None

    # Render User Message
    st.session_state.memory.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    with st.chat_message("assistant"):
        activity_status = st.empty()

        try:
            # ----------------------------------------------------
            # Step 1: Query Agent
            # ----------------------------------------------------
            pipeline_placeholder.markdown(get_pipeline_html("query"), unsafe_allow_html=True)
            activity_status.info("● **Query Agent** — Analyzing question intent and formulating search query...")
            
            query_output = query_agent(active_query, groq_client, model_choice)
            time.sleep(0.3)  # Brief visual cadence

            # ----------------------------------------------------
            # Step 2: Retrieval Agent
            # ----------------------------------------------------
            pipeline_placeholder.markdown(get_pipeline_html("retrieval"), unsafe_allow_html=True)
            activity_status.info("● **Retrieval Agent** — Searching FAISS index for relevant personal chunks...")
            
            retrieval_output = retrieval_agent(query_output, index, chunks, embed_model, top_k=5)
            time.sleep(0.3)

            # ----------------------------------------------------
            # Step 3: Research Agent (Modular / Optional)
            # ----------------------------------------------------
            research_output = None
            research_output = research_agent(
                query_output,
                groq_client,
                model_choice,
                enabled=True
                )
            # ----------------------------------------------------
            # Step 4: Document Analysis Agent
            # ----------------------------------------------------
            pipeline_placeholder.markdown(get_pipeline_html("analysis"), unsafe_allow_html=True)
            activity_status.info("● **Document Analysis Agent** — Reasoning over personal documents and synthesizing answer...")

            final_answer = document_analysis_agent(
                active_query,
                query_output,
                retrieval_output,
                groq_client,
                model_choice,
                research_data=research_output
            )

            # ----------------------------------------------------
            # Step 5: Final Completed Pipeline & Response
            # ----------------------------------------------------
            pipeline_placeholder.markdown(get_pipeline_html("done"), unsafe_allow_html=True)
            activity_status.empty()  # Clear transient status

            # Display final answer
            st.markdown(final_answer)

            # Compact Activity Box
            activity_meta = {
                "intent": query_output.get("intent", "General"),
                "query": query_output.get("search_query", active_query),
                "chunk_count": retrieval_output.get("chunk_count", 0),
                "sources": retrieval_output.get("sources", []),
                "top_score": retrieval_output.get("top_score", 0.0),
                "research_status": research_output.get("status") if research_output else "skipped",
                "research_summary": research_output.get("summary") if research_output else "Skipped"
            }

            st.markdown(f"""
            <div class="activity-box">
                <div class="activity-step">✅ <b>Query Agent:</b> {activity_meta['intent']} | Refined Search: <i>"{activity_meta['query']}"</i></div>
                <div class="activity-step">✅ <b>Retrieval Agent:</b> Retrieved {activity_meta['chunk_count']} chunks from {len(activity_meta['sources'])} sources (Top score: {activity_meta['top_score']})</div>
                <div class="activity-step">✅ <b>Document Analysis Agent:</b> Personalized reasoning synthesized</div>
                <div class="activity-step">{'✅' if activity_meta['research_status'] == 'completed' else '⚪'} <b>Research Agent:</b> {activity_meta['research_summary']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Retrieved Context Expander
            with st.expander("🔎 Retrieved Personal Context", expanded=False):
                for item in retrieval_output.get("chunks", []):
                    st.markdown(f"**Source:** `{item['source']}` | **Chunk ID:** `{item['chunk_id']}` | **Similarity Score:** `{item['score']}`")
                    st.code(item["text"], language=None)
                    st.markdown("---")

            # Store in Session Memory
            st.session_state.memory.append({
                "role": "assistant",
                "content": final_answer,
                "activity": activity_meta,
                "retrieved_chunks": retrieval_output.get("chunks", [])
            })

        except Exception as e:
            pipeline_placeholder.markdown(get_pipeline_html("idle"), unsafe_allow_html=True)
            activity_status.empty()
            st.error(f"An error occurred while processing your request: {str(e)}")
