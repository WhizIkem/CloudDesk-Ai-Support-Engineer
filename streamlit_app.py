import os
import time
import streamlit as st
from huggingface_hub import InferenceClient

from retrieval_pipeline import (
    load_config,
    get_vector_store,
    run_rag_pipeline,
)


st.set_page_config(
    page_title="CloudDesk AI Support Engineer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for better styling
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
    }
    .response-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-card {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .error-card {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .header-container { 
        text-align: center; 
        padding: 20px 0; 
        margin-bottom: 30px;
    }
    .header-title {
        font-size: 48px;
        font-weight: bold;
        margin: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)

# Load Streamlit Cloud secrets into environment variables

try:
    for key in [
        "HF_TOKEN",
        "HF_MODEL",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
    ]:
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass


@st.cache_resource
def initialise_pipeline():
    cfg = load_config()
    vstore = get_vector_store(cfg)

    client = (
        InferenceClient(api_key=cfg["hf_token"])
        if cfg.get("hf_token")
        else None
    )

    return cfg, vstore, client


# Header Section
st.markdown("""
<div class="header-container">
    <div class="header-title">🤖 CloudDesk AI Support Engineer</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.metric("📚 Knowledge Base", "105 chunks")
with col2:
    st.metric("🎯 Confidence Threshold", "60%")
with col3:
    st.metric("⚡ Embedding Model", "MiniLM-L6")

st.divider()

# Introduction
st.markdown("""
### 💬 How it Works
1. **Ask** a CloudDesk support question
2. **Retrieve** relevant documentation from the knowledge base
3. **Generate** an AI-powered response with confidence score
4. **Escalate** to human support if confidence is too low

> *Powered by RAG (Retrieval-Augmented Generation) | Instant answers from your knowledge base*
""")

# Example questions
st.markdown("### 📋 Try These Questions:")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("🟢 **Good Question**\n\nHow do I troubleshoot SSO SAML authentication?")
with col2:
    st.warning("🟡 **Partial Coverage**\n\nHow do I fix webhook delivery issues?")
with col3:
    st.error("🔴 **Out of Scope**\n\nHow do I reset my password?")

st.divider()

try:
    cfg, vstore, client = initialise_pipeline()

    question = st.chat_input("🔍 Ask a CloudDesk support question...", key="question_input")

    if question:
        # User message
        with st.chat_message("user", avatar="👤"):
            st.write(question)

        # Assistant response
        with st.chat_message("assistant", avatar="🤖"):
            start_time = time.perf_counter()

            with st.spinner("🔎 Searching knowledge base... This may take a few seconds."):
                result = run_rag_pipeline(
                    cfg,
                    vstore,
                    client,
                    question,
                )
                
            response_time = time.perf_counter() - start_time

            # Parse confidence for color coding
            confidence_value = float(result['confidence_pct'].rstrip('%'))
            
            # Determine status icon and color
            if result["requires_escalation"]:
                status_icon = "⚠️"
                status_color = "warning"
                status_text = "ESCALATED"
            elif confidence_value >= 80:
                status_icon = "✅"
                status_color = "success"
                status_text = "CONFIDENT"
            elif confidence_value >= 60:
                status_icon = "⚡"
                status_color = "info"
                status_text = "PARTIAL"
            else:
                status_icon = "❌"
                status_color = "error"
                status_text = "LOW CONFIDENCE"

            # Status badge
            st.markdown(f"### {status_icon} Response Status: **{status_text}**")
            
            # Answer card
            st.markdown("### 📝 Answer")
            with st.container(border=True):
                st.markdown(result["answer"])

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⏱️ Response Time", f"{response_time:.2f}s")
            with col2:
                confidence_css = "confidence-high" if confidence_value >= 80 else "confidence-medium" if confidence_value >= 60 else "confidence-low"
                st.markdown(f"<div class='{confidence_css}'>🎯 Confidence: {result['confidence_pct']}</div>", unsafe_allow_html=True)
            with col3:
                st.metric("📚 Documents Retrieved", "3")
            with col4:
                st.metric("🔤 Embedding Model", "MiniLM-L6")

            st.divider()

            # Escalation warning if needed
            if result["requires_escalation"]:
                st.warning(
                    "🚨 **Auto-Escalated to Tier-2 Support**\n\n"
                    "This question has been flagged for human review because the confidence score "
                    "is below the 60% threshold. A support engineer will review this shortly."
                )

            # Sources section
            if result["citations"]:
                st.markdown("### 📖 Sources & References")
                with st.expander("Click to view source documents"):
                    st.markdown(result["citations"])

except Exception as e:
    st.error("❌ The CloudDesk support assistant could not be started.")
    st.exception(e)