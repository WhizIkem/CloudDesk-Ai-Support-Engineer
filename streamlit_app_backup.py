import os
import time
import hashlib
import streamlit as st
from huggingface_hub import InferenceClient

from retrieval_pipeline import (
    load_config,
    get_vector_store,
    run_rag_pipeline,
)

from database import DatabaseManager
from features import (
    MetricsTracker,
    FeedbackSystem,
    KnowledgeUpdater,
    SessionManager,
    EscalationRules,
    CacheManager,
    TaggingSystem,
    NotificationHandler,
    SearchAnalytics
)


st.set_page_config(
    page_title="CloudDesk AI Support Engineer",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Custom CSS for enhanced visual appeal
def inject_custom_css():
    st.markdown("""
    <style>
    /* Main color scheme */
    :root {
        --primary-color: #0066cc;
        --secondary-color: #00d9ff;
        --success-color: #00cc66;
        --warning-color: #ff9900;
        --danger-color: #ff3333;
        --dark-bg: #0f1419;
        --light-bg: #f8f9fa;
        --border-color: #e0e0e0;
    }
    
    /* Page background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Custom title styling */
    .header-container {
        background: linear-gradient(135deg, #0066cc 0%, #00d9ff 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.15);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    .stChatMessage.user {
        background: linear-gradient(135deg, #f0f4ff 0%, #f5e6ff 100%);
        border-left: 4px solid #0066cc;
    }
    
    .stChatMessage.assistant {
        background: linear-gradient(135deg, #f0fff9 0%, #e6f9ff 100%);
        border-left: 4px solid #00d9ff;
    }
    
    /* Chat input styling */
    .stChatInput {
        border-radius: 12px !important;
        border: 2px solid #e0e0e0 !important;
        background-color: #ffffff !important;
    }
    
    .stChatInput:focus-within {
        border: 2px solid #0066cc !important;
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1) !important;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #e6f3ff 0%, #f0e6ff 100%);
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e6ffe6 0%, #f0fff0 100%);
        border-left: 4px solid #00cc66;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff9e6 0%, #fff5e6 100%);
        border-left: 4px solid #ff9900;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #ffe6e6 0%, #fff0f0 100%);
        border-left: 4px solid #ff3333;
    }
    
    /* Metric card styling */
    .metric-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #0066cc;
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.1);
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #0066cc;
        font-size: 1.8rem;
        font-weight: 800;
    }
    
    /* Divider styling */
    .stDivider {
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #e0e0e0, transparent);
        margin: 2rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 102, 204, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #f0f0f0 100%);
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #f0f4ff 0%, #f0f0f0 100%);
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #0066cc !important;
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


@st.cache_resource
def init_db():
    """Initialize database and features"""
    db = DatabaseManager()
    return db


def get_or_create_session(db: DatabaseManager):
    """Get or create session for current user"""
    if 'session_id' not in st.session_state:
        # Create anonymous user
        user_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        st.session_state.user_id = user_id
        
        session_manager = SessionManager(db)
        st.session_state.session_id = session_manager.create_session(user_id)
    
    return st.session_state.session_id, st.session_state.user_id


# Inject custom CSS
inject_custom_css()

# Header section
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🚀 CloudDesk AI Support Engineer</h1>
        <p class="header-subtitle">Intelligent assistance powered by retrieval-augmented generation</p>
    </div>
""", unsafe_allow_html=True)

# Description
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #555;">
        <p style="font-size: 1rem; line-height: 1.6;">
        Ask any CloudDesk support question and receive an intelligent, grounded response 
        based on our comprehensive support knowledge base. Our AI analyzes your query and 
        provides sources to support every answer.
        </p>
    </div>
    """, unsafe_allow_html=True)

try:
    cfg, vstore, client = initialise_pipeline()

    question = st.chat_input("💬 Ask a CloudDesk support question...", key="chat_input")

    if question:
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        # Display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            start_time = time.perf_counter()

            with st.spinner("🔍 Searching CloudDesk knowledge base..."):
                result = run_rag_pipeline(
                    cfg,
                    vstore,
                    client,
                    question,
                )
                
            response_time = time.perf_counter() - start_time

            # Main response
            st.markdown(result["answer"])
            
            # Metrics section
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">⏱️ Response Time</div>
                    <div class="metric-value">{response_time:.2f}s</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                confidence_value = result.get('confidence_pct', 'N/A')
                confidence_level = "🟢" if confidence_value != 'N/A' and float(confidence_value.rstrip('%')) >= 60 else "🟡"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{confidence_level} Confidence</div>
                    <div class="metric-value">{confidence_value}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                status = "⚠️ Escalated" if result["requires_escalation"] else "✅ Standard"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Status</div>
                    <div class="metric-value" style="font-size: 1.2rem;">{status}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Escalation warning if needed
            if result["requires_escalation"]:
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ Escalation Notice</strong><br>
                    This question has been escalated to Tier-2 Support because the retrieval 
                    confidence is below the 60% threshold. A human specialist will review this 
                    query to ensure you receive the most accurate assistance.
                </div>
                """, unsafe_allow_html=True)
            
            # Sources section
            if result["citations"]:
                with st.expander("📚 View Sources & References", expanded=False):
                    st.markdown(result["citations"])
            else:
                st.info("💡 No additional sources found for this query.")

except Exception as e:
    st.markdown("""
    <div class="danger-box">
        <strong>❌ Error Starting Assistant</strong><br>
        The CloudDesk support assistant could not be initialized.
    </div>
    """, unsafe_allow_html=True)
    st.exception(e)