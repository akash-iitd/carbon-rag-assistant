import os
import time
import sys
import logging
import shutil
from dotenv import load_dotenv
import streamlit as st
import pymupdf.layout
import pymupdf4llm

# --- LangChain Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DB_ROOT = "chroma_db_store"
PROJ_FILES = {
    "ARR Project": "arr_project_pdd.pdf",
    # Add more projects here easily:
    # "Solar Project": "solar_pdd.pdf"
}

# --- 1. Helper: Safe API Caller (Rate Limit Guardrail) ---
def safe_google_call(function, *args, **kwargs):
    """
    Executes a function with automatic retry logic for Rate Limit (429) errors.
    Waits 60 seconds if the limit is hit.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "resource_exhausted" in error_msg:
                st.warning(f"⚠️ API Rate Limit Hit. Waiting 60 seconds (Attempt {attempt+1}/{max_retries})...")
                time.sleep(60)
            else:
                raise e # Re-raise other errors immediately
    raise Exception("Max retries exceeded. Please try again later.")

# --- 2. Cached Resource: The RAG Engine ---
# @st.cache_resource ensures this runs ONLY ONCE per session.
@st.cache_resource(show_spinner="Initializing RAG Engine...")
def setup_rag_engine(selected_project_name):
    """
    Loads PDF, creates Embeddings, builds Vector Store, and returns the Retriever.
    Persists data to disk so it doesn't reload on page refresh.
    """
    try:
        file_path = PROJ_FILES[selected_project_name]
        collection_name = f"vec_{selected_project_name.lower().replace(' ', '_')}"
        persist_dir = os.path.join(DB_ROOT, collection_name)

        # A. Setup Embeddings (Local - No API Cost)
        print("[-] Loading Local Embeddings...")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

        # B. Load Vector Store
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )

        # C. Ingest Data (Only if empty)
        existing_count = len(vector_store.get()['ids'])
        
        if existing_count == 0:
            st.info(f"📂 Indexing {selected_project_name} for the first time...")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF not found at: {file_path}")

            # 1. Read & Convert to Markdown
            md_text = pymupdf4llm.to_markdown(file_path)
            if isinstance(raw_md_text, bytes):
                md_text = raw_md_text.decode("utf-8")
            else:
                md_text = raw_md_text.replace("\ufffd", "")

            # 2. Advanced Splitting
            headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_docs = markdown_splitter.split_text(md_text)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            splits = text_splitter.split_documents(md_docs)
            
            # 3. Add to DB (Batching isn't strictly needed for Local Embeddings, but good practice)
            # Since we use Local Embeddings, we don't need the 20s sleep here! It's fast and free.
            vector_store.add_documents(splits)
            st.success(f"✅ Successfully indexed {len(splits)} chunks!")

        # D. Build Retrievers
        all_docs = vector_store.get()
        # Guardrail: Handle empty database
        if not all_docs['documents']:
            return None, "Database is empty."

        # Handle None metadata using the 'or {}' fix
        doc_objects = [
            Document(page_content=txt, metadata=meta or {}) 
            for txt, meta in zip(all_docs['documents'], all_docs['metadatas'])
        ]

        # BM25 (Keyword Search)
        bm25_retriever = BM25Retriever.from_documents(doc_objects)
        bm25_retriever.k = 3

        # Vector (Semantic Search)
        vector_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "lambda_mult": 0.5}
        )

        # Ensemble
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
        
        return ensemble_retriever, None

    except Exception as e:
        return None, str(e)

# --- 3. UI Layout ---
st.set_page_config(page_title="Carbon RAG Agent", layout="wide")

st.title("🌱 Carbon Offset Project Assistant")

# Sidebar: Project Selection
with st.sidebar:
    st.header("Configuration")
    selected_project = st.selectbox("Select Project Document:", list(PROJ_FILES.keys()))
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("**Status:**")
    
    # Initialize/Load the RAG Engine
    retriever, error_msg = setup_rag_engine(selected_project)
    
    if error_msg:
        st.error(f"System Error: {error_msg}")
        st.stop()
    else:
        st.success("System Ready (Index Loaded)")

# --- 4. Chat Logic with History ---

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about the Carbon Project Design Document (PDD)."}
    ]

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt_text := st.chat_input("Ask a question about the project..."):
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # 2. Generate Answer
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Build Chain on the fly (lightweight)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                temperature=0, 
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            template = """
            You are a Carbon Offsets Project expert. Answer based STRICTLY on the context below.
            If the answer is not in the context, say "I do not have sufficient information in the provided document."
            
            Context:
            {context}

            Chat History:
            {chat_history}

            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)

            # Format history for the prompt
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])

            chain = (
                {
                    "context": retriever, 
                    "chat_history": lambda x: history_str,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            # 3. Invoke with Safety Wrapper (Rate Limit Guard)
            with st.spinner("Analyzing document..."):
                response = safe_google_call(chain.invoke, prompt_text)
            
            message_placeholder.markdown(response)
            
            # 4. Save Bot Message
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"An error occurred: {e}")