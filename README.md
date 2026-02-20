<div align="center">

# 🌱 Carbon RAG Assistant

**An AI-powered Retrieval-Augmented Generation chatbot for querying Carbon Offset Project Design Documents (PDDs)**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com)

</div>

---

## 📖 Project Overview

Carbon RAG Assistant is a domain-specific conversational AI agent built for the **carbon credit and sustainability** industry. It ingests Carbon Offset Project Design Documents (PDDs) — complex, multi-section regulatory PDFs — and enables users to ask natural-language questions grounded strictly in the document's content.

The system uses a **Hybrid Retrieval** pipeline combining dense semantic search (vector embeddings) with sparse keyword search (BM25), ensuring both conceptual understanding and exact-match precision when answering domain-specific queries.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit (Chat UI with session history) |
| **LLM** | Google Gemini 2.5 Flash via LangChain |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| **Vector Store** | ChromaDB (persistent, on-disk) |
| **Retrieval** | LangChain Ensemble Retriever (BM25 + MMR Vector Search) |
| **PDF Parsing** | PyMuPDF4LLM (Markdown-aware PDF extraction) |
| **Text Splitting** | Markdown Header Splitter + Recursive Character Splitter |
| **Framework** | LangChain (LCEL chains, prompt templates, output parsers) |

---

## ✨ Core Features

- **🔗 Hybrid RAG Pipeline** — Combines BM25 keyword retrieval (40% weight) with MMR-based semantic vector search (60% weight) via an Ensemble Retriever for high-precision, context-aware answers.
- **📄 Intelligent PDF Ingestion** — Converts complex regulatory PDFs to Markdown using PyMuPDF4LLM, then applies two-stage splitting (Markdown headers → Recursive character) to preserve document structure.
- **💾 Persistent Vector Store** — ChromaDB persists embeddings to disk, eliminating redundant re-indexing on page refresh or re-launch.
- **🛡️ Rate Limit Guardrail** — Built-in retry logic with exponential backoff for Google API 429/Resource Exhausted errors.
- **💬 Conversational Memory** — Maintains rolling chat history (last 5 messages) injected into the prompt for multi-turn context.
- **📂 Multi-Project Ready** — Modular project selector in the sidebar for switching between different PDDs without code changes.
- **🔒 Hallucination Guard** — System prompt enforces strict grounding: the LLM only answers from retrieved context or explicitly states insufficient information.

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.9+
- A [Google AI Studio API Key](https://aistudio.google.com/app/apikey) (free tier available)

### 1. Clone the Repository
```bash
git clone https://github.com/akash-iitd/carbon-rag-assistant.git
cd carbon-rag-assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Select a project document from the sidebar and start asking questions.

---

## 📁 Project Structure

```
carbon-rag-assistant/
├── app.py                 # Main Streamlit application (RAG pipeline + UI)
├── requirements.txt       # Python dependencies
├── arr_project_pdd.pdf    # Sample Carbon Offset PDD document
├── chroma_db_store/       # Auto-generated persistent vector store
└── .env                   # API key configuration (user-created)
```

---

## 🏗️ Architecture

```
PDF → PyMuPDF4LLM → Markdown → Header Splitter → Chunk Splitter
                                                        ↓
                                              HuggingFace Embeddings
                                                        ↓
User Query → Ensemble Retriever (BM25 + ChromaDB MMR) → Context
                                                        ↓
                                    Gemini 2.5 Flash + Chat History → Answer
```

---

<div align="center">
  <sub>Built with ❤️ for the Carbon Credit & Sustainability industry</sub>
</div>
