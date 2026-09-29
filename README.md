# Project-40: Personal Knowledge Multi-Agent Document Assistant

A production-grade, lightweight personal document assistant powered by a modular multi-agent architecture, FAISS vector search, Groq LLM API, and Streamlit.

---

## 🏛️ System Architecture

```text
       ┌───────────────────────────────┐
       │     Personal .txt Files       │
       │    (Initial 6 Documents)      │
       └──────────────┬────────────────┘
                      ▼
       ┌───────────────────────────────┐
       │   RAG Engine: Sliding-Window  │
       │   Chunking (chunk_text)       │
       └──────────────┬────────────────┘
                      ▼
       ┌───────────────────────────────┐
       │ SentenceTransformer Embeddings│
       │     (all-MiniLM-L6-v2)        │
       └──────────────┬────────────────┘
                      ▼
       ┌───────────────────────────────┐
       │    FAISS IndexFlatIP Store    │
       │   (Cosine Similarity Search)  │
       └──────────────┬────────────────┘
                      │
  User Question       │
       ▼              │
┌──────────────┐      │
│ Query Agent  │──────┤
└──────┬───────┘      │
       ▼              │
┌──────────────────┐  │
│ Retrieval Agent  │◄─┘
└──────┬───────────┘
       │
       ├─────────────────────────┐
       │                         ▼
       │               ┌──────────────────┐
       │               │  Research Agent  │ (Modular / Industry Trends)
       │               └────────┬─────────┘
       ▼                        │
┌───────────────────────────┐   │
│  Document Analysis Agent  │◄──┘
└─────────────┬─────────────┘
              ▼
    Personalized Answer
```

---

## 🤖 The Required Agents

1. **Query Agent (`query_agent`)**:
   - Analyzes user questions to determine the core intent (e.g. `skill_assessment`, `gap_analysis`, `career_roadmap`).
   - Identifies document focus areas and optimizes a search query for vector retrieval.
   - Evaluates whether industry trend research is required.
   - Formats a structured data object passed to downstream agents.

2. **Retrieval Agent (`retrieval_agent`)**:
   - Queries the local FAISS vector index using the Query Agent's refined keywords.
   - Retrieves top-$k$ relevant personal chunks with cosine similarity scores.
   - Formulates grounded context attributed to exact source documents and chunk IDs.

3. **Document Analysis Agent (`document_analysis_agent`)**:
   - Ingests the user query, Query Agent intent, Retrieval Agent context, and Research Agent output.
   - Cross-references Aditya Varma's documented skill ratings, projects, academic status, and goals.
   - Produces a grounded, transparent, personalized analysis.

4. **Research Agent (`research_agent`)**:
   - Modular agent for 2026 industry requirements, market expectations, and skill benchmarks.
   - Cleanly skips when questions are purely personal, never faking live activity.

---

## 📂 Project Structure

```text
project-40/
├── files/                              # Personal knowledge base (untouched)
│   ├── ai_ml_reference.txt
│   ├── career_roadmap.txt
│   ├── goals_and_tasks.txt
│   ├── personal_profile.txt
│   ├── projects.txt
│   └── study_notes.txt
├── rag.py                              # RAG engine: chunk_text, FAISS builder, retrieval
├── agents.py                           # Independent agents: Query, Retrieval, Analysis, Research
├── app.py                              # Streamlit application with custom animated pipeline
├── requirements.txt                    # Project dependencies
├── .env                                # Local environment file with GROQ_API_KEY
├── .env.example                        # Template environment file
├── run_windows.bat                     # Windows one-click launcher
└── README.md                           # Documentation
```

---

## 🚀 How to Run

1. **Launch with One Click**:
   Double click `run_windows.bat` or run:
   ```cmd
   py -3.13 -m streamlit run app.py
   ```

2. **Enter Questions**:
   Try questions like:
   - *"What skills do I currently have?"*
   - *"What should I learn before AI agents?"*
   - *"What skills should I learn to become an AI/ML engineer?"*
   - *"Based on my documents, what are my current skill gaps?"*
