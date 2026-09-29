import os
import sys
from pathlib import Path
from dotenv import load_dotenv

print("Testing Project-40 Personal Knowledge Multi-Agent Pipeline...")

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY present: {bool(api_key)}")

from rag import load_documents, load_model, build_vector_index, retrieve_chunks
from agents import query_agent, retrieval_agent, research_agent, document_analysis_agent, get_groq_client, DEFAULT_MODEL

# 1. Test documents
chunks, stats = load_documents(Path("files"))
print(f"Detected {stats['doc_count']} documents, {stats['chunk_count']} chunks.")
print("Files:", stats["files"])
assert stats["doc_count"] == 6, f"Expected 6 documents, found {stats['doc_count']}"

# 2. Test Embedding & FAISS
print("Loading embedding model...")
model = load_model("all-MiniLM-L6-v2")
print("Building FAISS index...")
index, dim = build_vector_index(chunks, model)
print(f"Index built successfully. Vector dimension: {dim}")

# 3. Test Groq Client
client = get_groq_client(api_key=api_key)
test_question = "What skills do I currently have?"
print(f"\n--- Testing Question: '{test_question}' ---")

# Step 1: Query Agent
print("\n[Step 1] Running Query Agent...")
q_out = query_agent(test_question, client, DEFAULT_MODEL)
print("Query Agent Output:", q_out)
assert "search_query" in q_out

# Step 2: Retrieval Agent
print("\n[Step 2] Running Retrieval Agent...")
r_out = retrieval_agent(q_out, index, chunks, model, top_k=3)
print(f"Retrieved {r_out['chunk_count']} chunks from sources: {r_out['sources']} (Top score: {r_out['top_score']})")
assert r_out["chunk_count"] > 0

# Step 3: Research Agent
print("\n[Step 3] Running Research Agent...")
res_out = research_agent(q_out, client, DEFAULT_MODEL, enabled=True)
print("Research Agent Status:", res_out["status"])

# Step 4: Document Analysis Agent
print("\n[Step 4] Running Document Analysis Agent...")
analysis = document_analysis_agent(test_question, q_out, r_out, client, DEFAULT_MODEL, res_out)
print("\nAnalysis Result:\n", analysis[:400].encode('ascii', errors='replace').decode('ascii'), "...\n")
assert len(analysis) > 50

print("ALL PIPELINE TESTS PASSED SUCCESSFULLY!")
