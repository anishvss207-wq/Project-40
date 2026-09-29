import os
import json
import re
import httpx
from groq import Groq
from rag import retrieve_chunks

DEFAULT_MODEL = "openai/gpt-oss-20b"

def get_groq_client(api_key: str) -> Groq:
    """Initialize Groq client with SSL-resilient http_client for Windows."""
    return Groq(api_key=api_key, http_client=httpx.Client(verify=False))

def call_llm(client: Groq, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """
    Direct Groq chat completion call reusing the existing workshop pattern.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def query_agent(user_question: str, client: Groq, model: str) -> dict:
    """
    Agent 1: Query Agent
    Analyzes the user's natural language question, determines intent, extracts key entities,
    decides whether external industry research is required, and formulates an optimized
    vector search query for personal knowledge retrieval.
    """
    system_prompt = """You are the specialized Query Agent for Aditya Varma's Personal Knowledge System.
Your job is to analyze the user's question, determine intent, extract key concepts, and formulate an optimized search query for FAISS vector search across personal documents.

Available personal document topics:
- personal_profile: personal details, skill ratings (out of 10), tools, education, background.
- career_roadmap: target roles (AI/ML Engineer), required skills, skill gaps, milestones to 2028.
- goals_and_tasks: Semester 5 academic goals, LeetCode 400 target, exam dates, project status.
- projects: Satellite Image Analysis, AI Code Error Detection, IPL Prediction, Smart Farming, RAG Assistant.
- study_notes: personal notes on Python, DSA, SQL, ML, Deep Learning, NLP with understanding levels.
- ai_ml_reference: definitions, architectures, pros/cons, and technical reference.

Respond ONLY with a valid JSON object in this exact format:
{
  "intent": "<short intent: e.g. skill_inquiry, career_planning, gap_analysis, project_details, study_help>",
  "search_query": "<refined keywords optimized for vector search in personal files>",
  "focus_areas": ["<relevant topics from above>"],
  "needs_research": <true if the user asks for external industry trends/standards/job market, false otherwise>,
  "query_summary": "<one sentence summarizing what the user is asking>"
}
Do not include any Markdown fences or text outside the JSON object."""

    raw_response = call_llm(client, model, system_prompt, f"User Question: {user_question}", temperature=0.1)

    # Robust JSON extraction
    try:
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
        else:
            data = json.loads(raw_response)
    except Exception:
        # Safe fallback if LLM produces unexpected format
        data = {
            "intent": "general_knowledge_inquiry",
            "search_query": user_question,
            "focus_areas": ["personal_profile", "career_roadmap", "projects", "study_notes"],
            "needs_research": any(k in user_question.lower() for k in ["market", "job", "trend", "industry", "become an ai/ml engineer"]),
            "query_summary": user_question
        }

    # Ensure required keys exist
    data.setdefault("search_query", user_question)
    data.setdefault("intent", "general_inquiry")
    data.setdefault("focus_areas", ["personal_profile", "career_roadmap"])
    data.setdefault("needs_research", False)
    data.setdefault("query_summary", user_question)

    return data

def retrieval_agent(query_data: dict, index, chunks: list, embed_model, top_k: int = 5) -> dict:
    """
    Agent 2: Retrieval Agent
    Takes the refined query from the Query Agent, queries the FAISS index, retrieves the most
    relevant personal chunks, formats them with metadata, and prepares them for the Analysis Agent.
    """
    search_query = query_data.get("search_query", "")
    retrieved = retrieve_chunks(search_query, index, chunks, embed_model, top_k=top_k)

    sources = sorted(list({c["source"] for c in retrieved})) if retrieved else []
    top_score = retrieved[0]["score"] if retrieved else 0.0

    # Build clean context block with source document headers
    context_lines = []
    for item in retrieved:
        context_lines.append(
            f"--- Document: {item['source']} (Chunk #{item['chunk_id']}) | Similarity: {item['score']} ---\n"
            f"{item['text']}\n"
        )
    formatted_context = "\n".join(context_lines)

    return {
        "chunks": retrieved,
        "sources": sources,
        "top_score": top_score,
        "chunk_count": len(retrieved),
        "formatted_context": formatted_context
    }

def research_agent(query_data: dict, client: Groq, model: str, enabled: bool = True) -> dict:
    """
    Modular Research Agent:
    Provides current industry trends and market benchmark knowledge when required.
    Does NOT pretend live research occurred if disabled.
    """
    needs_research = query_data.get("needs_research", False)

    if not enabled:
        return {
            "status": "disabled",
            "summary": "External research tool is toggled off in settings.",
            "market_insights": None
        }

    if not needs_research:
        return {
            "status": "skipped",
            "summary": "Skipped (query is fully covered by personal knowledge base).",
            "market_insights": None
        }

    # Synthesize current industry standards for AI/ML roles
    system_prompt = """You are the Industry Research Agent.
Your task is to provide concise, factual 2026 industry standards and benchmark requirements for modern AI/ML, Generative AI, and Agent engineering roles.
Highlight what product companies currently require:
- Production LLMs & Agentic patterns (tool calling, reflection, routing)
- Evaluation frameworks (RAG triaging, hallucination mitigation)
- System design, PyTorch loops, MLOps, CI/CD, and coding standards
Keep your summary factual, concise, and structured in bullet points."""

    user_prompt = f"Research market standards and current requirements related to: {query_data.get('query_summary', '')}"
    try:
        insights = call_llm(client, model, system_prompt, user_prompt, temperature=0.2)
        return {
            "status": "completed",
            "summary": "Synthesized current AI/ML industry skill requirements and benchmarks.",
            "market_insights": insights
        }
    except Exception as e:
        return {
            "status": "error",
            "summary": f"Research lookup unavailable: {str(e)}",
            "market_insights": None
        }

def document_analysis_agent(user_question: str, query_data: dict, retrieval_data: dict, client: Groq, model: str, research_data: dict = None) -> str:
    """
    Agent 3: Document Analysis Agent
    Takes the user question, the Query Agent's intent, the Retrieval Agent's context,
    and optional Research Agent output to perform deep personalized reasoning.
    Produces an actionable, grounded, and personalized answer.
    """
    context = retrieval_data.get("formatted_context", "").strip()

    if not context:
        return "I could not find relevant information in your personal documents to answer this question. Please ensure your documents are loaded into the knowledge base."

    research_section = ""
    if research_data and research_data.get("status") == "completed" and research_data.get("market_insights"):
        research_section = f"""
==================================================
CURRENT INDUSTRY RESEARCH (BENCHMARKS & TRENDS):
==================================================
{research_data['market_insights']}
"""

    system_prompt = f"""You are the Document Analysis Agent in Aditya Varma's Personal Knowledge Multi-Agent System.
Your job is to provide high-quality, personalized answers strictly grounded in Aditya's personal documents.

Guidelines:
1. PERSONALIZED REASONING:
   - Refer specifically to Aditya's documented information: skill ratings out of 10 (e.g. Python 8, ML 7, DL 6, GenAI 5.5, RAG 5, AI agents 4), specific projects (EuroSAT Satellite Image Analysis, IPL Prediction, Personal Document RAG Assistant), academic status (3rd Year B.Tech AI & ML at CAIT, Semester 5, CGPA 8.4), and specific goals (e.g. 400 LeetCode problems, SGPA 8.7).
2. SKILL GAP ANALYSIS:
   - When asked what to learn or what gaps exist, compare Aditya's CURRENT documented skills against target requirements (AI/ML Engineer role or industry trends) and explicitly identify what is missing or needs strengthening (e.g. graphs/DP, PyTorch training loops, pytest unit testing, advanced agent workflows).
3. ATTRIBUTION & CITATIONS:
   - Ground your answer in the provided documents. Mention the relevant source documents (e.g., [personal_profile.txt], [career_roadmap.txt], [projects.txt]) where appropriate.
4. STRUCTURE:
   - Format cleanly with markdown headings, bullet points, and an 'Actionable Next Steps' section.
5. NO HALLUCINATION:
   - Do not invent facts that contradict the personal context."""

    user_prompt = f"""User Question:
{user_question}

Query Intent:
{query_data.get('intent', 'general')} (Focus: {', '.join(query_data.get('focus_areas', []))})

==================================================
RETRIEVED PERSONAL DOCUMENT CONTEXT:
==================================================
{context}
{research_section}

Please analyze the context and provide a personalized, comprehensive answer for Aditya."""

    answer = call_llm(client, model, system_prompt, user_prompt, temperature=0.25)
    return answer
