import os
import re
import time
import uuid
import json
import hashlib
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError
from pinecone import Pinecone
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ============================================================
# ENV & CONFIG
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "dc-articles")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "articles")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "6"))
MEMORY_TTL_SECONDS = int(os.getenv("MEMORY_TTL_SECONDS", "3600"))
MEMORY_MAX_SESSIONS = int(os.getenv("MEMORY_MAX_SESSIONS", "500"))

# Enable hybrid search
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
USE_CROSS_ENCODER = os.getenv("USE_CROSS_ENCODER", "true").lower() == "true"
USE_QUERY_EXPANSION = os.getenv("USE_QUERY_EXPANSION", "true").lower() == "true"

if not all([OPENAI_API_KEY, PINECONE_API_KEY]):
    raise ValueError("Missing required environment variables")

# ============================================================
# CLIENTS
# ============================================================

openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

executor = ThreadPoolExecutor(max_workers=4)

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="DebtCircle AI Chatbot", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODELS
# ============================================================

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    top_k: Optional[int] = Field(8, ge=1, le=20)
    session_id: Optional[str] = None
    stream: bool = False

class Source(BaseModel):
    title: str
    url: str
    date: str
    slug: str
    score: float
    relevance_score: Optional[float] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str
    processing_time_ms: int

# ============================================================
# CONVERSATION MEMORY
# ============================================================

@dataclass
class ConversationState:
    updated_at: float = field(default_factory=time.time)
    messages: List[Dict[str, str]] = field(default_factory=list)
    last_retrieval_question: str = ""
    embedding_cache: Dict[str, List[float]] = field(default_factory=dict)
    
    def add_turn(self, user: str, assistant: str, retrieval_question: str):
        self.messages.extend([
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ])
        max_messages = max(2, MEMORY_MAX_TURNS * 2)
        self.messages = self.messages[-max_messages:]
        self.last_retrieval_question = retrieval_question
        self.updated_at = time.time()

_memory_lock = threading.RLock()
_conversation_memory: Dict[str, ConversationState] = {}

def cleanup_memory():
    now = time.time()
    with _memory_lock:
        expired = [
            sid for sid, state in _conversation_memory.items()
            if now - state.updated_at > MEMORY_TTL_SECONDS
        ]
        for sid in expired:
            _conversation_memory.pop(sid, None)
        
        if len(_conversation_memory) > MEMORY_MAX_SESSIONS:
            ordered = sorted(
                _conversation_memory.items(),
                key=lambda item: item[1].updated_at
            )
            remove_count = len(_conversation_memory) - MEMORY_MAX_SESSIONS
            for sid, _ in ordered[:remove_count]:
                _conversation_memory.pop(sid, None)

def get_or_create_session(session_id: Optional[str]) -> str:
    cleanup_memory()
    sid = (session_id or "").strip() or uuid.uuid4().hex
    with _memory_lock:
        if sid not in _conversation_memory:
            _conversation_memory[sid] = ConversationState()
        else:
            _conversation_memory[sid].updated_at = time.time()
    return sid

def get_session_state(session_id: str) -> ConversationState:
    cleanup_memory()
    with _memory_lock:
        state = _conversation_memory.get(session_id)
        if not state:
            state = ConversationState()
            _conversation_memory[session_id] = state
        state.updated_at = time.time()
        return state

# ============================================================
# EMBEDDING CACHE
# ============================================================

embedding_cache: Dict[str, List[float]] = {}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def create_query_embedding(text: str) -> List[float]:
    """Generate query embedding with caching."""
    cache_key = hashlib.md5(text.lower().encode()).hexdigest()
    
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIMENSION
        )
        vector = response.data[0].embedding
        embedding_cache[cache_key] = vector
        
        # Limit cache size
        if len(embedding_cache) > 1000:
            oldest = next(iter(embedding_cache))
            del embedding_cache[oldest]
        
        return vector
    except Exception as e:
        print(f"Embedding error: {e}")
        raise HTTPException(status_code=503, detail="Embedding service unavailable")

# ============================================================
# HYBRID SEARCH (BM25)
# ============================================================

if USE_HYBRID_SEARCH:
    try:
        from pinecone_text.sparse import BM25Encoder
        bm25_encoder = BM25Encoder.default()
    except ImportError:
        USE_HYBRID_SEARCH = False
        print("BM25Encoder not available, falling back to dense search")

# ============================================================
# CROSS-ENCODER RERANKER
# ============================================================

cross_encoder = None
if USE_CROSS_ENCODER:
    try:
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("Cross-encoder loaded successfully")
    except ImportError:
        USE_CROSS_ENCODER = False
        print("Cross-encoder not available, using semantic reranking only")

# ============================================================
# FOLLOW-UP DETECTION
# ============================================================

FOLLOWUP_PREFIXES = (
    "what about", "how about", "and what", "and for",
    "and next", "what if", "for next", "for this",
    "next year", "next financial year", "this year",
    "this financial year", "same for"
)

def looks_like_followup(question: str) -> bool:
    q = question.lower().strip()
    if q.startswith(FOLLOWUP_PREFIXES):
        return True
    
    pronoun_patterns = [
        r"\b(it|its|they|their|that|those|these|same)\b",
        r"\b(next|previous|last)\s+(year|financial year|fy)\b",
    ]
    if any(re.search(pattern, q) for pattern in pronoun_patterns):
        return True
    
    # Very short questions
    return len(q.split()) <= 5

def resolve_followup_query(
    question: str,
    history: List[Dict[str, str]],
    last_retrieval_question: str
) -> str:
    """Resolve follow-up questions using LLM."""
    if not history or not looks_like_followup(question):
        return question
    
    history_text = format_history_for_prompt(history)
    
    system_prompt = """You rewrite follow-up questions for a DebtCircle financial-news search engine.
Return ONE standalone search query only.

Rules:
1. Resolve omitted company/issuer, instrument, topic and financial-year context
2. Preserve relative phrases like 'this financial year' and 'next financial year'
3. If user introduces a new company, use the new entity
4. Do not answer the question or add facts"""

    user_prompt = f"""RECENT CONVERSATION:
{history_text}

PREVIOUS SEARCH TOPIC:
{last_retrieval_question or '(none)'}

NEW USER QUESTION:
{question}

STANDALONE SEARCH QUERY:"""
    
    try:
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            max_tokens=100,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        rewritten = response.choices[0].message.content.strip().strip('"')
        if rewritten:
            return rewritten
    except Exception as e:
        print(f"Follow-up rewrite failed: {e}")
    
    return f"{last_retrieval_question}. Follow-up: {question}" if last_retrieval_question else question

# ============================================================
# QUERY UNDERSTANDING
# ============================================================

RECENCY_WORDS = {"latest", "recent", "recently", "newest", "today", "current"}
SEARCH_NOISE_WORDS = RECENCY_WORDS | {"news", "update", "updates", "headline", "headlines"}

def is_chronology_query(question: str) -> bool:
    q = question.lower()
    if any(re.search(rf"\b{re.escape(word)}\b", q) for word in RECENCY_WORDS):
        return True
    phrases = ["most recent", "this week", "this month", "last week", "last month"]
    return any(phrase in q for phrase in phrases)

def clean_semantic_query(question: str) -> str:
    q = question.lower().strip()
    phrases = ["most recent", "this week", "this month", "last week", "last month"]
    for phrase in phrases:
        q = q.replace(phrase, " ")
    tokens = re.findall(r"[a-zA-Z0-9₹$%&.\-]+", q)
    cleaned = [token for token in tokens if token not in SEARCH_NOISE_WORDS]
    while cleaned and cleaned[0] in {"on", "about", "for"}:
        cleaned.pop(0)
    return " ".join(cleaned).strip() or question.strip()

# ============================================================
# TOPIC/INTENT MATCHING
# ============================================================

TOPIC_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at",
    "by", "for", "from", "to", "and", "or", "with", "about", "what", "show",
    "tell", "me", "give", "please", "latest", "recent", "recently", "newest",
    "today", "current", "news", "update", "updates", "headline", "headlines",
    "this", "next", "last", "previous", "coming", "upcoming", "financial",
    "fiscal", "year", "fy"
}

def topic_tokens(text: str) -> List[str]:
    if not text:
        return []
    raw = re.findall(r"[A-Za-z0-9]+", text.lower())
    tokens = []
    for token in raw:
        if token in TOPIC_STOPWORDS or re.fullmatch(r"20\d{2}", token):
            continue
        if re.fullmatch(r"\d{2}", token):
            continue
        tokens.append(token)
    return tokens

def strip_time_scope(question: str) -> str:
    """Remove time/recency words from query."""
    q = question.lower()
    
    phrases = [
        "most recent", "this financial year", "current financial year",
        "next financial year", "last financial year", "previous financial year",
        "coming financial year", "upcoming financial year", "this fiscal year",
        "current fiscal year", "next fiscal year", "last fiscal year",
        "previous fiscal year", "coming fiscal year", "upcoming fiscal year",
        "this week", "this month", "last week", "last month"
    ]
    for phrase in phrases:
        q = q.replace(phrase, " ")
    
    q = re.sub(r"\b(?:fy\s*)?20\d{2}\s*[-/]\s*\d{2,4}\b", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\bfy\s*['\-]?\s*(?:20)?\d{2}\b", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\b20\d{2}\b", " ", q)
    for word in SEARCH_NOISE_WORDS:
        q = re.sub(rf"\b{re.escape(word)}\b", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"\b(for|in|on|about)\s*$", "", q).strip()
    return q or question.strip()

def topic_match_score(question: str, match) -> float:
    """Calculate topic overlap score for reranking."""
    metadata = match.metadata or {}
    query = set(topic_tokens(strip_time_scope(question)))
    if not query:
        return 0.0
    
    title_tokens = set(topic_tokens(metadata.get("title", "")))
    body_text = " ".join([
        str(metadata.get("summary") or ""),
        str(metadata.get("text") or "")
    ])
    body_tokens = set(topic_tokens(body_text))
    
    title_overlap = len(query & title_tokens) / max(len(query), 1)
    body_overlap = len(query & body_tokens) / max(len(query), 1)
    score = (0.70 * title_overlap) + (0.30 * body_overlap)
    
    # Boost for bond issuance intent
    q_tokens = set(topic_tokens(question))
    issuance_intent = ("bond" in q_tokens and 
                      any(w in q_tokens for w in ["issue", "plan", "raise"]))
    
    if issuance_intent:
        searchable = " ".join([
            str(metadata.get("title") or ""),
            str(metadata.get("summary") or ""),
            str(metadata.get("text") or "")
        ]).lower()
        
        has_bond = bool(re.search(r"\b(bond|bonds|ncd|ncds|at1|tier[\s\-]*[iI]|debenture)\b", searchable))
        has_raise = bool(re.search(r"\b(issue|issuance|raise|fundraise|allot)\b", searchable))
        
        if has_bond and has_raise:
            score += 0.25
        elif has_bond:
            score += 0.10
    
    return min(score, 1.0)

# ============================================================
# FINANCIAL YEAR HANDLING
# ============================================================

IST = timezone(timedelta(hours=5, minutes=30))

def current_financial_year_end(now: Optional[datetime] = None) -> int:
    now = now or datetime.now(IST)
    if now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    else:
        now = now.astimezone(IST)
    return now.year + 1 if now.month >= 4 else now.year

def extract_requested_year(question: str) -> Optional[int]:
    """Extract financial year ending year from question."""
    q = question.lower().strip()
    current_fy_end = current_financial_year_end()
    
    # Relative FY phrases
    if "next financial year" in q or "next fiscal year" in q or "next fy" in q:
        return current_fy_end + 1
    if "this financial year" in q or "current financial year" in q or "this fy" in q:
        return current_fy_end
    if "last financial year" in q or "previous financial year" in q or "last fy" in q:
        return current_fy_end - 1
    
    # FY range
    match = re.search(r"\b(?:fy\s*)?(20\d{2})\s*[-/]\s*(\d{2,4})\b", q)
    if match:
        start = int(match.group(1))
        end_part = match.group(2)
        if len(end_part) == 2:
            end = (start // 100) * 100 + int(end_part)
            if end < start:
                end += 100
        else:
            end = int(end_part)
        return end
    
    # Full year
    match = re.search(r"\b(20\d{2})\b", q)
    if match:
        return int(match.group(1))
    
    # FY27 format
    match = re.search(r"\bfy\s*['\-]?\s*(\d{2})\b", q)
    if match:
        return 2000 + int(match.group(1))
    
    return None

def year_match_score(match, requested_year: Optional[int]) -> float:
    """Score match based on financial year presence."""
    if not requested_year:
        return 0.0
    
    metadata = match.metadata or {}
    searchable = " ".join([
        str(metadata.get("title") or ""),
        str(metadata.get("summary") or ""),
        str(metadata.get("text") or "")
    ]).lower()
    
    previous = requested_year - 1
    short = str(requested_year)[-2:]
    
    patterns = [
        f"fy{short}", f"fy {short}",
        f"fy{previous}-{short}", f"fy {previous}-{short}",
        f"{previous}-{short}", f"{previous}-{requested_year}",
        f"financial year {previous}-{short}",
        f"financial year {previous}-{requested_year}",
        f"fiscal year {previous}-{short}",
        f"fiscal year {previous}-{requested_year}",
    ]
    
    if any(p in searchable for p in patterns):
        return 1.0
    if str(requested_year) in searchable:
        return 0.90
    return 0.0

# ============================================================
# RANKING & RERANKING
# ============================================================

def parse_article_date(metadata) -> datetime:
    if not metadata:
        return datetime.min.replace(tzinfo=timezone.utc)
    
    value = metadata.get("date") or metadata.get("published_at") or ""
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    
    value = str(value).strip()
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)

def deduplicate_matches(matches):
    best_by_post = {}
    for match in matches:
        metadata = match.metadata or {}
        post_key = metadata.get("post_id") or metadata.get("slug") or metadata.get("title") or match.id
        current = best_by_post.get(post_key)
        if current is None or float(match.score or 0) > float(current.score or 0):
            best_by_post[post_key] = match
    return list(best_by_post.values())

def semantic_relevance_filter(matches, threshold=0.6):
    if not matches:
        return []
    best_score = max(float(m.score or 0) for m in matches)
    relevance_floor = max(best_score * threshold, best_score - 0.15)
    return [m for m in matches if float(m.score or 0) >= relevance_floor] or matches

def rank_matches(question: str, matches, top_k: int):
    """Rank matches with semantic, topic, and recency signals."""
    if not matches:
        return []
    
    unique_matches = deduplicate_matches(matches)
    if not unique_matches:
        return []
    
    chronology = is_chronology_query(question)
    requested_year = extract_requested_year(question)
    relevant = semantic_relevance_filter(unique_matches) or unique_matches
    
    now = datetime.now(IST)
    scored = []
    
    for match in relevant:
        metadata = match.metadata or {}
        semantic = float(match.score or 0)
        topic = topic_match_score(question, match)
        fy = year_match_score(match, requested_year)
        
        article_date = parse_article_date(metadata)
        
        # Recency boost
        recency = 0.0
        if article_date.year > 1:
            age_days = max(0, (now.astimezone(timezone.utc) - article_date).days)
            if age_days <= 7:
                recency = 1.0
            elif age_days <= 30:
                recency = 0.80
            elif age_days <= 90:
                recency = 0.55
            elif age_days <= 180:
                recency = 0.30
            elif age_days <= 365:
                recency = 0.15
        
        # Weighted score
        final_score = semantic
        final_score += topic * 0.22
        if requested_year:
            final_score += fy * 0.10
        if chronology:
            final_score += recency * 0.12
        elif requested_year:
            final_score += recency * 0.03
        
        scored.append((match, final_score, semantic, topic, fy, recency, article_date))
    
    scored.sort(key=lambda item: (item[1], item[6], item[2]), reverse=True)
    return [item[0] for item in scored[:top_k]]

# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def rerank_with_cross_encoder(question: str, matches, top_k: int):
    """Use cross-encoder for precise reranking."""
    if not cross_encoder or not matches:
        return matches
    
    try:
        # Prepare pairs
        pairs = []
        valid_matches = []
        for match in matches:
            text = (match.metadata or {}).get("text", "")[:512]
            if text:
                pairs.append((question, text))
                valid_matches.append(match)
        
        if not pairs:
            return matches
        
        # Get cross-encoder scores
        scores = cross_encoder.predict(pairs)
        
        # Combine with semantic scores
        for match, score in zip(valid_matches, scores):
            match.score = (float(match.score or 0) + float(score)) / 2
        
        # Sort by combined score
        valid_matches.sort(key=lambda x: x.score, reverse=True)
        return valid_matches[:top_k]
    except Exception as e:
        print(f"Cross-encoder reranking failed: {e}")
        return matches

# ============================================================
# PINECONE SEARCH
# ============================================================

def search_pinecone(question: str, top_k: int = 8):
    """Search Pinecone with hybrid search support."""

    chronology = is_chronology_query(question)
    requested_year = extract_requested_year(question)

    if chronology or requested_year:
        semantic_question = strip_time_scope(question)
    else:
        semantic_question = question.strip()

    search_top_k = 50 if (chronology or requested_year) else max(top_k, 16)
    search_top_k = min(search_top_k, 100)

    query_vector = create_query_embedding(semantic_question)

    if USE_HYBRID_SEARCH:
        sparse_vector = bm25_encoder.encode_queries(question)

        alpha = 0.5

        hybrid_dense = [
            value * alpha
            for value in query_vector
        ]

        hybrid_sparse = {
            "indices": sparse_vector["indices"],
            "values": [
                value * (1 - alpha)
                for value in sparse_vector["values"]
            ]
        }

        results = index.query(
            namespace=PINECONE_NAMESPACE,
            vector=hybrid_dense,
            sparse_vector=hybrid_sparse,
            top_k=search_top_k,
            include_metadata=True
        )

    else:
        results = index.query(
            namespace=PINECONE_NAMESPACE,
            vector=query_vector,
            top_k=search_top_k,
            include_metadata=True
        )

    return results
# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(question: str) -> str:
    """Expand query with relevant synonyms."""
    if not USE_QUERY_EXPANSION:
        return question
    
    synonyms = {
        "bonds": ["debentures", "fixed income", "debt securities"],
        "issuance": ["issue", "offering", "raising"],
        "nbfc": ["non-banking financial company", "financial institution"],
        "yield": ["return", "interest rate", "coupon"],
        "bank": ["banking institution", "lender"],
    }
    
    expanded = question
    for word, syns in synonyms.items():
        if word in question.lower():
            expanded += f" {' '.join(syns)}"
    
    return expanded

# ============================================================
# CONTEXT & ANSWER GENERATION
# ============================================================

def build_context(matches) -> str:
    context_parts = []
    for number, match in enumerate(matches, start=1):
        metadata = match.metadata or {}
        context_parts.append(f"""
SOURCE {number}

Title:
{metadata.get("title", "")}

Date:
{metadata.get("date", "")}

Published At:
{metadata.get("published_at", "")}

Author:
{metadata.get("author", "")}

Summary:
{metadata.get("summary", "")}

Article Content:
{metadata.get("text", "")}
""".strip())
    return "\n\n--------------------\n\n".join(context_parts)

def format_history_for_prompt(history) -> str:
    if not history:
        return "(no previous conversation)"
    lines = []
    for message in history:
        role = "User" if message.get("role") == "user" else "DebtCircle AI"
        content = str(message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "(no previous conversation)"

def generate_answer(
    question: str,
    context: str,
    history: Optional[List[Dict]] = None,
    standalone_question: Optional[str] = None
) -> str:
    """Generate answer using RAG."""
    history = history or []
    history_text = format_history_for_prompt(history)
    requested_year = extract_requested_year(standalone_question or question)
    
    fy_instruction = ""
    if requested_year:
        previous = requested_year - 1
        short = str(requested_year)[-2:]
        fy_instruction = f"""
The user's query contains year {requested_year}.
For Indian financial-market reporting, FY{short} may refer to FY {previous}-{short}.
Use this mapping ONLY when the context supports it.
Clearly state the fiscal year in the answer.
"""
    
    system_prompt = f"""
You are DebtCircle AI, an expert assistant for debt markets, fixed income, bonds, NBFCs,
banking, credit markets, macroeconomics and financial news.

You must answer using ONLY the DebtCircle article context supplied to you.

{fy_instruction}

Important rules:
1. Do not invent information.
2. Do not use external knowledge when the supplied articles do not support the answer.
3. If articles don't contain enough information, say:
   "I could not find sufficient information in the DebtCircle articles to answer this question."
4. Combine information from multiple relevant articles when useful.
5. For latest/recent queries, prioritize the newest relevant article.
6. For queries with a year/FY, answer for that requested year/FY only when supported.
7. Mention important dates, companies, regulators, yields, rates and financial figures when available.
8. Do not mention internal systems (Pinecone, embeddings, vector search, etc.).
9. Do not say "according to the context".
10. Be concise but informative.
"""
    
    user_prompt = f"""
RECENT CONVERSATION:
{history_text}

STANDALONE RETRIEVAL QUESTION:
{standalone_question or question}

DEBTCIRCLE ARTICLE CONTEXT:
{context}

CURRENT USER QUESTION:
{question}

Answer the CURRENT USER QUESTION. Use conversation history only to resolve references.
Use the newly retrieved DebtCircle article context as the factual source.
"""
    
    try:
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Answer generation failed: {e}")
        raise HTTPException(status_code=503, detail="Answer generation unavailable")

# ============================================================
# SOURCE PREPARATION
# ============================================================

def prepare_sources(matches) -> List[Dict]:
    sources = []
    seen = set()
    
    for match in matches:
        metadata = match.metadata or {}
        post_key = metadata.get("post_id") or metadata.get("slug") or metadata.get("title") or match.id
        
        if post_key in seen:
            continue
        seen.add(post_key)
        
        sources.append({
            "title": metadata.get("title", ""),
            "url": metadata.get("url", ""),
            "date": metadata.get("date", ""),
            "slug": metadata.get("slug", ""),
            "score": round(float(match.score or 0), 4),
            "relevance_score": None  # Could add cross-encoder score
        })
    
    return sources

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {"status": "running", "service": "DebtCircle AI Chatbot", "version": "3.0.0"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "pinecone_index": PINECONE_INDEX,
        "namespace": PINECONE_NAMESPACE,
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL,
        "hybrid_search": USE_HYBRID_SEARCH,
        "cross_encoder": USE_CROSS_ENCODER,
        "current_financial_year_end": current_financial_year_end(),
        "memory_sessions": len(_conversation_memory)
    }

@app.get("/memory/{session_id}")
def memory_state(session_id: str):
    state = get_session_state(session_id)
    return {
        "session_id": session_id,
        "message_count": len(state.messages),
        "last_retrieval_question": state.last_retrieval_question,
        "messages": state.messages
    }

@app.delete("/memory/{session_id}")
def delete_memory(session_id: str):
    with _memory_lock:
        existed = session_id in _conversation_memory
        _conversation_memory.pop(session_id, None)
    return {"session_id": session_id, "cleared": existed}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start_time = time.time()
    
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    top_k = max(1, min(request.top_k or 8, 20))
    
    # Session management
    session_id = get_or_create_session(request.session_id)
    session_state = get_session_state(session_id)
    history = session_state.messages
    
    # Resolve follow-up
    retrieval_question = resolve_followup_query(
        question=question,
        history=history,
        last_retrieval_question=session_state.last_retrieval_question
    )
    
    # Expand query
    expanded_question = expand_query(retrieval_question)
    
    try:
        # Search
        results = search_pinecone(expanded_question, top_k=50)
        matches = results.matches
        
        if not matches:
            return ChatResponse(
                answer="I could not find sufficient information in the DebtCircle articles to answer this question.",
                sources=[],
                session_id=session_id,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Rerank
        relevant_matches = rank_matches(expanded_question, matches, top_k=25)
        
        # Cross-encoder reranking
        if USE_CROSS_ENCODER:
            relevant_matches = rerank_with_cross_encoder(expanded_question, relevant_matches, top_k)
        else:
            relevant_matches = relevant_matches[:top_k]
        
        if not relevant_matches:
            return ChatResponse(
                answer="I could not find sufficient information in the DebtCircle articles to answer this question.",
                sources=[],
                session_id=session_id,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Generate answer
        context = build_context(relevant_matches)
        answer = generate_answer(
            question=question,
            context=context,
            history=history,
            standalone_question=retrieval_question
        )
        
        # Save conversation
        session_state.add_turn(question, answer, retrieval_question)
        
        # Prepare sources
        sources = prepare_sources(relevant_matches)
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
    
    except Exception as e:
        print(f"CHAT ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# OPTIONAL: ASYNC STREAMING ENDPOINT
# ============================================================

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming version for better user experience."""
    from fastapi.responses import StreamingResponse
    import asyncio
    
    async def generate():
        # Process query and stream chunks
        # Implementation would use async processing
        yield "data: Starting...\n\n"
        # ... streaming implementation
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)