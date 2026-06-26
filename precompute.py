"""
Redrob Hackathon — PRECOMPUTE STAGE
Outputs: precomputed.json — all per-candidate signals needed for ranking

This stage is allowed to be slow (loads 100K candidates, runs hard filters,
builds candidate texts, computes bi-encoder embeddings for all survivors,
and computes all structured/behavioral/alias/consistency scores). Per the
hackathon spec, pre-computation may exceed the 5-minute ranking budget as
long as it is documented and run separately from the ranking step.

Usage:
    python precompute.py

Requirements:
    pip install sentence-transformers numpy
"""

import json
import re
import math
import time
import os
# os.environ["HF_HUB_OFFLINE"] = "1"

from datetime import datetime, date, timedelta
from collections import defaultdict

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
CANDIDATES_FILE = "candidates.jsonl"
PRECOMPUTED_FILE = "precomputed.json"

YOE_LOW = 4
YOE_HIGH = 9
EXEMPT_ACTIVE_DAYS = 30
EXEMPT_SEMANTIC_PCT = 0.15

# ── CLASSIFICATION SETS ────────────────────────────────────────────────────────

PRODUCT_INDUSTRIES = {
    "software", "fintech", "food delivery", "e-commerce", "saas",
    "ai/ml", "adtech", "insurance tech", "healthtech", "gaming",
    "healthtech ai", "conversational ai", "ai services", "voice ai",
    "internet", "media", "consumer electronics", "edtech",
}

VECTOR_DB_SKILLS = {
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "chroma", "pgvector", "vespa", "redis",
}

EMBEDDING_SKILLS = {
    "sentence transformers", "sentence-transformers", "openai embeddings",
    "bge", "e5", "embedding", "embeddings", "vector search",
    "semantic search", "dense retrieval", "bi-encoder", "cross-encoder",
    "neural search", "text embeddings", "word2vec",
}

RETRIEVAL_SYSTEMS_SKILLS = {
    "search infrastructure", "information retrieval systems",
    "bm25", "indexing algorithms", "search backend",
    "vector representations", "ranking systems", "search & discovery",
    "content matching", "text encoders", "haystack",
    "search and discovery", "learning to rank", "reranking systems",
    "hybrid search", "sparse retrieval", "search & ranking",
    "search engine", "document retrieval", "passage retrieval",
    "approximate nearest neighbor", "ann", "inverted index",
    "tf-idf", "bm25f", "colbert", "splade", "dpr",
}

ML_AI_SKILLS = {
    "nlp", "machine learning", "deep learning", "pytorch", "tensorflow",
    "transformers", "bert", "gpt", "llm", "fine-tuning llms", "lora",
    "qlora", "peft", "rag", "retrieval", "ranking", "recommendation",
    "xgboost", "neural network", "mlops", "feature engineering",
    "a/b testing", "ndcg", "mrr", "information retrieval", "reranking",
    "hugging face", "scikit-learn",
}

CV_SPEECH_SKILLS = {
    "computer vision", "image classification", "object detection",
    "image segmentation", "ocr", "speech recognition", "tts",
    "text-to-speech", "speech synthesis", "asr", "robotics", "ros",
    "gans", "image generation", "stable diffusion", "video understanding",
    "action recognition", "yolo",
}

FRAMEWORK_SIGNALS = {
    "langchain", "llamaindex", "llama index", "autogpt",
    "flowise", "langflow", "dspy", "crewai",
}

SYSTEMS_DEPTH_SKILLS = VECTOR_DB_SKILLS | EMBEDDING_SKILLS | {
    "kafka", "kubernetes", "docker", "ray", "triton",
    "grpc", "fastapi", "distributed systems", "serving",
}

BONUS_SKILLS = {
    "lora": 0.040, "qlora": 0.040, "peft": 0.040,
    "fine-tuning llms": 0.035, "xgboost": 0.030, "lightgbm": 0.025,
    "learning to rank": 0.030, "ndcg": 0.035, "mrr": 0.030,
    "kafka": 0.015, "kubernetes": 0.015, "ray": 0.025,
    "triton": 0.020, "mlops": 0.020, "nlp": 0.020,
}

RETRIEVAL_SIGNAL_WORDS = {
    "recommendation", "ranking", "retrieval", "search",
    "information retrieval", "relevance", "embedding", "recommender",
    "similarity", "nearest neighbour", "nearest neighbor",
}

PRODUCTION_ML_SIGNAL_WORDS = {
    "drift", "monitoring", "production", "evaluation", "a/b test",
    "experiment", "metrics", "model serving", "deployment", "regression",
    "feedback loop", "offline", "online",
}

ALL_SIGNAL_WORDS = RETRIEVAL_SIGNAL_WORDS | PRODUCTION_ML_SIGNAL_WORDS

PROF_WEIGHTS = {
    "expert": 1.0, "advanced": 0.85,
    "intermediate": 0.60, "beginner": 0.35,
}

ALIAS_DICT = {
    "ndcg": (0.035, ["normalised discounted cumulative gain",
                     "normalized discounted cumulative gain",
                     "graded relevance", "ranking metric"]),
    "mrr": (0.035, ["mean reciprocal rank", "reciprocal rank",
                    "first relevant result"]),
    "map": (0.035, ["mean average precision", "average precision"]),
    "dcg": (0.035, ["discounted cumulative gain", "cumulative gain"]),
    "bge": (0.025, ["baai general embedding", "general text embedding"]),
    "e5": (0.025, ["multilingual e5", "text embedding e5", "e5 model"]),
    "colbert": (0.025, ["late interaction", "maxsim", "token-level retrieval"]),
    "dpr": (0.025, ["dense passage retrieval", "bi-encoder retrieval"]),
    "splade": (0.025, ["sparse learned embedding", "sparse retrieval"]),
}

PUNE_NOIDA_CITIES = {"pune", "noida"}
WELCOME_CITIES = {"hyderabad", "pune", "mumbai", "delhi", "ncr", "noida", "gurgaon", "gurugram", "new delhi"}

with open("jd_distilled.txt", encoding="utf-8") as f:
    DISTILLED_JD = f.read().strip()

with open("ideal_candidate.txt", encoding="utf-8") as f:
    IDEAL_JD = f.read().strip()

with open("negative_jd.txt", encoding="utf-8") as f:
    NEGATIVE_JD = f.read().strip()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def nsk(name):
    return name.lower().strip()


def get_skill_names(skills):
    return {nsk(s["name"]) for s in skills}


def get_product_months(career_history):
    return sum(
        j.get("duration_months", 0)
        for j in career_history
        if (j.get("industry") or "").lower().strip() in PRODUCT_INDUSTRIES
    )


def first_n_sentences(text, n):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(parts[:n])


def days_since(date_str):
    try:
        return (date.today() - datetime.strptime(date_str, "%Y-%m-%d").date()).days
    except Exception:
        return 9999


# ── STAGE 1: HARD FILTERS ─────────────────────────────────────────────────────

def is_hard_kill(r):
    p = r["profile"]
    s = r["redrob_signals"]
    ch = r["career_history"]
    sk = r["skills"]

    if not s["open_to_work_flag"]:
        return True, "not open to work"
    if p["years_of_experience"] < 3:
        return True, "YOE < 3"
    if get_product_months(ch) < 24:
        return True, "< 24 months at product companies"

    names = get_skill_names(sk)
    has_ai_tags = bool(names & (ML_AI_SKILLS | VECTOR_DB_SKILLS | EMBEDDING_SKILLS | RETRIEVAL_SYSTEMS_SKILLS))

    # NEW: Prose fallback for Empty Skills edge case in Hard Filters
    if not has_ai_tags:
        all_prose = " ".join((j.get("description") or "") + " " + (j.get("title") or "") for j in ch).lower()
        has_ai_prose = any(w in all_prose for w in {
            "machine learning", "nlp", "recommendation", "retrieval",
            "ranking", "pinecone", "vector", "llm", "embedding"
        })
        if not has_ai_prose:
            return True, "no ML/AI skills (in tags or prose)"

    cv_count = len(names & CV_SPEECH_SKILLS)
    ai_total = len(
        names & (ML_AI_SKILLS | CV_SPEECH_SKILLS | VECTOR_DB_SKILLS | EMBEDDING_SKILLS | RETRIEVAL_SYSTEMS_SKILLS))
    if ai_total > 3 and cv_count / ai_total > 0.60:
        return True, "primarily CV/speech/robotics"

    return False, None


# ── STAGE 2: CONSISTENCY SCORE ────────────────────────────────────────────────

def honeypot_date_logic_score(r):
    sk = r["skills"]
    ch = r["career_history"]
    mult = 1.0
    issues = []

    total_career_months = sum(j.get("duration_months", 0) for j in ch)
    for s in sk:
        if s.get("duration_months", 0) > total_career_months + 6:
            issues.append(f"skill '{s['name']}' duration ({s.get('duration_months')}mo) "
                          f"exceeds total career history ({total_career_months}mo)")

    current_jobs = [j for j in ch if j.get("is_current") and j.get("end_date") is None]
    if len(current_jobs) > 1:
        issues.append(f"{len(current_jobs)} simultaneous 'current' jobs claimed")

    def parse_date(s_):
        if not s_: return None
        try:
            return datetime.strptime(s_, "%Y-%m-%d").date()
        except Exception:
            return None

    dated_jobs = []
    for j in ch:
        start = parse_date(j.get("start_date"))
        end = parse_date(j.get("end_date")) if not j.get("is_current") else date.today()
        if start and end:
            dated_jobs.append((start, end, j.get("company", "")))

    dated_jobs.sort(key=lambda x: x[0])
    overlap_count = 0
    for i in range(1, len(dated_jobs)):
        prev_start, prev_end, prev_company = dated_jobs[i - 1]
        cur_start, cur_end, cur_company = dated_jobs[i]
        if cur_start < prev_end - timedelta(days=31):
            overlap_count += 1

    if overlap_count > 0:
        issues.append(f"{overlap_count} overlapping job date range(s)")

    n_issues = len(issues)
    if n_issues >= 2:
        mult = 0.35
    elif n_issues == 1:
        mult = 0.70

    return mult, issues


def yoe_career_mismatch_check(r):
    ch = r["career_history"]
    p = r["profile"]
    career_months = sum(j.get("duration_months", 0) for j in ch)
    stated_years = p.get("years_of_experience", 0)
    stated_months = stated_years * 12

    if stated_months <= 0: return 1.0, []

    ratio = career_months / stated_months
    issues = []
    mult = 1.0

    if ratio < 0.35:
        mult = 0.40
        issues.append(f"stated {stated_years:.1f}y experience but career_history only covers {career_months}mo")
    elif ratio < 0.55:
        mult = 0.65
        issues.append(f"stated {stated_years:.1f}y experience but career_history only covers {career_months}mo")
    elif ratio < 0.65:
        mult = 0.82
        issues.append(f"stated {stated_years:.1f}y experience vs {career_months / 12:.1f}y in career_history")
    elif ratio > 1.20:
        mult = 0.85
        issues.append(f"career_history ({career_months / 12:.1f}y) exceeds stated {stated_years:.1f}y experience")

    return mult, issues


def profile_consistency_score(r):
    sk = r["skills"]
    ch = r["career_history"]
    score = 1.0

    PROF_MIN = {"expert": 12, "advanced": 6, "intermediate": 3}
    violations = sum(
        1 for s in sk
        if s.get("duration_months", 0) < PROF_MIN.get(
            s.get("proficiency", "beginner").lower(), 0)
    )
    if violations >= 5:
        score *= 0.40
    elif violations >= 3:
        score *= 0.65
    elif violations >= 1:
        score *= 0.88

    endorsements = [s.get("endorsements", 0) for s in sk]
    if len(endorsements) >= 5 and max(endorsements) > 20:
        mean_e = sum(endorsements) / len(endorsements)
        if mean_e > 0:
            std_e = math.sqrt(sum((e - mean_e) ** 2 for e in endorsements) / len(endorsements))
            cv = std_e / mean_e
            if cv < 0.10 and mean_e > 30:
                score *= 0.45
            elif cv < 0.20 and mean_e > 25:
                score *= 0.70

    descs = [j.get("description", "").strip() for j in ch if j.get("description", "").strip()]
    if len(descs) >= 2:
        unique_ratio = len(set(descs)) / len(descs)
        if unique_ratio < 0.5:
            score *= 0.85
        elif unique_ratio < 0.75:
            score *= 0.92

    return max(0.05, min(1.0, score))


# ── STAGE 3: CANDIDATE TEXT (for bi-encoder) ──────────────────────────────────

def build_candidate_text(r):
    p = r["profile"]
    sk = r["skills"]
    ch = r["career_history"]
    parts = []

    if p.get("headline"): parts.append(p["headline"])
    if p.get("summary"):  parts.append(p["summary"])

    sorted_ch = sorted(ch, key=lambda j: (not j.get("is_current", False), 0))
    cumulative = 0
    for job in sorted_ch:
        dur = job.get("duration_months", 0)
        title = job.get("title", "")
        company = job.get("company", "")
        desc = job.get("description", "")
        industry = job.get("industry", "")
        age = cumulative

        if age < 36:
            parts.append(f"{title} at {company} ({industry}): {desc}")
        elif age < 72:
            desc_lower = desc.lower()
            if any(w in desc_lower for w in ALL_SIGNAL_WORDS):
                compressed = first_n_sentences(desc, 4) if desc else ""
            else:
                compressed = first_n_sentences(desc, 2) if desc else ""
            parts.append(f"{title} at {company}: {compressed}")
        else:
            desc_lower = desc.lower()
            if any(w in desc_lower for w in ALL_SIGNAL_WORDS):
                relevant = [
                    s for s in re.split(r'(?<=[.!?])\s+', desc)
                    if any(w in s.lower() for w in ALL_SIGNAL_WORDS)
                ]
                if relevant:
                    parts.append(f"{title} at {company}: {' '.join(relevant[:2])}")
                else:
                    parts.append(f"{title} at {company}")
            else:
                parts.append(f"{title} at {company}")
        cumulative += dur

    parts.append(" ".join(s["name"] for s in sk))
    return " ".join(filter(None, parts))


def build_ce_excerpt(r):
    """
    Excerpt fed to the cross-encoder at rank time. Uses a HYBRID SORT
    to prioritize current jobs AND highly relevant past jobs, preventing
    older ML roles from being silently truncated out of the top 3.
    """
    p = r["profile"]
    ch = r["career_history"]
    parts = []

    if p.get("headline"):
        parts.append(p["headline"])
    summary = p.get("summary", "")
    if summary:
        parts.append(" ".join(summary.split()[:90]))

    def sort_key(j):
        desc_lower = (j.get("description") or "").lower()
        title_lower = (j.get("title") or "").lower()
        # High signal = heavily rewarded in the sort
        signal_count = sum(1 for w in ALL_SIGNAL_WORDS if w in desc_lower or w in title_lower)

        return (
            not j.get("is_current", False),  # 0 if True (floats to top)
            -signal_count,  # More signal words = floats higher
            -j.get("duration_months", 0)  # Longer tenure tie breaker
        )

    sorted_ch = sorted(ch, key=sort_key)

    for job in sorted_ch[:3]:
        desc = job.get("description", "")
        title = job.get("title", "")
        company = job.get("company", "")
        trimmed = " ".join(desc.split()[:90])
        parts.append(f"{title} at {company}: {trimmed}")

    return " ".join(filter(None, parts))


# ── ALIAS SCORE ───────────────────────────────────────────────────────────────

def compute_alias_score(candidate_text_lower):
    total = 0.0
    for term, (weight, aliases) in ALIAS_DICT.items():
        if term in candidate_text_lower or any(a in candidate_text_lower for a in aliases):
            total += weight
    return min(1.0, total)


# ── BEHAVIORAL SCORE ──────────────────────────────────────────────────────────

def compute_availability_mult(r):
    s = r["redrob_signals"]
    mult = 1.0
    flags = []

    days = days_since(s.get("last_active_date", ""))
    if days > 180:
        mult *= 0.35;
        flags.append(f"inactive {days}d (>6mo) — likely unreachable")
    elif days > 90:
        mult *= 0.75;
        flags.append(f"inactive {days}d — reduced availability confidence")

    rrr = s.get("recruiter_response_rate", None)
    if rrr is not None:
        if rrr <= 0.05:
            mult *= 0.40;
            flags.append(f"recruiter response rate {rrr:.0%} — essentially unreachable")
        elif rrr <= 0.20:
            mult *= 0.75;
            flags.append(f"recruiter response rate {rrr:.0%} — low responsiveness")
        elif rrr <= 0.40:
            mult *= 0.90;
            flags.append(f"recruiter response rate {rrr:.0%} — below average responsiveness")

    art = s.get("avg_response_time_hours", None)
    if art is not None and art > 0:
        if art > 240:
            mult *= 0.85;
            flags.append(f"avg response time {art:.0f}h — very slow to reply")
        elif art > 120:
            mult *= 0.93;
            flags.append(f"avg response time {art:.0f}h — slow to reply")

    return max(0.10, min(1.0, mult)), flags


def credibility_score(signals):
    score = 0.0
    if signals.get("verified_email"):     score += 0.15
    if signals.get("verified_phone"):     score += 0.10
    if signals.get("linkedin_connected"): score += 0.10

    gh = signals.get("github_activity_score", -1)
    if gh >= 0:
        score += 0.20 * min(gh / 80, 1.0)
        if gh > 50:
            score += 0.02

    ic = signals.get("interview_completion_rate", -1)
    if ic >= 0: score += 0.15 * ic

    oa = signals.get("offer_acceptance_rate", -1)
    if oa >= 0: score += 0.08 * oa

    score += min(0.08, signals.get("saved_by_recruiters_30d", 0) * 0.02)
    completeness = signals.get("profile_completeness_score", 0)
    score += 0.08 * (completeness / 100)
    apps = signals.get("applications_submitted_30d", 0)
    score += min(0.05, apps * 0.005)

    return min(1.0, score)


# ── STRUCTURED SCORE ──────────────────────────────────────────────────────────

def yoe_score(yoe, is_exempt):
    if is_exempt:  return 0.80
    if yoe < 3:    return 0.00
    if yoe < 5:    return 0.45
    if yoe < 6:    return 0.70
    if yoe < 7:    return 0.85
    if yoe < 9:    return 1.00
    if yoe < 11:   return 0.82
    return 0.65


def product_ratio_score(ch):
    total = sum(j.get("duration_months", 0) for j in ch) or 1
    product = get_product_months(ch)
    ratio = product / total
    if product >= 48: ratio = min(1.0, ratio * 1.2)
    return min(1.0, ratio)


def python_skill_score(skills, career_history):
    # Base check via tags
    for s in skills:
        if nsk(s["name"]) == "python":
            prof = PROF_WEIGHTS.get(s.get("proficiency", "beginner"), 0.35)
            end = min(s.get("endorsements", 0), 60) / 60
            dur = min(s.get("duration_months", 0), 72) / 72
            return 0.50 * prof + 0.25 * end + 0.25 * dur

    # NEW: Empty Skills Prose Fallback
    all_prose = " ".join((j.get("description") or "") for j in career_history).lower()
    if "python" in all_prose:
        return 0.40
    return 0.0


def vector_db_skill_score(skills, career_history):
    best_vdb = best_emb = best_sys = 0.0
    for s in skills:
        name = nsk(s["name"])
        prof = PROF_WEIGHTS.get(s.get("proficiency", "beginner"), 0.35)
        dur = min(s.get("duration_months", 0), 48) / 48
        end = min(s.get("endorsements", 0), 50) / 50
        sc = 0.45 * prof + 0.35 * dur + 0.20 * end
        if name in VECTOR_DB_SKILLS:         best_vdb = max(best_vdb, sc)
        if name in EMBEDDING_SKILLS:         best_emb = max(best_emb, sc)
        if name in RETRIEVAL_SYSTEMS_SKILLS: best_sys = max(best_sys, sc)

    present = [v for v in (best_vdb, best_emb, best_sys) if v > 0]
    if len(present) >= 2:
        avg_of_present = sum(present) / len(present)
        return min(1.0, avg_of_present + 0.10)
    if len(present) == 1:
        return present[0]

    # NEW: Empty Skills Prose Fallback
    all_prose = " ".join((j.get("description") or "") for j in career_history).lower()
    has_vdb = any(w in all_prose for w in
                  {"pinecone", "qdrant", "weaviate", "milvus", "opensearch", "pgvector", "faiss", "elasticsearch"})
    has_emb = any(w in all_prose for w in {"embedding", "sentence-transformer", "dense retrieval"})
    has_sys = any(w in all_prose for w in {"retrieval", "ranking", "bm25", "hybrid search"})

    prose_present = sum([has_vdb, has_emb, has_sys])
    if prose_present >= 2:
        return 0.55
    elif prose_present == 1:
        return 0.35

    return 0.0


def notice_score(days):
    if days <= 15:  return 1.00
    if days <= 30:  return 0.90
    if days <= 45:  return 0.75
    if days <= 60:  return 0.55
    if days <= 90:  return 0.38
    return 0.25


def location_score(profile, signals):
    loc = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()

    if any(c in loc for c in PUNE_NOIDA_CITIES):
        return 1.00
    if any(c in loc for c in WELCOME_CITIES):
        return 0.95
    if signals.get("willing_to_relocate") and country == "india":
        return 1.00
    if country != "india":
        return 0.25
    return 0.20


def assessment_score(signals):
    scores = list(signals.get("skill_assessment_scores", {}).values())
    if not scores: return 0.35
    return (sum(scores) / len(scores)) / 100


def compute_bonus(skills):
    names = get_skill_names(skills)
    return min(0.10, sum(v for k, v in BONUS_SKILLS.items() if k in names))


def eval_score(skills, career_history):
    skill_names = get_skill_names(skills)
    has_eval_skill = any(x in skill_names for x in {
        "ndcg", "mrr", "map", "learning to rank", "a/b testing",
        "offline evaluation", "precision", "recall",
    })

    all_prose = " ".join((j.get("description") or "") for j in career_history).lower()
    has_eval_prose = any(w in all_prose for w in {
        "ndcg", "mrr", "a/b test", "offline eval", "ranking metric",
        "relevance judgment", "evaluation pipeline", "precision at",
        "recall at", "mean average", "mean reciprocal",
    })

    if has_eval_skill and has_eval_prose: return 1.0
    if has_eval_skill or has_eval_prose:  return 0.65
    return 0.20


def compute_structured_score(r, is_exempt):
    p = r["profile"]
    s = r["redrob_signals"]
    ch = r["career_history"]
    sk = r["skills"]
    components = {
        "yoe": (yoe_score(p["years_of_experience"], is_exempt), 0.14),
        "product": (product_ratio_score(ch), 0.22),
        "vector_db": (vector_db_skill_score(sk, ch), 0.25),
        "python": (python_skill_score(sk, ch), 0.08),
        "eval": (eval_score(sk, ch), 0.11),
        "notice": (notice_score(s["notice_period_days"]), 0.05),
        "location": (location_score(p, s), 0.00),
        "assessment": (assessment_score(s), 0.05),
        "credibility": (credibility_score(s), 0.10),
    }
    base = sum(v * w for v, w in components.values())
    bonus = compute_bonus(sk)
    return min(1.0, base + bonus), components


# ── DISQUALIFIER MULTIPLIERS ──────────────────────────────────────────────────

def compute_disqualifier_mult(r):
    s = r["redrob_signals"]
    ch = r["career_history"]
    sk = r["skills"]
    mult = 1.0
    flags = []

    tenures = [j["duration_months"] for j in ch if j.get("duration_months", 0) > 0]
    if len(tenures) >= 3:
        avg = sum(tenures) / len(tenures)
        if avg < 12:
            mult *= 0.50;
            flags.append("severe job hopping")
        elif avg < 18:
            mult *= 0.78;
            flags.append("job hopping pattern")

    names = get_skill_names(sk)
    fw_count = len(names & FRAMEWORK_SIGNALS)
    sys_depth = len(names & SYSTEMS_DEPTH_SKILLS)
    if fw_count > 0 and sys_depth < 2:
        mult *= 0.72;
        flags.append("framework enthusiast, limited systems depth")

    gh = s.get("github_activity_score", -1)
    has_certs = len(r.get("certifications", [])) > 0
    has_assess = len(s.get("skill_assessment_scores", {})) > 0
    if gh <= 5 and not has_certs and not has_assess:
        mult *= 0.85;
        flags.append("no external validation")

    ai_skills = [s_ for s_ in sk if nsk(s_["name"]) in ML_AI_SKILLS]
    ai_months = sum(s_.get("duration_months", 0) for s_ in ai_skills)
    pre_llm = any(
        any(w in (j.get("description", "")).lower()
            for w in ["recommendation", "ranking", "retrieval", "search", "relevance"])
        and j.get("duration_months", 0) > 12
        for j in ch
    )
    if ai_months < 18 and not pre_llm:
        mult *= 0.70;
        flags.append("limited AI depth or recency")

    full_history_text = " ".join(
        (j.get("description") or "") + " " + (j.get("title") or "") for j in ch
    ).lower()

    def _word_boundary_present(word, text):
        pattern = r"\b" + re.escape(word) + r"\b"
        return re.search(pattern, text) is not None

    SKILL_CATEGORIES = {
        "retrieval_vector": {
            "skills": VECTOR_DB_SKILLS | EMBEDDING_SKILLS | RETRIEVAL_SYSTEMS_SKILLS | {
                "retrieval", "ranking", "recommendation", "information retrieval", "reranking", "rag",
            },
            "concept_words": RETRIEVAL_SIGNAL_WORDS | PRODUCTION_ML_SIGNAL_WORDS | {
                "vector", "index", "bm25", "hybrid", "pinecone", "qdrant", "weaviate",
                "milvus", "elasticsearch", "opensearch", "faiss", "pgvector", "chroma",
                "vespa", "bge", "e5", "sentence-transformer", "sentence transformer",
            },
        },
        "ml_general": {
            "skills": {
                "nlp", "machine learning", "deep learning", "pytorch", "tensorflow",
                "transformers", "bert", "gpt", "llm", "fine-tuning llms", "lora",
                "qlora", "peft", "xgboost", "neural network", "mlops",
                "feature engineering", "scikit-learn", "hugging face",
            },
            "concept_words": {
                "model", "models", "ml", "machine learning", "pipeline", "predict",
                "training", "trained", "neural", "deep learning", "nlp", "language model",
                "fine-tuning", "finetuning", "feature", "monitoring", "deployment",
                "production", "experiment",
            },
        },
        "cv_speech": {
            "skills": CV_SPEECH_SKILLS,
            "concept_words": {
                "vision", "image", "video", "speech", "audio", "voice", "ocr",
                "detection", "segmentation", "classification model", "generative",
                "diffusion", "generative adversarial", "robotics",
            },
        },
    }

    uncorroborated = 0
    checked = 0
    for s_ in sk:
        skill_key = nsk(s_["name"])
        for cat in SKILL_CATEGORIES.values():
            if skill_key in cat["skills"]:
                checked += 1
                if not any(_word_boundary_present(w, full_history_text) for w in cat["concept_words"]):
                    uncorroborated += 1
                break

    if checked > 0:
        uncorroborated_ratio = uncorroborated / checked
        if uncorroborated_ratio > 0.75:
            mult *= 0.55;
            flags.append("most AI/ML skills unsupported by career history")
        elif uncorroborated_ratio > 0.5:
            mult *= 0.80;
            flags.append("some AI/ML skills unsupported by career history")

    notice = s.get("notice_period_days", 0)
    if notice > 90:
        mult *= 0.75
        flags.append(f"notice {notice}d — long wait for Series A hiring")

    work_mode = (s.get("preferred_work_mode") or "").lower()
    willing_relocate = s.get("willing_to_relocate", False)
    if work_mode == "remote" and not willing_relocate:
        mult *= 0.88
        flags.append("prefers remote — JD requires hybrid/onsite")

    return mult, flags


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    t_overall_start = time.time()
    stage_times = {}

    def mark(stage_name, t_stage_start):
        elapsed = time.time() - t_stage_start
        stage_times[stage_name] = elapsed
        print(f"      [{stage_name} took {elapsed:.1f}s]")
        return time.time()

    print("=" * 62)
    print("  REDROB PRECOMPUTE STAGE")
    print("=" * 62)

    print("\n[1/6] Loading candidates...")
    t = time.time()
    records = []
    with open(CANDIDATES_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"      {len(records):,} records loaded")
    t = mark("1_load_candidates", t)

    print("\n[2/6] Hard filters...")
    t = time.time()
    survivors = []
    kill_counts = defaultdict(int)

    # ── SANDBOX OVERRIDE LOGIC ──────────────────────────────────────────────
    IS_SANDBOX = len(records) <= 100
    if IS_SANDBOX:
        print("      [SANDBOX MODE DETECTED: <= 100 candidates. Bypassing hard filters.]")
        survivors = records
    else:
        for r in records:
            killed, reason = is_hard_kill(r)
            if killed:
                kill_counts[reason] += 1
            else:
                survivors.append(r)
    # ─────────────────────────────────────────────────────────────────────────

    print(f"      Survivors: {len(survivors):,} / {len(records):,}")
    for reason, cnt in sorted(kill_counts.items(), key=lambda x: -x[1]):
        print(f"        x {reason}: {cnt:,}")
    t = mark("2_hard_filters", t)

    print("\n[3/6] Profile consistency scores...")
    t = time.time()
    consistency = {}
    honeypot_date_issues = {}
    yoe_mismatch_issues = {}
    for r in survivors:
        cid = r["candidate_id"]
        base_consistency = profile_consistency_score(r)
        date_mult, issues = honeypot_date_logic_score(r)
        yoe_mult, yoe_issues = yoe_career_mismatch_check(r)
        consistency[cid] = max(0.05, min(1.0, base_consistency * date_mult * yoe_mult))
        if issues: honeypot_date_issues[cid] = issues
        if yoe_issues: yoe_mismatch_issues[cid] = yoe_issues
    suspects = sum(1 for v in consistency.values() if v < 0.45)
    print(f"      Honeypot suspects (< 0.45): {suspects}")
    print(f"      Candidates with date-logic issues: {len(honeypot_date_issues)}")
    print(f"      Candidates with YOE/career-history mismatch: {len(yoe_mismatch_issues)}")
    t = mark("3_consistency_scores", t)

    print("\n[4/6] Building candidate texts, alias scores, structured/behavioral scores...")
    t = time.time()
    cand_texts = {}
    alias_scores = {}
    behavioral = {}
    structured = {}
    disqualifier = {}

    for r in survivors:
        cid = r["candidate_id"]
        text = build_candidate_text(r)
        cand_texts[cid] = text
        alias_scores[cid] = compute_alias_score(text.lower())
        behavioral[cid] = compute_availability_mult(r)
        dis_m, flags = compute_disqualifier_mult(r)
        disqualifier[cid] = {"mult": dis_m, "flags": flags}
    t = mark("4_text_and_quick_scores", t)

    print("\n[5/6] Semantic scoring (bi-encoder)...")
    t = time.time()
    t_model = time.time()
    from sentence_transformers import SentenceTransformer
    bi_encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    bi_encoder.max_seq_length = 512
    print(f"      [bi-encoder model load took {time.time() - t_model:.1f}s]")
    ids = [r["candidate_id"] for r in survivors]
    docs = [cand_texts[cid] for cid in ids]
    jd_emb = bi_encoder.encode(
        ["Represent this sentence for searching relevant passages: " + DISTILLED_JD],
        normalize_embeddings=True, show_progress_bar=False
    )
    ideal_emb = bi_encoder.encode(
        ["Represent this sentence for searching relevant passages: " + IDEAL_JD],
        normalize_embeddings=True, show_progress_bar=False
    )
    neg_emb = bi_encoder.encode(
        ["Represent this sentence for searching relevant passages: " + NEGATIVE_JD],
        normalize_embeddings=True, show_progress_bar=False
    )
    sem_scores = {}
    for i in range(0, len(docs), 512):
        batch = docs[i:i + 512]
        batch_ids = ids[i:i + 512]
        embs = bi_encoder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        pos_sims = (embs @ jd_emb.T).flatten()
        ideal_sims = (embs @ ideal_emb.T).flatten()
        neg_sims = (embs @ neg_emb.T).flatten()
        for cid, pos, ideal, neg in zip(batch_ids, pos_sims, ideal_sims, neg_sims):
            neg_penalty = min(0.08, 0.15 * max(0.0, float(neg) - 0.35))
            sem_scores[cid] = (0.70 * float(pos) + 0.30 * float(ideal) - neg_penalty)
        print(f"      {min(i + 512, len(docs)):,}/{len(docs):,}", end="\r")
    print(f"      Done                                    ")
    t = mark("5_semantic_scoring_total", t)

    sem_thresh = sorted(sem_scores.values(), reverse=True)[
        int(len(sem_scores) * EXEMPT_SEMANTIC_PCT)
    ]

    print("\n[6/6] Structured scores (with YOE exemption logic) + CE excerpts...")
    t = time.time()
    for r in survivors:
        cid = r["candidate_id"]
        s = r["redrob_signals"]
        yoe = r["profile"]["years_of_experience"]

        needs_exempt = yoe <= YOE_LOW or yoe >= YOE_HIGH
        recently_active = days_since(s.get("last_active_date", "")) <= EXEMPT_ACTIVE_DAYS
        is_exempt = needs_exempt and recently_active and sem_scores[cid] >= sem_thresh

        str_s, comps = compute_structured_score(r, is_exempt)
        structured[cid] = {"score": str_s, "components": comps}

    ce_excerpts = {r["candidate_id"]: build_ce_excerpt(r) for r in survivors}
    t = mark("6_structured_and_ce_excerpts", t)

    # ── Assemble precomputed payload ──────────────────────────────────────────
    payload = {
        "records": {r["candidate_id"]: r for r in survivors},
        "sem_scores": sem_scores,
        "beh_scores": behavioral,
        "structured": structured,
        "alias_scores": alias_scores,
        "consistency": consistency,
        "disqualifier": disqualifier,
        "ce_excerpts": ce_excerpts,
        "honeypot_date_issues": honeypot_date_issues,
        "yoe_mismatch_issues": yoe_mismatch_issues,
        "meta": {
            "n_total": len(records),
            "n_survivors": len(survivors),
            "kill_counts": dict(kill_counts),
            "honeypot_suspects": suspects,
            "date_logic_issue_count": len(honeypot_date_issues),
            "yoe_mismatch_issue_count": len(yoe_mismatch_issues),
        }
    }

    print(f"\nWriting {PRECOMPUTED_FILE} ...")
    t = time.time()
    with open(PRECOMPUTED_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    t = mark("7_write_json", t)

    total_elapsed = time.time() - t_overall_start
    print(f"\n{'=' * 62}")
    print(f"  Precompute complete.")
    print(f"  Survivors cached: {len(survivors):,}")
    print(f"  Output: {PRECOMPUTED_FILE}")
    print(f"\n  Stage timing breakdown:")
    for stage, secs in stage_times.items():
        pct = (secs / total_elapsed) * 100 if total_elapsed > 0 else 0
        print(f"    {stage:<32} {secs:>7.1f}s  ({pct:4.1f}%)")
    print(f"  {'TOTAL':<32} {total_elapsed:>7.1f}s")
    print(f"  Next step: run rank.py")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    run()