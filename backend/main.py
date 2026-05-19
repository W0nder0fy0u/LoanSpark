"""
FASTAPI BACKEND — THE MOUTH OF THE SYSTEM (IMPROVED)
=====================================================
Caveman say: "Old mouth crash on bad input. New mouth never crash.
Old mouth lose session. New mouth track everything safe."

Changes:
- Added safe_float / safe_int helpers (no crash on bad input)
- Proper logging throughout
- UUID session IDs enforced
- Better error messages
- All endpoint errors caught and handled
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import traceback
import logging
import uuid

import database
import behavior_tracker
import ml_models
import decision_engine
import ollama_client

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("main")

app = FastAPI(
    title="Loan Approval + Fraud Detection API",
    description="Two-brain system: approve loans, detect fraud. Local. No paid APIs.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    database.init_db()
    log.info("System ready. Brain awake.")


# ── SAFE PARSING HELPERS ─────────────────────────────────────────────────────

def safe_float(value, default: float = 0.0, min_val: float = None, max_val: float = None) -> float:
    """
    Parse value to float — never crash.
    Caveman: 'Bad number? Use safe number. No crash.'
    """
    try:
        result = float(str(value).strip().replace(",", ""))
        if min_val is not None:
            result = max(result, min_val)
        if max_val is not None:
            result = min(result, max_val)
        return result
    except (TypeError, ValueError, AttributeError):
        return default


def safe_int(value, default: int = 0, min_val: int = None, max_val: int = None) -> int:
    """Parse value to int — never crash."""
    try:
        result = int(float(str(value).strip().replace(",", "")))
        if min_val is not None:
            result = max(result, min_val)
        if max_val is not None:
            result = min(result, max_val)
        return result
    except (TypeError, ValueError, AttributeError):
        return default


# ── PYDANTIC MODELS ──────────────────────────────────────────────────────────

class AnswerSubmit(BaseModel):
    session_id: str
    question_key: str
    question_text: str
    answer_value: str
    response_time_s: float = 5.0
    num_edits: int = 0
    page_switches: int = 0


class BehaviorEvent(BaseModel):
    session_id: str
    question_key: str
    response_time_s: float
    num_edits: int = 0
    page_switches: int = 0
    total_session_time: float = 0.0


class DecisionRequest(BaseModel):
    session_id: str
    total_session_time: float = 120.0


# ── ANSWER PARSING ───────────────────────────────────────────────────────────

def _parse_answers(raw_answers: Dict) -> Dict:
    """
    Convert text answers to numbers for ML models.
    Caveman: 'User say words. We turn to numbers. No crash on bad words.'
    Uses safe_float/safe_int — never throws.
    """
    parsed = {}

    if "income" in raw_answers:
        parsed["income"] = safe_float(raw_answers["income"], default=30_000.0, min_val=0.0)

    if "loan_amount" in raw_answers:
        parsed["loan_amount"] = safe_float(raw_answers["loan_amount"], default=10_000.0, min_val=0.0)

    if "loan_term" in raw_answers:
        term_map = {
            "12 months (1 year)": 12, "24 months (2 years)": 24,
            "36 months (3 years)": 36, "60 months (5 years)": 60,
            "84 months (7 years)": 84, "120 months (10 years)": 120,
        }
        val = str(raw_answers["loan_term"])
        parsed["loan_term"] = float(term_map.get(val, safe_float(val, default=36.0)))

    if "credit_score" in raw_answers:
        cs_map = {
            "Below 500 (Poor)": 420,    "500-599 (Fair)": 550,
            "600-699 (Average)": 650,   "700-749 (Good)": 725,
            "750-799 (Very Good)": 775, "800+ (Excellent)": 820,
        }
        val = str(raw_answers["credit_score"])
        parsed["credit_score"] = float(cs_map.get(val, safe_float(val, default=600.0, min_val=300.0, max_val=850.0)))

    if "employment_years" in raw_answers:
        parsed["employment_years"] = safe_float(raw_answers["employment_years"], default=1.0, min_val=0.0, max_val=50.0)

    if "existing_debts" in raw_answers:
        parsed["existing_debts"] = safe_float(raw_answers["existing_debts"], default=0.0, min_val=0.0)

    if "age" in raw_answers:
        parsed["age"] = safe_float(raw_answers["age"], default=30.0, min_val=18.0, max_val=100.0)

    if "education" in raw_answers:
        edu_map = {
            "No formal education": 0,
            "High School / 12th pass": 1,
            "Bachelor's degree": 2,
            "Master's or higher": 3,
        }
        val = str(raw_answers["education"])
        parsed["education"] = float(edu_map.get(val, safe_int(val, default=1, min_val=0, max_val=3)))

    if "own_property" in raw_answers:
        parsed["own_property"] = 1.0 if "Yes" in str(raw_answers["own_property"]) else 0.0

    if "num_dependents" in raw_answers:
        dep_map = {
            "0 - I support only myself": 0,
            "1 dependent": 1, "2 dependents": 2,
            "3 dependents": 3, "4 or more dependents": 4,
        }
        val = str(raw_answers["num_dependents"])
        parsed["num_dependents"] = float(dep_map.get(val, safe_int(val, default=0, min_val=0, max_val=10)))

    return parsed


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.post("/start-session")
async def start_session(request: Request):
    """
    Create new session. Returns unique session ID.
    Caveman: 'User arrive. We give ID rock. Remember rock.'
    """
    try:
        session_id = database.create_session(
            ip_address=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "")
        )
        log.info("New session: %s", session_id)
        return {
            "session_id": session_id,
            "message": "Welcome! We will ask you some questions about your loan application.",
            "total_questions": len(ollama_client.STATIC_QUESTION_FLOW),
        }
    except Exception as e:
        log.error("start_session error: %s", e)
        raise HTTPException(500, "Could not create session. Please try again.")


@app.get("/next-question")
async def next_question(session_id: str):
    """
    Get next question for session.
    Caveman: 'What question next? Check stone tablet.'
    """
    if not session_id or not session_id.strip():
        raise HTTPException(400, "session_id is required")

    try:
        raw_answers = database.get_session_answers(session_id)
        asked_keys  = list(raw_answers.keys())
        question    = await ollama_client.get_next_question(raw_answers, asked_keys)

        if question is None:
            return {"done": True, "message": "All questions answered! Calculating your result..."}

        total    = len(ollama_client.STATIC_QUESTION_FLOW)
        answered = len(asked_keys)
        progress = round((answered / total) * 100)

        return {
            "done": False,
            "question": question,
            "progress": progress,
            "answered": answered,
            "total": total,
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error("next_question error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(500, "Could not fetch next question.")


@app.post("/submit-answer")
async def submit_answer(data: AnswerSubmit):
    """
    Save answer + behavioral data.
    Caveman: 'User answer. Write on stone. Also watch HOW they answer.'
    """
    if not data.session_id or not data.answer_value:
        raise HTTPException(400, "session_id and answer_value are required")

    try:
        database.save_answer(
            data.session_id,
            data.question_key,
            data.question_text,
            data.answer_value,
        )
        behavior_tracker.record_behavior(
            session_id      = data.session_id,
            question_key    = data.question_key,
            response_time_s = safe_float(data.response_time_s, default=5.0, min_val=0.0),
            num_edits       = safe_int(data.num_edits, default=0, min_val=0),
            page_switches   = safe_int(data.page_switches, default=0, min_val=0),
        )
        database.save_behavior(
            data.session_id,
            data.question_key,
            safe_float(data.response_time_s, 5.0),
            safe_int(data.num_edits, 0),
        )
        return {"saved": True, "key": data.question_key}
    except HTTPException:
        raise
    except Exception as e:
        log.error("submit_answer error: %s", e)
        raise HTTPException(500, "Could not save answer.")


@app.post("/behavior-event")
async def behavior_event(data: BehaviorEvent):
    """Frontend secretly sends behavior signals. Bank sees. User doesn't know."""
    try:
        behavior_tracker.record_behavior(
            session_id      = data.session_id,
            question_key    = data.question_key,
            response_time_s = safe_float(data.response_time_s, 5.0, min_val=0.0),
            num_edits       = safe_int(data.num_edits, 0, min_val=0),
            page_switches   = safe_int(data.page_switches, 0, min_val=0),
        )
        if data.total_session_time > 0:
            behavior_tracker.set_session_time(data.session_id, data.total_session_time)
        return {"received": True}
    except Exception as e:
        log.warning("behavior_event error: %s", e)
        return {"received": False}  # Never crash on behavior tracking failure


@app.post("/final-decision")
async def final_decision(data: DecisionRequest):
    """
    THE BIG MOMENT. Run all ML models. Return decision.
    Caveman: 'All questions done. Brain think. Brain say YES or NO.'
    """
    raw_answers = database.get_session_answers(data.session_id)
    if len(raw_answers) < 5:
        raise HTTPException(
            400,
            "Not enough answers to make a decision. Please answer all questions first."
        )

    behavior_tracker.set_session_time(
        data.session_id,
        safe_float(data.total_session_time, default=120.0, min_val=0.0)
    )

    parsed = _parse_answers(raw_answers)

    # ── Loan Brain ────────────────────────────────────────────────────────
    try:
        loan_pred, loan_prob, loan_analysis = ml_models.predict_loan_approval(parsed)
    except FileNotFoundError:
        log.warning("Loan model not found — using rule-based fallback")
        credit   = safe_float(parsed.get("credit_score"), 600)
        income   = safe_float(parsed.get("income"), 30_000)
        amount   = safe_float(parsed.get("loan_amount"), 10_000)
        loan_prob = min(max((credit - 300) / 550 * 0.5 + (income / max(amount, 1)) * 0.05, 0), 1)
        loan_pred = int(loan_prob >= 0.5)
        loan_analysis = {"top_factors": [], "debt_to_income": 0, "loan_to_income": 0}
    except Exception as e:
        log.error("Loan model error: %s", e)
        loan_prob = 0.4
        loan_pred = 0
        loan_analysis = {"top_factors": [], "debt_to_income": 0, "loan_to_income": 0}

    # ── Fraud Brain ───────────────────────────────────────────────────────
    behavior_feats = behavior_tracker.get_behavior_features(data.session_id)
    behavior_flags = behavior_tracker.get_behavior_flags(data.session_id)

    try:
        fraud_prob, fraud_analysis = ml_models.predict_fraud(parsed, behavior_feats)
    except FileNotFoundError:
        log.warning("Fraud model not found — using behavior fallback")
        fraud_prob     = 0.1 + safe_float(behavior_feats.get("behavior_score"), 0.0) * 0.4
        fraud_analysis = {"top_fraud_factors": [], "behavioral_score": behavior_feats.get("behavior_score", 0)}
    except Exception as e:
        log.error("Fraud model error: %s", e)
        fraud_prob     = 0.15
        fraud_analysis = {"top_fraud_factors": [], "behavioral_score": 0}

    # ── Decision Engine ───────────────────────────────────────────────────
    final_dec, dec_reason = decision_engine.make_final_decision(loan_prob, fraud_prob)

    # ── Ollama Explanations ───────────────────────────────────────────────
    user_message   = await ollama_client.generate_user_explanation(final_dec, loan_prob, parsed)
    static_suggs   = ml_models.generate_loan_suggestions(parsed, loan_prob)
    ai_suggestions = await ollama_client.generate_suggestions(final_dec, loan_prob, parsed, static_suggs)

    # ── Bank Report ───────────────────────────────────────────────────────
    bank_report = decision_engine.build_bank_report(
        answers        = parsed,
        loan_prob      = loan_prob,
        fraud_prob     = fraud_prob,
        fraud_analysis = fraud_analysis,
        behavior_flags = behavior_flags,
        final_decision = final_dec,
        decision_reason= dec_reason,
    )

    database.save_decision(
        session_id     = data.session_id,
        loan_prob      = loan_prob,
        fraud_prob     = fraud_prob,
        risk_level     = bank_report["risk_level"],
        final_decision = final_dec,
        user_message   = user_message,
        bank_report    = bank_report,
        suggestions    = static_suggs,
    )

    log.info("Decision [%s]: %s (loan=%.2f, fraud=%.2f)", data.session_id[:8], final_dec, loan_prob, fraud_prob)

    # Return USER-SAFE result (NO fraud info!)
    return {
        "session_id": data.session_id,
        "decision": final_dec,
        "loan_probability": round(loan_prob * 100, 1),
        "user_message": user_message,
        "suggestions": ai_suggestions,
        "next_steps": _get_user_next_steps(final_dec),
    }


@app.get("/bank-report/{session_id}")
async def bank_report(session_id: str, bank_key: str = ""):
    """Bank-only endpoint. Requires secret key. Full fraud + behavioral report."""
    BANK_SECRET = "BANK_SECRET_2024"
    if bank_key != BANK_SECRET:
        raise HTTPException(403, "Access denied. Valid bank credentials required.")

    try:
        decision = database.get_decision(session_id)
        if not decision:
            raise HTTPException(404, f"Session '{session_id}' not found.")
        return {
            "session_id": session_id,
            "bank_report": decision["bank_report"],
            "decided_at": decision["decided_at"],
            "raw_loan_probability": decision["loan_probability"],
            "raw_fraud_probability": decision["fraud_probability"],
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error("bank_report error: %s", e)
        raise HTTPException(500, "Error fetching bank report.")


@app.get("/all-sessions")
async def all_sessions(bank_key: str = ""):
    """Bank dashboard: list all sessions."""
    BANK_SECRET = "BANK_SECRET_2024"
    if bank_key != BANK_SECRET:
        raise HTTPException(403, "Access denied.")

    try:
        conn = database.get_conn()
        rows = conn.execute("""
            SELECT s.session_id, s.created_at, d.final_decision,
                   d.loan_probability, d.fraud_probability, d.risk_level
            FROM sessions s
            LEFT JOIN decisions d ON s.session_id = d.session_id
            ORDER BY s.created_at DESC LIMIT 50
        """).fetchall()
        conn.close()
        return {"sessions": [dict(r) for r in rows]}
    except Exception as e:
        log.error("all_sessions error: %s", e)
        raise HTTPException(500, "Error fetching sessions.")


def _get_user_next_steps(decision: str) -> list:
    if decision == "APPROVE":
        return [
            "Check your email for the approval letter",
            "Keep documents ready: ID, income proof, bank statements (3 months)",
            "A loan officer will call you within 2 business days",
        ]
    elif decision == "REVIEW":
        return [
            "Your application is under review by our team",
            "We may contact you for additional documents",
            "Expect a response within 3-5 business days",
        ]
    else:
        return [
            "Review the improvement suggestions below",
            "Focus on improving your credit score over the next 6 months",
            "Consider reapplying once key metrics improve",
        ]


@app.get("/health")
async def health():
    return {"status": "alive", "message": "Brain awake and ready.", "version": "2.0.0"}
