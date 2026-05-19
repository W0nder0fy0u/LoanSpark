"""Ollama client for question flow, explanations, and safe fallback handling."""

import httpx
import json
import asyncio
import re
from typing import Dict, List, Optional
import logging

log = logging.getLogger(__name__)

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"


async def _call_ollama(prompt: str, max_tokens: int = 500) -> str:
    """
    Send prompt to Mistral. Return text response.
    Returns empty string if Ollama unavailable — never crashes.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": max_tokens,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
    except httpx.ConnectError:
        log.info("Ollama offline — using static fallback")
        return ""
    except httpx.TimeoutException:
        log.warning("Ollama timed out")
        return ""
    except Exception as e:
        log.warning("Ollama error: %s", e)
        return ""


def _safe_parse_json(raw: str) -> Optional[Dict]:
    """
    Robustly extract JSON from LLM response.
    Handles: wrapped in markdown, extra text before/after, single quotes.
    """
    if not raw:
        return None

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Try to find JSON object with regex
    matches = re.findall(r'\{[^{}]*\}', cleaned, re.DOTALL)
    for candidate in matches:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try fixing single quotes
            try:
                fixed = candidate.replace("'", '"')
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue

    # Last resort: try the full cleaned string
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ── PROMPT TEMPLATES ─────────────────────────────────────────────────────────

QUESTION_PROMPT = """You are a friendly loan officer chatbot for an Indian bank.
Based on the answers collected so far, generate the SINGLE BEST next question to ask.

Answers so far:
{answers_json}

Questions already asked: {asked_keys}

Rules:
- Ask ONE question only
- Be conversational and warm
- Ask in plain English
- Focus on what's most important still missing
- If you have enough data, respond with exactly: DONE

Respond with JSON only (no extra text, no markdown):
{{"key": "field_name", "question": "Your question here?", "type": "number|text|choice", "choices": ["opt1","opt2"] or null}}
"""

EXPLAIN_APPROVAL_PROMPT = """You are a helpful loan advisor at an Indian bank.
The applicant has been given this result: {decision} with {probability}% probability.

Their key details:
- Income: ₹{income}/year
- Loan requested: ₹{loan_amount}
- Credit score: {credit_score}
- Employment: {employment_years} years

Write a SHORT, warm, encouraging message (3-4 sentences) to the applicant explaining their result.
Be honest but kind. Do NOT mention fraud detection or risk scoring.
If approved: celebrate and mention next steps.
If reviewing: explain what happens next.
If rejected: be empathetic and give hope.
"""

SUGGESTIONS_PROMPT = """You are a helpful loan advisor.
The applicant was {decision} with {probability}% probability.

Based on their profile:
{profile_summary}

Give 2-3 specific, actionable tips to improve their loan application chances.
Be brief, specific, and encouraging. Number them 1, 2, 3.
Do NOT mention fraud detection or behavior tracking.
"""


# ── QUESTION FLOW ─────────────────────────────────────────────────────────────

STATIC_QUESTION_FLOW = [
    {
        "key": "income",
        "question": "What is your annual income? (in ₹)",
        "type": "number",
        "choices": None,
        "placeholder": "e.g. 600000"
    },
    {
        "key": "loan_amount",
        "question": "How much loan are you applying for? (in ₹)",
        "type": "number",
        "choices": None,
        "placeholder": "e.g. 500000"
    },
    {
        "key": "loan_term",
        "question": "Over how many months would you like to repay?",
        "type": "choice",
        "choices": [
            "12 months (1 year)", "24 months (2 years)", "36 months (3 years)",
            "60 months (5 years)", "84 months (7 years)", "120 months (10 years)"
        ],
        "placeholder": None
    },
    {
        "key": "credit_score",
        "question": "What is your approximate credit score (CIBIL)?",
        "type": "choice",
        "choices": [
            "Below 500 (Poor)", "500-599 (Fair)", "600-699 (Average)",
            "700-749 (Good)", "750-799 (Very Good)", "800+ (Excellent)"
        ],
        "placeholder": None
    },
    {
        "key": "employment_years",
        "question": "How many years have you been working at your current job?",
        "type": "number",
        "choices": None,
        "placeholder": "e.g. 3.5"
    },
    {
        "key": "existing_debts",
        "question": "What is your total existing debt or loans? (enter 0 if none, in ₹)",
        "type": "number",
        "choices": None,
        "placeholder": "e.g. 200000"
    },
    {
        "key": "age",
        "question": "What is your age?",
        "type": "number",
        "choices": None,
        "placeholder": "e.g. 32"
    },
    {
        "key": "education",
        "question": "What is your highest level of education?",
        "type": "choice",
        "choices": [
            "No formal education", "High School / 12th pass",
            "Bachelor's degree", "Master's or higher"
        ],
        "placeholder": None
    },
    {
        "key": "own_property",
        "question": "Do you own any property (house, land, etc.)?",
        "type": "choice",
        "choices": ["Yes, I own property", "No, I don't own property"],
        "placeholder": None
    },
    {
        "key": "num_dependents",
        "question": "How many dependents do you financially support? (spouse, children, parents)",
        "type": "choice",
        "choices": [
            "0 - I support only myself", "1 dependent", "2 dependents",
            "3 dependents", "4 or more dependents"
        ],
        "placeholder": None
    },
]


async def get_next_question(answers: Dict, asked_keys: List[str]) -> Optional[Dict]:
    """
    Get next question. Try Ollama first; fall back to static list.
    """
    collected = set(asked_keys)
    missing   = [q for q in STATIC_QUESTION_FLOW if q["key"] not in collected]

    if not missing:
        return None  # All done

    # Try Ollama after we have enough context
    if len(asked_keys) >= 3 and answers:
        try:
            prompt = QUESTION_PROMPT.format(
                answers_json=json.dumps(answers, indent=2),
                asked_keys=", ".join(asked_keys),
            )
            raw = await asyncio.wait_for(_call_ollama(prompt, max_tokens=200), timeout=8.0)

            if raw and raw.strip().upper() == "DONE":
                return None

            if raw:
                data = _safe_parse_json(raw)
                if data and data.get("key") not in collected and data.get("question"):
                    # Ensure required fields are present
                    data.setdefault("type", "text")
                    data.setdefault("choices", None)
                    data.setdefault("placeholder", None)
                    return data

        except asyncio.TimeoutError:
            log.info("Ollama question timeout — using static fallback")
        except Exception as e:
            log.warning("Ollama question error: %s", e)

    # Static fallback
    return missing[0]


async def generate_user_explanation(decision: str, loan_prob: float, answers: Dict) -> str:
    """Generate friendly explanation for user. Falls back gracefully."""
    try:
        prompt = EXPLAIN_APPROVAL_PROMPT.format(
            decision=decision,
            probability=round(loan_prob * 100, 1),
            income=int(float(answers.get("income", 0))),
            loan_amount=int(float(answers.get("loan_amount", 0))),
            credit_score=answers.get("credit_score", "N/A"),
            employment_years=answers.get("employment_years", "N/A"),
        )
        response = await asyncio.wait_for(_call_ollama(prompt, max_tokens=300), timeout=15.0)
        if response:
            return response
    except Exception as e:
        log.warning("generate_user_explanation error: %s", e)

    # Static fallback
    pct = round(loan_prob * 100, 1)
    if decision == "APPROVE":
        return (
            f"Great news! Your loan application looks strong with a {pct}% confidence score. "
            "Your income and credit history are in good shape. "
            "A loan officer will contact you within 2 business days to complete verification."
        )
    elif decision == "REVIEW":
        return (
            f"Your application (score: {pct}%) is under review by our team. "
            "We may need a few additional documents to complete the assessment. "
            "Expect to hear from us within 3-5 business days."
        )
    else:
        return (
            f"We're sorry — based on your current profile ({pct}% score), "
            "we're unable to approve this loan application at this time. "
            "Please review the improvement suggestions below and consider reapplying in 6 months."
        )


async def generate_suggestions(
    decision: str, loan_prob: float, answers: Dict, static_suggestions: list
) -> str:
    """Generate improvement suggestions. Falls back to ML-generated suggestions."""
    try:
        profile = "\n".join([
            f"- Income: ₹{float(answers.get('income', 0)):,.0f}/year",
            f"- Loan amount: ₹{float(answers.get('loan_amount', 0)):,.0f}",
            f"- Credit score range: {answers.get('credit_score', 'N/A')}",
            f"- Employment: {answers.get('employment_years', 0)} years",
            f"- Existing debts: ₹{float(answers.get('existing_debts', 0)):,.0f}",
        ])
        prompt = SUGGESTIONS_PROMPT.format(
            decision=decision,
            probability=round(loan_prob * 100, 1),
            profile_summary=profile,
        )
        response = await asyncio.wait_for(_call_ollama(prompt, max_tokens=300), timeout=15.0)
        if response:
            return response
    except Exception as e:
        log.warning("generate_suggestions error: %s", e)

    # Fallback: use ML-generated suggestions
    return "\n".join([f"{i+1}. {s}" for i, s in enumerate(static_suggestions)])
