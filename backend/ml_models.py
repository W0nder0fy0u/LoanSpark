"""
ML MODELS (THE TWO BRAINS) — NOW WITH PROPER SCALING!
=======================================================
Old: raw features fed to model. No scaler. Unstable.
New: Pipeline object has scaler built in. Same transform train AND inference.

Caveman say: "Old brain eat raw food. Get sick. New brain cook food first. Stay healthy."
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import logging

log = logging.getLogger(__name__)

# Import feature lists from training module (single source of truth)
from train_data import LOAN_FEATURES, FRAUD_FEATURES

MODEL_DIR = Path(__file__).parent / "models"

# Lazy-loaded pipeline bundles
_loan_bundle  = None
_fraud_bundle = None


def _load(name: str):
    """Load pipeline bundle from disk. Raise clear error if missing."""
    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"Run: python train_data.py  (from project directory)"
        )
    return joblib.load(path)


def _get_loan():
    global _loan_bundle
    if _loan_bundle is None:
        _loan_bundle = _load("loan_model.pkl")
    return _loan_bundle


def _get_fraud():
    global _fraud_bundle
    if _fraud_bundle is None:
        _fraud_bundle = _load("fraud_model.pkl")
    return _fraud_bundle


def _safe_float(val, default=0.0):
    """Safe float cast — never crash."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def predict_loan_approval(answers: Dict) -> Tuple[int, float, Dict]:
    """
    INPUT:  parsed answers dict
    OUTPUT: (approved 0/1, probability 0-1, analysis dict)

    Caveman: "We ask brain: give loan? Brain say YES or NO. With percent."
    Pipeline includes scaler — no mismatch possible.
    """
    bundle   = _get_loan()
    pipeline = bundle["pipeline"]  # Pipeline: scaler + model

    income           = _safe_float(answers.get("income"), 30_000)
    credit_score     = _safe_float(answers.get("credit_score"), 600)
    employment_years = _safe_float(answers.get("employment_years"), 2)
    loan_amount      = _safe_float(answers.get("loan_amount"), 10_000)
    loan_term        = _safe_float(answers.get("loan_term"), 36)
    age              = _safe_float(answers.get("age"), 30)
    existing_debts   = _safe_float(answers.get("existing_debts"), 0)
    num_dependents   = _safe_float(answers.get("num_dependents"), 0)
    education        = _safe_float(answers.get("education"), 1)
    own_property     = _safe_float(answers.get("own_property"), 0)

    # Derived features — must match training
    debt_to_income = (existing_debts + loan_amount * 0.02) / max(income / 12, 1)
    loan_to_income = loan_amount / max(income, 1)

    row = pd.DataFrame([{
        "income": income, "credit_score": credit_score,
        "employment_years": employment_years, "loan_amount": loan_amount,
        "loan_term": loan_term, "age": age, "existing_debts": existing_debts,
        "num_dependents": num_dependents, "education": education,
        "own_property": own_property, "debt_to_income": debt_to_income,
        "loan_to_income": loan_to_income,
    }])[LOAN_FEATURES]

    prob = float(pipeline.predict_proba(row)[0][1])
    pred = int(prob >= 0.50)

    # Feature importances from the underlying model
    raw_model   = pipeline.named_steps["model"]
    importance  = dict(zip(LOAN_FEATURES, raw_model.feature_importances_))
    top_factors = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

    return pred, round(prob, 4), {
        "top_factors": top_factors,
        "debt_to_income": round(debt_to_income, 3),
        "loan_to_income": round(loan_to_income, 3),
    }


def predict_fraud(answers: Dict, behavior: Dict) -> Tuple[float, Dict]:
    """
    INPUT:  financial answers + behavioral features
    OUTPUT: (fraud_probability 0-1, risk analysis)

    Caveman: "Spy brain check person. Is person liar? Brain say number."
    BANK ONLY — user never sees this.
    """
    bundle   = _get_fraud()
    pipeline = bundle["pipeline"]  # Pipeline: scaler + model

    income           = _safe_float(answers.get("income"), 30_000)
    credit_score     = _safe_float(answers.get("credit_score"), 600)
    employment_years = _safe_float(answers.get("employment_years"), 2)
    loan_amount      = _safe_float(answers.get("loan_amount"), 10_000)
    existing_debts   = _safe_float(answers.get("existing_debts"), 0)

    debt_to_income = (existing_debts + loan_amount * 0.02) / max(income / 12, 1)
    loan_to_income = loan_amount / max(income, 1)

    avg_response_time   = _safe_float(behavior.get("avg_response_time"), 5.0)
    num_edits           = _safe_float(behavior.get("num_edits"), 0)
    inconsistency_score = _safe_float(behavior.get("inconsistency_score"), 0.0)
    session_duration    = _safe_float(behavior.get("session_duration"), 60.0)
    num_page_switches   = _safe_float(behavior.get("num_page_switches"), 0)
    behavior_score      = _safe_float(behavior.get("behavior_score"), 0.0)

    row = pd.DataFrame([{
        "income": income, "credit_score": credit_score,
        "employment_years": employment_years, "loan_amount": loan_amount,
        "existing_debts": existing_debts, "debt_to_income": debt_to_income,
        "loan_to_income": loan_to_income, "avg_response_time": avg_response_time,
        "num_edits": num_edits, "inconsistency_score": inconsistency_score,
        "session_duration": session_duration, "num_page_switches": num_page_switches,
        "behavior_score": behavior_score,
    }])[FRAUD_FEATURES]

    fraud_prob = float(pipeline.predict_proba(row)[0][1])

    raw_model  = pipeline.named_steps["model"]
    importance = dict(zip(FRAUD_FEATURES, raw_model.feature_importances_))

    return round(fraud_prob, 4), {
        "top_fraud_factors": sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5],
        "behavioral_score": round(behavior_score, 3),
    }


def generate_loan_suggestions(answers: Dict, loan_prob: float) -> list:
    """
    Caveman say: 'Tell user how to get better. Be honest but kind.'
    """
    suggestions  = []
    income       = _safe_float(answers.get("income"), 30_000)
    credit_score = _safe_float(answers.get("credit_score"), 600)
    loan_amount  = _safe_float(answers.get("loan_amount"), 10_000)
    emp_years    = _safe_float(answers.get("employment_years"), 2)
    existing     = _safe_float(answers.get("existing_debts"), 0)

    if credit_score < 600:
        suggestions.append(
            f"Improve your credit score (currently {int(credit_score)}). "
            "Pay bills on time for 6 months to gain +50-100 points."
        )
    if loan_amount > income * 3:
        reduced = int(income * 2.5)
        suggestions.append(
            f"Consider reducing loan amount to ₹{reduced:,} (currently ₹{int(loan_amount):,}). "
            "Keeps your debt-to-income ratio healthy."
        )
    if existing > income * 0.4:
        suggestions.append(
            f"Pay down existing debts (₹{int(existing):,}). "
            "High existing debt significantly hurts your profile."
        )
    if emp_years < 1:
        suggestions.append(
            "Aim for at least 1 year of employment stability before reapplying. "
            "Employment history is a major approval factor."
        )
    if not answers.get("own_property") or answers.get("own_property") == "0":
        suggestions.append(
            "Having property as collateral significantly increases approval chances."
        )
    if not suggestions:
        suggestions.append(
            "Your profile looks strong! You may qualify for a slightly higher loan amount."
        )

    return suggestions
