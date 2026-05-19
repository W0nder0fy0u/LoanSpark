"""
DECISION ENGINE
===============
Caveman say: "We take loan number. We take fraud number.
We put in magic box. Box say: APPROVE or REJECT or REVIEW."

Simple logic but powerful.
"""

from typing import Dict, Tuple


def compute_risk_level(fraud_prob: float) -> str:
    """
    Fraud probability → risk label.
    """
    if fraud_prob >= 0.70:
        return "CRITICAL"
    elif fraud_prob >= 0.50:
        return "HIGH"
    elif fraud_prob >= 0.30:
        return "MEDIUM"
    elif fraud_prob >= 0.15:
        return "LOW"
    else:
        return "MINIMAL"


def make_final_decision(loan_prob: float, fraud_prob: float) -> Tuple[str, str]:
    """
    Combine loan approval + fraud risk → final verdict.

    Returns: (decision, reason)

    Logic Table:
    ┌──────────────┬──────────────┬──────────────┐
    │  Loan Prob   │  Fraud Prob  │  Decision    │
    ├──────────────┼──────────────┼──────────────┤
    │  High (≥65%) │  Low (<30%)  │  APPROVE     │
    │  High (≥65%) │  Med (30-50%)│  REVIEW      │
    │  Any         │  High (≥50%) │  REJECT      │
    │  Low (<50%)  │  Any         │  REJECT      │
    │  Med (50-65%)│  Low (<30%)  │  REVIEW      │
    └──────────────┴──────────────┴──────────────┘
    """
    risk = compute_risk_level(fraud_prob)

    # Fraud override (bank safety first)
    if fraud_prob >= 0.70:
        return "REJECT", "Fraud risk is critically high — automatic rejection."
    if fraud_prob >= 0.50:
        return "REJECT", "High fraud probability detected — requires investigation before proceeding."

    # Now check loan merit
    if loan_prob >= 0.65:
        if fraud_prob < 0.30:
            return "APPROVE", "Strong financial profile with minimal fraud risk."
        else:
            return "REVIEW", "Good financial profile but moderate fraud indicators require manual review."
    elif loan_prob >= 0.50:
        if fraud_prob < 0.20:
            return "REVIEW", "Borderline financial profile — manual review recommended."
        else:
            return "REJECT", "Borderline finances combined with fraud concerns."
    else:
        return "REJECT", "Financial profile does not meet minimum approval criteria."


def build_bank_report(
    answers: Dict,
    loan_prob: float,
    fraud_prob: float,
    fraud_analysis: Dict,
    behavior_flags: list,
    final_decision: str,
    decision_reason: str
) -> Dict:
    """
    Full bank report. Secret from user.
    Caveman: 'This only for bank eyes.'
    """
    risk_level = compute_risk_level(fraud_prob)

    # Financial red flags
    financial_flags = []
    income      = float(answers.get("income", 30000))
    loan_amount = float(answers.get("loan_amount", 10000))
    credit      = float(answers.get("credit_score", 600))
    debts       = float(answers.get("existing_debts", 0))

    dti = (debts + loan_amount * 0.02) / max(income / 12, 1)
    lti = loan_amount / max(income, 1)

    if credit < 500:
        financial_flags.append(f"Very low credit score: {int(credit)}")
    elif credit < 600:
        financial_flags.append(f"Below-average credit score: {int(credit)}")

    if dti > 3:
        financial_flags.append(f"Dangerous debt-to-income ratio: {dti:.1f}x")
    elif dti > 2:
        financial_flags.append(f"High debt-to-income ratio: {dti:.1f}x")

    if lti > 8:
        financial_flags.append(f"Loan amount extremely high vs income: {lti:.1f}x annual income")
    elif lti > 5:
        financial_flags.append(f"Loan-to-income ratio is concerning: {lti:.1f}x")

    emp = float(answers.get("employment_years", 2))
    if emp < 0.5:
        financial_flags.append("Very recent employment (< 6 months)")

    return {
        "final_decision": final_decision,
        "decision_reason": decision_reason,
        "risk_level": risk_level,
        "loan_approval_probability": f"{loan_prob * 100:.1f}%",
        "fraud_probability": f"{fraud_prob * 100:.1f}%",
        "financial_flags": financial_flags if financial_flags else ["No major financial red flags"],
        "behavioral_analysis": behavior_flags,
        "top_fraud_factors": [
            f"{k.replace('_', ' ').title()}: {v:.3f}"
            for k, v in fraud_analysis.get("top_fraud_factors", [])[:3]
        ],
        "behavioral_score": fraud_analysis.get("behavioral_score", 0),
        "recommendation": _get_recommendation(final_decision, risk_level, loan_prob, fraud_prob),
        "action_items": _get_action_items(final_decision, risk_level),
        "applicant_summary": {
            "income": f"₹{float(answers.get('income', 0)):,.0f}",
            "loan_requested": f"₹{float(answers.get('loan_amount', 0)):,.0f}",
            "credit_score": answers.get("credit_score", "N/A"),
            "employment": f"{answers.get('employment_years', 0)} years",
        }
    }


def _get_recommendation(decision: str, risk: str, loan_prob: float, fraud_prob: float) -> str:
    if decision == "APPROVE":
        return "Proceed with standard loan processing. Verify documents and disburse."
    elif decision == "REVIEW":
        if risk in ("MEDIUM", "HIGH"):
            return "Escalate to fraud team for behavioral verification before approval."
        return "Manual underwriter review needed. Request additional income documentation."
    else:  # REJECT
        if fraud_prob >= 0.5:
            return "Flag account in fraud database. Consider reporting to credit bureau."
        return "Send standard rejection notice. Offer guidance to reapply in 6 months."


def _get_action_items(decision: str, risk: str) -> list:
    if decision == "APPROVE":
        return [
            "Verify identity documents (Aadhaar/PAN)",
            "Confirm bank statements for last 3 months",
            "Process disbursement upon verification"
        ]
    elif decision == "REVIEW":
        return [
            "Request income verification documents",
            "Conduct phone verification call",
            "Review fraud team behavioral report",
            "Manager approval required before proceeding"
        ]
    else:
        actions = ["Issue rejection letter with reason codes"]
        if risk in ("HIGH", "CRITICAL"):
            actions.append("File Suspicious Activity Report (SAR)")
            actions.append("Block IP and flag session for investigation")
        actions.append("Update credit bureau if applicable")
        return actions
