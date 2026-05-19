"""
FEATURED TRAINING DATA MODULE — NOW STRONGER!
=============================================
Old: 5000 rows. Weak. No scaler. Brain confused.
New: 12000 rows. Realistic. Scaler used. Brain happy.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

np.random.seed(42)
N = 12_000  # Larger dataset for more stable training

# Feature lists for model inputs
LOAN_FEATURES = [
    "income", "credit_score", "employment_years", "loan_amount",
    "loan_term", "age", "existing_debts", "num_dependents",
    "education", "own_property", "debt_to_income", "loan_to_income",
]

FRAUD_FEATURES = [
    "income", "credit_score", "employment_years", "loan_amount",
    "existing_debts", "debt_to_income", "loan_to_income",
    "avg_response_time", "num_edits", "inconsistency_score",
    "session_duration", "num_page_switches", "behavior_score",
]


def make_loan_data(n=N):
    """Generate realistic Indian fintech loan applicant data."""
    rng = np.random.default_rng(42)

    income = rng.lognormal(mean=10.8, sigma=0.55, size=n).clip(12_000, 500_000)
    credit_score = rng.normal(loc=640, scale=90, size=n).clip(300, 850).astype(int)
    employment_years = rng.exponential(scale=5.5, size=n).clip(0, 45)
    loan_amount = rng.lognormal(mean=10.4, sigma=0.65, size=n).clip(5_000, 800_000)
    loan_term = rng.choice([12, 24, 36, 48, 60, 84, 120], size=n)
    age = rng.normal(loc=37, scale=11, size=n).clip(21, 72).astype(int)
    existing_debts = (
        rng.lognormal(mean=9.0, sigma=0.85, size=n) *
        rng.choice([0, 1], p=[0.25, 0.75], size=n)
    ).clip(0, 300_000)
    num_dependents = rng.choice([0, 1, 2, 3, 4], p=[0.30, 0.28, 0.25, 0.12, 0.05], size=n)
    education = rng.choice([0, 1, 2, 3], p=[0.08, 0.32, 0.40, 0.20], size=n)
    own_property = rng.choice([0, 1], p=[0.42, 0.58], size=n)

    monthly_income = income / 12
    debt_to_income = (existing_debts + loan_amount * 0.02) / monthly_income.clip(1)
    loan_to_income = loan_amount / income.clip(1)

    score = (
        (credit_score - 300) / 550 * 35
        + np.log1p(income) / np.log1p(500_000) * 25
        - debt_to_income.clip(0, 4) * 14
        - loan_to_income.clip(0, 6) * 9
        + (employment_years / 45) * 10
        + own_property * 5
        + education * 2
        - num_dependents * 1.5
        + rng.normal(0, 4.5, n)
    )
    approved = (score > 38).astype(int)

    return pd.DataFrame({
        "income": income.round(2),
        "credit_score": credit_score,
        "employment_years": employment_years.round(2),
        "loan_amount": loan_amount.round(2),
        "loan_term": loan_term,
        "age": age,
        "existing_debts": existing_debts.round(2),
        "num_dependents": num_dependents,
        "education": education,
        "own_property": own_property,
        "debt_to_income": debt_to_income.round(4),
        "loan_to_income": loan_to_income.round(4),
        "approved": approved,
    })


def make_fraud_data(n=N):
    """Extend loan data with behavioral signals. ~22% fraud rate."""
    rng = np.random.default_rng(99)
    df = make_loan_data(n)

    avg_response_time = rng.lognormal(mean=2.4, sigma=0.85, size=n).clip(1, 300)
    num_edits = rng.poisson(lam=1.8, size=n).clip(0, 25)
    inconsistency_score = rng.beta(a=2, b=8, size=n)
    session_duration = rng.lognormal(mean=3.6, sigma=0.65, size=n).clip(10, 3600)
    num_page_switches = rng.poisson(lam=0.6, size=n).clip(0, 20)

    # Composite behavior score (0-1, higher = more suspicious)
    behavior_score = (
        (avg_response_time / 300).clip(0, 1) * 0.25
        + (num_edits / 25).clip(0, 1) * 0.35
        + inconsistency_score * 0.25
        + (num_page_switches / 20).clip(0, 1) * 0.15
    ).clip(0, 1)

    fraud_score = (
        (df["debt_to_income"] > 3).astype(float) * 22
        + (df["credit_score"] < 450).astype(float) * 18
        + (df["loan_amount"] / df["income"].clip(1) > 10).astype(float) * 18
        + (avg_response_time > 30).astype(float) * 9
        + (num_edits > 5).astype(float) * 14
        + inconsistency_score * 20
        + (num_page_switches > 3).astype(float) * 9
        + (df["employment_years"] < 0.1).astype(float) * 8
        + behavior_score * 15
        + rng.normal(0, 7, n)
    ).clip(0, 100)

    is_fraud = (fraud_score > 44).astype(int)

    df["avg_response_time"] = avg_response_time.round(3)
    df["num_edits"] = num_edits
    df["inconsistency_score"] = inconsistency_score.round(4)
    df["session_duration"] = session_duration.round(2)
    df["num_page_switches"] = num_page_switches
    df["behavior_score"] = behavior_score.round(4)
    df["fraud_score"] = fraud_score.round(2)
    df["is_fraud"] = is_fraud
    return df


def train_loan_model(df):
    """Train loan approval Pipeline (scaler + model in one object)."""
    X, y = df[LOAN_FEATURES], df["approved"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.85, random_state=42, min_samples_leaf=20,
        )),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    log.info("\n[LOAN MODEL] ROC-AUC: %.4f", roc_auc_score(y_test, y_prob))
    log.info("\n" + classification_report(y_test, y_pred))
    return pipeline, LOAN_FEATURES


def train_fraud_model(df):
    """Train fraud detection Pipeline (scaler + model in one object)."""
    X, y = df[FRAUD_FEATURES], df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=250, max_depth=9, random_state=42,
            class_weight="balanced", min_samples_leaf=15, n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    log.info("\n[FRAUD MODEL] ROC-AUC: %.4f", roc_auc_score(y_test, y_prob))
    log.info("\n" + classification_report(y_test, y_pred))
    return pipeline, FRAUD_FEATURES


if __name__ == "__main__":
    log.info("=== TRAINING RITUAL ===")
    loan_df = make_loan_data()
    fraud_df = make_fraud_data()
    log.info("Loan data:  %d rows | %.1f%% approved", len(loan_df), loan_df["approved"].mean()*100)
    log.info("Fraud data: %d rows | %.1f%% fraud",   len(fraud_df), fraud_df["is_fraud"].mean()*100)

    # Save generated dataset
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    fraud_df.to_csv(os.path.join(data_dir, "loan_dataset.csv"), index=False)
    log.info("Dataset saved → data/loan_dataset.csv")

    loan_pipeline, loan_features   = train_loan_model(loan_df)
    fraud_pipeline, fraud_features = train_fraud_model(fraud_df)

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump({"pipeline": loan_pipeline,  "features": loan_features},  os.path.join(model_dir, "loan_model.pkl"))
    joblib.dump({"pipeline": fraud_pipeline, "features": fraud_features}, os.path.join(model_dir, "fraud_model.pkl"))
    log.info("Brains saved to models/ directory. Done.")
