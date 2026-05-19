"""SQLite storage helpers for sessions, answers, behavior logs, and decisions."""

import sqlite3
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "loan_system.db"


def get_conn() -> sqlite3.Connection:
    """Get a fresh SQLite connection. Each call gets own connection (thread-safe)."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    # Enable WAL journaling
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables on first run."""
    try:
        conn = get_conn()
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                created_at   TEXT NOT NULL,
                status       TEXT DEFAULT 'active',
                ip_address   TEXT,
                user_agent   TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS answers (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT NOT NULL,
                question_key  TEXT NOT NULL,
                question_text TEXT,
                answer_value  TEXT,
                answered_at   TEXT,
                UNIQUE(session_id, question_key),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS behavior_logs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       TEXT NOT NULL,
                question_key     TEXT,
                response_time_s  REAL,
                num_edits        INTEGER DEFAULT 0,
                logged_at        TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id        TEXT NOT NULL UNIQUE,
                loan_probability  REAL,
                fraud_probability REAL,
                risk_level        TEXT,
                final_decision    TEXT,
                user_message      TEXT,
                bank_report       TEXT,
                suggestions       TEXT,
                decided_at        TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        conn.commit()
        conn.close()
        log.info("Database ready at %s", DB_PATH)
    except Exception as e:
        log.error("init_db error: %s", e)
        raise


def create_session(ip_address: str = None, user_agent: str = None) -> str:
    """Create new session with UUID. Returns session_id."""
    session_id = str(uuid.uuid4())
    try:
        conn = get_conn()
        conn.execute(
            "INSERT INTO sessions (session_id, created_at, ip_address, user_agent) VALUES (?,?,?,?)",
            (session_id, datetime.utcnow().isoformat(), ip_address, user_agent),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.error("create_session error: %s", e)
        raise
    return session_id


def save_answer(session_id: str, question_key: str, question_text: str, answer_value: str):
    """
    Save or update an answer. Uses upsert so re-submissions overwrite old value.
    """
    try:
        conn = get_conn()
        conn.execute(
            """INSERT INTO answers (session_id, question_key, question_text, answer_value, answered_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(session_id, question_key)
               DO UPDATE SET answer_value=excluded.answer_value, answered_at=excluded.answered_at""",
            (session_id, question_key, question_text, str(answer_value), datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.error("save_answer error: %s", e)
        raise


def save_behavior(session_id: str, question_key: str, response_time_s: float, num_edits: int):
    """Log behavioral data entry."""
    try:
        conn = get_conn()
        conn.execute(
            """INSERT INTO behavior_logs
               (session_id, question_key, response_time_s, num_edits, logged_at)
               VALUES (?,?,?,?,?)""",
            (session_id, question_key, float(response_time_s), int(num_edits), datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("save_behavior error (non-fatal): %s", e)


def get_session_answers(session_id: str) -> dict:
    """Return all answers for a session as {key: value} dict."""
    try:
        conn = get_conn()
        rows = conn.execute(
            "SELECT question_key, answer_value FROM answers WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
        conn.close()
        return {r["question_key"]: r["answer_value"] for r in rows}
    except Exception as e:
        log.error("get_session_answers error: %s", e)
        return {}


def get_session_behavior(session_id: str) -> list:
    """Return all behavior log entries for a session."""
    try:
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM behavior_logs WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        log.error("get_session_behavior error: %s", e)
        return []


def save_decision(
    session_id: str,
    loan_prob: float,
    fraud_prob: float,
    risk_level: str,
    final_decision: str,
    user_message: str,
    bank_report: dict,
    suggestions: list,
):
    """Save final ML decision. Upsert — safe to call multiple times."""
    try:
        conn = get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO decisions
               (session_id, loan_probability, fraud_probability, risk_level, final_decision,
                user_message, bank_report, suggestions, decided_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                session_id, loan_prob, fraud_prob, risk_level, final_decision,
                user_message, json.dumps(bank_report), json.dumps(suggestions),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.error("save_decision error: %s", e)
        raise


def get_decision(session_id: str) -> dict:
    """Retrieve stored decision for a session."""
    try:
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM decisions WHERE session_id=?", (session_id,)
        ).fetchone()
        conn.close()
        if row:
            d = dict(row)
            d["bank_report"] = json.loads(d["bank_report"])
            d["suggestions"] = json.loads(d["suggestions"])
            return d
        return None
    except Exception as e:
        log.error("get_decision error: %s", e)
        return None
