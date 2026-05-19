"""Behavior tracker for session behavior and fraud risk."""

from dataclasses import dataclass, field
from typing import Dict, List
import statistics
import logging

log = logging.getLogger(__name__)


@dataclass
class BehaviorProfile:
    session_id: str
    response_times: List[float] = field(default_factory=list)
    edit_counts: List[int]      = field(default_factory=list)
    question_keys: List[str]    = field(default_factory=list)
    total_session_time: float   = 0.0
    page_switches: int          = 0

    def add_event(self, question_key: str, response_time_s: float, num_edits: int):
        """Record one question's behavioral data."""
        # Clamp to sane ranges — no crashes from wild values
        safe_time  = max(0.5, min(float(response_time_s), 600.0))
        safe_edits = max(0, min(int(num_edits), 50))
        self.response_times.append(safe_time)
        self.edit_counts.append(safe_edits)
        self.question_keys.append(question_key)

    def compute_features(self) -> Dict:
        """
        Compute behavioral features that match fraud model training features.
        """
        if not self.response_times:
            return {
                "avg_response_time": 5.0,
                "num_edits": 0,
                "inconsistency_score": 0.0,
                "session_duration": 0.0,
                "num_page_switches": 0,
                "behavior_score": 0.0,
            }

        avg_time   = statistics.mean(self.response_times)
        total_edits = sum(self.edit_counts)

        # Inconsistency: coefficient of variation in response times
        if len(self.response_times) > 1:
            std_time      = statistics.stdev(self.response_times)
            inconsistency = min(std_time / max(avg_time, 1.0), 1.0)
        else:
            inconsistency = 0.0

        # Composite behavior score — MUST match formula in make_fraud_data()
        behavior_score = (
            min(avg_time / 300.0, 1.0)       * 0.25   # slow = suspicious
            + min(total_edits / 25.0, 1.0)   * 0.35   # many edits = nervous
            + inconsistency                   * 0.25   # variance = inconsistent
            + min(self.page_switches / 20.0, 1.0) * 0.15  # tab switching
        )
        behavior_score = round(min(max(behavior_score, 0.0), 1.0), 4)

        return {
            "avg_response_time": round(avg_time, 3),
            "num_edits": total_edits,
            "inconsistency_score": round(inconsistency, 4),
            "session_duration": round(self.total_session_time, 2),
            "num_page_switches": self.page_switches,
            "behavior_score": behavior_score,
        }

    def get_risk_flags(self) -> List[str]:
        """Human-readable risk reasons for bank dashboard."""
        flags = []
        feats = self.compute_features()

        if feats["avg_response_time"] > 45:
            flags.append("Very slow responses — may be looking up information externally")
        elif feats["avg_response_time"] > 25:
            flags.append("Slow response time across multiple questions")

        if feats["num_edits"] > 6:
            flags.append("Excessive answer editing — possible data fabrication")
        elif feats["num_edits"] > 3:
            flags.append("Multiple answer changes detected")

        if feats["inconsistency_score"] > 0.5:
            flags.append("Highly inconsistent response timing — anomalous pattern")
        elif feats["inconsistency_score"] > 0.3:
            flags.append("Moderately inconsistent response timing")

        if feats["num_page_switches"] > 3:
            flags.append("Frequent tab/window switching during session")

        if feats["behavior_score"] > 0.6:
            flags.append(f"High composite behavior risk score: {feats['behavior_score']:.2f}/1.00")
        elif feats["behavior_score"] > 0.35:
            flags.append(f"Moderate behavior risk score: {feats['behavior_score']:.2f}/1.00")

        if not flags:
            flags.append("No significant behavioral anomalies detected")

        return flags


# In-memory store: session_id → BehaviorProfile
_profiles: Dict[str, BehaviorProfile] = {}


def get_or_create_profile(session_id: str) -> BehaviorProfile:
    if session_id not in _profiles:
        _profiles[session_id] = BehaviorProfile(session_id=session_id)
    return _profiles[session_id]


def record_behavior(
    session_id: str,
    question_key: str,
    response_time_s: float,
    num_edits: int,
    page_switches: int = 0,
):
    """Record behavioral event for a question answer."""
    try:
        profile = get_or_create_profile(session_id)
        profile.add_event(question_key, response_time_s, num_edits)
        profile.page_switches += max(0, int(page_switches))
    except Exception as e:
        log.warning("behavior record failed: %s", e)


def get_behavior_features(session_id: str) -> Dict:
    """Return computed behavioral features dict for fraud model input."""
    try:
        return get_or_create_profile(session_id).compute_features()
    except Exception as e:
        log.warning("get_behavior_features failed: %s", e)
        return {
            "avg_response_time": 5.0,
            "num_edits": 0,
            "inconsistency_score": 0.0,
            "session_duration": 0.0,
            "num_page_switches": 0,
            "behavior_score": 0.0,
        }


def get_behavior_flags(session_id: str) -> List[str]:
    """Return human-readable fraud flags for bank report."""
    try:
        return get_or_create_profile(session_id).get_risk_flags()
    except Exception as e:
        log.warning("get_behavior_flags failed: %s", e)
        return ["Behavioral data unavailable"]


def set_session_time(session_id: str, total_time: float):
    """Update total session duration."""
    try:
        get_or_create_profile(session_id).total_session_time = max(0.0, float(total_time))
    except Exception as e:
        log.warning("set_session_time failed: %s", e)
