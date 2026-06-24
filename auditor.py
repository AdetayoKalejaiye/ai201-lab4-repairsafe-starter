"""
auditor.py

Appends a structured JSONL record of every interaction to the audit log file,
and prints a concise one-line summary to the terminal for real-time monitoring.
"""

import json
import os
from datetime import datetime, timezone

from config import LOG_FILE

_QUESTION_MAX_CHARS = 300
_RESPONSE_PREVIEW_MAX_CHARS = 200
_REASON_MAX_CHARS = 300


def _truncate(text: str, max_chars: int) -> str:
    """Truncate to max_chars, appending an ellipsis if anything was cut."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\u2026"


def log_interaction(question: str, tier: str, response: str, reason: str = "") -> None:
    """
    Append a structured record of this interaction to the audit log.

    NOTE ON SIGNATURE: the spec's Input/Output Contract lists only
    (question, tier, response), but the Log Entry Fields table also
    requires "classifier_reasoning" — which classify_safety_tier() returns
    as result["reason"] and has no other way of reaching this function.
    This adds an optional `reason` parameter (default "") so existing
    callers that only pass the original three arguments still work, but a
    caller that wants a complete audit trail should pass through
    classify_safety_tier()'s "reason" value.

    Writes one JSON object per line to LOG_FILE (logs/audit.jsonl) using
    json.dumps() (no pretty-printing, so the file stays valid JSONL),
    creating the logs/ directory first if it doesn't exist yet. Also
    prints a one-line summary to the terminal.

    Logged fields:
      - "timestamp"            : ISO 8601 UTC, e.g. "2025-11-01T14:22:01Z"
      - "tier"                  : safety tier assigned to this question
      - "question"              : user's question, truncated to 300 chars
      - "response_preview"      : first 200 characters of the response
      - "classifier_reasoning"  : classifier's stated reason, truncated to 300 chars
      - "flagged"               : bool — see note below

    "flagged" logic: the spec describes this as flagging questions that
    could plausibly come from a bad actor, but doesn't say how to decide
    that, and this system has no separate intent-detection step. The only
    existing signal is the classifier's own tier, so flagged is set to
    (tier == "refuse"). This is a coarse triage signal for a human
    reviewer — most "refuse" questions will be ordinary people asking
    about a genuinely dangerous repair, not bad actors, so this field
    should be read as "worth a second look," not as an accusation.
    """
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    record = {
        "timestamp": timestamp,
        "tier": tier,
        "question": _truncate(question, _QUESTION_MAX_CHARS),
        "response_preview": _truncate(response, _RESPONSE_PREVIEW_MAX_CHARS),
        "classifier_reasoning": _truncate(reason, _REASON_MAX_CHARS),
        "flagged": tier == "refuse",
    }

    # Create logs/ on first run (or if it's ever deleted) rather than
    # assuming it exists. exist_ok=True makes this safe to call on every
    # invocation, including from multiple processes at once.
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    question_preview = _truncate(question, 60)
    flag_marker = " \u26a0 flagged" if record["flagged"] else ""
    print(
        f'[LOGGED] tier={tier} | "{question_preview}" \u2192 '
        f"{len(response)} chars{flag_marker}"
    )