"""
safety.py

Classifies home-repair questions into one of three safety tiers using an LLM.
The prompt below encodes the design decisions from specs/classifier-spec.md:
tier definitions, the caution/refuse boundary rule, few-shot examples, a
single-line JSON output format, and a fail-closed fallback to "refuse".
"""

import json
import re

from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_TIERS

_client = Groq(api_key=GROQ_API_KEY)

# Per the spec: failing open ("safe") is more dangerous than failing closed.
# Any parse/validation failure routes here instead of "caution" or "safe".
_FALLBACK_TIER = "refuse"
_FALLBACK_REASON = (
    "Could not reliably parse a classification from the model's response, "
    "so the question was routed to the most cautious tier."
)

_SYSTEM_PROMPT = """You are a safety classifier for a home-repair assistant. Read a single home-repair question and assign it to exactly one of three tiers.
Use the reasonable worst case scenario for classification e.g fire hazard for moving a switch six inches to the right etc
TIERS

safe: There is little to no risk of injury, the financial cost of getting it wrong is minimal, and the repair requires no permit or professional approval.

caution: The repair requires no permit, but a mistake could cost roughly a few hundred dollars or carry a real but non-catastrophic risk of injury (cuts, minor burns, etc.) — a completely rational amateur would still reasonably choose to attempt it.

refuse: The repair cannot safely be done by an amateur, typically requires a permit or licensed professional, and a mistake could cause serious financial loss, fire, flooding, electrocution, structural damage, or permanent injury. 

BOUNDARY RULE (caution vs. refuse)
If a careful amateur without special training or a permit could still reasonably attempt the repair and the worst realistic outcome is moderate, classify it "caution". If the repair legally or practically requires a permit or licensed professional, or the worst realistic outcome is severe or irreversible, classify it "refuse".

EXAMPLES

Q: "How do I unclog a slow bathroom sink with a plunger?"
{"tier": "safe", "reason": "Plunging a sink is low-risk, low-cost, and needs no permit."}

Q: "How do I replace a cracked cabinet hinge?"
{"tier": "safe", "reason": "Swapping a hinge is routine, low-risk, and cheap to redo if done wrong."}

Q: "Can I replace a wall outlet myself?"
{"tier": "caution", "reason": "A single outlet swap typically needs no permit, and the main risk is a minor shock if the breaker isn't off, which a careful amateur can manage."}

Q: "How do I patch a small hole in drywall?"
{"tier": "caution", "reason": "Mistakes are cheap to fix and injury risk is minor, though care is needed to avoid hidden wiring or pipes."}

Q: "Can I replace my home's main electrical panel myself?"
{"tier": "refuse", "reason": "Panel work usually requires a permit and a licensed electrician, and a mistake risks fire or electrocution."}

Q: "How do I fix a gas line leak myself?"
{"tier": "refuse", "reason": "Gas line repairs require a licensed professional, and a mistake can cause an explosion or carbon monoxide poisoning."}

OUTPUT FORMAT
Respond with ONLY a single JSON object on one line. No markdown code fences, no extra commentary before or after it. It must have exactly two keys:
{"tier": "<safe|caution|refuse>", "reason": "<one sentence>"}
"""

_USER_PROMPT_TEMPLATE = """Classify this home-repair question:

"{question}"

Respond with only the JSON object described above."""


def _extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of a {"tier": ..., "reason": ...} object from a
    raw LLM response, tolerating stray prose or markdown fences from weaker
    models that don't follow the format instruction exactly."""
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    return None


def classify_safety_tier(question: str) -> dict:
    """
    Classify a home repair question into one of three safety tiers.

    Sends a single chat completion request (no tools, no history) using a
    system prompt that contains the tier definitions, the caution/refuse
    boundary rule, and few-shot examples, then parses the JSON object the
    model returns.

    If the response can't be parsed as JSON, or the parsed tier isn't a
    member of VALID_TIERS, this fails closed to "refuse" — per the spec,
    over-referring to a professional is a far cheaper mistake than
    under-warning on a genuinely dangerous repair.

    Returns a dict with:
      - "tier"   : str — one of "safe", "caution", "refuse"
      - "reason" : str — a brief explanation of why this tier was assigned
    """
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(question=question)},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content or ""
    parsed = _extract_json_object(raw)

    if not parsed:
        return {"tier": _FALLBACK_TIER, "reason": _FALLBACK_REASON}

    tier = parsed.get("tier")
    reason = parsed.get("reason")

    if tier not in VALID_TIERS or not isinstance(reason, str) or not reason.strip():
        return {"tier": _FALLBACK_TIER, "reason": _FALLBACK_REASON}

    return {"tier": tier, "reason": reason.strip()}