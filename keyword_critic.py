import json
import os
import re
import time
from typing import List, Dict, Tuple, Any
from api_client_manager import get_next_api_client

# Depending on your SDK version, ClientError may live here:
try:
    from google.genai import ClientError
except ImportError:
    from google.api_core import exceptions as google_exceptions
    ClientError = google_exceptions.GoogleAPIError

# The LLM client imported from your existing code
from google.genai import Client



def _extract_json(raw: str) -> Any:
    """
    Extract the first JSON object in a raw LLM response.
    """
    raw = raw.strip()
    # if it starts with ```json, strip code fences
    if raw.startswith("```json"):
        raw = raw.partition("```json")[2]
    raw = raw.strip('`')
    # find outer braces
    start = raw.find('{')
    end = raw.rfind('}') + 1
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in response")
    json_blob = raw[start:end]
    return json.loads(json_blob)


def critique_list(
    label: str,
    candidates: List[str],
    model: str = "gemini-2.0-flash",
    reject_threshold: float = 0.0
) -> Tuple[List[str], Dict[str, str]]:
    """
    Ask a Critic AI to prune and refine a flat list of keyword candidates.
    Returns (kept_terms, suggestions_for_rejected).
    """
    # Build prompt
    cand_json = json.dumps(candidates, ensure_ascii=False)
    prompt = (
        f"You are an expert academic research critic.\n"
        f"You have this list of candidate keywords for {label}:\n"
        f"  {cand_json}\n\n"
        f"Your task:\n"
        f"1) Remove any terms that are overly broad, redundant, or off-topic.\n"
        f"2) For each removed term, suggest exactly one more precise replacement.\n"
        f"Output JUST a JSON object with keys: 'keep', 'reject', and 'suggest'.\n"
        f"Example output:\n"
        f"{{\n"
        f"  \"keep\": [\"term1\", \"term2\"],\n"
        f"  \"reject\": [\"term3\"],\n"
        f"  \"suggest\": {{\"term3\": \"better term\"}}\n"
        f"}}\n"
    )
    try:
        active_client = get_next_api_client() # Get the next client
        resp = active_client.models.generate_content(model=model, contents=prompt)
        time.sleep(2)
        raw = resp.text
        data = _extract_json(raw)
    except Exception as e:
        # on any failure, fallback: keep all
        print(f"⚠️  Critic failed for '{label}': {e}")
        return candidates, {}

    # Validate structure
    keep = data.get('keep', [])
    reject = data.get('reject', [])
    suggest = data.get('suggest', {})

    # Case-insensitive matching: map raw candidates lower → original
    lower_map = {c.lower(): c for c in candidates}

    # Normalize keep/reject entries
    norm_keep: List[str] = []
    norm_reject: List[str] = []
    for k in keep:
        orig = lower_map.get(k.lower())
        if orig:
            norm_keep.append(orig)
    for r in reject:
        orig = lower_map.get(r.lower())
        if orig:
            norm_reject.append(orig)

    # Build suggestions map using original casing
    norm_suggest: Dict[str, str] = {}
    for r, v in suggest.items():
        r_orig = lower_map.get(r.lower())
        if r_orig:
            norm_suggest[r_orig] = v

    # Ensure every candidate is either kept or rejected
    seen = set(norm_keep) | set(norm_reject)
    missing = set(candidates) - seen
    # treat missing as kept
    for m in missing:
        norm_keep.append(m)

    return norm_keep, norm_suggest


def critique_map(
    label: str,
    candidates_map: Dict[str, List[str]],
    model: str = "gemini-2.0-flash",
    reject_threshold: float = 0.0
) -> Dict[str, List[str]]:
    """
    Apply critique_list to each sub-list in a mapping.
    Returns a new map subtheme -> refined keywords.
    """
    refined: Dict[str, List[str]] = {}
    for subtheme, candidates in candidates_map.items():
        full_label = f"{label} ▶ {subtheme}"
        kept, suggestions = critique_list(full_label, candidates, model=model)
        # include any suggested replacements at end
        refined[subtheme] = kept + list(suggestions.values())
    return refined
