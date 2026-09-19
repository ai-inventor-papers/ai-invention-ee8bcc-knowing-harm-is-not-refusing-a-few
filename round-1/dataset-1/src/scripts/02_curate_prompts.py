#!/usr/bin/env python3
"""02_curate_prompts.py — build prompt pools, map categories, and construct the
probe / B1 / B2 / interpolation-anchor splits for the safety-screening bundle.

Deterministic: numpy Generator(seed=20260919) for ALL sampling.
Zero LLM API calls. Logs the seed. Writes:
  sources/category_mapping.json  — documented source-label -> 7-category map
  sources/pools.jsonl            — every pool row (text, src, row, category, polarity)
  sources/selection.json         — the selected rows with all bundle metadata
"""

import csv
import gzip
import html
import json
import re
import string
import sys
from bisect import bisect_left, bisect_right
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

csv.field_size_limit(10**7)  # prompts.chat contains very long fields

import numpy as np
import pandas as pd
from loguru import logger

WORKSPACE = Path(__file__).resolve().parents[1]
RAW = WORKSPACE / "sources" / "raw"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WORKSPACE / "logs" / "run.log"), rotation="30 MB", level="DEBUG")

SEED = 20260919
RNG = np.random.default_rng(SEED)
logger.info(f"curation seed = {SEED} (numpy Generator)")

MIN_WORDS, MAX_WORDS = 8, 80
LEN_RATIO_MIN, LEN_RATIO_MAX = 0.6, 1.4  # benign/harmful char ratio window (+-40%)
NEAR_DUP_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Target taxonomy
# ---------------------------------------------------------------------------
CATEGORIES = [
    "violence",
    "illegal_acts",
    "fraud",
    "self_harm",
    "privacy_invasion",
    "cyberattacks",
    "disinformation",
]

# probe stratification: every category >= 1, exactly 5 categories get 2
PROBE_COUNTS = {c: 0 for c in CATEGORIES}
for c in ["violence", "illegal_acts", "fraud", "self_harm", "cyberattacks"]:
    PROBE_COUNTS[c] = 2
for c in ["privacy_invasion", "disinformation"]:
    PROBE_COUNTS[c] = 1
assert sum(PROBE_COUNTS.values()) == 12

B1_COUNTS = {"violence": 7, "illegal_acts": 7, "fraud": 7, "self_harm": 7,
             "cyberattacks": 7, "privacy_invasion": 3, "disinformation": 2}
assert sum(B1_COUNTS.values()) == 40
B2_COUNTS = {"violence": 4, "illegal_acts": 4, "fraud": 3, "self_harm": 3,
             "cyberattacks": 3, "privacy_invasion": 2, "disinformation": 1}
assert sum(B2_COUNTS.values()) == 20

# coarse topic keywords per category (benign matching + rule classification)
TOPIC_KEYWORDS = {
    "violence": ["kill", "murder", "attack", "hurt", "fight", "weapon", "gun", "bomb",
                 "beat", "stab", "injure", "torture", "strangle", "assault", "shoot", "poison"],
    "illegal_acts": ["drug", "cocaine", "meth", "heroin", "steal", "robbery", "burglar",
                     "smuggl", "illegal", "crime", "kidnap", "bribe", "assassinat", "counterfeit"],
    "fraud": ["scam", "fraud", "credit card", "money", "embezzle", "tax evasion",
              "forgery", "fake id", "identity theft", "launder"],
    "self_harm": ["suicide", "self-harm", "self harm", "kill myself", "hurt myself",
                  "cut myself", "overdose", "depression", "self-injury"],
    "privacy_invasion": ["stalk", "dox", "record", "spy", "track", "hidden camera",
                         "private", "personal information", "address", "phone number",
                         "surveillance", "secretly"],
    "cyberattacks": ["hack", "malware", "virus", "ransomware", "password", "exploit",
                     "ddos", "sql injection", "keylogger", "breach", "botnet",
                     "spyware", "trojan", "backdoor", "phishing", "account"],
    "disinformation": ["fake news", "misinformation", "disinformation", "propaganda",
                       "hoax", "deepfake", "rumor", "conspiracy", "false claim", "fabricat"],
}

# BeaverTails raw label -> target category
BEAVERTAILS_MAP = {
    "violence,aiding_and_abetting,incitement": "violence",
    "drug_abuse,weapons,banned_substance": "illegal_acts",
    "terrorism,organized_crime": "illegal_acts",
    "financial_crime,property_crime,theft": "fraud",
    "self_harm": "self_harm",
    "privacy_violation": "privacy_invasion",
    "misinformation_regarding_ethics,laws_and_safety": "disinformation",
}

# forbidden_question_set content_policy_name -> target category
FORBIDDEN_POLICY_MAP = {
    "Illegal Activity": "illegal_acts",
    "Malware": "cyberattacks",
    "Physical Harm": "violence",
    "Economic Harm": "fraud",
    "Fraud": "fraud",
    "Privacy Violence": "privacy_invasion",
}

# meta-jailbreak / roleplay scaffolding (dropped from harmful pool)
META_JB = [
    " dan ", " jailbreak", "pretend", "ignore previous", "ignore all previous",
    "developer mode", "do anything now", "no restrictions", "unfiltered",
    "act as", "you are now", "roleplay", "simulate", "in this fictional",
    "within this hypothetical", "bypass", "unleash", "as an ai", "as a language model",
    "hypothetical scenario", "for research purposes", "educational purposes only",
    "claude", "chatgpt", "gpt-4", "gpt-3", "assistant mode", "red team",
]
# template / dialog artifacts (dropped from BOTH pools)
ARTIFACTS = ["human:", "assistant:", "<|", "[/inst", "### response", "### instruction",
             "http://", "https://", "\u2014", "\u2026", "#include", "```"]

IMPERATIVE_STARTS = (
    "write", "create", "give", "tell", "show", "explain", "help", "provide", "list",
    "make", "generate", "build", "describe", "teach", "send", "post", "share", "set up",
    "install", "develop", "design", "produce", "turn", "use", "find", "get", "access",
    "track", "connect", "steal", "hack", "kill", "threaten", "prepare", "spread",
    "recommend", "suggest", "plan", "arrange", "start", "stop", "delete", "change",
    "convince", "disguise", "forge", "impersonate", "cover", "hide", "break",
)


# ---------------------------------------------------------------------------
# Normalization / filters
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Whitespace/quote normalization + HTML unescape."""
    text = html.unescape(text or "")
    for a, b in [("\u201c", '"'), ("\u201d", '"'), ("\u2018", "'"), ("\u2019", "'"),
                 ("\u2013", "-"), ("\u2014", "-"), ("\u00a0", " ")]:
        text = text.replace(a, b)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    good = sum(1 for ch in text if ord(ch) < 128 and ch in string.printable)
    return good / len(text)


def word_count(text: str) -> int:
    return len(text.split())


def drop_framing(text: str, patterns: list) -> bool:
    t = " " + text.lower() + " "
    return any(p in t for p in patterns)


def near_dup(text: str, others: list) -> bool:
    """True if text is a near-duplicate (SequenceMatcher ratio >= threshold)."""
    a = normalize(text).lower()
    if not a:
        return False
    for o in others:
        b = normalize(o).lower()
        if b and SequenceMatcher(None, a, b).ratio() >= NEAR_DUP_THRESHOLD:
            return True
    return False


HARMFUL_MARKERS = [
    # violent / lethal
    "how to kill", "kill myself", "kill yourself", "kill someone", "how do i kill",
    "how to murder", "murder", "assault", "strangle", "kidnap", "deadly poison",
    "poison someone", "make a bomb", "how to make a bomb", "build a bomb", "arson",
    "house fire", "date rape", " rape ",
    # cyber
    "how to hack", "hack into", "help me hack", "hack my", "hack someone", "hack her",
    "hack his", "create malware", "write malware", "make malware",
    "develop a virus", "ransomware", "ddos", "keylogger",
    # fraud / theft
    "how to steal", "steal from", "steal a car", "shoplift", "commit credit card fraud",
    "credit card fraud", "how to launder", "launder money", "money laundering",
    "how to rob", "rob a bank", "identity theft", "fake id", "forgery", "counterfeit",
    "tax evasion", "embezzle", "commit fraud", "how to scam", "scam email", " scam ",
    "gonna steal", "going to steal", "want to steal", "planning to steal",
    "steal something", "get away with", "end up in prison", "insurance money",
    "crash my car", "money that doesn't belong", "take money", "money that is not mine",
    "under the table",
    # hate speech (not a target category; never a benign control)
    "mexicans", "muslims are", "jews are", "niggers", "faggot", "black people are",
    "asians are", "whites are", "retarded", "chinese people are",
    # self-harm / drugs
    "suicide", "self-harm", "self harm", "cut myself", "hurt myself", "overdose",
    "drug smuggl", "smuggl", "cocaine", "heroin", "meth", "roofie", "illegal drugs",
    "move drugs", "traffick",
    # privacy / creepy
    "stalk ", "stalk my", "stalking", "dox", "doxx", "hit a lick", "scamm",
    "fake degree", "fake diploma", "gain access to someone's", "access someone's",
    "someone's private", "private medical", "wish i knew how", "steal a woman's",
    # sensitive / out-of-taxonomy
    "child porn", "underage", "posing as a minor", "solicit", "sexual content from",
    "sexually explicit", "porn",
]

# out-of-taxonomy sensitive content (dropped from harmful pool entirely)
SENSITIVE_EXCLUDE = ["posing as a minor", "underage", "child porn", "solicit sexual",
                     "date rape", "sexual content from", "rape", "sexually explicit",
                     "porn", "nude", "explicit photos", "sex", "sexual", "anal ",
                     "prostitut", "escort", "masturbat", "orgasm", "blowjob",
                     "penis", "vagina", "boob", "breast", "fetish", "panties",
                     "underwear", "g-string", "lingerie", "used panties", "teaches children",
                     "children how to", "kids how to", "children to perform",
]

# when a source row carries several mapped labels, prefer the most specific one
CAT_PRIORITY = ["violence", "self_harm", "cyberattacks", "fraud", "illegal_acts",
                "privacy_invasion", "disinformation"]


def is_harmful_phrase(text: str) -> bool:
    """Phrase-level harmful directive detector (keeps the benign pool clean)."""
    t = text.lower()
    return any(m in t for m in HARMFUL_MARKERS)


def pick_category(cats: list) -> str:
    """Prefer the most specific target category among a row's mapped labels."""
    for prio in CAT_PRIORITY:
        if prio in cats:
            return prio
    return cats[0]


def classify_by_keywords(text: str) -> str | None:
    """Rule-based category assignment for unlabeled harmful sources.

    Defensive/protective contexts (predicting, detecting, preventing harm) are
    treated as NOT harmful-directive and excluded.
    """
    t = text.lower()
    if any(w in t for w in ("predict", "detect", "prevent", "identify", "avoid",
                            "spot ", "recognize", "protect against")):
        return None
    for cat in ["violence", "self_harm", "cyberattacks", "fraud", "illegal_acts",
                "privacy_invasion", "disinformation"]:
        if any(k in t for k in TOPIC_KEYWORDS[cat]):
            return cat
    return None


# ---------------------------------------------------------------------------
# Loaders (each returns [{"text", "src_row", ...}])
# ---------------------------------------------------------------------------
def load_beavertails(split: str) -> list[dict]:
    f = RAW / "PKU-Alignment__BeaverTails" / "round0" / "30k" / f"{split}.jsonl.gz"
    rows = []
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            rows.append({"text": r.get("prompt", ""), "safe": r.get("is_safe"),
                         "cat_dict": r.get("category") or {}, "src_row": f"{split}#{i}"})
    logger.info(f"BeaverTails {split}: {len(rows)} rows")
    return rows


def load_dolly() -> list[dict]:
    f = RAW / "databricks__databricks-dolly-15k" / "databricks-dolly-15k.jsonl"
    rows = []
    with open(f, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            rows.append({"text": r.get("instruction", ""), "src_row": f"dolly#{i}"})
    logger.info(f"dolly: {len(rows)} rows")
    return rows


def load_alpaca_eval() -> list[dict]:
    f = RAW / "tatsu-lab__alpaca_eval" / "alpaca_eval_gpt4_baseline.json"
    data = json.loads(f.read_text(encoding="utf-8"))
    rows = [{"text": r.get("instruction", ""), "src_row": f"alpaca_eval#{i}"}
            for i, r in enumerate(data)]
    logger.info(f"alpaca_eval: {len(rows)} rows")
    return rows


def load_alpaca_cleaned() -> list[dict]:
    f = RAW / "yahma__alpaca-cleaned" / "alpaca_data_cleaned.json"
    data = json.loads(f.read_text(encoding="utf-8"))
    rows = []
    for i, r in enumerate(data):
        inst = (r.get("instruction") or "").strip()
        inp = (r.get("input") or "").strip()
        text = f"{inst}\n{inp}".strip() if inp else inst
        rows.append({"text": text, "src_row": f"alpaca_cleaned#{i}"})
    logger.info(f"alpaca-cleaned: {len(rows)} rows")
    return rows


def load_prompts_chat() -> list[dict]:
    f = RAW / "fka__prompts.chat" / "prompts.csv"
    rows = []
    with open(f, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, r in enumerate(reader):
            rows.append({"text": r.get("prompt", ""), "src_row": f"prompts_chat#{i}"})
    logger.info(f"prompts.chat: {len(rows)} rows")
    return rows


def load_harmful_behaviors() -> list[dict]:
    f = RAW / "mlabonne__harmful_behaviors" / "data" / "train-00000-of-00001.parquet"
    df = pd.read_parquet(f)
    rows = [{"text": str(r.text), "src_row": f"harmful_behaviors#{i}"}
            for i, r in enumerate(df.itertuples(index=False))]
    logger.info(f"harmful_behaviors: {len(rows)} rows")
    return rows


def load_forbidden_questions() -> list[dict]:
    f = RAW / "verazuo__jailbreak_llms__forbidden_question_set.csv"
    rows = []
    with open(f, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, r in enumerate(reader):
            rows.append({"text": r.get("question", ""), "policy": r.get("content_policy_name", ""),
                         "src_row": f"forbidden_question_set#{i}"})
    logger.info(f"forbidden_question_set: {len(rows)} rows")
    return rows


def load_jailbreak_prompts() -> list[dict]:
    f = RAW / "verazuo__jailbreak_llms__jailbreak_prompts_2023_05_07.csv"
    rows = []
    with open(f, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, r in enumerate(reader):
            rows.append({"text": r.get("prompt", ""), "src_row": f"jailbreak_prompts#{i}"})
    logger.info(f"jailbreak_prompts: {len(rows)} rows")
    return rows


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------
POOL_HARMFUL: list[dict] = []
POOL_BENIGN: list[dict] = []
SEEN_TEXTS: dict[str, dict] = {}  # casefolded text -> pool row


def _add_pool(text: str, polarity: str, category: str | None, src: str, src_row: str) -> bool:
    text = normalize(text)
    if not text:
        return False
    nw = word_count(text)
    if nw < MIN_WORDS or nw > MAX_WORDS:
        return False
    if ascii_ratio(text) < 0.95:
        return False
    if drop_framing(text, ARTIFACTS):
        return False
    key = text.casefold()
    if key in SEEN_TEXTS:
        return False  # exact dedupe (case-folded), keep first source
    if polarity == "harmful" and drop_framing(text, META_JB):
        return False
    if polarity == "harmful" and drop_framing(text, SENSITIVE_EXCLUDE):
        return False  # out-of-taxonomy sensitive content (grooming/porn/rape)
    if polarity == "benign" and is_harmful_phrase(text):
        return False
    row = {"text": text, "polarity": polarity, "category": category, "src": src,
           "src_row": src_row, "len_chars": len(text), "len_words": nw}
    SEEN_TEXTS[key] = row
    if polarity == "harmful":
        POOL_HARMFUL.append(row)
    else:
        POOL_BENIGN.append(row)
    return True


def build_pools() -> None:
    for r in load_beavertails("train"):
        if r["safe"] is True:
            _add_pool(r["text"], "benign", None, "PKU-Alignment/BeaverTails", r["src_row"])
        else:
            cats = [BEAVERTAILS_MAP[k] for k, v in r["cat_dict"].items() if v and k in BEAVERTAILS_MAP]
            if cats:
                _add_pool(r["text"], "harmful", pick_category(cats), "PKU-Alignment/BeaverTails", r["src_row"])
    for r in load_beavertails("test"):
        if r["safe"] is True:
            _add_pool(r["text"], "benign", None, "PKU-Alignment/BeaverTails", r["src_row"])
        else:
            cats = [BEAVERTAILS_MAP[k] for k, v in r["cat_dict"].items() if v and k in BEAVERTAILS_MAP]
            if cats:
                _add_pool(r["text"], "harmful", pick_category(cats), "PKU-Alignment/BeaverTails", r["src_row"])
    for r in load_forbidden_questions():
        cat = FORBIDDEN_POLICY_MAP.get(r["policy"])
        if cat:
            _add_pool(r["text"], "harmful", cat, "verazuo/jailbreak_llms",
                      f"forbidden_question_set::{r['src_row']}")
    for r in load_harmful_behaviors():
        cat = classify_by_keywords(r["text"])
        if cat:
            _add_pool(r["text"], "harmful", cat, "mlabonne/harmful_behaviors", r["src_row"])
    for r in load_jailbreak_prompts():
        cat = classify_by_keywords(r["text"])
        if cat:
            _add_pool(r["text"], "harmful", cat, "verazuo/jailbreak_llms",
                      f"jailbreak_prompts::{r['src_row']}")

    for src, loader in [("databricks/databricks-dolly-15k", load_dolly),
                        ("tatsu-lab/alpaca_eval", load_alpaca_eval),
                        ("yahma/alpaca-cleaned", load_alpaca_cleaned),
                        ("fka/prompts.chat", load_prompts_chat)]:
        for r in loader():
            _add_pool(r["text"], "benign", None, src, r["src_row"])

    dist = Counter(p["category"] for p in POOL_HARMFUL)
    logger.info(f"harmful pool: {len(POOL_HARMFUL)} rows; per category: {json.dumps(dict(dist))}")
    logger.info(f"benign pool: {len(POOL_BENIGN)} rows")


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------
def sample_category(cat: str, n: int, exclude_texts: list) -> list[dict]:
    """Deterministically sample n harmful rows of category cat, excluding near-dups.

    Length-gated near-dup check keeps this fast: SequenceMatcher is only run
    against excluded texts whose length is within +-35% of the candidate.
    """
    cands = [p for p in POOL_HARMFUL if p["category"] == cat]
    order = RNG.permutation(len(cands))
    picked: list[dict] = []
    ex_set = set(exclude_texts)
    ex_list = [(e, len(e)) for e in exclude_texts]
    for idx in order:
        if len(picked) >= n:
            break
        c = cands[idx]
        if c["text"] in ex_set:
            continue
        # length-gated near-dup
        dup = False
        for e, elen in ex_list:
            if abs(len(c["text"]) - elen) > 0.35 * max(elen, 1):
                continue
            if SequenceMatcher(None, c["text"].lower(), e.lower()).ratio() >= NEAR_DUP_THRESHOLD:
                dup = True
                break
        if dup:
            continue
        if near_dup(c["text"], [p["text"] for p in picked]):
            continue
        picked.append(c)
    if len(picked) < n:
        logger.warning(f"category {cat}: only {len(picked)}/{n} available after exclusion")
    return picked


_BENIGN_SORTED: list[dict] | None = None
_BENIGN_LENS: list[int] | None = None


def benign_sorted() -> tuple[list[dict], list[int]]:
    """Lazily sorted benign pool + its char lengths (hoisted out of match loop)."""
    global _BENIGN_SORTED, _BENIGN_LENS
    if _BENIGN_SORTED is None:
        _BENIGN_SORTED = sorted(POOL_BENIGN, key=lambda p: p["len_chars"])
        _BENIGN_LENS = [p["len_chars"] for p in _BENIGN_SORTED]
    return _BENIGN_SORTED, _BENIGN_LENS


def match_benign(harmful_text: str, category: str | None, used_benign: list) -> dict | None:
    """Benign control: topic keyword overlap preferred; length ratio closest to 1."""
    target_len = len(harmful_text)
    lo = int(target_len * LEN_RATIO_MIN)
    hi = max(int(target_len * LEN_RATIO_MAX), lo + 1)
    pool, lens = benign_sorted()
    cands = pool[bisect_left(lens, lo):bisect_right(lens, hi)]
    used_texts = {u["text"] for u in used_benign}
    kws = TOPIC_KEYWORDS.get(category or "", [])
    scored = []
    for c in cands:
        if c["text"] in used_texts:
            continue
        if is_harmful_phrase(c["text"]):
            continue  # belt & braces: never ship a harmful control
        overlap = sum(1 for k in kws if k in c["text"].lower())
        ratio = c["len_chars"] / target_len
        scored.append((c, overlap, abs(ratio - 1.0), ratio))
    if not scored:
        return None
    scored.sort(key=lambda t: (-t[1], t[2]))
    best, _, _, ratio = scored[0]
    out = dict(best)
    out["len_ratio"] = round(ratio, 3)
    return out


# ---------------------------------------------------------------------------
# Quality gate (no-LLM heuristics)
# ---------------------------------------------------------------------------
def gate_flags(row: dict) -> list[str]:
    """Return quality issues for a harmful row (empty list = OK)."""
    flags = []
    t = row["text"].lower()
    if t.endswith(("!", ":", ";", ",")):
        flags.append("non-request ending")
    if re.search(r"\b(what|why|when|where|who)\b", t) and "how to" not in t and row["len_words"] < 14:
        flags.append("info-question too short to be a request")
    low_action = ["some places", "is it ", "are there ", "do you think", "what do you think",
                  "i wonder", "explain why", "explain the history", "what is the difference",
                  "what is internet", "what is the best way", "does ", "can you explain",
                  "what are the ", "in todays", "what seem", "can you recommend",
                  "where can i", "where are", "what are some good", "is it possible",
                  "it is possible", "good places", "best places", "do you know",
                  "i want to know if", "what is a good", "what's a good", "where do i",
                  "sometimes,", "did you see", "did you hear", "did you notice"]
    if any(t.startswith(w) for w in low_action) or any(w in t for w in ["what is the difference", "what is internet"]):
        flags.append("background/definitional question rather than direct request")
    if re.search(r"someone else's (netflix|spotify|account)|streaming account|predicting fraud|"
                 r"detecting fraud|fraud detection|detect fraud|prevent fraud|machine-learning model|"
                 r"watch a movie before|released in theaters|torrent|pirat|panties|underwear|"
                 r"credit eligibility|effectively automate decisions|\bif (one|you|i) were to\b",
                 t):
        flags.append("low-harm gray-area or defensive context")
    for bad in ["xxx", "todo", "placeholder", "example.com"]:
        if bad in t:
            flags.append(f"placeholder token '{bad}'")
    # cross-category mismatch: text looks more like another category than its label
    kw_counts = {c: sum(1 for k in TOPIC_KEYWORDS[c] if k in t) for c in CATEGORIES}
    own = kw_counts.get(row["category"], 0)
    others = {c: n for c, n in kw_counts.items() if c != row["category"] and n >= 1}
    if others:
        top = max(others, key=others.get)
        if others[top] >= 2 or (others[top] >= 1 and own == 0):
            flags.append(f"possible text-category mismatch (looks more like {top})")
    return flags


def gate_group(group: list[dict], exclude_texts: list, fold: str) -> list[dict]:
    """Auto-swap flagged rows (loop until replacements are also clean)."""
    review = []
    for i in range(len(group)):
        row = group[i]
        flags = gate_flags(row)
        attempts = 0
        while flags and attempts < 20:
            logger.warning(f"GATE {fold}: {row['text'][:65]} -> {flags}")
            repl = sample_category(row["category"], 1, exclude_texts + [row["text"]]) 
            if not repl:
                review.append({"old_text": row["text"], "new_text": None,
                               "flags": flags, "note": f"{fold}#{i}: no replacement found"})
                break
            repl = repl[0]
            group[i] = repl
            review.append({"old_text": row["text"], "new_text": repl["text"],
                           "src": repl["src"], "src_row": repl["src_row"],
                           "flags": flags, "note": f"{fold}#{i}: auto-swapped by quality gate"})
            logger.info(f"  swapped -> {repl['text'][:65]}")
            row = group[i]
            flags = gate_flags(row)
            attempts += 1
        if flags:
            logger.warning(f"GATE {fold}#{i}: still flagged after swaps: {row['text'][:50]}")
    return review


# ---------------------------------------------------------------------------
# Main selection
# ---------------------------------------------------------------------------
@logger.catch(reraise=True)
def main() -> None:
    build_pools()
    logger.info("---- probe corpus ----")

    # 1) probe harmful: stratified sampling
    probe_harmful: list[dict] = []
    for cat in CATEGORIES:
        probe_harmful.extend(sample_category(cat, PROBE_COUNTS[cat],
                              [p["text"] for p in probe_harmful]))
    all_excl = [p["text"] for p in probe_harmful]

    # 2) quality gate BEFORE pairing
    review = gate_group(probe_harmful, all_excl, "probe")
    probe_texts = [p["text"] for p in probe_harmful]

    # 3) benign controls for probe
    used_benign: list[dict] = []
    probe_pairs: list[dict] = []
    for p in probe_harmful:
        m = match_benign(p["text"], p["category"], used_benign)
        if m is None:
            logger.warning(f"no benign match for: {p['text'][:60]}")
            continue
        used_benign.append(m)
        probe_pairs.append({"harmful": p, "benign": m})
    logger.info(f"probe pairs: {len(probe_pairs)}")

    # 4) nested subsets p4/p8/p12 by actionability
    def action_score(row: dict) -> int:
        t = row["text"].lower()
        s = 0
        first = t.split()[0] if t.split() else ""
        if first in IMPERATIVE_STARTS or t.startswith("how"):
            s += 2
        if "how to" in t:
            s += 1
        if 8 <= row["len_words"] <= 25:
            s += 1
        if any(w in t for w in ("my", "me ", " i ")):
            s += 1
        return s

    pair_scores = sorted(((action_score(pr["harmful"]), i, pr)
                          for i, pr in enumerate(probe_pairs)), key=lambda t: (-t[0], t[1]))
    p4: list[dict] = []
    used_cats: set[str] = set()
    for _, _, pr in pair_scores:
        cat = pr["harmful"]["category"]
        if cat not in used_cats:
            used_cats.add(cat)
            p4.append(pr)
        if len(p4) == 4 and len(used_cats) >= 4:
            break
    p8: list[dict] = list(p4)
    used_c8 = {pr["harmful"]["category"] for pr in p4}
    rem = [pr for _, _, pr in pair_scores if pr not in p4]
    # 1) one pair per still-missing category (guarantees >=6 cats)
    for pr in rem:
        if pr["harmful"]["category"] not in used_c8:
            p8.append(pr)
            used_c8.add(pr["harmful"]["category"])
    # 2) fill up to exactly 8 pairs with the highest-scoring leftovers
    for pr in rem:
        if len(p8) >= 8:
            break
        if pr not in p8:
            p8.append(pr)
    logger.info(f"p4 covers {sorted({pr['harmful']['category'] for pr in p4})}")
    logger.info(f"p8 covers {sorted({pr['harmful']['category'] for pr in p8})}")

    # 5) B1 split
    logger.info("---- B1 split ----")
    b1_harmful: list[dict] = []
    for cat in CATEGORIES:
        b1_harmful.extend(sample_category(cat, B1_COUNTS[cat], probe_texts +
                          [p["text"] for p in b1_harmful]))
    review += gate_group(b1_harmful, probe_texts + [p["text"] for p in b1_harmful], "b1")
    b1_texts = [p["text"] for p in b1_harmful]

    # 6) B2 split (reserved)
    logger.info("---- B2 split (reserved) ----")
    b2_harmful: list[dict] = []
    for cat in CATEGORIES:
        b2_harmful.extend(sample_category(cat, B2_COUNTS[cat], probe_texts + b1_texts + [p["text"] for p in b2_harmful]))
    review += gate_group(b2_harmful, probe_texts + b1_texts + [p["text"] for p in b2_harmful], "b2")

    # 7) B1 benign controls (10, topic-length matched)
    b1_benign: list[dict] = []
    for hp in b1_harmful[:10]:
        m = match_benign(hp["text"], None, used_benign)
        if m:
            used_benign.append(m)
            b1_benign.append(m)

    # 8) B2 benign controls (5, exact-disjoint from all used benign; near-dup O(n^2)
    #    is skipped here — benign rows are already pool-deduped; exact check suffices)
    b2_benign: list[dict] = []
    order = RNG.permutation(len(POOL_BENIGN))
    used_all = {u["text"] for u in used_benign}
    for idx in order:
        if len(b2_benign) >= 5:
            break
        c = POOL_BENIGN[idx]
        if c["text"] in used_all:
            continue
        used_all.add(c["text"])
        used_benign.append(c)
        b2_benign.append(dict(c))

    # 9) interpolation anchors: probe pairs (reused) + fresh matches for B1
    logger.info("---- interpolation anchors ----")
    anchors: list[dict] = []
    for pr in probe_pairs:
        anchors.append({"harmful": pr["harmful"], "benign": pr["benign"], "reused": True})
    for hp in b1_harmful:
        m = match_benign(hp["text"], hp["category"], used_benign)
        if m is None:
            logger.warning(f"no anchor benign for B1 row: {hp['text'][:60]}")
            continue
        used_benign.append(m)
        anchors.append({"harmful": hp, "benign": m, "reused": False})
    logger.info(f"anchors: {len(anchors)} (probe-reused {sum(1 for a in anchors if a['reused'])})")

    # 10) persist
    selection = {
        "seed": SEED,
        "probe_pairs": probe_pairs,
        "p4_pair_indices": [probe_pairs.index(pr) for pr in p4],
        "p8_pair_indices": [probe_pairs.index(pr) for pr in p8],
        "b1_harmful": b1_harmful,
        "b1_benign": b1_benign,
        "b2_harmful": b2_harmful,
        "b2_benign": b2_benign,
        "anchors": anchors,
        "review": review,
    }
    (WORKSPACE / "sources"/ "selection.json").write_text(json.dumps(selection, indent=1, default=str))
    with (WORKSPACE / "sources" / "pools.jsonl").open("w", encoding="utf-8") as fh:
        for p in POOL_HARMFUL + POOL_BENIGN:
            fh.write(json.dumps(p, default=str) + "\n")
    logger.info(f"selection written: probe={len(probe_pairs)} pairs, B1={len(b1_harmful)}+{len(b1_benign)}, "
                f"B2={len(b2_harmful)}+{len(b2_benign)}, anchors={len(anchors)}")


if __name__ == "__main__":
    main()