#!/usr/bin/env python3
"""Assemble research_out.json from data modules; extract passages verbatim from saved pages."""
import json, os, sys

WS = "/ai-inventor/aii_data/runs/run_hhQTjY478FHc/3_invention_loop/iter_1/gen_art/gen_art_research_1"
SCR = os.path.join(WS, "scratch")
sys.path.insert(0, SCR)

from d01 import V1, V2
from d02 import V3, V4
from d03 import V5, V6, V7
from d04 import V8, V9, V10
from d05 import BASELINE_SPECS
from d06 import HF_CHECKPOINTS
from d07 import SATURATION
from d08 import NUMBERS_TO_REPRODUCE
from d09a import SOURCES_META
from d09b import SOURCES_META2
from d09c import SOURCES_META3
from d09d import SOURCES_META4
from d09e import SOURCES_META5, FOLLOW_UP_QUESTIONS

PAGES = {}
pd = os.path.join(SCR, "pages")
for fn in os.listdir(pd):
    if fn.endswith(".txt"):
        with open(os.path.join(pd, fn), encoding="utf-8") as f:
            PAGES[fn[:-4]] = f.read()

def extract(page, start, end):
    t = PAGES.get(page, "")
    i = t.find(start)
    if i < 0:
        return None, "START_MISSING:" + start[:50]
    j_raw = t.find(end, i)
    if j_raw < 0:
        return None, "END_MISSING:" + end[:50]
    j = max(j_raw + len(end), i + len(start))
    s = t[i:j].strip()
    return s, None

errors = []
SOURCES = []
for meta in SOURCES_META + SOURCES_META2 + SOURCES_META3 + SOURCES_META4 + SOURCES_META5:
    passages = []
    for p in meta.get("passages", []):
        s, err = extract(meta["page"], p["start"], p["end"])
        if err:
            errors.append((meta["index"], err))
        else:
            passages.append({"quote": s, "locator": "abstract (page fetched 2026-09-19)"})
    if meta.get("page") is not None and not passages:
        errors.append((meta["index"], "NO_VALID_PASSAGES"))
    SOURCES.append({
        "index": meta["index"],
        "url": meta["url"],
        "title": meta["title"],
        "summary": meta["summary"],
        "authors": meta.get("authors"),
        "year": meta.get("year"),
        "supporting_passages": passages,
    })

ANSWER = {
 "verification_table": [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10],
 "baseline_specs": BASELINE_SPECS,
 "hf_checkpoints": HF_CHECKPOINTS,
 "saturation": SATURATION,
 "numbers_to_reproduce": NUMBERS_TO_REPRODUCE,
}

OUT = {
 "answer": ANSWER,
 "sources": SOURCES,
 "follow_up_questions": FOLLOW_UP_QUESTIONS,
}

if errors:
    print("PASSAGE ERRORS (fix markers):")
    for e in errors:
        print(" ", e)
    # still dump partial info for iteration
else:
    print("All passages extracted OK.")

with open(os.path.join(WS, "research_out.json"), "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=1, ensure_ascii=False)
print("wrote research_out.json,", len(SOURCES), "sources,", len(ANSWER["verification_table"]), "verification rows")
print("JSON valid:", json.load(open(os.path.join(WS, "research_out.json"), encoding="utf-8")) is not None)