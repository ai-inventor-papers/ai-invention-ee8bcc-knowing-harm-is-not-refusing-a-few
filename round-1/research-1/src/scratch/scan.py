#!/usr/bin/env python3
"""Scan a python data file for suspicious corruption markers (missing letters, doubled spaces, stray non-ascii)."""
import sys, re
BAD = ["arviv", "hugingface", "github .com", "git hub", "  ", "\u00ad", "Modls", "Languae",
       "Evlution", "Perspevtive", "tmplate_without_l", "acv", "Analysi", "a nd", "o f", "t he",
       "rpo", "reops", "modal", "modsl", "Dirction", "succss", "jilbreak", "th e", "wih ",
       "i n ", "a re", "o n ", "fo r ", "to rado", "Reusal", "refsal", "saety", "siga",
       "Meassring", "Measring", "priary", "abstarct", "ynt", "C-NF", "metr ics"]
path = sys.argv[1]
txt = open(path, encoding="utf-8").read()
for i, ln in enumerate(txt.split("\n"), 1):
    for b in BAD:
        if b in ln:
            print(f"{path}:{i}: BAD[{b}] :: {ln.strip()[:110]}")
# doubled-letter heuristic
for m in re.finditer(r"\b(\w{2,})\b", txt):
    w = m.group(1)
    if re.search(r"(.)\1\1", w):
        pass
print("scan done:", path)