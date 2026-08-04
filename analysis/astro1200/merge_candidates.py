#!/usr/bin/env python3
"""Astro-1200 — merge the 8 category candidate lists into one deduped pool (candidates.json)."""
import glob, json, os, re, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))

def norm(t):
    t = unicodedata.normalize('NFKD', str(t)).encode('ascii', 'ignore').decode()
    t = re.sub(r'[^a-z0-9 ]+', ' ', t.lower())
    return re.sub(r'\s+', ' ', t).strip()

def slug(s): return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')

seen, pool, per = set(), [], {}
for p in sorted(glob.glob(os.path.join(HERE, 'cand_*.json'))):
    cat = os.path.basename(p)[5:-5]
    kept = 0
    for t in json.load(open(p)):
        t = norm(t)
        k = slug(t)
        if not t or not k or len(t.split()) > 2 or k in seen: continue   # operator rule: 1-2 words max
        seen.add(k); pool.append(t); kept += 1
    per[cat] = kept
json.dump(pool, open(os.path.join(HERE, 'candidates.json'), 'w'), indent=0)
print(per, 'TOTAL', len(pool))
