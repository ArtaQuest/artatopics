#!/usr/bin/env python3
"""Astro-1200 — final GENERIC-TOPIC candidate pool.

candidates.json = union of (a) the entity-filtered survivors of the research-based pool
(generic_chunk_0..3.json — companies/brands/people/titles removed) and (b) the generic-topic
enumeration waves (gen_*.json). 1-2 words, deduped by slug, defunct blocklist re-applied."""
import glob, json, os, re, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))

def norm(t):
    t = unicodedata.normalize('NFKD', str(t)).encode('ascii', 'ignore').decode()
    t = re.sub(r'[^a-z0-9 ]+', ' ', t.lower())
    return re.sub(r'\s+', ' ', t).strip()

def slug(s): return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')

seen, pool, per = set(), [], {}
files = sorted(glob.glob(os.path.join(HERE, 'generic_chunk_*.json'))) + sorted(glob.glob(os.path.join(HERE, 'gen_*.json')))
for p in files:
    kept = 0
    for t in json.load(open(p)):
        t = norm(t)
        k = slug(t)
        if not t or not k or len(t.split()) > 2 or k in seen: continue
        seen.add(k); pool.append(t); kept += 1
    per[os.path.basename(p)] = kept
json.dump(pool, open(os.path.join(HERE, 'candidates.json'), 'w'), indent=0)
print(per, 'TOTAL', len(pool))
