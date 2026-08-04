#!/usr/bin/env python3
"""Merge the two SEPARATE registries into _allfit.json for the atlas build, keeping the pages cleanly separated:
- Skills  (_fields_weekly, axis='house')  → the Skills page (/professions). Keep their keys.
- Topics  (_topics_weekly, axis='topic')  → the Topics page (/skills).
A term that is BOTH a skill and a topic (e.g. 'art', 'finance', 'music') would collide on its slug key and the
topic would shadow the skill — so any topic whose key already belongs to a skill gets a '-t' suffix. Both entries
then survive with distinct keys, each tagged with its own axis, so neither page loses a field and neither bleeds
into the other. Env: AQ_SKILLS, AQ_TOPICS, AQ_OUT."""
import json, os

SK = os.environ.get("AQ_SKILLS", "analysis/_fields_weekly.json")
TP = os.environ.get("AQ_TOPICS", "analysis/_topics_weekly.json")
OUT = os.environ.get("AQ_OUT", "analysis/_allfit.json")

sk = json.load(open(SK)) if os.path.exists(SK) else {}
tp = json.load(open(TP)) if os.path.exists(TP) else {}

merged, coll = {}, 0
for k, v in sk.items():
    v = dict(v); v["axis"] = "house"; v["key"] = k
    merged[k] = v
for k, v in tp.items():
    key = k
    while key in merged:                       # collides with a skill (or an earlier topic) → distinct key
        key = key + "-t"; coll += 1
    v = dict(v); v["axis"] = "topic"; v["key"] = key
    merged[key] = v

json.dump(merged, open(OUT, "w"), indent=0)
print(f"[merge_atlas] {len(sk)} skills + {len(tp)} topics = {len(merged)}  ({coll} topic keys suffixed -t to avoid collision)")
