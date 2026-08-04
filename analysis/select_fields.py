#!/usr/bin/env python3
"""Topic-centric field selection — by COMMONNESS, not Google-Trends popularity.

Pipeline (replaces the old multi-level funnel + word pools entirely):
  1. Each of the 567 topics proposes 10 NOUN + 10 ADJECTIVE candidate keywords (topic_keywords / the _kwgen agents).
  2. We tally how COMMON each keyword is — how many topics independently proposed it (noun and adjective tallies
     kept separate, since a word's two camps are different roles).
  3. For EACH topic we keep its TOP-5 most common nouns and TOP-5 most common adjectives — the shared, central
     vocabulary, not the obscure one-offs.
  4. The UNION of every topic's kept keywords is the field set to analyse, each carrying: pos (noun→WHAT,
     adjective→HOW), its commonness (topic count), and the list of topics that proposed it (provenance).

Output → analysis/_fields.json  { keyword: {pos, freq, topics:[...]} }   (the daily-collection work-list).
The most-common keywords are analysed first.

  python3 analysis/select_fields.py
"""
import importlib.util as u, json, os
from collections import Counter, defaultdict

sw = u.module_from_spec(u.spec_from_file_location("sw", "analysis/_stopwords.py"))
u.spec_from_file_location("sw", "analysis/_stopwords.py").loader.exec_module(sw)

CAND = "analysis/_topic_candidates.json"
OUT = "analysis/_fields.json"
TOP_PER_TOPIC = 4     # keep each topic's 4 most common nouns + 4 most common adjectives


def norm(w):
    return (w or "").strip().lower()


def main():
    cand = json.load(open(CAND))
    # 1) global commonness tallies (how many topics proposed each word), per camp
    noun_freq, adj_freq = Counter(), Counter()
    for tk, c in cand.items():
        for w in {norm(x) for x in c.get("nouns", []) if not sw.is_junk(x)}:
            noun_freq[w] += 1
        for w in {norm(x) for x in c.get("adjectives", []) if not sw.is_junk(x)}:
            adj_freq[w] += 1
    # 2) per topic, keep the TOP-5 most common of each camp; accumulate provenance
    fields = {}  # keyword -> {pos, freq, topics:set}
    def keep(word, pos, freq, topic):
        f = fields.get(word)
        if not f:
            fields[word] = f = {"pos": pos, "freq": freq, "topics": set()}
        f["topics"].add(topic)
    for tk, c in cand.items():
        nouns = sorted({norm(x) for x in c.get("nouns", []) if not sw.is_junk(x)}, key=lambda w: (-noun_freq[w], w))[:TOP_PER_TOPIC]
        adjs  = sorted({norm(x) for x in c.get("adjectives", []) if not sw.is_junk(x)}, key=lambda w: (-adj_freq[w], w))[:TOP_PER_TOPIC]
        for w in nouns: keep(w, "noun", noun_freq[w], tk)
        for w in adjs:  keep(w, "adj",  adj_freq[w],  tk)
    # 3) finalise — commonness = # topics that PROPOSED it (the global tally); ordered most-common first
    out = {}
    for w, f in fields.items():
        gf = noun_freq[w] if f["pos"] == "noun" else adj_freq[w]
        out[w] = {"pos": f["pos"], "freq": gf, "topics": sorted(f["topics"])}
    out = dict(sorted(out.items(), key=lambda kv: (-kv[1]["freq"], kv[0])))
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=0)
    nn = sum(1 for v in out.values() if v["pos"] == "noun"); na = len(out) - nn
    top = list(out.items())[:20]
    print(f"[select_fields] {len(out)} unique fields to analyse ({nn} nouns, {na} adjectives), top-{TOP_PER_TOPIC}/topic → {OUT}")
    print("  most common:", ", ".join(f"{w}({v['freq']},{v['pos'][0]})" for w, v in top))


if __name__ == "__main__":
    main()
