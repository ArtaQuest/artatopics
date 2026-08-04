#!/usr/bin/env python3
"""Drain the operator's Studio "Houses" queue from prod and apply it to the offline analysis registries, so the
collector picks up operator add/remove requests. Run at the top of each grow_topics cycle.

  • ADD word  → prepended to analysis/_fields.json (collected next, single-field 15-param analysis, deployed).
  • REMOVE key → deleted from _fields.json + _fields_weekly.json + cached data (purged from the atlas next deploy).

The queue lives in the prod option `aq_fields_queue`; we read + CLEAR it atomically via `AQ\\Houses::drain()` over
the prod SSH wp-cli (no worker token needed). Safe to run when the collector is idle (grow_topics calls it inline).

  python3 analysis/drain_queue.py
"""
import glob, json, os, re, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", "artaquest"]
slug = lambda s: re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
ADJ_SUFFIX = ("al", "ic", "ous", "ive", "ful", "less", "ish", "ed", "ing", "ary", "ory", "able", "ible")


def guess_pos(word):
    w = word.split()[-1] if word else word                      # last token for a phrase
    return "adj" if w.endswith(ADJ_SUFFIX) and len(w) > 4 else "noun"


def drain():
    """Read + clear the prod queue via AQ\\Houses::drain(). Returns {add:[...], remove:[...]}."""
    try:
        r = subprocess.run(SSH + ["wp eval 'echo wp_json_encode(\\AQ\\Houses::drain());'"],
                           capture_output=True, text=True, timeout=60)
        for line in reversed(r.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
    except Exception as e:
        print(f"[drain] could not reach prod queue: {e}", flush=True)
    return {"add": [], "remove": []}


def main():
    q = drain()
    add = [str(w).strip().lower() for w in q.get("add", []) if str(w).strip()]
    remove = [str(k).strip().lower() for k in q.get("remove", []) if str(k).strip()]
    if not add and not remove:
        return
    fp, rp = f"{ROOT}/analysis/_fields.json", f"{ROOT}/analysis/_fields_weekly.json"
    fields = json.load(open(fp))
    reg = json.load(open(rp)) if os.path.exists(rp) else {}
    # ADD — prepend new words to the work-list (front = collected next)
    added = {w: {"pos": guess_pos(w), "freq": 999, "topics": []} for w in add if w not in fields}
    if added:
        added.update(fields); fields = added
    # REMOVE — delete from both registries + every cached artefact
    for k in remove:
        for cand in {k, slug(k)}:
            fields.pop(cand, None); reg.pop(cand, None)
        for p in (f"{ROOT}/analysis/data_weekly/{slug(k)}.csv", f"{ROOT}/analysis/data_monthly/{slug(k)}.csv"):
            try: os.remove(p)
            except OSError: pass
        for c in glob.glob(f"{ROOT}/analysis/data_chunks/{slug(k)}__*"):
            try: os.remove(c)
            except OSError: pass
    json.dump(fields, open(fp, "w"), indent=0)
    json.dump(reg, open(rp, "w"), indent=0)
    print(f"[drain] applied operator queue: +{len(added)} to analyse, -{len(remove)} purged", flush=True)


if __name__ == "__main__":
    main()
