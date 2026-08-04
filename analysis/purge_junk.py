#!/usr/bin/env python3
"""One-shot: strip every junk (stop-word / generic) keyword from the pool + all analysis registries + the
published data, then regenerate the atlas clean. Idempotent — safe to re-run. After this, is_junk() gates
every stage so junk can never re-enter.

  python3 analysis/purge_junk.py
"""
import importlib.util as u, json, os, subprocess

sw = u.module_from_spec(u.spec_from_file_location("sw", "analysis/_stopwords.py"))
u.spec_from_file_location("sw", "analysis/_stopwords.py").loader.exec_module(sw)
is_junk = sw.is_junk

POOL = "analysis/_pool_10k.txt"
# Registries keyed by word (dict) — drop junk keys.
DICTS = ["analysis/_pool_trend.json", "analysis/_houses_elig.json", "analysis/houses.json",
         "analysis/houses_weekly.json", "analysis/houses_daily.json", "analysis/topics.json"]


def purge_pool():
    words = [w.strip() for w in open(POOL) if w.strip()]
    keep = [w for w in words if not is_junk(w)]
    dropped = len(words) - len(keep)
    if dropped:
        open(POOL, "w").write("\n".join(keep) + "\n")
    print(f"  pool: {len(words)} → {len(keep)}  (-{dropped} junk)")


def purge_dict(path):
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    if not isinstance(d, dict):
        return
    junk = [k for k, v in d.items()
            if is_junk(k) or (isinstance(v, dict) and is_junk(v.get("key", "") or v.get("label", "")))]
    for k in junk:
        del d[k]
    if junk:
        json.dump(d, open(path, "w"), indent=1 if path.endswith(("houses.json", "houses_weekly.json", "houses_daily.json", "topics.json")) else None)
    print(f"  {os.path.basename(path)}: -{len(junk)} junk  ({', '.join(sorted(junk)[:8])}{'…' if len(junk) > 8 else ''})")


def main():
    print("[purge_junk] stripping junk keywords…")
    purge_pool()
    for p in DICTS:
        purge_dict(p)
    print("[purge_junk] regenerating published data…")
    subprocess.run(["python3", "analysis/export_research.py"], check=False)
    subprocess.run(["python3", "analysis/build_disciplines.py"], check=False)
    print("[purge_junk] done.")


if __name__ == "__main__":
    main()
