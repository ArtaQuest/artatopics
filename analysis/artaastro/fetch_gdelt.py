#!/usr/bin/env python3
"""
fetch_gdelt.py — build a GLOBAL DAILY AGGREGATE conflict series from the ENTIRE GDELT 1.0
event history (1979-01-01 -> present), for use as the honest, objective ground truth in the
ArtaAstro backtest.

Source: http://data.gdeltproject.org/events/  (GDELT 1.0 Event Database, CC-BY).
  - yearly  files 1979.zip .. 2005.zip
  - monthly files 200601.zip .. 201303.zip
  - daily   files YYYYMMDD.export.CSV.zip  (2013-04-01 -> today)
All share GDELT 1.0 column indices 0..34 (pre-2013 files have 57 cols, 2013+ have 58; the
trailing DATEADDED/SOURCEURL shift does not touch the columns we read):
    col 1  = SQLDATE        (YYYYMMDD, the day the event OCCURRED  -> our aggregation key)
    col 28 = EventRootCode  (CAMEO root; 18/19/20 = assault / fight / mass violence)
    col 29 = QuadClass      (1 verbal-coop, 2 material-coop, 3 verbal-conflict, 4 material-conflict)
    col 30 = GoldsteinScale (-10..+10 conflict<->cooperation)
    col 34 = AvgTone        (article tone, negative = adverse)

We STREAM each zip (download -> parse -> discard), never storing the 52 GB corpus. To avoid
double-counting historical re-reports, each record is bucketed by its SQLDATE and we accept a
record only from its "native" file group:
    * backfile (yearly+monthly): keep SQLDATE <  2013-04-01
    * daily stream            : keep SQLDATE >= 2013-04-01
Resumable: a done-list + a running aggregate are checkpointed so re-runs skip finished files.

Output: out/world_conflict_daily.csv with, per day:
    date, n_events, n_q1, n_q2, n_q3, n_q4, n_root1820, sum_goldstein, sum_tone
(the backtest derives normalized measures — material-conflict share, mean Goldstein, etc.)

Every network call is time-bounded (COORDINATION.md rule). Run in the background; it logs progress.
"""
import os, sys, io, csv, json, time, zipfile, threading, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(HERE, "out")
TMP    = os.environ.get("AQ_GDELT_TMP", "/private/tmp/claude-501/-Users-arash-Studio-artaquest/89a1c066-9c5c-48d0-b9ba-48281569cfb5/scratchpad/gdelt_tmp")
BASE   = "http://data.gdeltproject.org/events/"
SIZES  = os.environ.get("AQ_GDELT_SIZES", os.path.join(TMP, "..", "gdelt_filesizes.txt"))
CKPT   = os.path.join(OUT, "_gdelt_checkpoint.json")
DONE   = os.path.join(OUT, "_gdelt_done.txt")
LOG    = os.path.join(OUT, "_gdelt_fetch.log")
CUTOVER = 20130401           # SQLDATE partition between backfile and daily stream
WORKERS = int(os.environ.get("AQ_GDELT_WORKERS", "10"))
TIMEOUT = 180

os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
_lock = threading.Lock()
# agg[date] = [n, q1, q2, q3, q4, root1820, sum_goldstein, sum_tone]
agg = defaultdict(lambda: [0,0,0,0,0,0,0.0,0.0])
done = set()

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with _lock, open(LOG, "a") as f: f.write(line+"\n")

def load_state():
    if os.path.exists(DONE):
        done.update(l.strip() for l in open(DONE) if l.strip())
    if os.path.exists(CKPT):
        for d, v in json.load(open(CKPT)).items():
            agg[d] = v
    log(f"resume: {len(done)} files done, {len(agg)} days aggregated")

def save_state():
    with _lock:
        tmp = CKPT+".tmp"
        json.dump(agg, open(tmp,"w")); os.replace(tmp, CKPT)

def file_list():
    """Authoritative filenames from the cached directory listing."""
    names=[]
    for line in open(SIZES):
        line=line.strip()
        if not line: continue
        _, name = line.split(None,1)
        if "MASTERREDUCED" in name: continue
        if name.endswith(".zip"): names.append(name)
    # order: yearly, monthly, daily (roughly chronological is fine)
    return names

def parse_zip(data, keep_pred):
    """Aggregate one in-memory zip. keep_pred(sqldate:int)->bool selects the native SQLDATE range."""
    local = defaultdict(lambda: [0,0,0,0,0,0,0.0,0.0])
    z = zipfile.ZipFile(io.BytesIO(data))
    for member in z.namelist():
        for raw in z.read(member).split(b"\n"):
            if not raw: continue
            c = raw.split(b"\t")
            if len(c) < 35: continue
            d = c[1]
            if len(d)!=8 or not d.isdigit(): continue
            di = int(d)
            if not keep_pred(di): continue
            r = local[d.decode()]
            r[0]+=1
            try:
                q=c[29]
                if   q==b"1": r[1]+=1
                elif q==b"2": r[2]+=1
                elif q==b"3": r[3]+=1
                elif q==b"4": r[4]+=1
            except Exception: pass
            try:
                if c[28] in (b"18",b"19",b"20"): r[5]+=1
            except Exception: pass
            try: r[6]+=float(c[30])
            except Exception: pass
            try: r[7]+=float(c[34])
            except Exception: pass
    return local

def fetch_one(name):
    is_daily = name[0:8].isdigit() and name.endswith(".export.CSV.zip")
    keep = (lambda di: di>=CUTOVER) if is_daily else (lambda di: di<CUTOVER)
    url = BASE+name
    for attempt in range(4):
        try:
            req=urllib.request.Request(url, headers={"User-Agent":"ArtaAstro/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data=r.read()
            return name, parse_zip(data, keep)
        except Exception as e:
            if attempt==3:
                log(f"FAIL {name}: {e}")
                return name, None
            time.sleep(2*(attempt+1))

def main():
    load_state()
    names=[n for n in file_list() if n not in done]
    total=len(names)+len(done)
    log(f"files: {total} total, {len(names)} remaining, {WORKERS} workers")
    processed=0; t0=time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs={ex.submit(fetch_one,n):n for n in names}
        for fut in as_completed(futs):
            name, local = fut.result()
            processed+=1
            if local is not None:
                with _lock:
                    for d,v in local.items():
                        a=agg[d]
                        for i in range(6): a[i]+=v[i]
                        a[6]+=v[6]; a[7]+=v[7]
                    done.add(name)
                    with open(DONE,"a") as f: f.write(name+"\n")
            if processed % 25 == 0:
                save_state()
                rate=processed/max(1e-9,time.time()-t0)
                eta=(len(names)-processed)/max(1e-9,rate)
                log(f"{processed}/{len(names)}  days={len(agg)}  {rate:.1f} f/s  ETA {eta/60:.0f}m")
    save_state()
    # write final csv
    csvp=os.path.join(OUT,"world_conflict_daily.csv")
    with open(csvp,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["date","n_events","n_q1","n_q2","n_q3","n_q4","n_root1820","sum_goldstein","sum_tone"])
        for d in sorted(agg):
            v=agg[d]
            iso=f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            w.writerow([iso, v[0],v[1],v[2],v[3],v[4],v[5], f"{v[6]:.3f}", f"{v[7]:.3f}"])
    log(f"DONE. {len(agg)} days -> {csvp}")

if __name__=="__main__":
    main()
