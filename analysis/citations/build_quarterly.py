#!/usr/bin/env python3
"""BUILD QUARTERLY (operator 2026-07-22): harvested OpenAlex snapshot -> the citation datasets.

Consumes ~/.artaquest-dev/openalex-quarterly (see harvest_openalex.py) and writes
analysis/citations/:

  topics_taxonomy.csv                     4,516 topics -> subfield/field/domain (+names)
  openalex_topic_quarter_cohort.csv.gz    per (topic, year, quarter): works published,
                                          lifetime citations they earned, fwci sum/n
  openalex_topic_quarter_received.csv.gz  per (topic, year, quarter): citations RECEIVED
                                          that quarter (event = citing work's publication
                                          quarter, via the reference graph, full history)
  openalex_{subfield,field,domain,all}_quarter_{cohort,received}.csv.gz   rollups
  openalex_topic_year_received_check.csv.gz   counts_by_year validation (last ~10 yrs)
  arxiv_category_quarter_cohort.csv       arXiv categories: submissions + lifetime cites
  arxiv_category_quarter_received.csv     citations received per category per quarter
  build_stats.json

quarter: 0 = year-only date precision (OpenAlex stores year-only dates as Jan 1; ~half
the corpus at EVERY era — never read q0 as Q1), 1..4 = real quarters.

Designed for a 16 GB machine: the work->topic map (~320M entries) is sorted via 64
on-disk buckets into memory-mapped arrays; every aggregation is dense numpy.

  python3 analysis/citations/build_quarterly.py [--skip-edges]
"""
import argparse, csv, glob, gzip, json, os, re, time, zlib

import numpy as np

RUN = os.path.expanduser("~/.artaquest-dev/openalex-quarterly")
OUT = os.path.dirname(os.path.abspath(__file__))
ARX = os.path.expanduser("~/.artaquest-dev/arxiv-metadata-snapshot")
NO_TOPIC = 65534
EDGE_DT = np.dtype([("q", "<u2"), ("w", "<u8")])
MAP_DT = np.dtype([("w", "<u8"), ("t", "<u2")])
QMAX = 2031 * 5
NROW = 4600          # topic rows: 1..4516 live, 4599 = unclassified/unmapped
UNKNOWN = NROW - 1


def log(msg):
    print(f"[build {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- taxonomy
def taxonomy():
    path = f"{OUT}/topics_taxonomy.csv"
    if not os.path.exists(path):
        import pyarrow.fs as pafs, pyarrow.parquet as pq
        fs = pafs.S3FileSystem(anonymous=True, region="us-east-1")
        man = json.loads(fs.open_input_file("openalex/data/parquet/topics/manifest.json").read())
        rows = []
        for f in man["files"]:
            t = pq.read_table(fs.open_input_file(f["url"].replace("s3://", "")),
                              columns=["id", "display_name", "subfield", "field", "domain"])
            for r in t.to_pylist():
                rows.append([r["id"].rsplit("/T", 1)[1], r["display_name"],
                             r["subfield"]["id"].rsplit("/", 1)[1], r["subfield"]["display_name"],
                             r["field"]["id"].rsplit("/", 1)[1], r["field"]["display_name"],
                             r["domain"]["id"].rsplit("/", 1)[1], r["domain"]["display_name"]])
        rows.sort(key=lambda r: int(r[0]))
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["topic_id", "topic", "subfield_id", "subfield", "field_id", "field",
                        "domain_id", "domain"])
            w.writerows(rows)
        log(f"taxonomy: {len(rows)} topics -> {path}")
    tax = {}
    for r in csv.DictReader(open(path)):
        tax[int(r["topic_id"]) - 10000] = r
    return tax


# ---------------------------------------------------------------- cohort merge (dense)
def merge_partials():
    files = sorted(glob.glob(f"{RUN}/partials/*.npz"))
    log(f"merging {len(files)} partials (dense)")
    N = NROW * QMAX
    W = np.zeros(N, np.int64); S = np.zeros(N, np.int64)
    F = np.zeros(N, np.float64); FN = np.zeros(N, np.int64)
    BY = np.zeros(NROW * 2032, np.int64)
    for p in files:
        d = np.load(p)
        ck = d["ck"]
        flat = (np.minimum(ck >> 16, NROW - 1).astype(np.int64) * QMAX
                + np.minimum(ck & 0xFFFF, QMAX - 1))
        np.add.at(W, flat, d["cw"].astype(np.int64))
        np.add.at(S, flat, d["cs"].astype(np.int64))
        np.add.at(F, flat, d["cf"])
        np.add.at(FN, flat, d["cfn"].astype(np.int64))
        bk = d["bk"]
        if len(bk):
            bflat = (np.minimum(bk >> 16, NROW - 1).astype(np.int64) * 2032
                     + np.minimum(bk & 0xFFFF, 2031))
            np.add.at(BY, bflat, d["bc"].astype(np.int64))
    return (W.reshape(NROW, QMAX), S.reshape(NROW, QMAX), F.reshape(NROW, QMAX),
            FN.reshape(NROW, QMAX), BY.reshape(NROW, 2032))


# ---------------------------------------------------------------- rollups + CSV
def level_defs(tax):
    """[(level, ids, names, topicrow->levelrow index vector)] incl. the topic level itself."""
    out = []
    tids = [str(t + 10000) for t in sorted(tax)]
    tnames = {str(t + 10000): r["topic"] for t, r in tax.items()}
    tvec = np.full(NROW, len(tids), np.int64)
    for i, t in enumerate(sorted(tax)):
        tvec[t] = i
    out.append(("topic", tids + ["none"], {**tnames, "none": "unclassified"}, tvec))
    for lvl in ("subfield", "field", "domain"):
        ids = sorted({r[f"{lvl}_id"] for r in tax.values()}, key=int)
        names = {r[f"{lvl}_id"]: r[lvl] for r in tax.values()}
        ix = {k: i for i, k in enumerate(ids)}
        vec = np.full(NROW, len(ids), np.int64)
        for t, r in tax.items():
            vec[t] = ix[r[f"{lvl}_id"]]
        out.append((lvl, ids + ["none"], {**names, "none": "unclassified"}, vec))
    out.append(("all", ["all"], {"all": "all works"}, np.zeros(NROW, np.int64)))
    return out


def rollup(mat, vec, nrows):
    r = np.zeros((nrows, mat.shape[1]), mat.dtype)
    for tr in range(NROW):
        r[vec[tr]] += mat[tr]
    return r


def write_level_csv(path, lvl, ids, names, mats, cols, fmt=None):
    nz = np.nonzero(mats[0])
    with gzip.open(path, "wt", newline="") if path.endswith(".gz") else open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"{lvl}_id", lvl, "year", "quarter"] + cols)
        for r, qidx in zip(*nz):
            row = [ids[r], names.get(ids[r], "unclassified"), int(qidx) // 5, int(qidx) % 5]
            for j, m in enumerate(mats):
                v = m[r, qidx]
                row.append(round(float(v), 3) if m.dtype == np.float64 else int(v))
            w.writerow(row)
    log(f"{os.path.basename(path)}: {len(nz[0])} rows")


def write_cohort(W, S, F, FN, tax):
    for lvl, ids, names, vec in level_defs(tax):
        n = len(ids)
        mats = [rollup(m, vec, n) for m in (W, S, F, FN)]
        write_level_csv(f"{OUT}/openalex_{lvl}_quarter_cohort.csv.gz", lvl, ids, names,
                        mats, ["works", "cited_by_sum", "fwci_sum", "fwci_n"])


def write_received(M, tax):
    for lvl, ids, names, vec in level_defs(tax):
        mats = [rollup(M, vec, len(ids))]
        write_level_csv(f"{OUT}/openalex_{lvl}_quarter_received.csv.gz", lvl, ids, names,
                        mats, ["citations_received"])


# ---------------------------------------------------------------- map: bucketed disk sort
def load_map():
    kpath, vpath = f"{RUN}/map_keys.npy", f"{RUN}/map_vals.npy"
    if not (os.path.exists(kpath) and os.path.exists(vpath)):
        files = sorted(glob.glob(f"{RUN}/map/*.bin"))
        tot = sum(os.path.getsize(f) for f in files) // MAP_DT.itemsize
        log(f"sorting map: {len(files)} files, {tot / 1e6:.0f}M works, 64 disk buckets")
        bdir = f"{RUN}/map_buckets"
        os.makedirs(bdir, exist_ok=True)
        handles = [open(f"{bdir}/{b}.bin", "wb") for b in range(64)]
        for f in files:
            a = np.frombuffer(open(f, "rb").read(), dtype=MAP_DT)
            b = (a["w"] >> np.uint64(27)).astype(np.int64)
            b[b > 63] = 63
            order = np.argsort(b, kind="stable")
            a = a[order]; b = b[order]
            cuts = np.searchsorted(b, np.arange(65))
            for i in range(64):
                if cuts[i + 1] > cuts[i]:
                    handles[i].write(a[cuts[i]:cuts[i + 1]].tobytes())
        for h in handles:
            h.close()
        keys = np.lib.format.open_memmap(kpath, mode="w+", dtype=np.uint64, shape=(tot,))
        vals = np.lib.format.open_memmap(vpath, mode="w+", dtype=np.uint16, shape=(tot,))
        off = 0
        for b in range(64):
            a = np.frombuffer(open(f"{bdir}/{b}.bin", "rb").read(), dtype=MAP_DT)
            o = np.argsort(a["w"], kind="stable")
            keys[off:off + len(a)] = a["w"][o]
            vals[off:off + len(a)] = a["t"][o]
            off += len(a)
            os.remove(f"{bdir}/{b}.bin")
        keys.flush(); vals.flush()
        os.rmdir(bdir)
        log(f"map sorted -> {kpath}")
    keys = np.load(kpath, mmap_mode="r")
    vals = np.load(vpath, mmap_mode="r")
    log(f"map ready: {len(keys) / 1e6:.0f}M works")
    return keys, vals


# ---------------------------------------------------------------- the big edge join
def read_edges(f):
    """-> (q u16, w u64). npz = sorted+delta-encoded; legacy .bin.gz = raw struct stream."""
    if f.endswith(".npz"):
        d = np.load(f)
        return d["q"], np.cumsum(d["w"], dtype=np.uint64)
    e = np.frombuffer(zlib.decompress(open(f, "rb").read(), 47), dtype=EDGE_DT)
    return e["q"], e["w"]


def edge_join(keys, vals, arx_keys, arx_off, arx_cats, ncat):
    M = np.zeros((NROW, QMAX), np.int64)
    A = np.zeros((max(ncat, 1), QMAX), np.int64)
    files = sorted(glob.glob(f"{RUN}/edges/*.npz") + glob.glob(f"{RUN}/edges/*.bin.gz"),
                   key=lambda p: int(re.findall(r"(\d+)\.", os.path.basename(p))[0]))
    tot_e = matched = arx_hits = 0
    t0 = time.time()
    for i, f in enumerate(files):
        eq, w = read_edges(f)
        if not len(w):
            continue
        tot_e += len(w)
        idx = np.searchsorted(keys, w)
        idx[idx >= len(keys)] = 0
        hit = keys[idx] == w
        t = np.where(hit, vals[idx].astype(np.int64), UNKNOWN)
        t[t >= NROW] = UNKNOWN
        np.add.at(M, (t, eq.astype(np.int64)), 1)
        matched += int(hit.sum())
        if len(arx_keys):
            ai = np.searchsorted(arx_keys, w)
            ai[ai >= len(arx_keys)] = 0
            ah = arx_keys[ai] == w
            if ah.any():
                arx_hits += int(ah.sum())
                aw = ai[ah]; qh = eq[ah].astype(np.int64)
                starts = arx_off[aw]; cnt = (arx_off[aw + 1] - starts).astype(np.int64)
                total = int(cnt.sum())
                if total:
                    qrep = np.repeat(qh, cnt)
                    cum = np.cumsum(cnt)
                    pos = np.arange(total) - np.repeat(cum - cnt, cnt) + np.repeat(starts, cnt)
                    np.add.at(A, (arx_cats[pos], qrep), 1)
        if i % 200 == 0:
            log(f"edges {i}/{len(files)} · {tot_e / 1e9:.2f}B · match {matched / max(tot_e, 1) * 100:.1f}% "
                f"· arXiv {arx_hits / 1e6:.1f}M · {(time.time() - t0) / 60:.1f} min")
    log(f"edge join done: {tot_e / 1e9:.2f}B edges, matched {matched / max(tot_e, 1) * 100:.2f}%, "
        f"arXiv {arx_hits / 1e6:.1f}M")
    return M, A, tot_e, matched, arx_hits


# ---------------------------------------------------------------- arXiv layer
MON = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def rfc_quarter(s):
    try:
        p = s.split()
        return int(p[3]), (MON[p[2]] - 1) // 3 + 1
    except Exception:  # noqa: BLE001
        return None, None


def build_arxiv():
    import pyarrow.parquet as pq
    ids, dois, cats_list, dates = [], [], [], []
    for f in sorted(glob.glob(f"{ARX}/train-*.parquet")):
        t = pq.read_table(f, columns=["id", "doi", "categories", "versions"])
        ids += t.column("id").to_pylist()
        dois += t.column("doi").to_pylist()
        cats_list += t.column("categories").to_pylist()
        dates += [v[0]["created"] if v else None for v in t.column("versions").to_pylist()]
    log(f"arXiv corpus: {len(ids)} papers")
    all_cats = sorted({c for cs in cats_list if cs for c in cs.split()})
    cat_ix = {c: i for i, c in enumerate(all_cats)}
    log(f"arXiv categories: {len(all_cats)}")
    yq = [rfc_quarter(d) if d else (None, None) for d in dates]

    harv = {}
    for f in glob.glob(f"{RUN}/arxiv/*.csv"):
        for line in open(f):
            p = line.rstrip("\n").split(",")
            if len(p) >= 6 and p[1]:
                harv[p[1]] = (int(p[0]), int(p[5]))
    log(f"harvested arXiv-linked OpenAlex works with a DOI: {len(harv)}")

    paper_wids = [()] * len(ids)
    paper_cited = np.zeros(len(ids), np.int64)
    matched = 0
    for i, (aid, jdoi) in enumerate(zip(ids, dois)):
        hits = []
        h1 = harv.get(f"10.48550/arxiv.{aid.lower()}")
        if h1:
            hits.append(h1)
        if jdoi:
            h2 = harv.get(jdoi.lower().strip())
            if h2 and (not h1 or h2[0] != h1[0]):
                hits.append(h2)
        if hits:
            paper_wids[i] = tuple(h[0] for h in hits)
            paper_cited[i] = sum(h[1] for h in hits)
            matched += 1
    log(f"arXiv match: {matched}/{len(ids)} papers ({matched / len(ids) * 100:.1f}%)")

    cells = {}
    for i, (cs, (y, q)) in enumerate(zip(cats_list, yq)):
        if not cs or y is None:
            continue
        for c in cs.split():
            a = cells.setdefault((c, y, q), [0, 0, 0])
            a[0] += 1
            if paper_wids[i]:
                a[1] += 1; a[2] += int(paper_cited[i])
    with open(f"{OUT}/arxiv_category_quarter_cohort.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "year", "quarter", "submissions", "matched_openalex",
                    "cited_by_sum"])
        for (c, y, q), (n, m, s) in sorted(cells.items()):
            w.writerow([c, y, q, n, m, s])
    log(f"arXiv cohort: {len(cells)} cells")

    wid_list, cat_arrs = [], []
    for i, ws in enumerate(paper_wids):
        if not ws or not cats_list[i]:
            continue
        cl = np.array([cat_ix[c] for c in cats_list[i].split()], np.int64)
        for wd in ws:
            wid_list.append(wd); cat_arrs.append(cl)
    wid_arr = np.array(wid_list, np.uint64)
    order = np.argsort(wid_arr, kind="stable")
    arx_keys = wid_arr[order]
    catlists = [cat_arrs[int(j)] for j in order]
    arx_off = np.zeros(len(catlists) + 1, np.int64)
    arx_off[1:] = np.cumsum([len(x) for x in catlists])
    arx_cats = np.concatenate(catlists) if catlists else np.zeros(0, np.int64)
    return arx_keys, arx_off, arx_cats, all_cats


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-edges", action="store_true")
    args = ap.parse_args()
    tax = taxonomy()

    W, S, F, FN, BY = merge_partials()
    write_cohort(W, S, F, FN, tax)
    with gzip.open(f"{OUT}/openalex_topic_year_received_check.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["topic_id", "year", "citations_received_openalex_counts_by_year"])
        for tr, yr in zip(*np.nonzero(BY)):
            w.writerow([int(tr) + 10000 if tr < UNKNOWN else "none", int(yr), int(BY[tr, yr])])
    log("counts_by_year check written")
    if args.skip_edges:
        log("skip-edges: DONE")
        return

    arx_keys, arx_off, arx_cats, all_cats = build_arxiv()
    keys, vals = load_map()
    M, A, tot_e, matched, arx_hits = edge_join(keys, vals, arx_keys, arx_off, arx_cats,
                                               len(all_cats))
    del keys, vals
    write_received(M, tax)
    with open(f"{OUT}/arxiv_category_quarter_received.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "year", "quarter", "citations_received"])
        for c, q in zip(*np.nonzero(A)):
            w.writerow([all_cats[c], int(q) // 5, int(q) % 5, int(A[c, q])])
    log("arXiv received written")
    json.dump({"edges": int(tot_e), "matched": int(matched), "arx_hits": int(arx_hits),
               "snapshot": "2026-06-26"}, open(f"{OUT}/build_stats.json", "w"), indent=1)
    log("ALL DONE")


if __name__ == "__main__":
    main()
