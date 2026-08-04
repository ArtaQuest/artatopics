#!/usr/bin/env python3
"""CITATIONS HARVEST (operator 2026-07-22): the OpenAlex snapshot -> quarterly citation dataset.

Streams the FULL OpenAlex works snapshot (510M records, parquet on s3://openalex, anonymous)
reading ONLY the ~10% of bytes we need (column projection over ranged GETs), and emits per
snapshot file:

  partials/<i>.npz  cohort aggregate: (topic, year*5+q) -> works, cited_by sum, fwci sum/n
                    + received-per-year validation: (topic, year) -> counts_by_year cites
  edges/<i>.bin     the REAL citation graph, 10 bytes/edge: (citing quarter u2, cited work u8)
  map/<i>.bin       (work u8, primary_topic u2) for every topical non-xpac work
  arxiv/<i>.csv     every arXiv work: wid, doi, date, cited_by_count, counts_by_year
  done/<i>.json     checkpoint marker + per-file stats (safe to re-run: skips done files)

Quarter encoding qidx = year*5 + q with q=0 meaning YEAR-ONLY dated (publication_date is
Jan 1 — OpenAlex stores year-precision dates as YYYY-01-01, ~half the corpus at every era;
真 Jan-1 publications are the ~0.3% collateral, stated plainly) and q=1..4 real quarters.
XPAC works (is_xpac, the 2026 extended corpus excluded from default API counts) are dropped.
Works with no primary_topic aggregate under topic sentinel 65534 so totals stay complete.

  python3 analysis/citations/harvest_openalex.py [--workers 10] [--limit N]

Run dir: ~/.artaquest-dev/openalex-quarterly (needs ~35 GB free; transfer ~65 GB).
Then: python3 analysis/citations/build_quarterly.py
"""
import argparse, json, os, sys, time
import multiprocessing as mp

import numpy as np

RUN = os.path.expanduser("~/.artaquest-dev/openalex-quarterly")
COLS = ["id", "doi", "indexed_in", "publication_date", "publication_year", "type",
        "primary_topic", "is_paratext", "is_retracted", "is_xpac",
        "referenced_works", "cited_by_count", "counts_by_year", "fwci"]
PFX = 22          # len('https://openalex.org/W') — id/topic urls sliced then cast
NO_TOPIC = 65534
YMIN, YMAX = 1000, 2030
EDGE_DT = np.dtype([("q", "<u2"), ("w", "<u8")])
MAP_DT = np.dtype([("w", "<u8"), ("t", "<u2")])

_fs = None


def wid_u64(arr):
    """'https://openalex.org/W123' StringArray -> uint64 numpy (0 where null/malformed)."""
    import pyarrow.compute as pc
    s = pc.utf8_slice_codeunits(arr, PFX, 32767)
    ok = pc.match_substring_regex(s, r"^\d+$")
    s = pc.if_else(pc.and_kleene(ok, pc.is_valid(s)), s, "0")
    return pc.cast(s, "uint64").to_numpy(zero_copy_only=False)


def process_file(job):
    idx, url, nrec = job
    import pyarrow.compute as pc
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq
    global _fs
    if _fs is None:
        _fs = pafs.S3FileSystem(anonymous=True, region="us-east-1")
    t0 = time.time()
    f = pq.ParquetFile(_fs.open_input_file(url.replace("s3://", "")), pre_buffer=True)
    tbl = f.read(columns=COLS, use_threads=True)
    n0 = tbl.num_rows

    keep = pc.fill_null(pc.invert(pc.fill_null(tbl.column("is_xpac"), False)), True)
    tbl = tbl.filter(keep)
    n = tbl.num_rows

    wid = wid_u64(tbl.column("id"))
    pt = pc.struct_field(tbl.column("primary_topic"), "id")
    ts = pc.utf8_slice_codeunits(pc.fill_null(pt, ""), PFX, 32767)
    tok = pc.match_substring_regex(ts, r"^\d+$")
    traw = pc.cast(pc.if_else(tok, ts, "75534"), "uint32").to_numpy(zero_copy_only=False)
    topic = np.where((traw >= 10001) & (traw < 75534), traw - 10000, NO_TOPIC).astype(np.uint16)

    # quarter index from publication_date (date32 = epoch days), year fallback -> q0
    pd_col = tbl.column("publication_date").to_numpy(zero_copy_only=False)
    py = tbl.column("publication_year").fill_null(0).to_numpy(zero_copy_only=False).astype(np.int32)
    dt = pd_col.astype("datetime64[D]")
    valid_d = ~np.isnat(dt)
    Y = np.where(valid_d, dt.astype("datetime64[Y]").astype(int) + 1970, py)
    M = np.ones(n, np.int32)
    D = np.ones(n, np.int32)
    M[valid_d] = (dt[valid_d].astype("datetime64[M]").astype(int) % 12) + 1
    D[valid_d] = (dt[valid_d] - dt[valid_d].astype("datetime64[M]")).astype(int) + 1
    q = np.where(valid_d & ~((M == 1) & (D == 1)), (M - 1) // 3 + 1, 0).astype(np.int32)
    inyr = (Y >= YMIN) & (Y <= YMAX)
    qidx = np.where(inyr, Y * 5 + q, 0).astype(np.uint16)     # qidx 0 = undatable, dropped

    cbc = tbl.column("cited_by_count").fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    fw = tbl.column("fwci").to_numpy(zero_copy_only=False).astype(np.float64)
    fw_ok = ~np.isnan(fw)

    # ---- cohort aggregate over (topic, qidx)
    ok = qidx > 0
    key = topic[ok].astype(np.uint32) << np.uint32(16) | qidx[ok].astype(np.uint32)
    uk, inv = np.unique(key, return_inverse=True)
    cw = np.bincount(inv, minlength=len(uk)).astype(np.uint32)
    cs = np.bincount(inv, weights=cbc[ok], minlength=len(uk)).astype(np.uint64)
    cf = np.bincount(inv, weights=np.where(fw_ok[ok], fw[ok], 0.0), minlength=len(uk))
    cfn = np.bincount(inv, weights=fw_ok[ok].astype(np.float64), minlength=len(uk)).astype(np.uint32)

    # ---- received-per-year validation from counts_by_year (last ~10 years only)
    cby = tbl.column("counts_by_year")
    fl = pc.list_flatten(cby)
    par = pc.list_parent_indices(cby).to_numpy(zero_copy_only=False)
    if len(par):
        yy = pc.struct_field(fl, "year").to_numpy(zero_copy_only=False).astype(np.int64)
        cc = pc.struct_field(fl, "cited_by_count").fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        byk = topic[par].astype(np.uint32) << np.uint32(16) | np.clip(yy, 0, 65535).astype(np.uint32)
        ub, binv = np.unique(byk, return_inverse=True)
        bc = np.bincount(binv, weights=cc, minlength=len(ub)).astype(np.uint64)
    else:
        ub = np.zeros(0, np.uint32); bc = np.zeros(0, np.uint64)

    # ---- edges: (citing quarter, cited wid) for dated citing works
    refs = tbl.column("referenced_works")
    rpar = pc.list_parent_indices(refs).to_numpy(zero_copy_only=False)
    redge = np.zeros(0, EDGE_DT)
    if len(rpar):
        rw = wid_u64(pc.list_flatten(refs))
        eok = (qidx[rpar] > 0) & (rw > 0)
        redge = np.empty(int(eok.sum()), EDGE_DT)
        redge["q"] = qidx[rpar][eok]
        redge["w"] = rw[eok]

    # ---- map: wid -> topic (topical works only)
    mok = (topic != NO_TOPIC) & (wid > 0)
    mrec = np.empty(int(mok.sum()), MAP_DT)
    mrec["w"] = wid[mok]
    mrec["t"] = topic[mok]

    # ---- arXiv works: indexed_in contains 'arxiv' OR datacite arXiv doi
    ii = tbl.column("indexed_in")
    ipar = pc.list_parent_indices(ii).to_numpy(zero_copy_only=False)
    ax = np.zeros(n, bool)
    if len(ipar):
        hit = pc.equal(pc.list_flatten(ii), "arxiv").fill_null(False).to_numpy(zero_copy_only=False)
        ax[ipar[hit]] = True
    doi_l = pc.utf8_lower(pc.fill_null(tbl.column("doi"), ""))
    ax |= pc.match_substring(doi_l, "10.48550/arxiv").to_numpy(zero_copy_only=False)
    ax &= wid > 0
    arx_rows = []
    if ax.any():
        ai = np.nonzero(ax)[0]
        pos = {int(a): k for k, a in enumerate(ai)}
        cbys = [""] * len(ai)
        if len(par):
            yy_all = pc.struct_field(fl, "year").to_numpy(zero_copy_only=False)
            cc_all = pc.struct_field(fl, "cited_by_count").fill_null(0).to_numpy(zero_copy_only=False)
            for j in np.nonzero(np.isin(par, ai))[0]:
                k = pos[int(par[j])]
                cbys[k] += ("|" if cbys[k] else "") + f"{yy_all[j]}:{cc_all[j]}"
        dois = doi_l.take(ai).to_pylist()
        for k, a in enumerate(ai):
            d = dois[k].replace("https://doi.org/", "")
            date = f"{Y[a]:04d}-{M[a]:02d}-{D[a]:02d}" if qidx[a] > 0 and q[a] > 0 else ""
            arx_rows.append(f"{wid[a]},{d},{date},{Y[a]},{q[a]},{cbc[a]},{cbys[k]}\n")

    # ---- atomic writes
    def wr(path, data):
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)

    np.savez_compressed(f"{RUN}/partials/{idx}.npz.tmp.npz",
                        ck=uk, cw=cw, cs=cs, cf=cf, cfn=cfn, bk=ub, bc=bc)
    os.replace(f"{RUN}/partials/{idx}.npz.tmp.npz", f"{RUN}/partials/{idx}.npz")
    # edges sorted by cited wid + delta-encoded: ~5x smaller than raw gz AND join-cache-friendly.
    # (files harvested before this change are plain .bin.gz — the build reads both.)
    if len(redge):
        o = np.argsort(redge["w"], kind="stable")
        eq = np.ascontiguousarray(redge["q"][o])
        ew = np.ascontiguousarray(redge["w"][o])
        dw = np.empty_like(ew); dw[0] = ew[0]; dw[1:] = np.diff(ew)
    else:
        eq = np.zeros(0, np.uint16); dw = np.zeros(0, np.uint64)
    np.savez_compressed(f"{RUN}/edges/{idx}.npz.tmp.npz", q=eq, w=dw)
    os.replace(f"{RUN}/edges/{idx}.npz.tmp.npz", f"{RUN}/edges/{idx}.npz")
    wr(f"{RUN}/map/{idx}.bin", mrec.tobytes())
    wr(f"{RUN}/arxiv/{idx}.csv", "".join(arx_rows).encode())
    stats = {"idx": idx, "rows": int(n0), "kept": int(n), "xpac": int(n0 - n),
             "edges": int(len(redge)), "mapped": int(len(mrec)), "arxiv": int(len(arx_rows)),
             "undated": int((qidx == 0).sum()), "secs": round(time.time() - t0, 1)}
    wr(f"{RUN}/done/{idx}.json", json.dumps(stats).encode())
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    man = json.load(open(f"{RUN}/manifest.json"))
    files = sorted(man["files"], key=lambda f: -f["meta"]["content_length"])
    jobs = []
    for i, fobj in enumerate(files):
        if os.path.exists(f"{RUN}/done/{i}.json"):
            continue
        jobs.append((i, fobj["url"], fobj["meta"]["record_count"]))
    total = len(files)
    ndone = total - len(jobs)
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"[harvest] {total} files, {ndone} done, {len(jobs)} queued this pass, "
          f"{man['record_count']:,} records total", flush=True)
    t0, nrec, nfail = time.time(), 0, 0
    with mp.Pool(args.workers) as pool:
        for k, st in enumerate(pool.imap_unordered(process_file_safe, jobs)):
            if st is None:
                nfail += 1
                continue
            nrec += st["rows"]
            if k % 20 == 0 or k == len(jobs) - 1:
                el = time.time() - t0
                done_n = total - len(jobs) + k + 1
                print(f"[harvest] {done_n}/{total} files · {nrec/1e6:.1f}M rows this run · "
                      f"{el/60:.1f} min · {nrec/max(el,1)/1000:.0f}k rows/s · fails {nfail}", flush=True)
    print(f"[harvest] DONE this pass · fails {nfail} (re-run to retry fails)", flush=True)


def process_file_safe(job):
    for attempt in range(3):
        try:
            return process_file(job)
        except Exception as e:  # noqa: BLE001 — S3 hiccups: retry, then leave for a re-run
            time.sleep(5 * (attempt + 1))
            err = e
    sys.stderr.write(f"[harvest] FAIL {job[0]} {job[1]}: {err}\n")
    return None


if __name__ == "__main__":
    main()
