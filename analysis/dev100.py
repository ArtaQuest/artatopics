#!/usr/bin/env python3
"""FAST dev loop: fit a fixed 100-field sample with the CURRENT model (whatever trends_fit.kern_at is) and report R2.
No collection, no deploy, no rebuild. Use to iterate on the model quickly.  python3 analysis/dev100.py"""
import sys, json, numpy as np; sys.path.insert(0, "analysis")
import trends_fit as tf
lon = tf.ephemeris()
d = json.load(open("analysis/_fields_weekly.json"))
labels = [v["label"] for v in d.values() if v.get("res") == "weekly"]
r2 = []
for lab in labels:
    t, y = tf.load_y(lab)
    if y is None: continue
    try: r2.append(tf.fit_topic(y, lon)["r2"])
    except: pass
    if len(r2) >= 100: break
r2 = np.array(r2)
print(f"[dev100] {len(r2)} fields · mean R2 {r2.mean():.3f} · median {np.median(r2):.3f} · >0.5 {100*(r2>0.5).mean():.0f}% · >0.7 {100*(r2>0.7).mean():.0f}%")
