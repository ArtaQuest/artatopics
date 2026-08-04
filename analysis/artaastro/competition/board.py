#!/usr/bin/env python3
"""board.py — the plain-R2 scorer, a faithful mirror of the platform's `Competitions::per_target_r2s`
(metric `r2`): per measure ("target", the part before "|" in a solution key), R2 over its scored
holdout rows (an unpredicted row -> the measure mean, i.e. R2-neutral); leaderboard = mean over
measures. Zero-variance measures are skipped.
"""
import numpy as np


def truth_from_solution(solution):
    tbt = {}
    for k, v in solution.items():
        tp, rid = k.split("|"); tbt.setdefault(tp, {})[int(rid)] = float(v)
    return tbt


def score_r2(truth_by_topic, pred_by_topic_id):
    """truth_by_topic = {topic: {id: target}}; pred_by_topic_id = {topic: {id: value}}.
       Returns (mean_R2 or None, {topic: r2})."""
    per = {}
    for topic, ids in truth_by_topic.items():
        keys = list(ids); ys = np.array([ids[k] for k in keys], float)
        mean = ys.mean(); sst = float(np.sum((ys - mean) ** 2))
        if sst <= 1e-12:
            continue
        pm = pred_by_topic_id.get(topic, {})
        pr = np.array([pm.get(k, mean) for k in keys], float)
        per[topic] = 1.0 - float(np.sum((ys - pr) ** 2)) / sst
    return (float(np.mean(list(per.values()))) if per else None), per
