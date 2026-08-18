"""KL divergence of the predicted pie from the true pie, averaged over years — Kaggle metric.

solution / submission: DataFrame with `year` + one column per field, each row a distribution.
Score = mean over rows of KL(true || predicted) in nats. Lower is better. Predictions are
clipped and renormalised so a row that does not sum to 1 is scored on its normalised shape.
"""
import numpy as np, pandas as pd

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    sol = solution.set_index(row_id_column_name).sort_index()
    sub = submission.set_index(row_id_column_name).reindex(sol.index)
    if sub.isnull().any().any():
        raise ValueError("submission is missing rows or columns present in the solution")
    sub = sub[sol.columns]
    P = np.clip(sub.to_numpy(float), 1e-9, None); P = P / P.sum(1, keepdims=True)
    Q = np.clip(sol.to_numpy(float), 0, None); Q = Q / Q.sum(1, keepdims=True)
    kl = (Q * (np.log(np.maximum(Q, 1e-12)) - np.log(P))).sum(1)
    return float(np.mean(kl))
