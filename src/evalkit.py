"""Offline evaluation framework: temporal split, ranking metrics, per-group reports.

Any parquet with the REES46 columns works (event_time, event_type, product_id,
category_id, brand, price, user_id, user_session).

A recommender is any object with:
    fit(train: DataFrame) -> None
    recommend(user_ids: list[int], k: int) -> dict[int, list[int]]
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

EVENT_WEIGHTS = {"view": 1.0, "cart": 3.0, "purchase": 5.0}
POSITIVE_EVENTS = ("cart", "purchase")  # purchase alone is too sparse (see progress report)
GROUPS = ("unseen", "light", "heavy")


def load_events(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["event_type"] = df["event_type"].astype("string")
    return df.sort_values("event_time", kind="stable").reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame, test_start: str, test_end: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train = events before test_start; test = events in [test_start, test_end)."""
    t0 = pd.Timestamp(test_start)
    train = df[df["event_time"] < t0]
    test = df[df["event_time"] >= t0]
    if test_end is not None:
        test = test[test["event_time"] < pd.Timestamp(test_end)]
    return train, test


def user_groups(train: pd.DataFrame, users, heavy_min: int = 5) -> pd.Series:
    """unseen: no train events | light: 1..heavy_min-1 | heavy: >= heavy_min."""
    counts = train.groupby("user_id").size()
    n = counts.reindex(list(users)).fillna(0).astype(int)
    out = pd.Series("light", index=n.index, dtype="object")
    out[n == 0] = "unseen"
    out[n >= heavy_min] = "heavy"
    return out


def ground_truth(test: pd.DataFrame, positives=POSITIVE_EVENTS) -> dict[int, set[int]]:
    pos = test[test["event_type"].isin(positives)]
    return pos.groupby("user_id")["product_id"].agg(set).to_dict()


# ---- metrics (single user) -------------------------------------------------
# Definitions follow the course slides L8 "Đánh giá hệ thống gợi ý" (IT4613) where
# they differ from other common variants; slide numbers are noted on each metric.

def recall_at_k(recs: list[int], truth: set[int], k: int) -> float:
    """L8 slide 12: TP_u / min(P_u, K), so a user with more than K positives can still reach 1."""
    return len(set(recs[:k]) & truth) / min(len(truth), k)


def precision_at_k(recs: list[int], truth: set[int], k: int) -> float:
    """L8 slide 9: hits in the Top-K / K."""
    return len(set(recs[:k]) & truth) / k


def f1_at_k(recs: list[int], truth: set[int], k: int) -> float:
    """L8 slide 11: harmonic mean of precision@K and recall@K."""
    p, r = precision_at_k(recs, truth, k), recall_at_k(recs, truth, k)
    return 2 * p * r / (p + r) if p + r else 0.0


def ap_at_k(recs: list[int], truth: set[int], k: int) -> float:
    """L8 slides 24-25: mean of precision@rank over the ranks that are hits
    (0 when there is no hit). MAP = mean over users."""
    hits, total = 0, 0.0
    for rank, p in enumerate(recs[:k], 1):
        if p in truth:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


def rankscore_at_k(recs: list[int], truth: set[int], k: int, alpha: float = 2.0) -> float:
    """L8 slide 19: sum_{hits} 2^{-(rank-1)/alpha} / sum_{i=1..|T|} 2^{-(i-1)/alpha};
    alpha = ranking half-life (exponential decay with rank)."""
    got = sum(2 ** (-(r - 1) / alpha) for r, p in enumerate(recs[:k], 1) if p in truth)
    best = sum(2 ** (-(i - 1) / alpha) for i in range(1, len(truth) + 1))
    return got / best


def liftindex_at_k(recs: list[int], truth: set[int], k: int) -> float:
    """L8 slide 20: the list is split into 10 equal parts weighted 1, 0.9, ..., 0.1
    (linear decay); liftindex = weighted hits / number of hits (0 when no hit).
    For a list of length L, rank r gets weight 1 - (r-1)/L (slide 22: L=5 gives 0.8, 0.6, 0.4)."""
    L = len(recs[:k]) or k
    w = [1 - (r - 1) / L for r, p in enumerate(recs[:k], 1) if p in truth]
    return sum(w) / len(w) if w else 0.0


def hit_at_k(recs: list[int], truth: set[int], k: int) -> float:
    return float(any(p in truth for p in recs[:k]))


def mrr_at_k(recs: list[int], truth: set[int], k: int) -> float:
    for rank, p in enumerate(recs[:k], 1):
        if p in truth:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(recs: list[int], truth: set[int], k: int) -> float:
    """L8 slide 21 DCG. IDCG here uses min(|T|, K) ideal positions (standard), whereas
    the slide sums over the h hits; both agree on the slide's example (h = |T| = 3).
    With the slide's version a list whose few hits sit at the top scores 1 no matter how
    many relevant items it missed, so the standard IDCG is kept."""
    dcg = sum(1.0 / math.log2(r + 1) for r, p in enumerate(recs[:k], 1) if p in truth)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(truth), k) + 1))
    return dcg / idcg


METRICS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "f1": f1_at_k,
    "hit": hit_at_k,
    "mrr": mrr_at_k,
    "map": ap_at_k,
    "ndcg": ndcg_at_k,
    "rankscore": rankscore_at_k,
    "liftindex": liftindex_at_k,
}


def three_way_split(df: pd.DataFrame, val_start: str, test_start: str, test_end: str | None = None):
    """Returns (fit_for_val, val, fit_for_test, test).

    Choose hyper-parameters with models fit on `fit_for_val` and scored on `val`;
    then refit on `fit_for_test` (= everything before test_start) and score `test` once.
    """
    fit_val, val = temporal_split(df, val_start, test_start)
    fit_test, test = temporal_split(df, test_start, test_end)
    return fit_val, val, fit_test, test


# ---- uncertainty -------------------------------------------------------------
def _cluster_sums(values, clusters):
    values = np.asarray(values, dtype=float)
    if clusters is None:
        return values, np.ones_like(values)
    codes, _ = pd.factorize(pd.Series(clusters))
    return np.bincount(codes, weights=values), np.bincount(codes).astype(float)


def bootstrap_ci(values, clusters=None, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Mean with a (1-alpha) percentile CI. Resamples whole clusters (sessions /
    baskets) when given, since queries from the same cluster are correlated."""
    s, c = _cluster_sums(values, clusters)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return s.sum() / c.sum(), lo, hi


def paired_bootstrap(a, b, clusters=None, n_boot: int = 1000, seed: int = 0):
    """Difference mean(a) - mean(b) on the same queries: (diff, lo, hi, p_one_sided),
    where p = share of resamples with diff <= 0."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    s, c = _cluster_sums(d, clusters)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return s.sum() / c.sum(), lo, hi, float((boots <= 0).mean())


# ---- beyond accuracy ---------------------------------------------------------
def beyond_accuracy(rec_lists, train: pd.DataFrame, k: int = 10) -> dict:
    """coverage: share of train catalogue ever recommended in the Top-K;
    novelty: mean -log2(share of train events) of recommended items (higher = less mainstream);
    pop_share: share of recommendations that come from the top-1% most popular items."""
    pop = train.groupby("product_id").size()
    p = pop / pop.sum()
    top1 = set(pop.nlargest(max(1, len(pop) // 100)).index)
    recs = [i for r in rec_lists for i in r[:k]]
    if not recs:
        return {"coverage": 0.0, "novelty": 0.0, "pop_share": 0.0}
    floor = 1.0 / pop.sum()
    return {
        "coverage": len(set(recs) & set(pop.index)) / len(pop),
        "novelty": float(np.mean([-np.log2(p.get(i, floor)) for i in recs])),
        "pop_share": float(np.mean([i in top1 for i in recs])),
    }


@dataclass
class EvalResult:
    name: str
    table: pd.DataFrame  # index: group (incl. "all"), columns: n_users + metric@k
    per_user: pd.DataFrame | None = None
    recs: dict | None = None

    def __str__(self) -> str:
        return f"== {self.name} ==\n{self.table.round(4).to_string()}"

    def ci(self, metric: str = "hit@10") -> tuple[float, float, float]:
        return bootstrap_ci(self.per_user[metric].to_numpy())


def evaluate(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    ks=(10, 20),
    name: str | None = None,
    fit: bool = True,
) -> EvalResult:
    """Fit on train, score on test positives, report overall and per user group."""
    if fit:
        model.fit(train)
    truth = ground_truth(test)
    users = list(truth)
    groups = user_groups(train, users)
    recs = model.recommend(users, max(ks))

    rows = []
    for u in users:
        r = recs.get(u, [])
        row = {"user_id": u, "group": groups[u]}
        for k in ks:
            for m, fn in METRICS.items():
                row[f"{m}@{k}"] = fn(r, truth[u], k)
        rows.append(row)
    per_user = pd.DataFrame(rows)

    cols = [c for c in per_user.columns if "@" in c]
    by_group = per_user.groupby("group")[cols].mean().reindex(list(GROUPS))
    by_group.insert(0, "n_users", per_user.groupby("group").size().reindex(list(GROUPS)))
    overall = per_user[cols].mean().to_frame("all").T
    overall.insert(0, "n_users", len(per_user))
    table = pd.concat([overall, by_group.dropna(how="all")])
    return EvalResult(name or type(model).__name__, table, per_user, recs)


def weighted_scores(df: pd.DataFrame) -> pd.Series:
    """Per-event score: view 1, cart 3, purchase 5 (matches the project outline)."""
    return df["event_type"].map(EVENT_WEIGHTS).fillna(0.0)
