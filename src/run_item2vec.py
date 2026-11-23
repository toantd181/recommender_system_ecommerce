"""Next-item evaluation for session-based (guest) recommenders.

For each test session s (repeats collapsed, len >= 2) and each position t,
query with s[:t] and check whether s[t] is in the Top-K. Metrics: HitRate@K, MRR@K.

Usage:
    .venv/bin/python src/run_item2vec.py data/nov_u5.parquet --test-start 2019-11-24
"""
import argparse
import time

import numpy as np
import pandas as pd

from evalkit import load_events, temporal_split
from item2vec import CoVisitation, CoVisitThenItem2Vec, Item2Vec, PopularSession, session_sequences


def next_item_queries(test: pd.DataFrame, max_sessions: int, seed: int = 0):
    seqs = session_sequences(test)["items"]
    if len(seqs) > max_sessions:
        seqs = seqs.sample(max_sessions, random_state=seed)
    prefixes, targets = [], []
    for items in seqs:
        for t in range(1, len(items)):
            prefixes.append(items[:t])
            targets.append(items[t])
    return prefixes, np.array(targets), len(seqs)


def score(recs: list[list[int]], targets: np.ndarray, ks=(10, 20)) -> dict:
    out = {}
    ranks = np.array([r.index(t) + 1 if t in r else 0 for r, t in zip(recs, targets)])
    for k in ks:
        hit = (ranks > 0) & (ranks <= k)
        out[f"hit@{k}"] = hit.mean()
        out[f"mrr@{k}"] = np.where(hit, 1.0 / np.maximum(ranks, 1), 0.0).mean()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--test-start", default="2019-11-24")
    ap.add_argument("--max-sessions", type=int, default=20000)
    args = ap.parse_args()

    df = load_events(args.parquet)
    train, test = temporal_split(df, args.test_start)
    prefixes, targets, n_sess = next_item_queries(test, args.max_sessions)
    print(f"test sessions: {n_sess:,} | next-item queries: {len(prefixes):,}")

    covisit, i2v_side = CoVisitation(), Item2Vec(side_info=True)
    models = {
        "popular": PopularSession(),
        "covisit": covisit,
        "item2vec": Item2Vec(),
        "item2vec+side_info": i2v_side,
        # reuses the two models above (already fitted by then)
        "covisit->item2vec (share=2)": CoVisitThenItem2Vec(covisit, i2v_side, i2v_share=2),
    }
    rows = {}
    for name, m in models.items():
        t0 = time.time()
        if isinstance(m, CoVisitThenItem2Vec):
            m.popular = m.covisit.popular
        else:
            m.fit(train)
        recs = m.recommend_batch(prefixes, 20)
        rows[name] = score(recs, targets) | {"sec": round(time.time() - t0, 1)}
        print(f"{name}: done in {rows[name]['sec']}s", flush=True)

    # coverage: how often the target item is even known to item2vec
    known = np.mean([t in models["item2vec"].row for t in targets])
    print(f"\ntarget items present in item2vec vocab: {known:.1%}")
    print(pd.DataFrame(rows).T.round(4).to_string())


if __name__ == "__main__":
    main()
