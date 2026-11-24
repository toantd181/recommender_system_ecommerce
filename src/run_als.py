"""Tune ALS on the validation week, then score the test week once.

    .venv/bin/python src/run_als.py                 # needs data/nov_u5.parquet (+ oct_u5.parquet)

Protocol follows experiments.py: history < 17/11 -> validation 17-23/11 (choose settings);
history < 24/11 -> test 24-30/11 (single run). Selection metric: ndcg@10 over users WITH
train history (ALS cannot serve `unseen` users; the router sends them elsewhere).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from als import ALSRecommender  # noqa: E402
from baselines import PopularByCategory  # noqa: E402
from evalkit import beyond_accuracy, bootstrap_ci, evaluate, paired_bootstrap, three_way_split  # noqa: E402

VAL_START, TEST_START = "2019-11-17", "2019-11-24"
LOG: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def with_history(res) -> pd.DataFrame:
    return res.per_user[res.per_user["group"] != "unseen"]


def score(res, metric="ndcg@10") -> float:
    return float(with_history(res)[metric].mean())


COLS = ["event_time", "event_type", "product_id", "category_id", "user_id"]  # all ALS needs


def load_small(*paths: str) -> pd.DataFrame:
    """Like evalkit.load_events but only the needed columns (memory is tight on this machine)."""
    df = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)
    df["event_type"] = df["event_type"].astype("string")
    return df.sort_values("event_time", kind="stable").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nov", default="data/nov_u5.parquet")
    ap.add_argument("--oct", default="data/oct_u5.parquet")
    ap.add_argument("--out", default="results/als")
    args = ap.parse_args()

    sources = {"nov": (args.nov,), "oct+nov": (args.oct, args.nov)}
    cache: dict[str, pd.DataFrame] = {}

    def get(name: str) -> pd.DataFrame:  # keep at most one dataset in memory
        if name not in cache:
            cache.clear()
            cache[name] = load_small(*sources[name])
        return cache[name]

    base = dict(factors=64, regularization=0.05, alpha=20.0, half_life_days=14.0, filter_seen="purchased")
    grid = [
        ("data", ["nov", "oct+nov"]),
        ("filter_seen", [None, "purchased", "all"]),
        ("half_life_days", [None, 7.0, 14.0, 30.0]),
        ("alpha", [5.0, 20.0, 60.0]),
        ("factors", [32, 64, 128]),
        ("regularization", [0.01, 0.05, 0.2]),
    ]
    best = dict(base, data="nov")

    log(f"######## STAGE 1 – validation {VAL_START} → {TEST_START}")
    # coordinate search: vary one setting at a time, keep the best, move on
    for key, values in grid:
        scores = {}
        for v in values:
            cfg = dict(best, **{key: v})
            fit, val, _, _ = three_way_split(get(cfg["data"]), VAL_START, TEST_START)
            t0 = time.time()
            kw = {k: x for k, x in cfg.items() if k != "data"}
            res = evaluate(ALSRecommender(**kw), fit, val)
            scores[v] = score(res)
            log(f"  {key}={v!s:>8}  data={cfg['data']:<8} ndcg@10(history users)={scores[v]:.4f}  "
                f"hit@10={score(res, 'hit@10'):.4f}  [{time.time() - t0:.0f}s]")
        best[key] = max(scores, key=scores.get)
        log(f"-> best {key}: {best[key]}")
    # baseline on the same validation split / data, for reference
    fit, val, _, _ = three_way_split(get(best["data"]), VAL_START, TEST_START)
    pb = evaluate(PopularByCategory(7), fit, val)
    log(f"validation PopularByCategory ndcg@10(history users)={score(pb):.4f} hit@10={score(pb, 'hit@10'):.4f}")
    log("CHOSEN:", best)

    log(f"\n######## STAGE 2 – test {TEST_START} → end (single run)")
    _, _, fit, test = three_way_split(get(best["data"]), VAL_START, TEST_START)
    kw = {k: x for k, x in best.items() if k != "data"}
    als = evaluate(ALSRecommender(**kw), fit, test, name="ALS")
    pbc = evaluate(PopularByCategory(7), fit, test, name="PopularByCategory")
    log(als)
    log(pbc)
    h_als, h_pbc = with_history(als), with_history(pbc)
    for m in ("hit@10", "ndcg@10", "recall@10"):
        a = bootstrap_ci(h_als[m].to_numpy())
        b = bootstrap_ci(h_pbc[m].to_numpy())
        d = paired_bootstrap(h_als[m].to_numpy(), h_pbc[m].to_numpy())
        log(f"{m} (users with history, n={len(h_als)}): ALS {a[0]:.4f} [{a[1]:.4f}, {a[2]:.4f}] | "
            f"PopularByCategory {b[0]:.4f} [{b[1]:.4f}, {b[2]:.4f}] | diff {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] p={d[3]:.3f}")
    log("beyond accuracy ALS:", {k: round(v, 4) for k, v in beyond_accuracy(list(als.recs.values()), fit).items()})
    log("beyond accuracy PopularByCategory:", {k: round(v, 4) for k, v in beyond_accuracy(list(pbc.recs.values()), fit).items()})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out + ".txt").write_text("\n".join(LOG) + "\n")
    Path(args.out + ".json").write_text(json.dumps({"chosen": best}, default=str, indent=1))
    log("saved", args.out + ".txt")


if __name__ == "__main__":
    main()
