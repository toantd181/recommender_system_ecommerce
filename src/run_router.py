"""Tune content-based, pick the module for each user group on validation, test once.

    PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python src/run_router.py

Stage 1 (history < 17/11, validation 17-23/11): coordinate search for ContentBased (metric: ndcg@10
on `light` users, the new-member group), then for each group (unseen/light/heavy) choose the best of
{PopularRecent, PopularByCategory, ALS, ContentBased, rank-fusion ALS+content} by ndcg@10.
Stage 2 (history < 24/11, test 24-30/11): the router built from those choices is scored once.
ALS settings come from results/als.json (chosen by run_als.py).
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
from baselines import PopularByCategory, PopularRecent  # noqa: E402
from content_based import ContentBased  # noqa: E402
from evalkit import (GROUPS, beyond_accuracy, bootstrap_ci, evaluate, paired_bootstrap,  # noqa: E402
                     three_way_split)
from router import HybridRouter, RankFusion  # noqa: E402

VAL_START, TEST_START = "2019-11-17", "2019-11-24"
COLS = ["event_time", "event_type", "product_id", "category_id", "brand", "price", "user_id"]
LOG: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def group_mean(res, group: str, metric: str = "ndcg@10") -> float:
    pu = res.per_user
    return float(pu.loc[pu["group"] == group, metric].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nov", default="data/nov_u5.parquet")
    ap.add_argument("--als", default="results/als.json")
    ap.add_argument("--out", default="results/router")
    args = ap.parse_args()

    als_cfg = {k: v for k, v in json.loads(Path(args.als).read_text())["chosen"].items() if k != "data"}
    als_cfg["threads"] = 6
    df = pd.read_parquet(args.nov, columns=COLS)
    df["event_type"] = df["event_type"].astype("string")
    df = df.sort_values("event_time", kind="stable").reset_index(drop=True)
    fit_v, val, fit_t, test = three_way_split(df, VAL_START, TEST_START)

    # ---- 1a. content-based tuning ------------------------------------------------------
    log(f"######## STAGE 1a – ContentBased tuning, validation {VAL_START} → {TEST_START}  (metric: ndcg@10, light users)")
    best = dict(w_brand=0.5, w_price=0.5, beta=1.0, cand_days=14)
    grid = [("beta", [0.0, 0.5, 1.0, 2.0, 4.0]), ("w_brand", [0.0, 0.5, 1.0, 2.0]),
            ("w_price", [0.0, 0.5, 1.0, 2.0]), ("cand_days", [7, 14, 30])]
    for key, values in grid:
        scores = {}
        for v in values:
            t0 = time.time()
            r = evaluate(ContentBased(**dict(best, **{key: v})), fit_v, val)
            scores[v] = group_mean(r, "light")
            log(f"  {key}={v!s:>5}  ndcg@10(light)={scores[v]:.4f}  hit@10(light)={group_mean(r, 'light', 'hit@10'):.4f}"
                f"  ndcg@10(heavy)={group_mean(r, 'heavy'):.4f}  [{time.time() - t0:.0f}s]")
        best[key] = max(scores, key=scores.get)
        log(f"-> best {key}: {best[key]}")
    log("CONTENT CHOSEN:", best)

    # ---- 1b. module per group ---------------------------------------------------------
    log("\n######## STAGE 1b – best module per user group (validation, ndcg@10 / hit@10)")

    def build(fit_df):
        als, cb = ALSRecommender(**als_cfg), ContentBased(**best)
        als.fit(fit_df)
        cb.fit(fit_df)
        pr, pc = PopularRecent(7), PopularByCategory(7)
        pr.fit(fit_df)
        pc.fit(fit_df)
        mods = {"PopularRecent": pr, "PopularByCategory": pc, "ALS": als, "ContentBased": cb}
        for wa, wc in [(1, 1), (2, 1), (1, 2)]:
            mods[f"Fusion(ALS×{wa}, Content×{wc})"] = RankFusion([als, cb], [wa, wc])
        return mods

    mods = build(fit_v)
    val_res = {n: evaluate(m, fit_v, val, name=n, fit=False) for n, m in mods.items()}
    rows = {n: {g: f"{group_mean(r, g):.4f} / {group_mean(r, g, 'hit@10'):.4f}" for g in GROUPS}
            for n, r in val_res.items()}
    log(pd.DataFrame(rows).T.to_string())
    chosen = {g: max(val_res, key=lambda n: group_mean(val_res[n], g)) for g in GROUPS}
    log("ROUTING CHOSEN:", chosen)

    # ---- 2. test, single run ------------------------------------------------------------
    log(f"\n######## STAGE 2 – test {TEST_START} → end (single run)")
    mods = build(fit_t)
    router = HybridRouter({g: mods[n] for g, n in chosen.items()})
    router.fit(fit_t)
    res = {"Router": evaluate(router, fit_t, test, name="Router", fit=False)}
    for n in ("PopularByCategory", "ALS", "ContentBased"):
        res[n] = evaluate(mods[n], fit_t, test, name=n, fit=False)
    for n, r in res.items():
        log(r)
    hit = {n: r.per_user.set_index("user_id") for n, r in res.items()}
    log("\nhit@10 / ndcg@10 with 95% CI (all test users with cart/purchase):")
    for n, r in hit.items():
        h, d = bootstrap_ci(r["hit@10"].to_numpy()), bootstrap_ci(r["ndcg@10"].to_numpy())
        log(f"  {n:18s} hit@10 {h[0]:.4f} [{h[1]:.4f}, {h[2]:.4f}]   ndcg@10 {d[0]:.4f} [{d[1]:.4f}, {d[2]:.4f}]")
    for other in ("PopularByCategory", "ALS"):
        for m in ("hit@10", "ndcg@10"):
            d = paired_bootstrap(hit["Router"][m].to_numpy(), hit[other][m].loc[hit["Router"].index].to_numpy())
            log(f"  Router - {other:18s} {m}: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] p={d[3]:.3f}")
    log("  beyond accuracy Router:", {k: round(v, 4) for k, v in
                                      beyond_accuracy(list(res["Router"].recs.values()), fit_t).items()})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out + ".txt").write_text("\n".join(LOG) + "\n")
    Path(args.out + ".json").write_text(json.dumps({"content": best, "routing": chosen, "als": als_cfg},
                                                   indent=1, default=str))
    log("saved", args.out + ".txt")


if __name__ == "__main__":
    main()
