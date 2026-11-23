"""Full experiment: choose settings on a validation week, score the test week once.

    fit_val  = everything before VAL_START      val  = [VAL_START, TEST_START)
    fit_test = everything before TEST_START     test = [TEST_START, end]

Stage 1 (validation): data (Nov only vs Oct+Nov), co-visitation variants, hybrid
share, FP-Growth settings. Stage 2 (test): chosen settings vs baselines with 95%
cluster-bootstrap CIs, paired differences and beyond-accuracy metrics.

Usage:
    .venv/bin/python src/experiments.py --nov data/nov_u5.parquet --octnov data/octnov_u5.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import PopularByCategory, PopularRecent
from evalkit import (
    beyond_accuracy, bootstrap_ci, evaluate, load_events, paired_bootstrap, three_way_split,
)
from fbt import FrequentlyBoughtTogether, make_baskets
from item2vec import CoVisitation, CoVisitThenItem2Vec, Item2Vec, PopularSession, session_sequences

VAL_START, TEST_START = "2019-11-17", "2019-11-24"
K = 10
OUT: dict = {}


def log(*a):
    print(*a, flush=True)


# ---- per-query scoring -------------------------------------------------------
def session_queries(df_eval: pd.DataFrame, max_sessions: int, seed: int = 0):
    seqs = session_sequences(df_eval)["items"]
    if len(seqs) > max_sessions:
        seqs = seqs.sample(max_sessions, random_state=seed)
    P, T, C = [], [], []
    for sid, items in enumerate(seqs):
        for t in range(1, len(items)):
            P.append(items[:t]); T.append(items[t]); C.append(sid)
    return P, np.array(T), np.array(C)


def hit_rr(recs, targets, k=K):
    rank = np.array([r.index(t) + 1 if t in r[:k] else 0 for r, t in zip(recs, targets)])
    return (rank > 0).astype(float), np.where(rank > 0, 1.0 / np.maximum(rank, 1), 0.0)


def fbt_queries(baskets):
    Q, T, C = [], [], []
    for bid, b in enumerate(baskets):
        for q in b:
            Q.append(q); T.append(set(b) - {q}); C.append(bid)
    return Q, T, np.array(C)


def fbt_hits(rec_fn, Q, T, k=K):
    recs = [rec_fn(q) for q in Q]
    hit = np.array([float(any(p in t for p in r[:k])) for r, t in zip(recs, T)])
    return hit, recs


def fbt_systems(m: FrequentlyBoughtTogether):
    def fbt(order):
        def run(q):
            m.same_cat_first = order
            return [i for i, _, _ in m.recommend_items([q], K)]
        return run
    return {
        "fbt (same-cat first)": fbt(True),
        "fbt (cat-rules first)": fbt(False),
        "popular_same_cat": lambda q: [i for i in m.cat_top.get(m.item_cat.get(q), []) if i != q][:K],
        "popular_global": lambda q: [i for i in m.global_top if i != q][:K],
    }


def fmt_ci(mean, lo, hi):
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


# ---- stage 1: validation -----------------------------------------------------
def stage_validation(nov: pd.DataFrame, octnov: pd.DataFrame, max_sessions: int) -> dict:
    log("\n######## STAGE 1 – validation week", VAL_START, "→", TEST_START)
    chosen = {}

    # 1a. does October help?  (val queries are identical: they come from November)
    log("\n-- 1a. Nov only vs Oct+Nov as history")
    rows = {}
    for name, df in [("nov", nov), ("oct+nov", octnov)]:
        fit, val, _, _ = three_way_split(df, VAL_START, TEST_START)
        P, T, C = session_queries(val, max_sessions)
        cv = CoVisitation().fit(fit)
        h, _ = hit_rr(cv.recommend_batch(P, K), T)
        m = FrequentlyBoughtTogether().fit(fit)
        Q, TT, _ = fbt_queries(make_baskets(val))
        fh, _ = fbt_hits(fbt_systems(m)["fbt (same-cat first)"], Q, TT)
        ev = evaluate(PopularByCategory(7), fit, val)
        rows[name] = {
            "session covisit hit@10": h.mean(),
            "fbt hit@10": fh.mean(),
            "fbt product rules": len(m.rules),
            "fbt queries with rule": np.mean([bool(m._prod_idx.get(frozenset({q}))) for q in Q]),
            "users unseen share": (ev.per_user["group"] == "unseen").mean(),
            "PopularByCategory hit@10": ev.table.loc["all", "hit@10"],
        }
    t = pd.DataFrame(rows)
    log(t.round(4).to_string())
    OUT["val_data"] = t.to_dict()
    # each module picks its history by its own validation metric
    def pick(metric):
        return "oct+nov" if rows["oct+nov"][metric] > rows["nov"][metric] else "nov"

    chosen["data_users"] = pick("PopularByCategory hit@10")
    chosen["data_session"] = pick("session covisit hit@10")
    chosen["data_fbt"] = pick("fbt hit@10")
    log("chosen data:", {k: v for k, v in chosen.items() if k.startswith("data_")})

    data = {"nov": nov, "oct+nov": octnov}
    fit, val, _, _ = three_way_split(data[chosen["data_session"]], VAL_START, TEST_START)
    P, T, C = session_queries(val, max_sessions)

    # 1b. co-visitation variants
    log("\n-- 1b. co-visitation variants (session next-item, val)")
    variants = {
        "plain w3": dict(),
        "plain w5": dict(window=5),
        "type": dict(type_weighted=True),
        "time": dict(time_weighted=True),
        "type+time": dict(type_weighted=True, time_weighted=True),
        "type+time w5": dict(type_weighted=True, time_weighted=True, window=5),
    }
    rows, fitted = {}, {}
    for name, kw in variants.items():
        t0 = time.time()
        m = CoVisitation(**kw).fit(fit)
        recs = m.recommend_batch(P, 20)
        h10, rr = hit_rr(recs, T, 10)
        h20, _ = hit_rr(recs, T, 20)
        rows[name] = {"hit@10": h10.mean(), "mrr@10": rr.mean(), "hit@20": h20.mean(), "sec": round(time.time() - t0)}
        fitted[name] = m
    t = pd.DataFrame(rows).T
    log(t.round(4).to_string())
    OUT["val_covisit"] = t.to_dict()
    best = t["hit@10"].idxmax()
    chosen["covisit"] = variants[best]
    log("chosen co-visitation:", best, variants[best])

    # 1c. hybrid share with the chosen co-visitation
    log("\n-- 1c. hybrid co-visitation -> Item2Vec, slots reserved for Item2Vec")
    i2v = Item2Vec(side_info=True).fit(fit)
    rows = {}
    for share in (0, 2, 4):
        h = CoVisitThenItem2Vec(fitted[best], i2v, i2v_share=share)
        for k in (10, 20):
            recs = h.recommend_batch(P, k)  # share applies within the K actually shown
            hk, rr = hit_rr(recs, T, k)
            rows.setdefault(f"share={share}", {})[f"hit@{k}"] = hk.mean()
            rows[f"share={share}"][f"mrr@{k}"] = rr.mean()
    t = pd.DataFrame(rows).T
    log(t.round(4).to_string())
    OUT["val_hybrid"] = t.to_dict()
    chosen["share"] = int(t["hit@10"].idxmax().split("=")[1])
    log("chosen share (by hit@10):", chosen["share"])

    # 1d. FP-Growth settings
    log(f"\n-- 1d. FP-Growth settings (leave-one-out on val baskets, history: {chosen['data_fbt']})")
    fit, val, _, _ = three_way_split(data[chosen["data_fbt"]], VAL_START, TEST_START)
    Q, TT, _ = fbt_queries(make_baskets(val))
    rows = {}
    for mc in (2, 3, 5):
        m = FrequentlyBoughtTogether(min_count=mc).fit(fit)
        for name, fn in fbt_systems(m).items():
            if name.startswith("fbt"):
                rows[f"min_count={mc} {name}"] = {"hit@10": fbt_hits(fn, Q, TT)[0].mean(), "rules": len(m.rules)}
    t = pd.DataFrame(rows).T
    log(t.round(4).to_string())
    OUT["val_fbt"] = t.to_dict()
    best = t["hit@10"].idxmax()
    chosen["fbt_min_count"] = int(best.split()[0].split("=")[1])
    chosen["fbt_same_cat_first"] = "same-cat first" in best
    log("chosen FP-Growth:", best)
    return chosen


# ---- stage 2: test -----------------------------------------------------------
def stage_test(data: dict, chosen: dict, max_sessions: int) -> None:
    log("\n######## STAGE 2 – test week", TEST_START, "→ end (single run with chosen settings)")

    # 2a. user-level Top-K (baselines; ALS / content-based slot in here later)
    _, _, fit, test = three_way_split(data[chosen["data_users"]], VAL_START, TEST_START)
    log(f"\n-- 2a. user Top-K, hit@10 with 95% CI (history: {chosen['data_users']})")
    res = {n: evaluate(m, fit, test) for n, m in
           [("PopularRecent", PopularRecent(7)), ("PopularByCategory", PopularByCategory(7))]}
    rows = {}
    for n, r in res.items():
        rows[n] = {"hit@10": fmt_ci(*r.ci("hit@10")), "ndcg@10": fmt_ci(*r.ci("ndcg@10"))}
        for g in ("unseen", "light", "heavy"):
            rows[n][f"hit@10 {g}"] = round(r.table.loc[g, "hit@10"], 4)
        rows[n] |= {k: round(v, 4) for k, v in beyond_accuracy(list(r.recs.values()), fit).items()}
    t = pd.DataFrame(rows).T
    log(t.to_string())
    d = paired_bootstrap(res["PopularByCategory"].per_user["hit@10"], res["PopularRecent"].per_user["hit@10"])
    log(f"PopularByCategory - PopularRecent: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}], p={d[3]:.3f}")
    OUT["test_users"] = t.to_dict()

    # 2b. session-based (guest)
    _, _, fit, test = three_way_split(data[chosen["data_session"]], VAL_START, TEST_START)
    log(f"\n-- 2b. session next-item, 95% CI (cluster bootstrap over sessions; history: {chosen['data_session']})")
    P, T, C = session_queries(test, max_sessions)
    cv_plain = CoVisitation().fit(fit)
    cv_best = CoVisitation(**chosen["covisit"]).fit(fit) if chosen["covisit"] else cv_plain
    i2v = Item2Vec(side_info=True).fit(fit)
    systems = {
        "popular": PopularSession().fit(fit),
        "item2vec+side_info": i2v,
        "covisit (plain)": cv_plain,
        f"covisit (chosen {chosen['covisit'] or 'plain'})": cv_best,
        f"hybrid share={chosen['share']}": CoVisitThenItem2Vec(cv_best, i2v, i2v_share=chosen["share"]),
    }
    hits, rows = {}, {}
    for n, m in systems.items():
        if isinstance(m, CoVisitThenItem2Vec):
            m.popular = m.covisit.popular
        recs = m.recommend_batch(P, K)
        h, rr = hit_rr(recs, T)
        hits[n] = h
        rows[n] = {"hit@10": fmt_ci(*bootstrap_ci(h, C)), "mrr@10": fmt_ci(*bootstrap_ci(rr, C))}
        rows[n] |= {k: round(v, 4) for k, v in beyond_accuracy(recs, fit).items()}
    t = pd.DataFrame(rows).T
    log(t.to_string())
    OUT["test_session"] = t.to_dict()
    final = list(systems)[-1]
    for other in ("covisit (plain)", "item2vec+side_info"):
        d = paired_bootstrap(hits[final], hits[other], C)
        log(f"{final} - {other}: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}], p={d[3]:.3f}")

    # 2c. frequently bought together
    _, _, fit, test = three_way_split(data[chosen["data_fbt"]], VAL_START, TEST_START)
    log(f"\n-- 2c. frequently bought together, 95% CI (cluster bootstrap over baskets; history: {chosen['data_fbt']})")
    m = FrequentlyBoughtTogether(min_count=chosen["fbt_min_count"]).fit(fit)
    Q, TT, CB = fbt_queries(make_baskets(test))
    sys_ = fbt_systems(m)
    chosen_name = "fbt (same-cat first)" if chosen["fbt_same_cat_first"] else "fbt (cat-rules first)"
    hits, rows, recs_by = {}, {}, {}
    for n, fn in sys_.items():
        h, recs = fbt_hits(fn, Q, TT)
        hits[n], recs_by[n] = h, recs
        rows[n] = {"hit@10": fmt_ci(*bootstrap_ci(h, CB))}
        rows[n] |= {k: round(v, 4) for k, v in beyond_accuracy(recs, fit).items()}
    t = pd.DataFrame(rows).T
    log(f"test baskets: {CB.max() + 1:,}, queries: {len(Q):,}, product rules: {len(m.rules):,}")
    log(t.to_string())
    d = paired_bootstrap(hits[chosen_name], hits["popular_same_cat"], CB)
    log(f"{chosen_name} - popular_same_cat: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}], p={d[3]:.3f}")
    OUT["test_fbt"] = t.to_dict()

    # 2d. substitutes vs complements
    log("\n-- 2d. substitutes vs complements (share of pairs in the SAME category_id)")
    cat = m.item_cat
    truth_same = np.mean([cat.get(q) == cat.get(p) for q, t in zip(Q, TT) for p in t if q in cat and p in cat])
    log(f"test baskets, query vs other basket items: {truth_same:.1%} same category")
    for n in (chosen_name, "popular_same_cat"):
        r = [cat.get(q) == cat.get(p) for q, rr in zip(Q, recs_by[n]) for p in rr if q in cat and p in cat]
        log(f"{n} recommendations: {np.mean(r):.1%} same category")
    rule_same = np.mean([cat.get(next(iter(a))) == cat.get(next(iter(c)))
                         for a, c in zip(m.rules.antecedents, m.rules.consequents)
                         if len(a) == 1 and len(c) == 1])
    log(f"FP-Growth product rules: {rule_same:.1%} same category")
    OUT["substitutes"] = {"truth_same_cat": truth_same, "rules_same_cat": rule_same}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nov", default="data/nov_u5.parquet")
    ap.add_argument("--octnov", default="data/octnov_u5.parquet")
    ap.add_argument("--max-sessions", type=int, default=20000)
    ap.add_argument("--out", default="results/experiments.json")
    args = ap.parse_args()

    nov, octnov = load_events(args.nov), load_events(args.octnov)
    chosen = stage_validation(nov, octnov, args.max_sessions)
    OUT["chosen"] = chosen
    log("\nCHOSEN:", chosen)
    stage_test({"nov": nov, "oct+nov": octnov}, chosen, args.max_sessions)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(OUT, indent=1, default=str, ensure_ascii=False))
    log("\nsaved", args.out)


if __name__ == "__main__":
    main()
