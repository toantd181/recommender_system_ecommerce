"""Learning-to-rank re-ranker for session (guest) recommendations.

Candidates per query (session prefix):
    co-visitation top-N  ∪  Item2Vec top-N  ∪  earlier items of the same session (revisits)
Features: co-visitation score/rank, Item2Vec similarity/rank, revisit distance,
same category / brand as the last item, price ratio, item popularity, cart rate, ...
Model: LightGBM LambdaRank.

Protocol (no test leakage):
    candidate models fit on data < VAL_START, ranker trained on validation-week queries
    candidate models refit on data < TEST_START, ranker applied once to test-week queries

Usage (after experiments.py, whose chosen co-visitation config it reuses):
    .venv/bin/python src/ltr_session.py data/nov_u5.parquet   (history chosen in experiments.json)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from evalkit import bootstrap_ci, load_events, paired_bootstrap, three_way_split
from experiments import K, TEST_START, VAL_START, hit_rr, session_queries
from item2vec import CoVisitation, CoVisitThenItem2Vec, Item2Vec, SessionRecommender

FEATURES = [
    "cv_score", "cv_rank", "cv_score_rel", "i2v_sim", "i2v_rank", "revisit", "steps_back",
    "same_cat", "same_brand", "price_ratio", "log_pop", "cart_rate", "sess_len", "n_cv",
]


def log(*a):
    print(*a, flush=True)


def item_table(fit: pd.DataFrame) -> pd.DataFrame:
    g = fit.groupby("product_id")
    pos = fit["event_type"].isin(["cart", "purchase"]).groupby(fit["product_id"]).sum()
    n = g.size()
    meta = fit.drop_duplicates("product_id", keep="last").set_index("product_id")
    return pd.DataFrame({
        "log_pop": np.log1p(n),
        "cart_rate": (pos + 1) / (n + 20),  # smoothed
        "cat": meta["category_id"],
        "brand": meta["brand"].fillna("unknown"),
        "price": meta["price"],
    })


def build(cv: CoVisitation, i2v: Item2Vec, items: pd.DataFrame, P, T=None,
          n_cv=30, n_i2v=30, n_rev=10) -> pd.DataFrame:
    # Item2Vec top lists, batched
    qv, ok = i2v._query_vectors(P)
    n_i2v = min(n_i2v, len(i2v.item_ids) - 1)  # argpartition needs kth < vocab size
    i2v_top = []
    for s in range(0, len(P), 2048):
        sc = qv[s:s + 2048] @ i2v.vecs.T
        top = np.argpartition(-sc, n_i2v, axis=1)[:, :n_i2v]
        order = np.take_along_axis(sc, top, 1).argsort(1)[:, ::-1]
        i2v_top += [i2v.item_ids[t[o]].tolist() if ok[s + i] else []
                    for i, (t, o) in enumerate(zip(top, order))]

    qid, cand, cvs, cvr, i2r, rev, steps, slen, ncv = ([] for _ in range(9))
    for q, prefix in enumerate(P):
        last = prefix[-1]
        d: dict[int, list] = {}
        cvl = [x for x in cv.recommend_session(prefix, n_cv) if x[2] != "Đang thịnh hành"]
        for r, (p, s, _) in enumerate(cvl):
            d[p] = [s, r, 999]
        for r, p in enumerate(i2v_top[q]):
            d.setdefault(p, [0.0, 999, 999])[2] = r
        pos = {p: i for i, p in enumerate(prefix)}  # last occurrence
        for p in list(dict.fromkeys(reversed(prefix[:-1])))[:n_rev]:
            d.setdefault(p, [0.0, 999, 999])
        d.pop(last, None)
        for p, (s, r1, r2) in d.items():
            qid.append(q); cand.append(p); cvs.append(s); cvr.append(r1); i2r.append(r2)
            rev.append(p in pos); steps.append(len(prefix) - 1 - pos[p] if p in pos else 99)
            slen.append(len(prefix)); ncv.append(len(cvl))

    f = pd.DataFrame({"qid": qid, "cand": cand, "cv_score": cvs, "cv_rank": cvr, "i2v_rank": i2r,
                      "revisit": rev, "steps_back": steps, "sess_len": slen, "n_cv": ncv})
    f["revisit"] = f["revisit"].astype(int)
    top_cv = f.groupby("qid")["cv_score"].transform("max")
    f["cv_score_rel"] = np.where(top_cv > 0, f["cv_score"] / top_cv, 0.0)

    # Item2Vec similarity for every candidate (not only its own top list)
    rows = f["cand"].map(i2v.row).fillna(-1).astype(int).to_numpy()
    qn = qv / np.maximum(np.linalg.norm(qv, axis=1, keepdims=True), 1e-9)
    sim = np.full(len(f), np.nan, dtype=np.float32)
    valid = (rows >= 0) & ok[f["qid"].to_numpy()]
    vi = np.where(valid)[0]
    for s in range(0, len(vi), 1_000_000):
        b = vi[s:s + 1_000_000]
        sim[b] = np.einsum("ij,ij->i", i2v.vecs[rows[b]], qn[f["qid"].to_numpy()[b]])
    f["i2v_sim"] = sim

    # item and last-item attributes
    it = items.reindex(f["cand"].to_numpy())
    last = items.reindex([p[-1] for p in P])
    lq = f["qid"].to_numpy()
    f["log_pop"] = it["log_pop"].to_numpy()
    f["cart_rate"] = it["cart_rate"].to_numpy()
    f["same_cat"] = (it["cat"].to_numpy() == last["cat"].to_numpy()[lq]).astype(int)
    ib = it["brand"].astype(object).fillna("unknown").to_numpy()
    lb = last["brand"].astype(object).fillna("unknown").to_numpy()[lq]
    f["same_brand"] = ((ib == lb) & (ib != "unknown")).astype(int)
    f["price_ratio"] = it["price"].to_numpy() / np.maximum(last["price"].to_numpy()[lq], 0.01)
    if T is not None:
        f["label"] = (f["cand"].to_numpy() == np.asarray(T)[lq]).astype(int)
    return f


def train_ranker(f: pd.DataFrame, C: np.ndarray, seed: int = 0) -> lgb.LGBMRanker:
    f = f[f.groupby("qid")["label"].transform("max") > 0]  # queries whose target is a candidate
    sess = C[f["qid"].to_numpy()]
    rng = np.random.default_rng(seed)
    held = set(rng.choice(np.unique(sess), size=len(np.unique(sess)) // 5, replace=False))
    is_val = np.isin(sess, list(held))
    tr, va = f[~is_val], f[is_val]
    model = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=500, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=seed, verbose=-1,
    )
    model.fit(
        tr[FEATURES], tr["label"], group=tr.groupby("qid", sort=False).size().to_numpy(),
        eval_set=[(va[FEATURES], va["label"])], eval_group=[va.groupby("qid", sort=False).size().to_numpy()],
        eval_at=[K], callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    log(f"ranker: {len(tr):,} train rows, best iteration {model.best_iteration_}, "
        f"inner-val ndcg@{K} {model.best_score_['valid_0'][f'ndcg@{K}']:.4f}")
    return model


def rerank(model, f: pd.DataFrame, n_queries: int) -> list[list[int]]:
    f = f.assign(s=model.predict(f[FEATURES]))
    f = f.sort_values(["qid", "s"], ascending=[True, False])
    top = f.groupby("qid").head(K).groupby("qid")["cand"].agg(list)
    return [top.get(q, []) for q in range(n_queries)]


class SessionLTR(SessionRecommender):
    """Deployable guest recommender: candidate models + trained ranker."""

    REASONS = {
        "revisit": "Bạn đã xem trước đó",
        "covisit": "Người xem sản phẩm này cũng xem",
        "i2v": "Vì bạn đang xem sản phẩm tương tự",
    }

    def __init__(self, cv: CoVisitation, i2v: Item2Vec, items: pd.DataFrame, ranker):
        self.cv, self.i2v, self.items, self.ranker = cv, i2v, items, ranker
        self.popular = cv.popular

    def recommend_session(self, items, k=10):
        items = list(items)
        if not items:
            return self._fill_popular([], set(), k)
        f = build(self.cv, self.i2v, self.items, [items])
        if len(f) == 0:
            return self._fill_popular([], {items[-1]}, k)
        f = f.assign(s=self.ranker.predict(f[FEATURES])).nlargest(k, "s")
        recs = []
        for r in f.itertuples():
            why = "revisit" if r.revisit else ("covisit" if r.cv_rank < 999 else "i2v")
            recs.append((int(r.cand), float(r.s), self.REASONS[why]))
        return self._fill_popular(recs, {items[-1]}, k)


def fit_ranker(df: pd.DataFrame, cv_kw: dict, n_sessions: int = 15000, verbose: bool = True):
    """Candidate models fit before VAL_START, ranker trained on validation-week queries."""
    fit_val, val, _, _ = three_way_split(df, VAL_START, TEST_START)
    cv, i2v = CoVisitation(**cv_kw).fit(fit_val), Item2Vec(side_info=True).fit(fit_val)
    P, T, C = session_queries(val, n_sessions, seed=1)
    f = build(cv, i2v, item_table(fit_val), P, T)
    if verbose:
        recall = f.groupby("qid")["label"].max().reindex(range(len(P)), fill_value=0).mean()
        log(f"val: {len(P):,} queries, {len(f):,} candidate rows, candidate recall {recall:.3f}")
    return train_ranker(f, C)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--chosen", default="results/experiments.json")
    ap.add_argument("--train-sessions", type=int, default=15000)
    ap.add_argument("--test-sessions", type=int, default=20000)
    ap.add_argument("--out", default="results/ltr_session.json")
    args = ap.parse_args()

    chosen = json.loads(Path(args.chosen).read_text())["chosen"]
    cv_kw, share = chosen["covisit"], chosen["share"]
    log("chosen co-visitation:", cv_kw, "| hybrid share:", share)

    df = load_events(args.parquet)
    fit_val, val, fit_test, test = three_way_split(df, VAL_START, TEST_START)

    # ---- train on the validation week
    t0 = time.time()
    ranker = fit_ranker(df, cv_kw, args.train_sessions)
    log(f"ranker trained in {time.time() - t0:.0f}s")
    imp = pd.Series(ranker.booster_.feature_importance("gain"), index=FEATURES)
    log("feature importance (gain share):\n" + (imp / imp.sum()).sort_values(ascending=False).round(3).to_string())

    # ---- apply once to the test week (same queries as experiments.py)
    cv, i2v = CoVisitation(**cv_kw).fit(fit_test), Item2Vec(side_info=True).fit(fit_test)
    P, T, C = session_queries(test, args.test_sessions)
    f = build(cv, i2v, item_table(fit_test), P, T)
    recall = f.groupby("qid")["label"].max().reindex(range(len(P)), fill_value=0).mean()
    log(f"test: {len(P):,} queries, {len(f):,} candidate rows, candidate recall {recall:.3f} (ceiling for any re-ranker)")

    hyb = CoVisitThenItem2Vec(cv, i2v, i2v_share=share)
    hyb.popular = cv.popular
    def with_revisits(recs, n_rev):
        """Heuristic: the n_rev most recent earlier items of the session first."""
        out = []
        for prefix, r in zip(P, recs):
            rev = [p for p in dict.fromkeys(reversed(prefix[:-1])) if p != prefix[-1]][:n_rev]
            out.append((rev + [p for p in r if p not in rev])[:K])
        return out

    cv_recs = cv.recommend_batch(P, K)
    hyb_recs = hyb.recommend_batch(P, K)
    systems = {
        "covisit (chosen)": cv_recs,
        f"hybrid share={share}": hyb_recs,
        "hybrid + 1 revisit first": with_revisits(hyb_recs, 1),
        "hybrid + 3 revisits first": with_revisits(hyb_recs, 3),
        "LTR re-ranker": rerank(ranker, f, len(P)),
    }
    hits, rows = {}, {}
    for n, recs in systems.items():
        h, rr = hit_rr(recs, T)
        hits[n] = h
        rows[n] = {f"hit@{K}": "{:.4f} [{:.4f}, {:.4f}]".format(*bootstrap_ci(h, C)),
                   f"mrr@{K}": "{:.4f} [{:.4f}, {:.4f}]".format(*bootstrap_ci(rr, C))}
    t = pd.DataFrame(rows).T
    log(t.to_string())
    out = {"table": t.to_dict(), "importance": (imp / imp.sum()).to_dict(), "candidate_recall": recall}
    best_heur = max(("hybrid + 1 revisit first", "hybrid + 3 revisits first"), key=lambda n: hits[n].mean())
    for other in ("covisit (chosen)", f"hybrid share={share}", best_heur):
        d = paired_bootstrap(hits["LTR re-ranker"], hits[other], C)
        log(f"LTR - {other}: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}], p={d[3]:.3f}")
        out[f"diff vs {other}"] = d
    # how much comes from revisits alone
    rev_share = np.mean([t_ in set(p[:-1]) for p, t_ in zip(P, T)])
    log(f"share of test targets that revisit an earlier item of the session: {rev_share:.1%}")
    out["revisit_target_share"] = rev_share
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    log("saved", args.out)


if __name__ == "__main__":
    main()
