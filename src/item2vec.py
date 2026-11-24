"""Session-based recommendations for guests: Item2Vec (Word2Vec over session item sequences).

side_info=True interleaves attribute tokens after every item, e.g.
    [p1, c:<category_id>, b:<brand>, p2, c:..., b:..., ...]
so products that share category/brand also share context, which helps rare items
(idea from note_hgy/Note.md: embed attributes and category, not just item ids).
Only product tokens are ever recommended.

Also provides CoVisitation, a strong count-based baseline for the same task.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from evalkit import weighted_scores


def session_sequences(df: pd.DataFrame, min_len: int = 2) -> pd.DataFrame:
    """One row per session: ordered product list with consecutive repeats collapsed."""
    d = df.sort_values(["user_session", "event_time"], kind="stable")
    d = d[d["product_id"] != d.groupby("user_session")["product_id"].shift()]
    seq = d.groupby("user_session", sort=False).agg(
        items=("product_id", list), start=("event_time", "min")
    )
    return seq[seq["items"].map(len) >= min_len]


def _popular(train: pd.DataFrame, days: int = 7, n: int = 200) -> list[int]:
    cutoff = train["event_time"].max() - pd.Timedelta(days=days)
    recent = train[train["event_time"] > cutoff]
    return weighted_scores(recent).groupby(recent["product_id"]).sum().nlargest(n).index.tolist()


class SessionRecommender:
    """Shared API: fit(train); recommend_session(items, k) -> [(product_id, score, reason)]."""

    def _fill_popular(self, recs: list, exclude: set, k: int) -> list:
        have = {i for i, _, _ in recs} | exclude
        for p in self.popular:
            if len(recs) >= k:
                break
            if p not in have:
                recs.append((p, 0.0, "Đang thịnh hành"))
        return recs

    def recommend_batch(self, sessions: list[list[int]], k: int) -> list[list[int]]:
        return [[i for i, _, _ in self.recommend_session(s, k)] for s in sessions]


class Item2Vec(SessionRecommender):
    def __init__(
        self,
        vector_size=64,
        window=5,
        min_count=3,
        epochs=10,
        negative=10,
        ns_exponent=0.75,
        sample=1e-3,
        side_info=False,
        recency_decay=0.7,
        last_n=5,
        workers=8,
        seed=42,
    ):
        self.params = dict(
            vector_size=vector_size, window=window, min_count=min_count, epochs=epochs,
            negative=negative, ns_exponent=ns_exponent, sample=sample, sg=1, workers=workers, seed=seed,
        )
        self.side_info = side_info
        self.recency_decay = recency_decay
        self.last_n = last_n

    def _corpus(self, train: pd.DataFrame, seqs: pd.DataFrame) -> list[list[str]]:
        if not self.side_info:
            return [[str(p) for p in items] for items in seqs["items"]]
        meta = train.drop_duplicates("product_id").set_index("product_id")
        cat, brand = meta["category_id"].to_dict(), meta["brand"].fillna("unknown").to_dict()
        corpus = []
        for items in seqs["items"]:
            toks = []
            for p in items:
                toks.append(str(p))
                toks.append(f"c:{cat.get(p)}")
                b = brand.get(p, "unknown")
                if b != "unknown":
                    toks.append(f"b:{b}")
            corpus.append(toks)
        return corpus

    def fit(self, train: pd.DataFrame) -> "Item2Vec":
        seqs = session_sequences(train)
        params = dict(self.params)
        if self.side_info:  # attribute tokens triple sequence length -> widen window to compensate
            params["window"] = params["window"] * 3
        self.w2v = Word2Vec(self._corpus(train, seqs), **params)

        kv = self.w2v.wv
        keys = [k for k in kv.index_to_key if not k[:2] in ("c:", "b:")]
        self.item_ids = np.array([int(k) for k in keys])
        vecs = np.stack([kv[k] for k in keys])
        self.vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        self.row = {p: i for i, p in enumerate(self.item_ids)}
        self.popular = _popular(train)
        return self

    def _query_vectors(self, sessions: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        q = np.zeros((len(sessions), self.vecs.shape[1]), dtype=np.float32)
        ok = np.zeros(len(sessions), dtype=bool)
        for i, s in enumerate(sessions):
            known = [self.row[p] for p in s[-self.last_n:] if p in self.row]
            if not known:
                continue
            w = self.recency_decay ** np.arange(len(known))[::-1]  # most recent weight 1
            q[i] = (w[:, None] * self.vecs[known]).sum(0)
            ok[i] = True
        return q, ok

    def recommend_batch(self, sessions, k, batch=1024):
        out = []
        for start in range(0, len(sessions), batch):
            chunk = sessions[start:start + batch]
            q, ok = self._query_vectors(chunk)
            scores = q @ self.vecs.T
            top = np.argpartition(-scores, k + 1, axis=1)[:, : k + 1]
            for i, s in enumerate(chunk):
                if not ok[i]:
                    out.append([p for p in self.popular if p != s[-1]][:k])
                    continue
                cand = top[i][np.argsort(-scores[i, top[i]])]
                recs = [int(self.item_ids[j]) for j in cand if self.item_ids[j] != s[-1]][:k]
                out.append(recs)
        return out

    def recommend_session(self, items, k=10):
        items = list(items)
        if not items:
            return self._fill_popular([], set(), k)
        q, ok = self._query_vectors([items])
        if not ok[0]:
            return self._fill_popular([], set(items), k)
        scores = self.vecs @ q[0]
        order = np.argsort(-scores)
        seen = set(items)
        recs = []
        for j in order:
            p = int(self.item_ids[j])
            if p not in seen:
                recs.append((p, float(scores[j]), "Vì bạn đang xem sản phẩm tương tự"))
                if len(recs) == k:
                    break
        return recs


def weighted_session_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Sessions as runs of consecutive identical products, each run carrying the
    strongest event weight inside it (view 1 / cart 3 / purchase 5) and its last time."""
    d = df.sort_values(["user_session", "event_time"], kind="stable")
    new_run = d["product_id"] != d.groupby("user_session")["product_id"].shift()
    run = new_run.cumsum()
    runs = pd.DataFrame({
        "user_session": d["user_session"].to_numpy(),
        "product_id": d["product_id"].to_numpy(),
        "w": weighted_scores(d).to_numpy(),
        "t": d["event_time"].to_numpy(),
        "run": run.to_numpy(),
    }).groupby("run", sort=False).agg(
        user_session=("user_session", "first"), product_id=("product_id", "first"),
        w=("w", "max"), t=("t", "max"),
    )
    return runs.groupby("user_session", sort=False).agg(
        items=("product_id", list), w=("w", list), t=("t", "max")
    )


class CoVisitation(SessionRecommender):
    """Count item pairs seen within `window` steps of each other in train sessions.

    type_weighted: a pair counts (w_a + w_b) / 2 with view 1 / cart 3 / purchase 5,
                   so pairs that reached the cart weigh more than pairs only viewed.
    time_weighted: a session's pairs are scaled 1 -> 1 + time_boost from the oldest to
                   the newest train session (recent co-visits matter more; OTTO-style).
    """

    def __init__(self, window=3, recency_decay=0.7, last_n=3, top_n=50,
                 type_weighted=False, time_weighted=False, time_boost=3.0):
        self.window = window
        self.recency_decay = recency_decay
        self.last_n = last_n
        self.top_n = top_n
        self.type_weighted = type_weighted
        self.time_weighted = time_weighted
        self.time_boost = time_boost

    def fit(self, train: pd.DataFrame) -> "CoVisitation":
        counts: dict[int, Counter] = defaultdict(Counter)
        if self.type_weighted or self.time_weighted:
            sess = weighted_session_runs(train)
            sess = sess[sess["items"].map(len) >= 2]
            t = sess["t"].astype("int64").to_numpy(dtype=float)
            span = max(t.max() - t.min(), 1.0)
            tw = 1.0 + self.time_boost * (t - t.min()) / span if self.time_weighted else np.ones(len(t))
            rows = zip(sess["items"], sess["w"], tw)
        else:
            rows = ((items, None, 1.0) for items in session_sequences(train)["items"])
        for items, ws, tf in rows:
            for i, a in enumerate(items):
                for j in range(i + 1, min(i + 1 + self.window, len(items))):
                    b = items[j]
                    if a == b:
                        continue
                    w = tf * ((ws[i] + ws[j]) / 2 if self.type_weighted else 1.0)
                    counts[a][b] += w
                    counts[b][a] += w
        self.nbrs = {a: c.most_common(self.top_n) for a, c in counts.items()}
        self.popular = _popular(train)
        return self

    def recommend_session(self, items, k=10):
        items = list(items)
        score: Counter = Counter()
        for age, p in enumerate(reversed(items[-self.last_n:])):
            w = self.recency_decay ** age
            for b, c in self.nbrs.get(p, []):
                score[b] += w * c
        last = items[-1] if items else None
        recs = [(p, float(s), "Người xem sản phẩm này cũng xem") for p, s in score.most_common(k + 1) if p != last][:k]
        return self._fill_popular(recs, {last} if last is not None else set(), k)

    def recommend_batch(self, sessions, k):
        return [[i for i, _, _ in self.recommend_session(s, k)] for s in sessions]


class CoVisitThenItem2Vec(SessionRecommender):
    """Co-visitation neighbours first; Item2Vec fills the rest instead of popular items.

    Co-visitation is precise for well-observed items, Item2Vec (with side info)
    still has something to say about rare ones.
    """

    def __init__(self, covisit: CoVisitation | None = None, i2v: Item2Vec | None = None, i2v_share: int = 0):
        self.covisit = covisit or CoVisitation()
        self.i2v = i2v or Item2Vec(side_info=True)
        self.i2v_share = i2v_share  # always reserve this many slots for Item2Vec

    def fit(self, train):
        self.covisit.fit(train)
        self.i2v.fit(train)
        self.popular = self.covisit.popular
        return self

    def recommend_session(self, items, k=10):
        cv = [r for r in self.covisit.recommend_session(items, k) if r[2] != "Đang thịnh hành"]
        cv = cv[: k - self.i2v_share] if self.i2v_share else cv
        have = {p for p, _, _ in cv}
        for p, s, r in self.i2v.recommend_session(items, k + len(have) + 1):
            if len(cv) >= k:
                break
            if p not in have and (not items or p != items[-1]):
                cv.append((p, s, r))
                have.add(p)
        return self._fill_popular(cv, {items[-1]} if items else set(), k)

    def recommend_batch(self, sessions, k):
        i2v = self.i2v.recommend_batch(sessions, 2 * k)
        out = []
        for s, extra in zip(sessions, i2v):
            cv = [p for p, _, r in self.covisit.recommend_session(s, k) if r != "Đang thịnh hành"]
            if self.i2v_share:
                cv = cv[: k - self.i2v_share]
            have = set(cv)
            cv += [p for p in extra if p not in have][: k - len(cv)]
            out.append(cv)
        return out


class PopularSession(SessionRecommender):
    def fit(self, train):
        self.popular = _popular(train)
        return self

    def recommend_session(self, items, k=10):
        return self._fill_popular([], {items[-1]} if items else set(), k)
