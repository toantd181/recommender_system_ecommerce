"""ALS for implicit feedback (old members) behind the evalkit recommender interface.

    fit(train: DataFrame) -> None
    recommend(user_ids, k) -> {user_id: [product_id, ...]}

Confidence follows the project outline: c_ui = 1 + alpha * r_ui, where r_ui is built from event
weights (view 1, cart 3, purchase 5), optionally decayed by event age (half-life in days).
Repeated views of the same product are capped, since a view is the weakest, most repetitive signal.
Users without train history cannot be served by ALS and fall back to `fallback` (a popularity
model by default); the router decides who actually reaches this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.cpu.als import AlternatingLeastSquares

from baselines import PopularRecent
from evalkit import EVENT_WEIGHTS


class ALSRecommender:
    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        alpha: float = 20.0,
        iterations: int = 15,
        half_life_days: float | None = 14.0,
        view_cap: float = 3.0,
        min_item_events: int = 3,
        filter_seen: str | None = "purchased",  # None | "purchased" | "all"
        fallback=None,
        threads: int = 4,
        seed: int = 0,
    ):
        assert filter_seen in (None, "purchased", "all")
        self.factors, self.regularization, self.alpha = factors, regularization, alpha
        self.iterations, self.half_life_days, self.view_cap = iterations, half_life_days, view_cap
        self.min_item_events, self.filter_seen = min_item_events, filter_seen
        self.fallback = fallback if fallback is not None else PopularRecent(7)
        self.threads, self.seed = threads, seed

    # ---- training ----------------------------------------------------------
    def _ratings(self, train: pd.DataFrame) -> pd.DataFrame:
        d = train[train["event_type"].isin(EVENT_WEIGHTS)]
        counts = d.groupby("product_id")["user_id"].transform("size")
        d = d[counts >= self.min_item_events]
        w = d["event_type"].map(EVENT_WEIGHTS).astype("float32")
        if self.half_life_days:
            age = (train["event_time"].max() - d["event_time"]).dt.total_seconds() / 86400.0
            w = w * np.power(0.5, age / self.half_life_days).astype("float32")
        g = pd.DataFrame({"u": d["user_id"].to_numpy(), "i": d["product_id"].to_numpy(),
                          "view": np.where(d["event_type"] == "view", w, 0.0),
                          "other": np.where(d["event_type"] != "view", w, 0.0)})
        g = g.groupby(["u", "i"], sort=False).sum().reset_index()
        g["r"] = np.log1p(np.minimum(g["view"], self.view_cap) + g["other"]).astype("float32")
        return g[["u", "i", "r"]]

    def fit(self, train: pd.DataFrame) -> None:
        self.fallback.fit(train)
        r = self._ratings(train)
        users, ui = np.unique(r["u"].to_numpy(), return_inverse=True)
        items, ii = np.unique(r["i"].to_numpy(), return_inverse=True)
        self.user_index = {u: k for k, u in enumerate(users)}
        self.items = items
        self.ui = sp.csr_matrix((r["r"].to_numpy(), (ui, ii)), shape=(len(users), len(items)), dtype="float32")

        self.excluded: dict[int, set[int]] = {}
        if self.filter_seen:
            ev = ["purchase"] if self.filter_seen == "purchased" else list(EVENT_WEIGHTS)
            seen = train[train["event_type"].isin(ev)]
            self.excluded = seen.groupby("user_id")["product_id"].agg(set).to_dict()

        self.model = AlternatingLeastSquares(
            factors=self.factors, regularization=self.regularization, alpha=self.alpha,
            iterations=self.iterations, num_threads=self.threads, random_state=self.seed,
        )
        self.model.fit(self.ui, show_progress=False)

    # ---- serving -----------------------------------------------------------
    def covers(self, user_id: int) -> bool:
        return user_id in self.user_index

    def recommend(self, user_ids, k: int) -> dict[int, list[int]]:
        known = [u for u in user_ids if u in self.user_index]
        out: dict[int, list[int]] = {}
        if known:
            idx = np.array([self.user_index[u] for u in known])
            extra = 0 if not self.filter_seen else 60
            ids, _ = self.model.recommend(idx, self.ui[idx], N=k + extra, filter_already_liked_items=False)
            for u, row in zip(known, ids):
                prods = self.items[row].tolist()
                ex = self.excluded.get(u)
                if ex:
                    prods = [p for p in prods if p not in ex]
                out[u] = prods[:k]
        missing = [u for u in user_ids if u not in out]
        if missing:
            out.update(self.fallback.recommend(missing, k))
        return out

    def similar_items(self, product_id: int, k: int = 10) -> list[int]:
        """Item-item neighbours from the ALS factors (usable for 'related products')."""
        j = np.searchsorted(self.items, product_id)
        if j >= len(self.items) or self.items[j] != product_id:
            return []
        ids, _ = self.model.similar_items(j, N=k + 1)
        return [int(self.items[x]) for x in ids if x != j][:k]
