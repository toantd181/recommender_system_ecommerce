"""Content-based recommender (new members) behind the evalkit interface.

Item vector = one-hot [category_id | brand | price bucket], each block weighted and the whole
vector L2-normalised. A user profile is the (decayed, event-weighted) sum of the vectors of the
items they touched; candidates are ranked by cosine(profile, item) + beta * popularity prior,
so a thin profile still gets sensible, popular items from the right category/brand/price band.
`category_code` is not used: it is unknown for 33% of events (see the EDA).

Users with no history cannot be profiled -> `fallback` (popularity). In the product, new members
can instead state favourite categories at sign-up: `recommend_from_items` builds a profile from
any list of products (e.g. the top sellers of the chosen categories, or items just viewed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from als import ALSRecommender
from baselines import PopularRecent
from evalkit import EVENT_WEIGHTS


def _onehot(codes: np.ndarray, n: int, weight: float) -> sp.csr_matrix:
    ok = codes >= 0
    rows = np.flatnonzero(ok)
    return sp.csr_matrix((np.full(len(rows), weight, dtype="float32"), (rows, codes[ok])),
                         shape=(len(codes), max(n, 1)))


class ContentBased:
    def __init__(self, w_cat: float = 1.0, w_brand: float = 0.5, w_price: float = 0.5, beta: float = 1.0,
                 price_bins: int = 10, cand_days: int = 14, max_cand: int = 5000,
                 half_life_days: float | None = 7.0, view_cap: float = 3.0, fallback=None):
        self.w_cat, self.w_brand, self.w_price, self.beta = w_cat, w_brand, w_price, beta
        self.price_bins, self.cand_days, self.max_cand = price_bins, cand_days, max_cand
        self.half_life_days, self.view_cap = half_life_days, view_cap
        self.fallback = fallback if fallback is not None else PopularRecent(7)

    def fit(self, train: pd.DataFrame) -> None:
        self.fallback.fit(train)
        it = train.groupby("product_id").agg(category_id=("category_id", "last"), brand=("brand", "last"),
                                             price=("price", "median"))
        self.items = it.index.to_numpy()
        cat, _ = pd.factorize(it["category_id"])
        brand, bu = pd.factorize(it["brand"].astype("object").where(it["brand"].notna()))
        lp = np.log1p(it["price"].astype(float))
        pb = pd.qcut(lp, self.price_bins, labels=False, duplicates="drop")
        pb = pb.fillna(-1).astype(int).to_numpy()
        X = sp.hstack([_onehot(cat, cat.max() + 1, self.w_cat), _onehot(brand, len(bu), self.w_brand),
                       _onehot(pb, pb.max() + 1, self.w_price)]).tocsr()
        self.X = normalize(X)

        # user profiles from event-weighted, decayed interactions
        r = ALSRecommender(half_life_days=self.half_life_days, view_cap=self.view_cap, min_item_events=1)._ratings(train)
        users, ui = np.unique(r["u"].to_numpy(), return_inverse=True)
        self.user_index = {u: k for k, u in enumerate(users)}
        ii = np.searchsorted(self.items, r["i"].to_numpy())
        R = sp.csr_matrix((r["r"].to_numpy(), (ui, ii)), shape=(len(users), len(self.items)), dtype="float32")
        self.profiles = normalize(R @ self.X)

        # candidates: recently popular items, with a popularity prior in [0, 1]
        cutoff = train["event_time"].max() - pd.Timedelta(days=self.cand_days)
        rec = train[train["event_time"] > cutoff]
        w = rec["event_type"].map(EVENT_WEIGHTS).fillna(0.0)
        pop = w.groupby(rec["product_id"]).sum().nlargest(self.max_cand)
        self.cand = np.searchsorted(self.items, pop.index.to_numpy())
        lg = np.log1p(pop.to_numpy())
        self.prior = (lg / lg.max()).astype("float32")
        self.cand_ids = self.items[self.cand]
        self.Xc_T = self.X[self.cand].T.tocsr()

    def covers(self, user_id: int) -> bool:
        return user_id in self.user_index

    def _rank(self, P: sp.csr_matrix, k: int) -> list[list[int]]:
        out = []
        for s in range(0, P.shape[0], 2000):
            S = (P[s:s + 2000] @ self.Xc_T).toarray() + self.beta * self.prior
            top = np.argpartition(-S, min(k, S.shape[1] - 1), axis=1)[:, :k]
            order = np.take_along_axis(S, top, 1).argsort(1)[:, ::-1]
            out += self.cand_ids[np.take_along_axis(top, order, 1)].tolist()
        return out

    def recommend(self, user_ids, k: int) -> dict[int, list[int]]:
        known = [u for u in user_ids if u in self.user_index]
        out: dict[int, list[int]] = {}
        if known:
            idx = np.array([self.user_index[u] for u in known])
            out = dict(zip(known, self._rank(self.profiles[idx], k)))
        missing = [u for u in user_ids if u not in out]
        if missing:
            out.update(self.fallback.recommend(missing, k))
        return out

    def recommend_from_items(self, product_ids, k: int, exclude=()) -> list[tuple[int, float]]:
        """Profile from an explicit list of products (sign-up favourites, items just viewed)."""
        j = np.searchsorted(self.items, list(product_ids))
        j = j[(j < len(self.items)) & (self.items[np.minimum(j, len(self.items) - 1)] == np.asarray(list(product_ids)))]
        if len(j) == 0:
            return []
        p = normalize(sp.csr_matrix(self.X[j].sum(axis=0)))
        S = (p @ self.Xc_T).toarray()[0] + self.beta * self.prior
        order = np.argsort(-S)
        ex = set(exclude)
        return [(int(self.cand_ids[c]), float(S[c])) for c in order if self.cand_ids[c] not in ex][:k]
