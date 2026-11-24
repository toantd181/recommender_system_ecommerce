"""Popularity baselines every other module must beat."""
from __future__ import annotations

import pandas as pd

from evalkit import weighted_scores


class PopularRecent:
    """Same Top-K for everyone: most popular products in the last `days` of train."""

    def __init__(self, days: int = 7):
        self.days = days
        self.top: list[int] = []

    def fit(self, train: pd.DataFrame) -> None:
        cutoff = train["event_time"].max() - pd.Timedelta(days=self.days)
        recent = train[train["event_time"] > cutoff]
        scores = weighted_scores(recent).groupby(recent["product_id"]).sum()
        self.top = scores.sort_values(ascending=False).index.tolist()[:500]

    def recommend(self, user_ids, k):
        return {u: self.top[:k] for u in user_ids}


class PopularByCategory:
    """Popular within the user's favourite category_id (from train); global popular fills gaps."""

    def __init__(self, days: int = 7, min_user_events: int = 1):
        self.days = days
        self.min_user_events = min_user_events

    def fit(self, train: pd.DataFrame) -> None:
        self.global_ = PopularRecent(self.days)
        self.global_.fit(train)

        cutoff = train["event_time"].max() - pd.Timedelta(days=self.days)
        recent = train[train["event_time"] > cutoff]
        s = weighted_scores(recent).groupby([recent["category_id"], recent["product_id"]]).sum()
        self.cat_top = {
            c: g.sort_values(ascending=False).index.get_level_values("product_id").tolist()[:200]
            for c, g in s.groupby(level="category_id")
        }

        w = weighted_scores(train)
        ucat = w.groupby([train["user_id"], train["category_id"]]).sum()
        ucat = ucat.reset_index(name="w").sort_values("w", ascending=False)
        self.user_cat = ucat.drop_duplicates("user_id").set_index("user_id")["category_id"].to_dict()

    def recommend(self, user_ids, k):
        out = {}
        for u in user_ids:
            recs = list(self.cat_top.get(self.user_cat.get(u), []))[:k]
            if len(recs) < k:
                seen = set(recs)
                recs += [p for p in self.global_.top if p not in seen][: k - len(recs)]
            out[u] = recs
        return out
