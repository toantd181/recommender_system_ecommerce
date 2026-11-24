"""Router: send each user to the module that suits their amount of history.

Groups follow evalkit.user_groups: unseen (no train events), light (1..heavy_min-1), heavy (>= heavy_min).
`modules` maps group -> any recommender with recommend(user_ids, k). `RankFusion` blends several
recommenders with weighted reciprocal-rank fusion, so scores of different models need not be comparable.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from evalkit import user_groups


class RankFusion:
    """Weighted reciprocal-rank fusion: score(item) = sum_m w_m / (c + rank_m(item))."""

    def __init__(self, models, weights=None, c: int = 10, depth: int = 50):
        self.models, self.c, self.depth = list(models), c, depth
        self.weights = list(weights) if weights is not None else [1.0] * len(self.models)

    def fit(self, train: pd.DataFrame) -> None:  # members are fitted by the caller
        pass

    def recommend(self, user_ids, k: int) -> dict[int, list[int]]:
        lists = [m.recommend(user_ids, self.depth) for m in self.models]
        out = {}
        for u in user_ids:
            score: dict[int, float] = defaultdict(float)
            for w, recs in zip(self.weights, lists):
                for rank, p in enumerate(recs.get(u, []), 1):
                    score[p] += w / (self.c + rank)
            out[u] = sorted(score, key=score.get, reverse=True)[:k]
        return out


class HybridRouter:
    def __init__(self, modules: dict, heavy_min: int = 5):
        assert set(modules) >= {"unseen", "light", "heavy"}
        self.modules, self.heavy_min = modules, heavy_min

    def fit(self, train: pd.DataFrame) -> None:
        self.counts = train.groupby("user_id").size()

    def group_of(self, users) -> pd.Series:
        n = self.counts.reindex(list(users)).fillna(0).astype(int)
        g = pd.Series("light", index=n.index, dtype="object")
        g[n == 0] = "unseen"
        g[n >= self.heavy_min] = "heavy"
        return g

    def recommend(self, user_ids, k: int) -> dict[int, list[int]]:
        groups = self.group_of(user_ids)
        out: dict[int, list[int]] = {}
        for g, module in self.modules.items():
            ids = [u for u in user_ids if groups[u] == g]
            if ids:
                out.update(module.recommend(ids, k))
        return out
