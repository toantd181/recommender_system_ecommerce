import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from baselines import PopularRecent
from content_based import ContentBased
from router import HybridRouter, RankFusion


class Fixed:
    def __init__(self, recs): self.recs = recs
    def fit(self, train): pass
    def recommend(self, user_ids, k): return {u: self.recs[:k] for u in user_ids}


def _train():
    rows, t0 = [], pd.Timestamp("2019-11-01")
    # brand a / category 1 / cheap, brand b / category 2 / expensive
    cat = {p: (1 if p < 10 else 2) for p in range(20)}
    for u in range(30):
        pool = range(0, 10) if u < 15 else range(10, 20)
        for p in list(pool)[: 3 + u % 3]:
            rows.append((t0, "view", p, cat[p], "a" if cat[p] == 1 else "b", 10.0 if cat[p] == 1 else 900.0, u))
    return pd.DataFrame(rows, columns=["event_time", "event_type", "product_id", "category_id", "brand", "price", "user_id"])


def test_content_based_follows_profile_and_falls_back():
    m = ContentBased(beta=0.0, max_cand=20, price_bins=2)
    m.fit(_train())
    r = m.recommend([0, 20, 999], 5)
    assert all(p < 10 for p in r[0]) and all(p >= 10 for p in r[20])
    assert r[999] == m.fallback.top[:5]
    assert all(p >= 10 for p, _ in m.recommend_from_items([12, 13], 5))


def test_router_sends_each_group_to_its_module():
    train = _train()
    mods = {"unseen": Fixed([1]), "light": Fixed([2]), "heavy": Fixed([3])}
    r = HybridRouter(mods, heavy_min=5)
    r.fit(train.iloc[:0].pipe(lambda d: pd.concat([d, pd.DataFrame({"user_id": [1] * 2 + [2] * 6})], ignore_index=True)))
    out = r.recommend([1, 2, 3], 1)
    assert out == {1: [2], 2: [3], 3: [1]}


def test_rank_fusion_prefers_items_both_models_rank_high():
    f = RankFusion([Fixed([1, 2, 3]), Fixed([2, 9, 1])], [1, 1])
    assert f.recommend([0], 3)[0][0] == 2
