import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from als import ALSRecommender


def _train():
    """Two taste groups: users 0-19 like products 0-9, users 20-39 like products 10-19."""
    rng, rows = np.random.default_rng(0), []
    t0 = pd.Timestamp("2019-11-01")
    for u in range(40):
        pool = range(0, 10) if u < 20 else range(10, 20)
        for p in rng.choice(list(pool), size=6, replace=False):
            rows.append((t0 + pd.Timedelta(hours=int(u)), "view", int(p), 1, u))
            if rng.random() < 0.5:
                rows.append((t0 + pd.Timedelta(hours=int(u)), "cart", int(p), 1, u))
    return pd.DataFrame(rows, columns=["event_time", "event_type", "product_id", "category_id", "user_id"])


def test_recommends_from_own_taste_group_and_falls_back_for_unseen():
    m = ALSRecommender(factors=8, iterations=10, min_item_events=1, filter_seen=None, threads=1)
    m.fit(_train())
    recs = m.recommend([0, 30, 999], 5)
    assert all(len(r) == 5 for r in recs.values())
    assert np.mean([p < 10 for p in recs[0]]) >= 0.8       # user 0 -> products 0-9
    assert np.mean([p >= 10 for p in recs[30]]) >= 0.8     # user 30 -> products 10-19
    assert m.covers(0) and not m.covers(999)
    assert recs[999] == m.fallback.top[:5]                 # unseen user -> popularity fallback


def test_filter_seen_removes_purchased_items():
    df = _train()
    extra = pd.DataFrame([(pd.Timestamp("2019-11-02"), "purchase", 3, 1, 0)], columns=df.columns)
    m = ALSRecommender(factors=8, iterations=10, min_item_events=1, filter_seen="purchased", threads=1)
    m.fit(pd.concat([df, extra], ignore_index=True))
    assert 3 not in m.recommend([0], 10)[0]
