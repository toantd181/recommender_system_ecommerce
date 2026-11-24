import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from evalkit import (recall_at_k, hit_at_k, mrr_at_k, ndcg_at_k, temporal_split,
                     user_groups, ground_truth, evaluate)
from baselines import PopularRecent, PopularByCategory


def test_metrics():
    recs, truth = [9, 1, 8, 2], {1, 2, 3}
    assert recall_at_k(recs, truth, 4) == 2 / 3
    assert hit_at_k(recs, truth, 1) == 0.0 and hit_at_k(recs, truth, 2) == 1.0
    assert mrr_at_k(recs, truth, 4) == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert abs(ndcg_at_k(recs, truth, 4) - dcg / idcg) < 1e-12
    assert ndcg_at_k([1, 2, 3], truth, 3) == 1.0


def _events():
    rows = []
    def add(t, u, p, e, c=1): rows.append((pd.Timestamp(t), e, p, c, u))
    for i in range(6): add(f"2019-11-0{i+1}", 1, 100 + i, "view")      # heavy user
    add("2019-11-02", 2, 100, "view")                                  # light user
    add("2019-11-05", 3, 100, "purchase")                              # popular item
    add("2019-11-09", 1, 100, "cart")                                  # test positives
    add("2019-11-09", 2, 100, "purchase")
    add("2019-11-09", 4, 100, "cart")                                  # unseen user
    add("2019-11-09", 4, 555, "view")                                  # view is not positive
    return pd.DataFrame(rows, columns=["event_time", "event_type", "product_id", "category_id", "user_id"])


def test_split_groups_truth():
    df = _events()
    train, test = temporal_split(df, "2019-11-08")
    assert train["event_time"].max() < pd.Timestamp("2019-11-08") <= test["event_time"].min()
    g = user_groups(train, [1, 2, 4])
    assert g.to_dict() == {1: "heavy", 2: "light", 4: "unseen"}
    assert ground_truth(test) == {1: {100}, 2: {100}, 4: {100}}


def test_baselines_end_to_end():
    train, test = temporal_split(_events(), "2019-11-08")
    for model in (PopularRecent(30), PopularByCategory(30)):
        res = evaluate(model, train, test, ks=(1, 5)).table
        assert res.loc["all", "hit@1"] == 1.0 and res.loc["all", "n_users"] == 3
        assert set(res.index) == {"all", "unseen", "light", "heavy"}


def test_bootstrap_and_split():
    import numpy as np
    from evalkit import bootstrap_ci, paired_bootstrap, three_way_split, beyond_accuracy
    v = np.r_[np.ones(50), np.zeros(50)]
    m, lo, hi = bootstrap_ci(v)
    assert m == 0.5 and lo < 0.5 < hi and hi - lo < 0.3
    # identical clusters: whole cluster resampled -> CI wider than ignoring clusters
    clusters = np.repeat(np.arange(10), 10)
    v2 = np.repeat(np.r_[np.ones(5), np.zeros(5)], 10)
    _, lo_c, hi_c = bootstrap_ci(v2, clusters)
    _, lo_i, hi_i = bootstrap_ci(v2)
    assert hi_c - lo_c > hi_i - lo_i
    d, lo, hi, p = paired_bootstrap(np.ones(100), np.zeros(100))
    assert d == 1.0 and p == 0.0
    df = _events()
    a, b, c, d_ = three_way_split(df, "2019-11-05", "2019-11-08")
    assert a.event_time.max() < pd.Timestamp("2019-11-05") <= b.event_time.min()
    assert b.event_time.max() < pd.Timestamp("2019-11-08") <= d_.event_time.min()
    assert len(c) == len(a) + len(b)
    ba = beyond_accuracy([[100], [100]], c, k=10)
    assert ba["pop_share"] == 1.0 and 0 < ba["coverage"] <= 1


def test_metrics_match_lecture_L8_examples():
    """Worked examples from slides L8 (IT4613, Đánh giá hệ thống gợi ý)."""
    from evalkit import (ap_at_k, liftindex_at_k, mrr_at_k, ndcg_at_k, precision_at_k,
                         rankscore_at_k, recall_at_k, f1_at_k)
    # slides 22-23: 5 recommendations, hits at ranks 2, 3, 4, |T| = 3
    recs, truth = [9, 1, 2, 3, 8], {1, 2, 3}
    assert round(rankscore_at_k(recs, truth, 5, alpha=2), 2) == 0.71
    assert round(rankscore_at_k(recs, truth, 5, alpha=1), 2) == 0.50
    assert round(ndcg_at_k(recs, truth, 5), 2) == 0.73
    assert round(liftindex_at_k(recs, truth, 5), 2) == 0.60
    # slide 24: AP for hits at 2,3,4 and at 1,4,5
    assert round(ap_at_k(recs, truth, 5), 3) == 0.639
    assert round(ap_at_k([1, 8, 9, 2, 3], truth, 5), 3) == 0.700
    # slide 25: MAP over two users = 0.56
    u1 = ap_at_k([1, 8, 2, 3, 9], {1, 2, 3}, 5)           # (1/1 + 2/3 + 3/4) / 3 = 0.8
    u2 = ap_at_k([7, 8, 9, 1, 2], {1, 2}, 5)              # (1/4 + 2/5) / 2 = 0.325
    assert round(u1, 3) == 0.806 and round(u2, 3) == 0.325
    assert round((u1 + u2) / 2, 2) == 0.57  # slide rounds AP_1 to 0.8 first -> 0.56
    # slide 17: MRR = (1/3 + 1/2 + 1) / 3 = 0.61
    mrr = (mrr_at_k([7, 8, 1], {1}, 3) + mrr_at_k([7, 1, 2], {1, 2}, 3) + mrr_at_k([1, 2, 7], {1, 2}, 3)) / 3
    assert round(mrr, 2) == 0.61
    # slide 12: recall divides by min(P_u, K); slides 9/11: precision and F1
    assert recall_at_k([1, 2], {1, 2, 3, 4, 5}, 2) == 1.0
    assert precision_at_k(recs, truth, 5) == 0.6
    assert round(f1_at_k(recs, truth, 5), 2) == 0.75
