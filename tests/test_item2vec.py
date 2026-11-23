import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from item2vec import CoVisitation, Item2Vec, session_sequences


def _df():
    rows, t = [], pd.Timestamp("2019-11-01")
    # phones (1,2) and accessories (3,4) co-viewed; laptops (5,6) in separate sessions
    for s in range(60):
        seq = [1, 3, 1, 4] if s % 2 else [2, 3, 2, 4]
        if s % 3 == 0:
            seq = [5, 6, 5]
        for i, p in enumerate(seq):
            rows.append((t + pd.Timedelta(seconds=i), "view", p, 10 + p // 5, "b", s, f"s{s}"))
    return pd.DataFrame(rows, columns=["event_time", "event_type", "product_id", "category_id", "brand", "user_id", "user_session"])


def test_session_sequences_collapse_repeats():
    df = pd.DataFrame({
        "event_time": pd.to_datetime(["2019-11-01 00:00:0%d" % i for i in range(5)]),
        "product_id": [1, 1, 2, 2, 1], "user_session": ["a"] * 4 + ["b"],
    })
    seq = session_sequences(df)
    assert seq["items"].tolist() == [[1, 2]]  # session b has length 1 -> dropped


def _clusters():
    """Two disjoint clusters (1-5 and 11-15); sessions are shuffles within one cluster."""
    import random
    rng, rows, t = random.Random(0), [], pd.Timestamp("2019-11-01")
    for s in range(400):
        items = rng.sample(range(1, 6) if s % 2 else range(11, 16), 4)
        for i, p in enumerate(items):
            rows.append((t + pd.Timedelta(seconds=i), "view", p, p // 10, "b", s, f"s{s}"))
    return pd.DataFrame(rows, columns=["event_time", "event_type", "product_id", "category_id", "brand", "user_id", "user_session"])


def test_covisit_neighbours():
    cv = CoVisitation().fit(_df())
    assert {p for p, _, _ in cv.recommend_session([5], k=1)} == {6}


def test_item2vec_neighbours_stay_in_cluster():
    # skip-gram similarity is second-order (shared contexts), hence a cluster toy set.
    # sample=0: in a 10-item vocab every token is "frequent" and gensim's default
    # downsampling (1e-3) would drop most of them.
    df = _clusters()
    for side in (False, True):
        m = Item2Vec(vector_size=16, min_count=1, epochs=20, sample=0, workers=1, side_info=side).fit(df)
        recs = [p for p, _, _ in m.recommend_session([1], k=4)]
        assert set(recs) == {2, 3, 4, 5}, (side, recs)
        assert m.recommend_batch([[1], [999]], 3)[1]  # unknown item -> popular fallback, not empty


def test_weighted_covisit_prefers_carted_pairs():
    t = pd.Timestamp("2019-11-01")
    rows = []
    # 1 co-viewed with 2 three times (views only), with 3 twice but carted
    for s in range(3):
        rows += [(t, "view", 1, s, f"a{s}"), (t + pd.Timedelta(seconds=1), "view", 2, s, f"a{s}")]
    for s in range(2):
        rows += [(t, "view", 1, s, f"b{s}"), (t + pd.Timedelta(seconds=1), "view", 3, s, f"b{s}"),
                 (t + pd.Timedelta(seconds=2), "cart", 3, s, f"b{s}")]
    df = pd.DataFrame(rows, columns=["event_time", "event_type", "product_id", "user_id", "user_session"])
    assert CoVisitation().fit(df).recommend_session([1], 1)[0][0] == 2
    assert CoVisitation(type_weighted=True).fit(df).recommend_session([1], 1)[0][0] == 3


def test_session_ltr_end_to_end():
    import numpy as np
    from ltr_session import FEATURES, SessionLTR, build, item_table, train_ranker
    df = _clusters()
    df["price"] = 10.0
    cv = CoVisitation().fit(df)
    i2v = Item2Vec(vector_size=16, min_count=1, epochs=5, sample=0, workers=1, side_info=True).fit(df)
    P = [[1, 2], [11, 12, 11], [3]] * 40
    T = [3, 12, 4] * 40  # second pattern: user goes back to 12 (a revisit)
    f = build(cv, i2v, item_table(df), P, T)
    assert set(FEATURES) <= set(f.columns) and f.groupby("qid")["label"].sum().max() == 1
    assert not ((f["cand"] == np.array([p[-1] for p in P])[f["qid"]])).any()  # never the last item
    ranker = train_ranker(f, np.arange(len(P)))
    m = SessionLTR(cv, i2v, item_table(df), ranker)
    recs = m.recommend_session([11, 12, 11], k=5)
    assert len(recs) == 5 and recs[0][2] == "Bạn đã xem trước đó" and recs[0][0] == 12
    assert len(m.recommend_session([], k=3)) == 3
