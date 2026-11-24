import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from fbt import FrequentlyBoughtTogether, make_baskets, mine_rules


def _df():
    rows, t = [], pd.Timestamp("2019-11-01")
    # camera (1) always bought with memory card (2); category 10 -> 20
    for s in range(6):
        rows += [(t, "purchase", 1, 10, s, f"s{s}"), (t, "purchase", 2, 20, s, f"s{s}")]
    # unrelated product 3 (cat 30) bought alone, viewed a lot -> globally popular
    for s in range(6, 20):
        rows += [(t, "view", 3, 30, s, f"v{s}")]
    # product 4 shares category 10 with the camera, popular there
    for s in range(20, 24):
        rows += [(t, "cart", 4, 10, s, f"c{s}")]
    return pd.DataFrame(rows, columns=["event_time", "event_type", "product_id", "category_id", "user_id", "user_session"])


def test_baskets_and_rules():
    baskets = make_baskets(_df())
    assert baskets == [frozenset({1, 2})] * 6
    rules = mine_rules(baskets, min_count=3)
    pairs = {(tuple(a), tuple(c)) for a, c in zip(rules.antecedents, rules.consequents)}
    assert pairs == {((1,), (2,)), ((2,), (1,))}
    assert (rules.confidence == 1.0).all()


def test_recommend_order_and_reasons():
    m = FrequentlyBoughtTogether(min_count=3, cat_min_count=3).fit(_df())
    recs = m.recommend_items([1], k=3)
    assert [i for i, _, _ in recs] == [2, 4, 3]
    assert [r for _, _, r in recs] == ["Thường được mua cùng", "Phổ biến cùng danh mục", "Đang thịnh hành"]
    assert all(i != 1 for i, _, _ in m.recommend_items([1], k=10))  # never recommend the query
