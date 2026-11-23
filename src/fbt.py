"""'Frequently Bought Together' via FP-Growth association rules.

Basket = distinct products a session added to cart or purchased (purchase-only
baskets are too few after sampling; switch with `events=("purchase",)`).

Recommendation for a query item set (current product page or cart):
  1. product-level rules  A -> B whose antecedent is contained in the query
  2. popular products in the query's own categories
  3. category-level rules catA -> catB, filled with popular products of catB
     (2 and 3 swap with same_cat_first=False)
  4. global popular
Each item gets a `reason` so the demo can explain it.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

from evalkit import weighted_scores

BASKET_EVENTS = ("cart", "purchase")


def make_baskets(df: pd.DataFrame, key: str = "product_id", events=BASKET_EVENTS) -> list[frozenset]:
    ev = df[df["event_type"].isin(events)]
    s = ev.groupby("user_session")[key].agg(frozenset)
    return [b for b in s if len(b) >= 2]


def mine_rules(
    baskets: list[frozenset], min_count: int = 3, min_confidence: float = 0.05, max_len: int = 3
) -> pd.DataFrame:
    """FP-Growth on items appearing in >= min_count baskets; returns mlxtend rules table."""
    counts = pd.Series([i for b in baskets for i in b]).value_counts()
    keep = set(counts[counts >= min_count].index)
    baskets = [b & keep for b in baskets]
    baskets = [b for b in baskets if len(b) >= 2]
    if not baskets:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    items = sorted(keep)
    col = {it: j for j, it in enumerate(items)}
    rows, cols = [], []
    for i, b in enumerate(baskets):
        for it in b:
            rows.append(i)
            cols.append(col[it])
    from scipy.sparse import csr_matrix

    mat = csr_matrix(([True] * len(rows), (rows, cols)), shape=(len(baskets), len(items)), dtype=bool)
    # mlxtend needs sparse integer column names starting at 0 -> use positions, map back after
    onehot = pd.DataFrame.sparse.from_spmatrix(mat, columns=range(len(items)))

    freq = fpgrowth(onehot, min_support=min_count / len(baskets), use_colnames=True, max_len=max_len)
    if len(freq) == 0 or freq["itemsets"].map(len).max() < 2:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    freq["itemsets"] = freq["itemsets"].map(lambda s: frozenset(items[j] for j in s))
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]].reset_index(drop=True)


def _index_rules(rules: pd.DataFrame) -> dict[frozenset, list[tuple]]:
    """antecedent -> [(consequent_item, confidence, lift)], single-item consequents only."""
    idx: dict[frozenset, list[tuple]] = defaultdict(list)
    for a, c, conf, lift in rules[["antecedents", "consequents", "confidence", "lift"]].itertuples(index=False):
        if len(c) == 1:
            idx[frozenset(a)].append((next(iter(c)), conf, lift))
    return idx


class FrequentlyBoughtTogether:
    def __init__(
        self,
        min_count=3,
        min_confidence=0.05,
        cat_min_count=5,
        events=BASKET_EVENTS,
        popular_days=7,
        same_cat_first=True,
        same_cat_n=100,
    ):
        self.same_cat_first = same_cat_first
        self.same_cat_n = same_cat_n
        self.min_count = min_count
        self.min_confidence = min_confidence
        self.cat_min_count = cat_min_count
        self.events = events
        self.popular_days = popular_days

    def fit(self, train: pd.DataFrame) -> "FrequentlyBoughtTogether":
        self.item_cat = train.drop_duplicates("product_id").set_index("product_id")["category_id"].to_dict()

        self.rules = mine_rules(make_baskets(train, "product_id", self.events), self.min_count, self.min_confidence)
        self.cat_rules = mine_rules(
            make_baskets(train, "category_id", self.events), self.cat_min_count, self.min_confidence, max_len=2
        )
        self._prod_idx = _index_rules(self.rules)
        self._cat_idx = _index_rules(self.cat_rules)

        cutoff = train["event_time"].max() - pd.Timedelta(days=self.popular_days)
        recent = train[train["event_time"] > cutoff]
        s = weighted_scores(recent).groupby([recent["category_id"], recent["product_id"]]).sum()
        self.cat_top = {
            c: g.sort_values(ascending=False).index.get_level_values("product_id").tolist()[:100]
            for c, g in s.groupby(level="category_id")
        }
        g = weighted_scores(recent).groupby(recent["product_id"]).sum()
        self.global_top = g.sort_values(ascending=False).index.tolist()[:200]
        return self

    def _matching(self, idx, query: frozenset):
        """Yield (item, confidence, lift) from rules whose antecedent ⊆ query."""
        for a, cons in idx.items():
            if a <= query:
                yield from cons

    def recommend_items(self, query_items, k: int = 10) -> list[tuple[int, float, str]]:
        """Return [(product_id, score, reason)] for a product page / cart."""
        query = frozenset(query_items)
        out: dict[int, tuple[float, str]] = {}

        def add(item, score, reason):
            if item not in query and item not in out and len(out) < k:
                out[item] = (score, reason)

        # 1. product rules: score = confidence, ties by lift
        cand: dict[int, tuple[float, float]] = {}
        for it, conf, lift in self._matching(self._prod_idx, query):
            if (conf, lift) > cand.get(it, (0.0, 0.0)):
                cand[it] = (conf, lift)
        for it, (conf, lift) in sorted(cand.items(), key=lambda x: x[1], reverse=True):
            add(it, 2.0 + conf, "Thường được mua cùng")

        qcats = frozenset(self.item_cat[i] for i in query if i in self.item_cat)

        def same_category():
            for c in qcats:
                for it in self.cat_top.get(c, [])[: self.same_cat_n]:
                    add(it, 1.0, "Phổ biến cùng danh mục")

        def category_rules():
            ccand: dict[int, float] = {}
            for c, conf, _ in self._matching(self._cat_idx, qcats):
                ccand[c] = max(conf, ccand.get(c, 0.0))
            for c, conf in sorted(ccand.items(), key=lambda x: x[1], reverse=True):
                for it in self.cat_top.get(c, [])[:k]:
                    add(it, 0.5 + conf / 2, "Danh mục thường mua kèm")

        # 2-3. baskets often hold several products of the same category, so by
        # default same-category popular comes before cross-category rules
        steps = (same_category, category_rules) if self.same_cat_first else (category_rules, same_category)
        for step in steps:
            step()
        # 4. global popular
        for it in self.global_top:
            add(it, 0.0, "Đang thịnh hành")

        return [(it, s, r) for it, (s, r) in out.items()]


def rule_report(rules: pd.DataFrame, top: int = 10) -> str:
    if len(rules) == 0:
        return "no rules"
    d = rules[["support", "confidence", "lift"]].describe().loc[["count", "mean", "50%", "max"]]
    t = rules.sort_values(["lift", "confidence"], ascending=False).head(top).copy()
    t["antecedents"] = t["antecedents"].map(lambda s: ",".join(map(str, sorted(s))))
    t["consequents"] = t["consequents"].map(lambda s: ",".join(map(str, sorted(s))))
    return f"{d.round(4).to_string()}\n\ntop {top} by lift:\n{t.round(4).to_string(index=False)}"
