"""Evaluate 'Frequently Bought Together' on held-out test baskets.

Protocol (leave-one-out): for every test basket (cart+purchase session with >=2
distinct products) and every item in it, query with that item and treat the
rest of the basket as ground truth. Compared against popularity baselines that
get the same query.

Usage:
    .venv/bin/python src/run_fbt.py data/nov_u5.parquet --test-start 2019-11-24
"""
import argparse

import pandas as pd

from evalkit import METRICS, load_events, temporal_split
from fbt import FrequentlyBoughtTogether, make_baskets, rule_report


def evaluate_fbt(model: FrequentlyBoughtTogether, baskets, ks=(5, 10)) -> pd.DataFrame:
    kmax = max(ks)

    def fbt_with(same_cat_first):
        def run(q):
            model.same_cat_first = same_cat_first
            return [i for i, _, _ in model.recommend_items(q, kmax)]
        return run

    systems = {
        "fbt": fbt_with(True),
        "fbt_catrules_first": fbt_with(False),
        "fbt_rules_only": lambda q: [i for i, _, r in model.recommend_items(q, kmax) if r == "Thường được mua cùng"],
        "popular_same_cat": lambda q: [
            i for i in model.cat_top.get(model.item_cat.get(next(iter(q))), []) if i not in q
        ][:kmax],
        "popular_global": lambda q: [i for i in model.global_top if i not in q][:kmax],
    }
    rows = []
    for b in baskets:
        for q in b:
            truth = set(b) - {q}
            for name, fn in systems.items():
                recs = fn({q})
                row = {"system": name, "has_rule": bool(model._prod_idx.get(frozenset({q})))}
                for k in ks:
                    for m, f in METRICS.items():
                        row[f"{m}@{k}"] = f(recs, truth, k)
                rows.append(row)
    res = pd.DataFrame(rows)
    cols = [c for c in res.columns if "@" in c]
    table = res.groupby("system", sort=False)[cols].mean()
    table.insert(0, "n_queries", res.groupby("system", sort=False).size())
    cov = res[res.system == "fbt"]["has_rule"].mean()
    model.same_cat_first = True
    # how each system does on queries that do / do not have a product rule
    split = res.groupby(["has_rule", "system"], sort=False)[[f"hit@{ks[0]}", f"hit@{kmax}"]].mean().unstack(0)
    return table, cov, split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--test-start", default="2019-11-24")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--min-confidence", type=float, default=0.05)
    ap.add_argument("--purchase-only", action="store_true")
    args = ap.parse_args()

    events = ("purchase",) if args.purchase_only else ("cart", "purchase")
    df = load_events(args.parquet)
    train, test = temporal_split(df, args.test_start)

    model = FrequentlyBoughtTogether(args.min_count, args.min_confidence, events=events).fit(train)
    print(f"train baskets: {len(make_baskets(train, events=events)):,}")
    print(f"product rules: {len(model.rules):,} | category rules: {len(model.cat_rules):,}\n")
    print("== product rules ==\n" + rule_report(model.rules) + "\n")
    print("== category rules ==\n" + rule_report(model.cat_rules, 5) + "\n")

    test_baskets = make_baskets(test, events=events)
    table, cov, split = evaluate_fbt(model, test_baskets)
    print(f"test baskets: {len(test_baskets):,} | queries with >=1 product rule: {cov:.1%}")
    print(table.round(4).to_string())
    print("\nhit rate split by whether the query item has a product rule:")
    print(split.round(4).to_string())


if __name__ == "__main__":
    main()
