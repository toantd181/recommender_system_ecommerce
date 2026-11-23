"""Fit every model the demo needs and pickle them to artifacts/.

Settings (history per module, co-visitation variant, FP-Growth min_count, hybrid
share) are read from results/experiments.json, i.e. what won on the validation week.

Usage:
    .venv/bin/python src/build_artifacts.py --nov data/nov_u5.parquet --octnov data/octnov_u5.parquet
"""
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from baselines import PopularByCategory, PopularRecent
from evalkit import load_events
from fbt import FrequentlyBoughtTogether
from item2vec import CoVisitation, CoVisitThenItem2Vec, Item2Vec
from ltr_session import SessionLTR, fit_ranker, item_table

DEFAULT_CHOSEN = {
    "data_users": "nov", "data_session": "nov", "data_fbt": "nov",
    "covisit": {}, "share": 2, "fbt_min_count": 3, "fbt_same_cat_first": True,
}


def product_catalog(df: pd.DataFrame) -> pd.DataFrame:
    meta = df.sort_values("event_time").drop_duplicates("product_id", keep="last")
    meta = meta[["product_id", "category_id", "category_code", "brand", "price"]].set_index("product_id")
    meta["category_code"] = meta["category_code"].fillna("unknown")
    meta["brand"] = meta["brand"].fillna("unknown")
    meta["main_category"] = meta["category_code"].str.split(".").str[0]
    counts = df.groupby("product_id").size().rename("n_events")
    return meta.join(counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nov", default="data/nov_u5.parquet")
    ap.add_argument("--octnov", default="data/octnov_u5.parquet")
    ap.add_argument("--chosen", default="results/experiments.json")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--min-product-events", type=int, default=20, help="catalog shown in the demo pickers")
    ap.add_argument("--no-ltr", action="store_true", help="skip the session re-ranker (faster build)")
    args = ap.parse_args()

    chosen = dict(DEFAULT_CHOSEN)
    if Path(args.chosen).exists():
        chosen |= json.loads(Path(args.chosen).read_text())["chosen"]
    print("settings:", chosen, flush=True)

    data = {"nov": load_events(args.nov)}
    if "oct+nov" in {chosen["data_users"], chosen["data_session"], chosen["data_fbt"]}:
        data["oct+nov"] = load_events(args.octnov)
    df_users, df_sess, df_fbt = (data[chosen[k]] for k in ("data_users", "data_session", "data_fbt"))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nov = data["nov"]
    catalog = product_catalog(df_fbt if len(df_fbt) > len(nov) else nov)
    user_hist = df_users.groupby("user_id").size()
    dense = user_hist[user_hist.between(20, 200)]
    artifacts = {
        "settings": chosen,
        "catalog": catalog,
        "demo_products": catalog[catalog["n_events"] >= args.min_product_events].index.tolist(),
        "demo_users": dense.sample(min(200, len(dense)), random_state=0).index.tolist(),
        "history": df_users[df_users["user_id"].isin(dense.index)][["user_id", "event_time", "event_type", "product_id"]],
    }

    def fit(name, model, df):
        print(f"fitting {name} on {len(df):,} events ...", flush=True)
        artifacts[name] = model.fit(df) or model
        return artifacts[name]

    fit("trending", PopularRecent(7), df_users)
    fit("old_member", PopularByCategory(7), df_users)  # placeholder until the ALS module is ready
    fbt = FrequentlyBoughtTogether(min_count=chosen["fbt_min_count"])
    fbt.same_cat_first = chosen["fbt_same_cat_first"]
    fit("fbt", fbt, df_fbt)
    cv = fit("covisit", CoVisitation(**chosen["covisit"]), df_sess)
    i2v = fit("item2vec", Item2Vec(side_info=True), df_sess)

    hybrid = CoVisitThenItem2Vec(cv, i2v, i2v_share=chosen["share"])
    hybrid.popular = cv.popular
    artifacts["hybrid"] = hybrid
    if not args.no_ltr:
        print("training session re-ranker on the validation week ...", flush=True)
        ranker = fit_ranker(df_sess, chosen["covisit"])
        artifacts["ltr"] = SessionLTR(cv, i2v, item_table(df_sess), ranker)

    path = out / "models.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {path} ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
