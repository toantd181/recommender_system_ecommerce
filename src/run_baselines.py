"""Evaluate baselines on a parquet file.

Usage:
    .venv/bin/python src/run_baselines.py data/nov_u5.parquet --test-start 2019-11-24
"""
import argparse

from baselines import PopularByCategory, PopularRecent
from evalkit import evaluate, load_events, temporal_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--test-start", default="2019-11-24")
    ap.add_argument("--test-end", default=None)
    args = ap.parse_args()

    df = load_events(args.parquet)
    train, test = temporal_split(df, args.test_start, args.test_end)
    print(f"train {len(train):,} events / test {len(test):,} events")
    for model in (PopularRecent(7), PopularByCategory(7)):
        print(evaluate(model, train, test))


if __name__ == "__main__":
    main()
