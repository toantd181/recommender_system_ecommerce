"""Collect the headline test-week numbers into results/summary.json for the demo's 'Đánh giá' tab.

Reads results/router.txt (stage-2 tables + CIs), results/ltr_session.json, results/experiments.json.
    .venv/bin/python src/make_summary.py
"""
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"


def parse_router(text: str) -> dict:
    test = text[text.index("STAGE 2"):]
    out = {}
    for name, block in re.findall(r"== (\w+) ==\n(.*?)(?=\n== |\n\nhit@10)", test, flags=re.S):
        lines = block.strip().splitlines()
        cols = lines[0].split()
        rows = {}
        for ln in lines[1:]:
            parts = ln.split()
            if parts[0] in ("all", "unseen", "light", "heavy"):
                rows[parts[0]] = dict(zip(cols, map(float, parts[1:])))
        out[name] = {g: {"n_users": int(r["n_users"]), "hit@10": r["hit@10"], "ndcg@10": r["ndcg@10"]}
                     for g, r in rows.items()}
    ci = {}
    for name, h, hl, hh, d, dl, dh in re.findall(
            r"^\s{2}(\w+)\s+hit@10 ([\d.]+) \[([\d.]+), ([\d.]+)\]\s+ndcg@10 ([\d.]+) \[([\d.]+), ([\d.]+)\]",
            test, flags=re.M):
        ci[name] = {"hit@10": [float(h), float(hl), float(hh)], "ndcg@10": [float(d), float(dl), float(dh)]}
    return {"groups": out, "ci": ci}


def main() -> None:
    router = parse_router((R / "router.txt").read_text())
    ltr = json.loads((R / "ltr_session.json").read_text())
    exp = json.loads((R / "experiments.json").read_text())
    summary = {
        "test_window": "2019-11-24 → 2019-11-30",
        "router": router,
        "session": {"hit@10": ltr["table"]["hit@10"], "candidate_recall": ltr["candidate_recall"]},
        "fbt": exp["test_fbt"]["hit@10"],
        "routing": json.loads((R / "router.json").read_text())["routing"],
    }
    (R / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print("saved", R / "summary.json")
    print(json.dumps(router["ci"], indent=1)[:500])


if __name__ == "__main__":
    main()
