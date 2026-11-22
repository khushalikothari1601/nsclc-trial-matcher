"""Rescore saved predictions against the current gold labels and scorer, without
calling the model. Works on results files written since predictions were saved.

    python eval/rescore.py                       # results.json -> results.json
    python eval/rescore.py --in old.json --out new.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_eval import load_gold, score_record  # noqa: E402
from schema import TrialCriteria  # noqa: E402

HERE = os.path.dirname(__file__)


def rescore(in_path: str, out_path: str, gold_dir: str) -> None:
    with open(in_path) as fh:
        data = json.load(fh)
    cases = {c.nct_id: c for c in load_gold(gold_dir)}

    runs, skipped = [], 0
    for r in data["runs"]:
        if "parse_error" in r:
            runs.append(r)
        elif "prediction" not in r or r["nct_id"] not in cases:
            skipped += 1
        else:
            pred = TrialCriteria.model_validate(r["prediction"])
            runs.append(score_record(cases[r["nct_id"]], pred, r["run"]))
    if skipped and not [r for r in runs if "field_scores" in r]:
        sys.exit(f"{in_path} has no saved predictions to rescore (it predates rescoring).")

    data["runs"] = runs
    data["rescored"] = True
    with open(out_path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    print(f"Rescored {len(runs)} run(s) into {out_path}" + (f"; skipped {skipped}" if skipped else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_path", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--out", default=None, help="default: overwrite --in")
    ap.add_argument("--gold-dir", default=os.path.join(HERE, "gold"))
    args = ap.parse_args()
    rescore(args.in_path, args.out or args.in_path, args.gold_dir)


if __name__ == "__main__":
    main()
