# score_generation.py — chrF + refusal rates for the Part C probe.
# Usage: python -I score_generation.py --in JSONL --out CSV
import argparse, collections, csv, json, os

import sacrebleu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    groups = collections.defaultdict(lambda: {"hyp": [], "ref": [], "refusals": 0, "meta": 0, "errors": 0, "n": 0})
    META_KW = ["let's check", "standard bhaibheri", "here is", "translation:", "note:", "according to"]
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["model"], r["language"])
            g = groups[key]
            g["n"] += 1
            if (not r.get("parsed")) or r.get("error"):
                g["errors"] += 1
                g["hyp"].append("")
                g["ref"].append(r["reference"])
                continue
            low = r["output"].lower()
            if r["refusal_flag"]:
                g["refusals"] += 1
            if any(k in low for k in META_KW):
                g["meta"] += 1
            g["hyp"].append(r["output"])
            g["ref"].append(r["reference"])

    rows = []
    for (model, lang), g in sorted(groups.items()):
        chrf = sacrebleu.corpus_chrf(g["hyp"], [g["ref"]], beta=2.0)
        rows.append({
            "model": model, "language": lang, "n_items": g["n"],
            "chrf": round(chrf.score, 1),
            "refusal_rate": round(100 * (g["refusals"] / g["n"]), 1),
            "error_rate": round(100 * (g["errors"] / g["n"]), 1),
            "meta_commentary_rate": round(100 * (g["meta"] / g["n"]), 1),
        })
        print(f"{model:24s} {lang:5s} chrF={chrf.score:5.1f}  refusals={rows[-1]['refusal_rate']}%  meta={rows[-1]['meta_commentary_rate']}%  errors={rows[-1]['error_rate']}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"WROTE {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
