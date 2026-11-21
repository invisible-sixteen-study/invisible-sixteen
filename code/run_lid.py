# run_lid.py — language-ID confusion for sampled constitutional languages.
# Uses langid (pure Python, 97-language catalogue). Sentence-level classification;
# reports predicted-language distribution per true language. Misclassification of
# related Bantu languages is a finding (Kreutzer et al. 2022 discipline).
# Usage: python -I run_lid.py --samples_root DIR --out CSV --max_sentences 300
import argparse, collections, csv, os, re

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

SAMPLES = [
    ("English", "eng", "kuj-paper/data/samples/english.txt"),
    ("Shona", "sna", "kuj-paper/data/samples/shona.txt"),
    ("Ndebele (Northern, Zimbabwe)", "nde", "kuj-paper/data/samples/northern_ndebele.txt"),
    ("Ndebele (Southern, SA)", "nbl", "kuj-paper/data/samples/southern_ndebele.txt"),
    ("Tswana", "tsn", "africa-ai-research/paper2/build/data/wiki/tn_wikipedia.txt"),
    ("Sotho (Southern)", "sot", "africa-ai-research/paper2/build/data/wiki/st_wikipedia.txt"),
    ("Xitsonga / Shangani", "tso", "africa-ai-research/paper2/build/data/wiki/ts_wikipedia.txt"),
    ("Chewa", "nya", "africa-ai-research/paper2/build/data/wiki/ny_wikipedia.txt"),
    ("Venda", "ven", "africa-ai-research/paper2/build/data/wiki/ve_wikipedia.txt"),
    ("Tonga (Zimbabwe)", "tog", "africa-ai-research/paper2/build/data/fineweb2/tog_Latn_fineweb2.txt"),
    ("Ndau", "ndc", "africa-ai-research/paper2/build/data/fineweb2/ndc_Latn_fineweb2.txt"),
    ("Kalanga", "kck", "africa-ai-research/paper2/build/data/fineweb2/kck_Latn_fineweb2.txt"),
]

# ISO 639-3 -> langid's 639-1 codes (for correct equality checks)
ALIAS = {
    "eng": {"en", "eng"}, "sna": {"sn", "sna"}, "nde": {"nd", "nde"},
    "nbl": {"nr", "nbl", "nb"}, "tsn": {"tn", "tsn"}, "sot": {"st", "sot"},
    "tso": {"ts", "tso"}, "nya": {"ny", "nya"}, "ven": {"ve", "ven"},
    "tog": {"tog"}, "ndc": {"ndc"}, "kck": {"kck"},
}

# candidate Bantu/neighbour codes to watch in the confusion output
WATCH = {"sn", "nd", "nr", "ny", "st", "tn", "ts", "ve", "so", "zu", "xh", "sw", "lg", "rw", "bm", "mg", "bem", "lo"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_sentences", type=int, default=300)
    args = ap.parse_args()

    import langid
    langid.set_languages(None)  # full 97-language catalogue

    rows = []
    for display, iso, rel in SAMPLES:
        path = os.path.join(args.samples_root, rel)
        if not os.path.exists(path):
            print(f"SKIP {display}: missing {path}")
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        sents = [s.strip() for s in SENT_SPLIT.split(text) if len(s.split()) >= 3]
        sents = sents[: args.max_sentences]
        if not sents:
            print(f"SKIP {display}: empty")
            continue
        pred = collections.Counter()
        conf = 0
        aliases = ALIAS.get(iso, {iso})
        for s in sents:
            code, _score = langid.classify(s)
            pred[code] += 1
            if code not in aliases:
                conf += 1
        rate = conf / len(sents)
        top = "; ".join(f"{c}={n}" for c, n in pred.most_common(5))
        rows.append({
            "language": display, "iso639-3": iso, "n_sentences": len(sents),
            "misclassification_rate": round(rate, 4),
            "top_predictions": top,
            "watchlist_hits": "; ".join(f"{c}={pred[c]}" for c in WATCH if pred[c]),
        })
        print(f"{display:30s} mis={rate:.2%}  top: {top}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"WROTE {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
