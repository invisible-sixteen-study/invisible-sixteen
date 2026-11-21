# run_fertility.py — tokenizer fertility for the 16 constitutional languages.
# Mirrors paper-1 schema (tokenizer,language,n_sentences,total_words,total_tokens,fertility,tokens_per_sentence)
# with add_special_tokens=False. Usage:
#   python -I run_fertility.py --samples_root <dir> --out CSV
import argparse, csv, os, re, sys

SAMPLES = [
    # (display name, ISO 639-3, relative path under samples_root)
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
    # Chibarwe (bwg), Koisan/Tshwao (no ISO), Nambya (nmq), Sign language (zib):
    # zero open text verified in 06-language-availability.md -> no fertility row (documented absence).
]

N_SENTENCES = 400
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def load_sentences(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    return sents[:N_SENTENCES]


def hf_tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    def enc(s):
        return tok.encode(s, add_special_tokens=False)
    return enc


def tiktoken_enc(name):
    import tiktoken
    enc = tiktoken.get_encoding(name)
    return enc.encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_root", required=True, help="parent dir containing kuj-paper/ and build/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    tokenizers = {
        "Qwen/Qwen2.5-0.5B": hf_tokenizer,
        "hf-internal-testing/llama-tokenizer": hf_tokenizer,
        "mistralai/Mistral-7B-v0.1": hf_tokenizer,
        "google-bert/bert-base-multilingual-cased": hf_tokenizer,
    }
    encoders = {}
    for name, factory in tokenizers.items():
        print(f"loading {name} ...", flush=True)
        encoders[name] = factory(name)
    encoders["openai/o200k_base (GPT-4o)"] = tiktoken_enc("o200k_base")

    for display, iso, rel in SAMPLES:
        path = os.path.join(args.samples_root, rel)
        if not os.path.exists(path):
            print(f"SKIP {display}: missing {path}")
            continue
        sents = load_sentences(path)
        n_words = sum(len(s.split()) for s in sents)
        if n_words == 0:
            print(f"SKIP {display}: empty")
            continue
        for tname, enc in encoders.items():
            n_tok = sum(len(enc(s)) for s in sents)
            fert = n_tok / n_words
            tps = n_tok / len(sents)
            rows.append({
                "tokenizer": tname, "language": display, "iso639-3": iso,
                "n_sentences": len(sents), "total_words": n_words,
                "total_tokens": n_tok, "fertility": round(fert, 4),
                "tokens_per_sentence": round(tps, 2),
            })
            print(f"  {tname:45s} {display:30s} fert={fert:.3f}")
        sys.stdout.flush()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"WROTE {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
