# run_perplexity.py — causal-LM perplexity per language (standard chunked NLL method).
# Usage:
#   python -I run_perplexity.py --model Qwen/Qwen2.5-0.5B --samples_root DIR --out CSV [--max_tokens 20000]
import argparse, csv, os, sys

import torch

SENT_SPLIT = None  # we evaluate on contiguous text, not sentences

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


@torch.no_grad()
def ppl_of_text(model, tok, text, block=512, max_tokens=20000):
    ids = tok.encode(text, add_special_tokens=False)
    ids = ids[:max_tokens]
    if len(ids) < block // 2:
        return None, len(ids)
    nll_sum, n_tokens = 0.0, 0
    for start in range(0, len(ids) - 1, block):
        chunk = ids[start:start + block + 1]
        if len(chunk) < 2:
            break
        inp = torch.tensor([chunk[:-1]])
        tgt = torch.tensor([chunk[1:]])
        out = model(inp)
        logits = out.logits
        nll = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), tgt.view(-1), reduction="sum"
        )
        nll_sum += nll.item()
        n_tokens += tgt.numel()
    avg_nll = nll_sum / n_tokens
    return float(torch.exp(torch.tensor(avg_nll))), n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=20000)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()

    rows = []
    for display, iso, rel in SAMPLES:
        path = os.path.join(args.samples_root, rel)
        if not os.path.exists(path):
            print(f"SKIP {display}: missing {path}")
            continue
        text = open(path, encoding="utf-8").read()
        ppl, n_tok = ppl_of_text(model, tok, text, max_tokens=args.max_tokens)
        if ppl is None:
            print(f"SKIP {display}: too few tokens ({n_tok})")
            continue
        rows.append({"model": args.model, "language": display, "iso639-3": iso,
                     "n_tokens": n_tok, "perplexity": round(ppl, 2)})
        print(f"  {display:30s} ppl={ppl:.2f} (n={n_tok})", flush=True)
        sys.stdout.flush()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"WROTE {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
