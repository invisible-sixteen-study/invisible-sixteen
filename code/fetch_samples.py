# fetch_samples.py — gather per-language text samples for fertility/perplexity/LID.
# Sources: Wikipedia editions (CC BY-SA; derived stats fine) and FineWeb-2 shards
# (ODC-BY 1.0; derived stats fine, no raw redistribution).
# Usage: python -I fetch_samples.py --wiki_out DIR --fw_out DIR --mode wiki|fineweb2|both
import json, os, sys, time, argparse
import requests

UA = {"User-Agent": "InvisibleSixteen-research/0.1 (academic language-availability audit; contact: alfred.tinonetsana@students.uz.ac.zw)"}

WIKI_CODES = ["tn", "st", "ts", "ny", "ve"]  # editions verified to exist (06-language-availability.md)
FW2_LANGS = ["tog_Latn", "ndc_Latn", "kck_Latn"]  # tiny shards: 2,858 / 779 / 132 rows
FW2_BASE = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2"
FW2_API = "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-2"
TARGET_WORDS = 15000
MAX_PAGES = 400


def fetch_wiki(code, out_dir):
    out_path = os.path.join(out_dir, f"{code}_wikipedia.txt")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        print(f"[wiki] {code}: already fetched, skipping")
        return
    api = f"https://{code}.wikipedia.org/w/api.php"
    texts, words, pages, seen = [], 0, 0, set()
    for _round in range(30):  # random generator returns fresh pages each call; dedupe by title
        if words >= TARGET_WORDS or pages >= MAX_PAGES:
            break
        params = {
            "action": "query", "format": "json",
            "generator": "random", "grnnamespace": 0, "grnlimit": 50,
            "prop": "extracts", "explaintext": 1, "exlimit": "max",
        }
        r = requests.get(api, params=params, headers=UA, timeout=30)
        r.raise_for_status()
        data = r.json()
        batch = data.get("query", {}).get("pages", {})
        if not batch:
            break
        for pid, pg in batch.items():
            title = pg.get("title", pid)
            if title in seen:
                continue
            seen.add(title)
            ext = pg.get("extract", "")
            if len(ext) > 200:  # skip stubs
                texts.append(ext.strip())
                words += len(ext.split())
                pages += 1
        time.sleep(0.3)
    joined = "\n\n".join(texts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(joined)
    print(f"[wiki] {code}: pages={pages} words={words} -> {out_path}")


def fetch_fineweb2(lang, out_dir):
    out_path = os.path.join(out_dir, f"{lang}_fineweb2.txt")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(f"[fw2] {lang}: already fetched, skipping")
        return
    # list shard files
    r = requests.get(f"{FW2_API}/tree/main/data/{lang}/train", headers=UA, timeout=30)
    r.raise_for_status()
    files = [x["path"] for x in r.json() if x["path"].endswith(".parquet")]
    if not files:
        # some configs keep data at data/<lang>/ root
        r = requests.get(f"{FW2_API}/tree/main/data/{lang}", headers=UA, timeout=30)
        r.raise_for_status()
        files = [x["path"] for x in r.json() if x["path"].endswith(".parquet")]
    if not files:
        print(f"[fw2] {lang}: no parquet files found")
        return
    total_rows, all_text = 0, []
    for rel in files:
        url = f"{FW2_BASE}/resolve/main/{rel}"
        rd = requests.get(url, headers=UA, timeout=120)
        rd.raise_for_status()
        tmp = os.path.join(out_dir, f"_tmp_{lang}.parquet")
        with open(tmp, "wb") as f:
            f.write(rd.content)
        import pyarrow.parquet as pq
        tbl = pq.read_table(tmp, columns=["text"])
        texts = tbl.column("text").to_pylist()
        all_text.extend(texts)
        total_rows += len(texts)
        os.remove(tmp)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))
    print(f"[fw2] {lang}: rows={total_rows} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki_out", required=True)
    ap.add_argument("--fw_out", required=True)
    ap.add_argument("--mode", default="both", choices=["wiki", "fineweb2", "both"])
    ap.add_argument("--codes", default=",".join(WIKI_CODES), help="comma-separated wiki codes")
    args = ap.parse_args()
    os.makedirs(args.wiki_out, exist_ok=True)
    os.makedirs(args.fw_out, exist_ok=True)
    if args.mode in ("wiki", "both"):
        for c in args.codes.split(","):
            try:
                fetch_wiki(c, args.wiki_out)
            except Exception as e:
                print(f"[wiki] {c}: ERROR {e}")
    if args.mode in ("fineweb2", "both"):
        for lang in FW2_LANGS:
            try:
                fetch_fineweb2(lang, args.fw_out)
            except Exception as e:
                print(f"[fw2] {lang}: ERROR {e}")


if __name__ == "__main__":
    main()
