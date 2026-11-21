# run_generation_probe.py — Part C: frontier-model translation probe.
# 5 reference languages x 30 items x 4 frontier models, temperature 0.
# Reuses kuj-paper/harness/llm_client.py (keys from kuj-paper/.env, never printed).
# Usage: python -I run_generation_probe.py --out JSONL
import argparse, json, os, sys, time

HARNESS = r"C:\Users\Administrator\Documents\Books\Data Sci\Research Project\kuj-paper\harness"
REFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "references", "reference_sets.json")

sys.path.insert(0, HARNESS)
import llm_client  # noqa: E402

REFUSAL_KW = ["i cannot", "i can't", "i can’t", "sorry", "unable", "as an ai",
              "i'm not able", "cannot translate", "cannot provide"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="gemini-3.5-flash,gemini-3.5-flash-lite,nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free,liquid/lfm-2.5-2.6b:free")
    args = ap.parse_args()

    refs = json.load(open(REFS, encoding="utf-8"))
    models = []
    mcfg = json.load(open(os.path.join(HARNESS, "models.json"), encoding="utf-8"))
    wanted = args.models.split(",")
    for m in mcfg["models"]:
        if m["id"] in wanted and m.get("include", True):
            models.append(m)
    print(f"models: {[m['id'] for m in models]}", flush=True)

    out_f = open(args.out, "w", encoding="utf-8")
    n_done = 0
    t_start = time.time()
    for m in models:
        for iso, pack in refs.items():
            lang_name = pack["name"]
            for item in pack["items"]:
                prompt = (f"Translate the following English sentence into {lang_name}. "
                          f"Reply with only the translation, nothing else.\n\nEnglish: {item['en']}")
                res = llm_client.ask(m, prompt, timeout=120, max_tokens=200)
                text = res.get("answer", "") or ""
                low = text.lower()
                refusal = bool(
                    (not res.get("parsed")) or any(k in low for k in REFUSAL_KW)
                    or text.strip() == item["en"].strip()
                )
                rec = {
                    "model": m["label"], "model_id": m["id"], "provider": m["provider"],
                    "language": iso, "lang_name": lang_name, "item_id": item["id"],
                    "source": item["en"], "reference": item["ref"],
                    "output": text, "parsed": res.get("parsed"),
                    "error": res.get("error"), "latency_s": res.get("latency_s"),
                    "refusal_flag": refusal,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                if n_done % 25 == 0:
                    print(f"{n_done} calls done ({time.time()-t_start:.0f}s)", flush=True)
                time.sleep(2)
    out_f.close()
    print(f"DONE {n_done} calls -> {args.out}")


if __name__ == "__main__":
    main()
