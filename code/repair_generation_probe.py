# repair_generation_probe.py — re-run failed probe cells with per-model pacing.
# - gemini-3.5-flash       : 429s -> sleep 8s + exponential backoff
# - nemotron-3-nano (free) : 429s + finish=length -> max_tokens 1500, sleep 8s + backoff
# - lfm-2.5-2.6b           : HTTP 404 (free variant removed from OpenRouter) -> skipped permanently
# Usage: python -I repair_generation_probe.py --in JSONL --out JSONL
import argparse, json, os, sys, time

HARNESS = r"C:\Users\Administrator\Documents\Books\Data Sci\Research Project\kuj-paper\harness"
sys.path.insert(0, HARNESS)
import llm_client  # noqa: E402

REFUSAL_KW = ["i cannot", "i can't", "i can’t", "sorry", "unable", "as an ai",
              "i'm not able", "cannot translate", "cannot provide"]

SKIP_MODELS = {"lfm-2.5-2.6b"}  # lfm: HTTP 404 (removed from OpenRouter Oct/Nov 2026)
MAX_TOKENS = {"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": 1500}
SLEEP_S = 15.0
BACKOFFS = [30, 120, 300]


def call_with_backoff(model_cfg, prompt, max_tokens):
    for attempt, wait in enumerate([0] + BACKOFFS):
        if wait:
            time.sleep(wait)
        res = llm_client.ask(model_cfg, prompt, timeout=180, max_tokens=max_tokens)
        if res.get("parsed") and not (res.get("error") or "").startswith("HTTP 429"):
            return res
        print(f"    retry {attempt+1} after {res.get('error', '')[:60]}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    mcfg = json.load(open(os.path.join(HARNESS, "models.json"), encoding="utf-8"))
    cfgs = {m["id"]: m for m in mcfg["models"]}

    n_done, n_fixed, n_skip = 0, 0, 0
    out_f = open(args.out, "w", encoding="utf-8")
    for r in records:
        if r["model_id"] in SKIP_MODELS:
            n_skip += 1
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            continue
        if r.get("parsed") and not r.get("error"):
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            continue
        m = cfgs.get(r["model_id"])
        if not m:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            continue
        prompt = (f"Translate the following English sentence into {r['lang_name']}. "
                  f"Reply with only the translation, nothing else.\n\nEnglish: {r['source']}")
        res = call_with_backoff(m, prompt, MAX_TOKENS.get(r["model_id"], 200))
        text = (res.get("answer") or "")
        low = text.lower()
        refusal = bool((not res.get("parsed")) or any(k in low for k in REFUSAL_KW)
                       or text.strip() == r["source"].strip())
        r.update({"output": text, "parsed": res.get("parsed"), "error": res.get("error"),
                  "latency_s": res.get("latency_s"), "refusal_flag": refusal,
                  "repaired": True})
        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out_f.flush()
        n_done += 1
        if res.get("parsed"):
            n_fixed += 1
        if n_done % 20 == 0:
            print(f"{n_done} repaired ({n_fixed} ok), {time.time() % 60:.0f}s", flush=True)
        time.sleep(SLEEP_S)
    out_f.close()
    print(f"REPAIR DONE: {n_done} re-run, {n_fixed} fixed, {n_skip} skipped (lfm 404)")


if __name__ == "__main__":
    main()
