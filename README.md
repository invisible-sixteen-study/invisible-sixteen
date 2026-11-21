# Invisible Sixteen — article artefacts

Code, data and outputs for the article *"Invisible sixteen: How well do large
language models handle Zimbabwe's constitutionally recognised languages?"*
by the article authors.

The article measures, for all sixteen languages recognised by Section 6 of
the Constitution of Zimbabwe (2013): (A) the availability of open text across
ten source classes; (B) processing cost (tokenizer fertility, perplexity,
language-ID confusion); and (C) frontier-model translation quality against
parallel references (chrF), with free-tier quota attrition documented.

## Repository map

```
code/        analysis pipeline (Python 3; see per-script headers)
  fetch_samples.py            Wikipedia + FineWeb-2 text sampler
  run_fertility.py            tokenizer fertility (5 tokenizers x 12 languages)
  run_lid.py                  language-ID confusion (langid)
  run_perplexity.py           causal-LM perplexity (Qwen2.5-0.5B)
  run_generation_probe.py     Part C translation probe (4 frontier endpoints)
  repair_generation_probe.py  quota-loss repair pass
  score_generation.py         chrF / refusal / error scoring (sacrebleu)
data/        audit and measurement outputs (CSV)
generation/  reference-set specifications, probe logs and scores
figure/      Figure 1 at 300 dpi
```

## Requirements

Python 3.10+: `pandas`, `requests`, `matplotlib`, `transformers`, `tiktoken`,
`langid`, `sacrebleu`, `pyarrow`, `torch` (CPU sufficient for the perplexity
runs). API keys for the generation probe: Gemini (AI Studio) and OpenRouter,
read from environment variables — never committed.

## Data sourcing and licences

- FineWeb-2 slices (ODC-BY 1.0), Wikipedia text (CC BY-SA 4.0) — derived
  statistics only are redistributed here.
- Verse-aligned Shona references: OPUS bible-uedin (CC0-claimed; the OPUS
  index lists the language as `sn`).
- Biblica Bible texts, Teleagents audio and any raw third-party text are
  NOT redistributed; only derived scores appear in this repository.
- `generation_merged.jsonl` contains model outputs and reference sentences
  used for scoring; if redistribution terms change, item IDs + scores remain.

## Citation

Author details withheld for double-blind review. *Invisible sixteen: How well do large language models handle
Zimbabwe's constitutionally recognised languages?* (article DOI to be added
on publication).

## Licence

Code: MIT. Data/score files: CC BY 4.0. Third-party source texts remain under
their own licences (see above).
