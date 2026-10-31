# Offline transcript-cleanup comparison

The runner `scripts/run-transcript-cleanup-benchmark.py` prepares immutable
requests and resumes inference against Ollama at `127.0.0.1:11434` only. It never
downloads models, updates the app, or writes to the source transcript. Store
private contexts and outputs under ignored `build/reports/`, not in Git.

This is a Mac quality experiment. Its times are not Android performance results,
and Ollama/GGUF output is not guaranteed to match a LiteRT-LM deployment.

## Research informing the prompts

These are research-informed adaptations, not a published prompt validated on
German Parakeet output or on these four exact checkpoints.

- [Generative Speech Recognition Error Correction with Large Language Models
  and Task-Activating Prompting](https://arxiv.org/abs/2309.15649) investigates
  post-ASR correction and in-context task activation. Its N-best setup differs
  from our single transcript; we do not manufacture alternative ASR hypotheses.
- [Contextual Modeling for Document-level ASR Error Correction](https://aclanthology.org/2024.lrec-main.341/)
  finds benefits from document context using a retrieval-based correction model.
  This motivates including original neighboring passages; it does not establish
  the effectiveness of our prompt.
- [An Exploratory Framework for LLM-assisted Human Annotation of Speech
  Datasets](https://www.isca-archive.org/interspeech_2025/johnson25_interspeech.pdf)
  tests human-corrected examples and entity-focused correction on English earnings
  calls. It chunks transcripts and tests rejecting outputs above a 25% word-change
  threshold. We only flag that threshold for review, retaining every raw output.
  Our illustrative examples are synthetic and are not human-verified reference
  excerpts from the evaluated recording.
- [LLM-Based Post-ASR Error Correction for Disordered Speech: released
  implementation](https://github.com/cmu-impactlab/LLM-Corrector-for-Aphasic-ASR/blob/main/Few-shot/models/azure_gpt.py)
  provides a correction playbook and output-only prompting. Its aphasic English
  speech task and model choices differ. We do not adopt its repetition-removal
  policy because this experiment should preserve conversational content.
- [ASR-EC Benchmark](https://arxiv.org/abs/2412.03075) reports that prompting alone
  can fail and introduce errors on Chinese ASR correction. Fluent output is not
  evidence that a correction is faithful to the audio.

## Experimental design

All source timestamp blocks are targets. Each receives the same user-confirmed
background/glossary, previous original block, next original block, and its own
original target. Generated corrections are never fed into later requests.

Two prompt variants are compared for every model/segment: conservative contextual
instructions, and those instructions plus short synthetic examples demonstrating
minimal edits, negation preservation, unchanged disfluencies, and abstention.
Synthetic names are separate from the recording glossary. Prompts are in English
and explicitly retain German content and English technical terms.

The glossary is evidence about spelling, not a global substitution table. Do not
infer relationships, speaker names, other private context, or missing statements.
Additional useful context would be user-confirmed place names, technical terms,
and a few audio-verified ASR/corrected sentence pairs. Unverified guesses should
not become reference answers or ground truth.

Default models: Gemma 4 E4B QAT, IBM Granite 4.2 3B Q4_K_M, Qwen3.5 4B and 2B
Ollama defaults. Quantizations are not identical; actual local model digests,
templates, model metadata, server version, requests, outputs and timings are
saved. For Qwen, record the actual quantization from `/api/show` rather than
assuming the size tag implies Q4. Model packages differ from LiteRT-LM packages.

Use a current Ollama release that supports all selected architectures. Start a
local server with `OLLAMA_NO_CLOUD=1 OLLAMA_HOST=127.0.0.1:11434 ollama serve`.
Install model files separately, without uploading any transcript. The runner
requires installed models and refuses remote-backed models and HTTP redirects.

Greedy decoding, fixed seed, 8,192 context tokens and 2,048 output tokens form a
shared operational baseline. Some vendors recommend other sampling parameters;
this baseline is not a reproduction of their published benchmarks. Check the
saved token counts and finish reasons for truncation. Do a short smoke run with
`--limit` before the full run. The runner does not measure peak memory.

```sh
python3 scripts/run-transcript-cleanup-benchmark.py prepare \
  --transcript /absolute/path/to/transcript.md \
  --context /absolute/path/to/private-context.json \
  --run build/reports/cleanup-comparison
python3 scripts/run-transcript-cleanup-benchmark.py run \
  --run build/reports/cleanup-comparison --limit 4
python3 scripts/run-transcript-cleanup-benchmark.py run \
  --run build/reports/cleanup-comparison
```

The context file is a JSON object containing user-confirmed background and a
glossary. `--models` accepts explicit local model tags during preparation.
`run --model TAG` runs only one model in the frozen matrix. Request changes
require a new run directory, as do changes to the source transcript/model identity.

## Reviewing results

Open `review.html` for original/output diffs and automatic flags. Each model and
prompt also gets a reconstructed transcript. Save edits to the review template
under another filename because `review-template.csv` is regenerated on resume.
Empty outputs and generation limits are visible flags, not silently accepted
corrections. Model output is retained even when the change rate exceeds 25%.

Without an audio-verified reference, do not report WER, accuracy, or a winner.
Word change rate is edit distance from the ASR input, not an error rate against
truth. Digits and negations are heuristic flags and can miss written-number or
meaning changes. Human review should classify repairs, introduced errors,
unnecessary stylistic changes, and uncertain edits separately. Preserve clean
passages as controls. A later annotated, held-out reference enables real WER/CER,
entity repair precision/recall, and damage-rate measurement.

## Deterministic name-only baseline

`compare-name-cleanup.py` compares an explicit alias table with the saved model
outputs. It requires no inference or network. Keep the private alias configuration
and generated reports under ignored `build/reports/`.

```sh
python3 scripts/compare-name-cleanup.py \
  --run build/reports/cleanup-comparison \
  --aliases /absolute/path/to/private-name-aliases.json \
  --output build/reports/cleanup-comparison/name-baseline
```

Configuration shape: `{"aliases":{"Nohra":"Nora","Nohras":"Noras"},
"review_only":["ambiguous word"]}`. Replacements are case-sensitive, whole-token,
single-pass operations. List possessive forms explicitly. All other characters
remain unchanged, and the runner verifies the original source hash before and
after generation. Review-only terms produce contextual candidates without edits.

The report includes every edit offset, reconstructed transcript, HTML comparison,
and per-model counts over the available matching segments. Remaining alias counts
and word edits after alias normalization are descriptive metrics, not accuracy.
Models may reduce alias counts by deleting content. The alias list itself encodes
contextual assumptions; do not treat an alias replacement as audio-verified truth.
Ordinary words that resemble names should remain review candidates.
