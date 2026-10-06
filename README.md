# Neural Machine Translation: EN ↔ DE ↔ SV

Seq2seq RNN/GRU/LSTM machine translation with Luong/Bahdanau attention,
trained from scratch on the Europarl parallel corpus, comparing word-level
and character-level tokenization across five ablation studies plus a
zero-shot pivot-translation experiment (German → Swedish via English, with
no direct DE-SV training data at all).

This started as a university NLP course project and was extended into a
small independent research pipeline: five sequential ablation studies
(architecture, embeddings, attention, translation direction,
generalization), two full tokenization strategies trained end-to-end, and
a from-scratch reproduction of the classic pivot-translation trick.

**[`nmt_pipeline.py`](nmt_pipeline.py)** is a single-file version of the
entire `src/` codebase (everything except the visualization/report-figure
scripts), generated directly from the modular files rather than
hand-duplicated, so the two never drift apart - for course submission
formats that expect one `.py` file. A `--task` flag selects which of the
original entry points to run (`preprocess`, `train`, `evaluate`, `pivot`,
`build-pivot-eval`, or `study` for the full orchestrator - see the file's
own module docstring for exact invocations). Verified end-to-end on real
GPU hardware, including the trickiest part: the orchestrator launches
training as a subprocess of *itself* via `torch.distributed.run`, which in
turn launches evaluation as another subprocess of itself for its automated
BLEU/METEOR backfill - both self-invocation chains were run for real, not
just reviewed.

## Results

Best-performing configuration per study, on the held-out 20% test split
(full table: [`results/word/evaluation_report_word.csv`](results/word/evaluation_report_word.csv),
[`results/char/evaluation_report_char.csv`](results/char/evaluation_report_char.csv)):

| Pipeline | Best config | Direction | Attention | BLEU | METEOR |
|---|---|---|---|---|---|
| Word | `WORD_E1` | EN → SV | Bahdanau | **4.32** | 23.54 |
| Word | `WORD_D2` | DE → EN | Bahdanau | 3.87 | 22.19 |
| Word | `WORD_C4` | EN → DE | Bahdanau | 3.35 | 20.53 |
| Char | `CHAR_C4` | EN → DE | Bahdanau | **2.14** | 9.42 |
| Char | `CHAR_D1` | EN → DE | Bahdanau | 1.24 | 10.17 |
| Char | `CHAR_E1` | EN → SV | Bahdanau | 1.30 | 4.30 |

Bahdanau attention won every attention ablation (Study C) in both
pipelines, and word-level tokenization clearly outperforms character-level
at this model scale and training budget — expected, since character
sequences are 3-6x longer for the same sentence, giving the model far more
steps to accumulate error over.

**Pivot chain (DE → EN → SV, zero-shot, 3,000 aligned test sentences):**

| Pipeline | DE→EN (intermediate) | DE→EN→SV (final) |
|---|---|---|
| Word | BLEU 3.92 | BLEU 1.14 |
| Char | BLEU 0.00 | BLEU 0.00 |

The char pipeline's `CHAR_D2` (DE→EN leg) essentially failed to learn that
direction (BLEU 0.0068 on its own test set — see the loss curve in
[`results/char/best_config_CHAR_D2_RNN.json`](results/char/best_config_CHAR_D2_RNN.json),
which plateaus after epoch 1 and never recovers). Chaining a broken first
leg into a second model compounds the failure rather than averaging it out:
[`report/figures/char_pivot_attention_v2.png`](report/figures/char_pivot_attention_v2.png)
shows the actual model output on a real sentence — the DE→EN leg produces a
single garbage character, and the EN→SV leg (correctly, working as intended)
has nothing meaningful to translate.

## Repo layout

```
src/            All pipeline code, shared between the word and char runs
notebooks/      RunPod bootstrap scripts used to launch real training runs
config/         Default hyperparameter profiles
results/        Every artifact (json/csv/png/log) from the real training runs
report/figures/ Final print-quality figures used in the write-up
bench.py        Standalone training-step throughput/VRAM microbenchmark
```

`src/` is shared between both pipelines — a `--token_type` flag selects
word vs. char throughout, rather than maintaining two copies of the
codebase:

| File | Purpose |
|---|---|
| `dataset.py` | Vocabulary building, tokenization, DataLoader/Sampler |
| `models.py` | Encoder / Decoder (Luong & Bahdanau attention) / Seq2Seq |
| `train.py` | Single-experiment training entry point (DDP-aware) |
| `evaluate.py` | BLEU/METEOR scoring, attention visualization, reporting |
| `embeddings.py` | Pretrained GloVe / word2vec-style embedding loading |
| `preprocess.py` | Europarl download, sampling, train/val/test splitting |
| `run_studies.py` | Master orchestrator - runs all 5 studies + tuning stages |
| `pivot.py` | DE→EN→SV pivot chain translator + quantitative eval |
| `build_pivot_eval_set.py` | Builds an aligned DE/EN/SV pivot evaluation set |
| `explore_data.py` | Task 1 corpus exploration (vocab size, length stats) |
| `visualize_attention.py`, `visualize_pivot.py`, `viz_style.py` | Print-quality attention heatmap figures for the report |
| `config.py`, `utils.py`, `auto_shutdown.py` | Shared config/logging/cache helpers |

`results/word/` and `results/char/` hold every `best_config_*.json` /
`*.csv` / `*.png` / `*.log` artifact produced by the real training runs;
`results/eda/` is the Task 1 data-exploration output.

## Running it yourself

```bash
git clone git@github.com:fabsGitHub/machine-translation-study.git
cd machine-translation-study
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**1. Get the data.** Raw corpora and pretrained embeddings are gitignored
(several GB, public downloads) - fetch them yourself:

```bash
# Europarl DE-EN and SV-EN parallel corpora into data/raw/
mkdir -p data/raw && cd data/raw
curl -O https://www.statmt.org/europarl/v7/de-en.tgz && tar xzf de-en.tgz
curl -O https://www.statmt.org/europarl/v7/sv-en.tgz && tar xzf sv-en.tgz
cd ../..

# Pretrained embeddings (only needed for --embedding_source glove/word2vec runs)
bash download_embeddings.sh
```

**2. Preprocess** (samples 10% of each corpus, builds train/val/test splits
for all three translation directions):

```bash
python src/preprocess.py
# or, for a fast no-download smoke test on tiny built-in mock sentences:
python src/preprocess.py --mock
```

**3. Run the full study pipeline** for one tokenization strategy:

```bash
python src/run_studies.py --study all --token_type word
python src/run_studies.py --study all --token_type char
```

This resumes automatically if interrupted (already-completed experiments
are skipped via their saved `completed: true` flag) and can also be pointed
at a single study, e.g. `--study C` for just the attention ablation.

**4. A single experiment**, or evaluating/visualizing an existing checkpoint:

```bash
python src/train.py --experiment MY_RUN --rnn_type LSTM --attention_type bahdanau \
    --src_lang en --trg_lang de --token_type word --epochs 10

python src/evaluate.py evaluate --checkpoint data/results/best_model_MY_RUN_LSTM.pt
python src/evaluate.py visualize --checkpoint data/results/best_model_MY_RUN_LSTM.pt \
    --sentence "ein kleiner hund läuft ."
```

## Automated checks

GitHub Actions compiles the modular source and standalone pipeline, then runs a standard-library regression test for terminal EOS trimming in both evaluators:

```bash
python -m compileall -q src nmt_pipeline.py tests
python -m unittest discover -s tests -v
```

These checks do not download data, train models, run GPU experiments, or regenerate the reported metrics. The tracked result tables remain historical until evaluation is rerun against the original checkpoints.

## Known limitations and result provenance

The evaluator now removes the generated end-of-sequence token from decoded hypotheses before scoring. The tracked CSV and JSON reports were produced before this correction and have not been regenerated. Treat those values as historical, pre-fix results; no corrected metric values are claimed in this change. Earlier project notes estimated that retaining EOS reduced BLEU/METEOR by about 2% relative. Re-run evaluation against the original checkpoints to produce post-fix scores. Checkpoints are not included in the repository.

- **Study D/E console banner:** the printed summary can show a stale embedding size from the previous study. This affects the banner only; the run configuration is passed separately. A related stale CLI parameter was corrected during development, as documented in the existing results and commit history.

## License

This project’s original source code is licensed under the MIT License; see [LICENSE](LICENSE). The Europarl corpus, pretrained embeddings, Python dependencies, and other third-party materials remain subject to their own licenses and terms.

## Development notes

Built and trained solo; Claude (Anthropic) was used as a coding assistant
throughout development, including for parts of the documentation in this
repository.
