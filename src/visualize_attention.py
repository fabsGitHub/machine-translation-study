"""
Task 4: attention/alignment visualization, sized for the ACM sigconf report.

Reuses the existing, working inference path (`build_model_from_checkpoint`,
`translate_sentence` from evaluate.py) rather than re-implementing model
loading or greedy decoding - this script is plotting-only. See viz_style.py
for why figures are built at final print size instead of relying on LaTeX to
shrink them.

Typical use (one example):
    .venv/bin/python src/visualize_attention.py \\
        --example data/results/best_model_CHAR_C4.pt "ein kleiner hund lauft ." "Bi-RNN+Bahdanau (CHAR_C4, BLEU 2.14, winner)" \\
        --out data/results/figures/char_attention_c4.png

Typical use (good example vs. a failure mode, side by side like the word
report's attention_heatmap.png + attention_c6_repetition.png pairing, but as
ONE correctly-sized figure instead of two separately-shrunk ones):
    .venv/bin/python src/visualize_attention.py \\
        --example data/results/best_model_CHAR_C4.pt "ein kleiner hund lauft ." "Bi-RNN+Bahdanau (CHAR_C4, BLEU 2.14, winner)" \\
        --example data/results/best_model_CHAR_C6.pt "put the european tongue back in ." "Bi-RNN+Bahdanau (CHAR_C6, BLEU 1.91)" \\
        --out data/results/figures/char_attention_comparison.png

IMPORTANT for char-level checkpoints (token_type="char" in the saved
config): pick SHORT example sentences (a handful of words). A char-level
attention matrix has one row/column per *character*, not per word, so a
normal-length sentence produces a 40-60 token sequence that cannot be
labeled legibly at any font size that still fits an ACM column - truncation
hides exactly the tail-end behavior (e.g. repetition) you're usually trying
to show. A short phrase (15-25 characters) keeps the whole matrix visible
and labeled.

Design note on LABEL: it is NOT rendered into the image. Panels are titled
with plain "(a)"/"(b)" markers only (matching visualize_pivot.py's "DE ->
EN"/"EN -> SV" panel titles - both are minimal, structural labels, nothing
about which model/architecture/BLEU produced the plot). LABEL is used only
to build a suggested LaTeX \\caption{...} printed to stdout at the end,
which is where model/architecture/BLEU metadata belongs - baking that into
the pixels wastes column width and can't be edited without regenerating the
image.
"""
import os
import sys
import argparse

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

from evaluate import build_model_from_checkpoint, translate_sentence
from viz_style import (
    set_report_style, plot_multi_attention, SINGLE_COL_IN, DOUBLE_COL_IN,
    latex_escape, panel_letter,
)


def capture_attention(checkpoint_path, src_sentence, device):
    """Loads a checkpoint, runs greedy decoding on `src_sentence`, and
    returns everything needed to plot it: source tokens (with <sos>/<eos>),
    the model's own output tokens, the attention matrix, and the config
    (used to report which architecture/attention type produced this)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, src_vocab, trg_vocab, cfg = build_model_from_checkpoint(checkpoint, device)

    if cfg.get("attention_type", "none") == "none":
        raise ValueError(
            f"{checkpoint_path} has attention_type='none' - nothing to visualize. "
            "Pick a checkpoint from an attention-enabled config (luong/bahdanau)."
        )

    token_type = cfg.get("token_type", "word")
    src_tokens = list(src_sentence) if token_type == "char" else src_sentence.strip().split()

    trg_tokens, attn_matrix = translate_sentence(model, src_tokens, src_vocab, trg_vocab, device)
    if attn_matrix is None or attn_matrix.size == 0:
        raise RuntimeError(f"No attention weights captured for {checkpoint_path} on: {src_sentence!r}")

    full_src_tokens = ["<sos>"] + src_tokens + ["<eos>"]
    return full_src_tokens, trg_tokens, attn_matrix, cfg


def main():
    parser = argparse.ArgumentParser(description="Task 4: attention heatmap visualization")
    parser.add_argument(
        "--example", nargs=3, action="append", metavar=("CHECKPOINT", "SENTENCE", "LABEL"),
        required=True,
        help="One example to plot: checkpoint path, source sentence, panel title/label. "
             "Repeat --example for a multi-panel comparison figure.",
    )
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument(
        "--layout", choices=["auto", "stacked", "side_by_side"], default="auto",
        help="auto: stacked (single-column) for 1 example, side-by-side (double-column, "
             "use inside figure* in LaTeX) for exactly 2 examples.",
    )
    parser.add_argument("--max_tokens", type=int, default=22,
                         help="Cap on tokens shown per axis before truncation kicks in.")
    args = parser.parse_args()

    set_report_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    panels = []
    caption_parts = []
    multi = len(args.example) > 1
    for i, (checkpoint_path, sentence, label) in enumerate(args.example):
        print(f"Loading {checkpoint_path} ...")
        src_tokens, trg_tokens, attn_matrix, cfg = capture_attention(checkpoint_path, sentence, device)
        print(f"  attention_type={cfg.get('attention_type')} token_type={cfg.get('token_type')} "
              f"-> output: {trg_tokens if cfg.get('token_type') != 'char' else ''.join(trg_tokens)}")
        # In-image title is a bare panel marker only ((a), (b), ...) - matches
        # visualize_pivot.py's "DE -> EN"/"EN -> SV" in being minimal and
        # structural. LABEL (the descriptive model/BLEU text) goes to the
        # suggested caption below instead, never into the plot itself.
        panels.append({
            "attn": attn_matrix,
            "src_tokens": src_tokens,
            "trg_tokens": trg_tokens,
            "title": panel_letter(i) if multi else None,
        })
        prefix = f"{panel_letter(i)} " if multi else ""
        caption_parts.append(f"{prefix}{latex_escape(label)}")

    if args.layout == "auto":
        width_in = DOUBLE_COL_IN if len(panels) == 2 else SINGLE_COL_IN
    elif args.layout == "side_by_side":
        width_in = DOUBLE_COL_IN
    else:
        width_in = SINGLE_COL_IN

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_multi_attention(panels, args.out, width_in=width_in, max_src=args.max_tokens, max_trg=args.max_tokens)
    print(f"\nSaved -> {args.out}")
    if width_in == DOUBLE_COL_IN:
        print("This figure targets the DOUBLE-column width (7.0in) - place it inside "
              "\\begin{figure*}...\\end{figure*} at width=\\linewidth, not a plain \\begin{figure}.")
    else:
        print("This figure targets the SINGLE-column width (3.34in) - "
              "\\includegraphics[width=\\linewidth]{...} inside a plain \\begin{figure}.")

    print("\nSuggested LaTeX caption (edit wording as needed, model/BLEU details "
          "belong here, not in the image):")
    print("\\caption{Attention alignment: " + "; ".join(caption_parts) + ".}")


if __name__ == "__main__":
    main()

