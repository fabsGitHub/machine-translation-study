"""
Task 5: pivot-chain (DE -> EN -> SV) attention visualization, sized for the
ACM sigconf report.

Fixes the specific readability bug in the word report's pivot_heatmap.png:
that figure crams two full heatmaps (2921x1319px) into a single ~3.34in
report column (`\\includegraphics[width=\\linewidth]` inside a plain
`figure`), a ~0.34x shrink that makes every tick label and cell number
unreadable in print. This script instead builds the two-leg comparison AT
the double-column width (7.0in) via viz_style's side-by-side layout, so it
must be placed inside `\\begin{figure*}...\\end{figure*}` (spans both
columns) rather than a plain `\\begin{figure}`. If keeping a single-column
figure is preferred instead, pass --layout stacked - each panel then gets
nearly the full single-column width by stacking vertically, at the cost of
a taller figure.

Usage:
    .venv/bin/python src/visualize_pivot.py \\
        --de_en_model data/results/best_model_CHAR_C4.pt \\
        --en_sv_model data/results/best_model_CHAR_D_EN_SV.pt \\
        --text "die europaeische kommission hat entschieden ." \\
        --out data/results/figures/char_pivot_attention.png \\
        --leg1_label "CHAR_D2, LSTM+Bahdanau, BLEU 3.87" \\
        --leg2_label "CHAR_E1, LSTM+Bahdanau, BLEU 4.32"

Panel titles stay the minimal "DE -> EN"/"EN -> SV" direction labels only -
--leg1_label/--leg2_label (both optional) are NOT rendered into the image,
they only feed the suggested LaTeX caption printed at the end. Model/BLEU
metadata belongs in the caption, not baked into the figure - see
visualize_attention.py's docstring for the same reasoning applied there.
"""
import os
import sys
import argparse

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

from pivot import PivotTranslator
from viz_style import set_report_style, plot_multi_attention, SINGLE_COL_IN, DOUBLE_COL_IN, latex_escape


def main():
    parser = argparse.ArgumentParser(description="Task 5: pivot-chain attention visualization")
    parser.add_argument("--de_en_model", required=True)
    parser.add_argument("--en_sv_model", required=True)
    parser.add_argument("--text", required=True, help="DE source sentence to translate through the pivot")
    parser.add_argument("--out", required=True)
    parser.add_argument("--token_type", choices=["word", "char"], default="word")
    parser.add_argument("--layout", choices=["side_by_side", "stacked"], default="side_by_side")
    parser.add_argument("--max_tokens", type=int, default=22,
                         help="Cap on tokens shown per axis before truncation kicks in. For "
                              "token_type=char, prefer a short --text over raising this - a long "
                              "char sequence stays illegible even truncated less aggressively.")
    parser.add_argument("--leg1_label", default=None,
                         help="Optional free text (model/BLEU) for the DE->EN leg, used only in "
                              "the suggested caption, never rendered into the image.")
    parser.add_argument("--leg2_label", default=None,
                         help="Optional free text (model/BLEU) for the EN->SV leg, used only in "
                              "the suggested caption, never rendered into the image.")
    args = parser.parse_args()

    set_report_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    translator = PivotTranslator(args.de_en_model, args.en_sv_model, device, token_type=args.token_type)
    result = translator.translate_with_attention(args.text)

    print(f"DE origin:  {result['de_sentence']}")
    print(f"EN pivot:   {result['en_sentence']}")
    print(f"SV output:  {result['sv_sentence']}")

    panels = [result["leg1"], result["leg2"]]
    width_in = DOUBLE_COL_IN if args.layout == "side_by_side" else SINGLE_COL_IN

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_multi_attention(panels, args.out, width_in=width_in,
                          max_src=args.max_tokens, max_trg=args.max_tokens)
    print(f"\nSaved -> {args.out}")
    if width_in == DOUBLE_COL_IN:
        print("This figure targets the DOUBLE-column width (7.0in) - place it inside "
              "\\begin{figure*}...\\end{figure*} at width=\\linewidth, not a plain \\begin{figure}.")
    else:
        print("This figure targets the SINGLE-column width (3.34in) - "
              "\\includegraphics[width=\\linewidth]{...} inside a plain \\begin{figure}.")

    leg1_desc = f" ({latex_escape(args.leg1_label)})" if args.leg1_label else ""
    leg2_desc = f" ({latex_escape(args.leg2_label)})" if args.leg2_label else ""
    print("\nSuggested LaTeX caption (edit wording as needed, model/BLEU details "
          "belong here, not in the image):")
    print(
        "\\caption{Attention alignment for both legs of the pivot chain on the "
        f"same example. Left: DE$\\to$EN{leg1_desc}. Right: EN$\\to$SV{leg2_desc}, "
        "whose source is the left panel's own output.}"
    )


if __name__ == "__main__":
    main()

