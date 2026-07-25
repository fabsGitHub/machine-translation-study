"""
Shared plotting style for report-ready figures (ACM sigconf two-column template).

The core problem this module solves: a matplotlib figure built with default
sizes/fonts and then shrunk in LaTeX to fit a ~3.3in column becomes
unreadable - axis ticks, legends and heatmap cell labels all get scaled down
by the same factor as the figure itself. The fix is to never let LaTeX do
that scaling: build the figure at its *final physical print size* (in
inches, matching the column width it will be placed at) with font sizes
chosen for that physical size, save at high DPI, and reference it in LaTeX
at `width=\\linewidth` (single column) or inside `figure*` at
`width=\\linewidth` (double column) with no extra shrink factor such as
`0.85\\linewidth`. 1 matplotlib point == 1/72 inch on the page regardless of
DPI, so this is the only way to control what size the text is on paper.

Column widths below were measured from this report's own compiled
main.log (`\\textwidth=506.295pt`, `\\columnsep=24.0pt`), not guessed:
    single column = (506.295 - 24.0) / 2 / 72.27 = 3.34in
    double column (figure*, full \\textwidth) = 506.295 / 72.27 = 7.00in
If the template changes, re-measure from main.log rather than reusing these.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Print geometry (inches) - figures are built at these exact physical sizes.
# ---------------------------------------------------------------------------
SINGLE_COL_IN = 3.34
DOUBLE_COL_IN = 7.00

# Font sizes (points) tuned for text printed at the physical sizes above.
# ACM sigconf body text is 10pt / captions ~9pt, so these deliberately stay
# in the same 7-10pt neighborhood rather than matplotlib's ~10-14pt defaults,
# which is what makes small-column figures unreadable in print.
FS_TICK = 7.5
FS_LABEL = 8.5
FS_TITLE = 9.5
FS_LEGEND = 7.5
FS_ANNOT = 6.8
FS_CELL = 6.5  # heatmap cell text, when annotated

# A colorblind-safe, print-stable qualitative palette (3 languages max here).
LANG_COLORS = {
    "de": "#2E4C8A",
    "en": "#B8792B",
    "sv": "#3A8451",
}
DEFAULT_SEQ_CMAP = "Blues"


def set_report_style():
    """Call once before building any figure. Pure matplotlib (no seaborn
    dependency) so these functions work in inference-only environments."""
    matplotlib.rcParams.update({
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "bold",
        "axes.labelsize": FS_LABEL,
        "axes.labelweight": "bold",
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "legend.frameon": False,
        "figure.dpi": 100,       # on-screen preview only; savefig dpi set explicitly
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })


def latex_escape(text):
    """Minimal escaping for dropping free text (experiment names, labels)
    into a LaTeX caption: underscores and a few other special characters
    would otherwise break compilation or silently vanish."""
    replacements = {
        "\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
        "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def panel_letter(i):
    return f"({chr(ord('a') + i)})"


def thousands_formatter(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def figsize(width_in=SINGLE_COL_IN, aspect=0.68):
    """aspect = height / width."""
    return (width_in, round(width_in * aspect, 3))


# ---------------------------------------------------------------------------
# Attention / alignment heatmap - the workhorse for Task 4 and Task 5.
# ---------------------------------------------------------------------------

def _truncate_seq(tokens, max_len):
    if len(tokens) <= max_len:
        return tokens, False
    return tokens[:max_len], True


def plot_attention_heatmap(
    attn, src_tokens, trg_tokens, ax, title=None,
    cmap=DEFAULT_SEQ_CMAP, max_src=22, max_trg=22,
    show_colorbar=True, cbar_ax=None, annotate="auto",
    xlabel="Source", ylabel="Target",
):
    """Draws one attention/alignment heatmap into `ax`.

    Designed to stay legible at single-column print width. Two defenses
    against illegible output:
      1. `max_src`/`max_trg` hard-caps the matrix size shown - beyond this,
         labeling every cell is not achievable at any font size that still
         fits a 3.3in column, so the sequence is truncated with a "+N more"
         note rather than silently shrinking the labels. For char-level
         attention (where "tokens" are individual characters, so sequences
         are 3-6x longer than word-level for the same sentence), pick a
         short example sentence up front rather than relying on truncation
         alone - truncation cuts off the tail of the output, which can hide
         exactly the failure mode (e.g. repetition) you wanted to show.
      2. `annotate` controls whether numeric weights are printed in-cell.
         "auto" annotates only when the matrix is small enough (<= 12x12)
         for the numbers to stay readable; otherwise the color alone carries
         the signal, which is standard practice for larger attention maps.
    """
    src_tokens, src_trunc = _truncate_seq(src_tokens, max_src)
    trg_tokens, trg_trunc = _truncate_seq(trg_tokens, max_trg)
    mat = attn[:len(trg_tokens), :len(src_tokens)]

    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0.0)

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right", rotation_mode="anchor",
                        fontsize=FS_TICK)
    ax.set_yticks(range(len(trg_tokens)))
    ax.set_yticklabels(trg_tokens, fontsize=FS_TICK)

    if src_trunc:
        ax.set_xlabel(f"{xlabel} (truncated)", fontsize=FS_LABEL, fontweight="bold")
    else:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL, fontweight="bold")
    if trg_trunc:
        ax.set_ylabel(f"{ylabel} (truncated)", fontsize=FS_LABEL, fontweight="bold")
    else:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL, fontweight="bold")

    if title:
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=4)

    do_annot = (max(mat.shape) <= 12) if annotate == "auto" else annotate
    if do_annot:
        thresh = mat.max() * 0.6 if mat.size else 0.5
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if v >= 0.08:  # skip near-zero cells to reduce clutter
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=FS_CELL,
                            color="white" if v > thresh else "#222222")

    ax.tick_params(length=2, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, cax=cbar_ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=FS_TICK - 0.5, length=2)
        cbar.outline.set_visible(False)
    return im


def plot_multi_attention(panels, out_path, width_in=SINGLE_COL_IN,
                          per_panel_aspect=0.9, ncols=None, suptitle=None,
                          max_src=22, max_trg=22, cmap=DEFAULT_SEQ_CMAP):
    """Renders N attention heatmaps as one report figure with correct spacing.

    `panels` is a list of dicts: {"attn", "src_tokens", "trg_tokens", "title"}.
    Uses matplotlib's constrained-layout engine (not tight_layout) because
    only constrained-layout reserves real space for each panel's own
    colorbar instead of letting it overlap the next panel's axis labels -
    that overlap is what makes naive multi-panel figures ugly.

    Layout choice:
      - ncols=1 (default when width_in == SINGLE_COL_IN): panels stacked
        vertically, each nearly the full single-column width - the safe
        default that needs no LaTeX changes (drop straight into `figure`).
      - ncols=len(panels) (default when width_in == DOUBLE_COL_IN): panels
        side by side, each ~half the double-column width - use inside
        `figure*` so it spans both columns; a paired A/B comparison (e.g.
        the two legs of a pivot chain) reads better side by side than
        stacked once there is enough width for it.
    """
    n = len(panels)
    if ncols is None:
        ncols = n if width_in >= DOUBLE_COL_IN - 0.5 else 1
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width_in, per_panel_aspect * width_in / ncols * nrows),
        layout="constrained",
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, panel in zip(axes, panels):
        plot_attention_heatmap(
            panel["attn"], panel["src_tokens"], panel["trg_tokens"], ax,
            title=panel.get("title"), cmap=panel.get("cmap", cmap),
            max_src=max_src, max_trg=max_trg,
            xlabel=panel.get("xlabel", "Source"), ylabel=panel.get("ylabel", "Target"),
        )
    for ax in axes[n:]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=FS_TITLE, fontweight="bold")

    fig.savefig(out_path)
    plt.close(fig)
    return out_path

