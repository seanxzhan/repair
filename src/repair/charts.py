"""Matplotlib charts for the non-spatial parts of the story, plus a tiler that
composes multiple polyscope screenshots into one labeled figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import ensure_out
from .templates import TEMPLATE_NAMES

TERMS = ["sound_removed", "structural", "fabrication", "grain", "defect"]


def save(fig, name) -> str:
    ensure_out()
    p = str(name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def tile(image_paths, titles, out_name, cols=3, suptitle=None):
    """Compose saved PNG screenshots into a labeled grid."""
    import matplotlib.image as mpimg
    n = len(image_paths)
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, title in zip(axes, image_paths, titles):
        ax.imshow(mpimg.imread(img))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return save(fig, out_name)


def bar_energy_terms(terms_list, labels, out_name, weights=None, title=None):
    """Grouped bars of the 5 energy terms across a few candidates (the tension figure)."""
    x = np.arange(len(TERMS))
    w = 0.8 / len(terms_list)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (t, lab) in enumerate(zip(terms_list, labels)):
        vals = [getattr(t, k) for k in TERMS]
        ax.bar(x + i * w, vals, w, label=lab)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(TERMS, rotation=15)
    ax.set_ylabel("term value (pre-weight)")
    ax.legend()
    ax.set_title(title or "Energy terms: minimum-removal vs clean-interface tension")
    return save(fig, out_name)


def bar_totals(names, totals, out_name, winner=None, title=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#c0413b" if n == winner else "#4a78b5" for n in names]
    ax.bar(names, totals, color=colors)
    ax.set_ylabel("total energy")
    ax.set_title(title or "Per-template total energy (red = oracle choice)")
    return save(fig, out_name)


def heatmap_landscape(xs, ys, Z, out_name, xlabel, ylabel, path=None,
                      feasible=None, title=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(xs, ys, Z, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="total energy")
    if feasible is not None:
        ax.contour(xs, ys, feasible.astype(float), levels=[0.5],
                   colors="white", linewidths=1.5, linestyles="--")
    if path is not None:
        px, py = np.asarray(path).T
        ax.plot(px, py, "-o", color="red", ms=3, lw=1.2, label="optimizer path")
        ax.legend(loc="upper right")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title or "Energy landscape (dashed = feasibility boundary)")
    return save(fig, out_name)


def hist_labels(samples, out_name):
    from collections import Counter
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    c = Counter(s.label_name for s in samples)
    axes[0].bar(TEMPLATE_NAMES, [c.get(n, 0) for n in TEMPLATE_NAMES], color="#4a78b5")
    axes[0].set_title("Oracle label distribution"); axes[0].set_ylabel("count")
    # by damage kind
    kinds = sorted({s.damage_kind for s in samples})
    bottom = np.zeros(len(TEMPLATE_NAMES))
    for k in kinds:
        ck = Counter(s.label_name for s in samples if s.damage_kind == k)
        vals = np.array([ck.get(n, 0) for n in TEMPLATE_NAMES])
        axes[1].bar(TEMPLATE_NAMES, vals, bottom=bottom, label=k)
        bottom += vals
    axes[1].set_title("Label by damage kind"); axes[1].legend()
    return save(fig, out_name)


def scatter_pca(samples, out_name):
    """2D PCA of the hand feature vectors, colored by oracle label (numpy SVD)."""
    X = np.stack([s.feats for s in samples]).astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    P = (X - X.mean(0)) @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = np.array([s.label for s in samples])
    for i, name in enumerate(TEMPLATE_NAMES):
        sel = labels == i
        ax.scatter(P[sel, 0], P[sel, 1], s=14, label=name, alpha=0.7)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Feature-space PCA, colored by oracle label"); ax.legend()
    return save(fig, out_name)


def confusion(cm, out_name, title=None):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(TEMPLATE_NAMES))); ax.set_xticklabels(TEMPLATE_NAMES, rotation=30)
    ax.set_yticks(range(len(TEMPLATE_NAMES))); ax.set_yticklabels(TEMPLATE_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("oracle label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title or "Template confusion")
    return save(fig, out_name)


def bar_regret(results: dict, out_name):
    """results: {split_name: {regret_prior, regret_random, regret_most_frequent, accuracy}}"""
    splits = list(results.keys())
    series = ["regret_oracle", "regret_prior", "regret_most_frequent", "regret_random"]
    labels = ["oracle", "prior (ours)", "most-frequent", "random"]
    x = np.arange(len(splits)); w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (s, lab) in enumerate(zip(series, labels)):
        ax.bar(x + i * w, [results[sp][s] for sp in splits], w, label=lab)
    ax.set_xticks(x + 1.5 * w); ax.set_xticklabels(splits)
    ax.set_ylabel("mean energy regret  (lower = better)")
    ax.set_title("Prior vs baselines across generalization splits")
    ax.legend()
    for i, sp in enumerate(splits):
        ax.text(x[i] + 1.5 * w, ax.get_ylim()[1] * 0.92,
                f"acc={results[sp]['accuracy']:.2f}", ha="center", fontsize=9)
    return save(fig, out_name)
