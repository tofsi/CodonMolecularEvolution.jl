#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_meta(meta_path: Path | None) -> dict:
    if meta_path is None or not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def get_source_for_sim(meta: dict, sim_id: int) -> str:
    for row in meta.get("shapes", []):
        if int(row.get("sim_id")) == int(sim_id):
            return str(row.get("source", ""))
    return ""


def iter_rate_matrices(npz_path: Path, meta: dict):
    z = np.load(npz_path, allow_pickle=True)
    sim_ids = np.asarray(z["sim_ids"], dtype=int)

    if "rates" in z.files:
        # Rectangular case: rates[simulation_index, site_index, parameter_index]
        rates = z["rates"]
        for i, sim_id in enumerate(sim_ids):
            yield int(sim_id), get_source_for_sim(meta, int(sim_id)), np.asarray(rates[i], dtype=float)
    else:
        # Non-rectangular case: rates_0000, rates_0001, ...
        for i, sim_id in enumerate(sim_ids):
            key = f"rates_{i:04d}"
            if key not in z.files:
                raise KeyError(f"Expected key {key!r} in {npz_path}")
            yield int(sim_id), get_source_for_sim(meta, int(sim_id)), np.asarray(z[key], dtype=float)


def compute_omega_matrix(rates: np.ndarray) -> np.ndarray:
    """
    rates[:, 0] = alpha
    rates[:, 1:] = class-specific beta/rate values
    omega = beta / alpha
    """
    if rates.ndim != 2 or rates.shape[1] < 2:
        raise ValueError(f"Expected 2D matrix with alpha + class columns, got shape {rates.shape}")

    alpha = rates[:, [0]]
    beta = rates[:, 1:]

    omega = np.full(beta.shape, np.nan, dtype=float)
    np.divide(beta, alpha, out=omega, where=(alpha != 0))
    return omega


def summarize_simulation(sim_id: int, source: str, rates: np.ndarray) -> dict:
    omega = compute_omega_matrix(rates)

    pos_by_class = omega > 1.0
    pos_any = np.any(pos_by_class, axis=1)

    max_omega = np.nanmax(omega, axis=1)
    mean_omega_by_class = np.nanmean(omega, axis=0)
    median_omega_by_class = np.nanmedian(omega, axis=0)
    prop_pos_by_class = np.nanmean(pos_by_class, axis=0)

    return {
        "sim_id": sim_id,
        "source": source,
        "n_sites": rates.shape[0],
        "n_classes": omega.shape[1],
        "omega": omega,
        "pos_by_class": pos_by_class,
        "pos_any": pos_any,
        "max_omega": max_omega,
        "mean_omega_by_class": mean_omega_by_class,
        "median_omega_by_class": median_omega_by_class,
        "prop_pos_by_class": prop_pos_by_class,
        "prop_pos_any": float(np.mean(pos_any)),
        "n_pos_any": int(np.sum(pos_any)),
    }


def build_simulation_summaries(npz_path: Path, meta_path: Path | None):
    meta = load_meta(meta_path)
    summaries = []

    for sim_id, source, rates in iter_rate_matrices(npz_path, meta):
        summaries.append(summarize_simulation(sim_id, source, rates))

    return summaries


def choose_candidate_simulations(summaries, target=0.5, top_n=20):
    scored = []
    for s in summaries:
        dist = abs(s["prop_pos_any"] - target)
        scored.append((dist, s))
    scored.sort(key=lambda x: x[0])
    return [s for _, s in scored[:top_n]]


def make_overview_page(pdf, summaries, target=0.5, top_n=20):
    candidates = choose_candidate_simulations(summaries, target=target, top_n=top_n)

    sim_ids = np.array([s["sim_id"] for s in summaries])
    prop_any = np.array([s["prop_pos_any"] for s in summaries])
    n_sites = np.array([s["n_sites"] for s in summaries])

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)

    ax.scatter(prop_any, n_sites, alpha=0.8)
    ax.axvline(target, linestyle="--")
    ax.set_xlabel("Fraction of sites with any omega class > 1")
    ax.set_ylabel("Number of sites")
    ax.set_title("All simulations: positive-site fraction vs alignment size")

    for s in candidates[:10]:
        ax.annotate(
            f"sim {s['sim_id']}",
            (s["prop_pos_any"], s["n_sites"]),
            fontsize=8,
        )

    text_lines = ["Top candidates closest to target {:.2f}:".format(target)]
    for s in candidates[:10]:
        text_lines.append(
            f"sim {s['sim_id']}: prop_pos_any={s['prop_pos_any']:.3f}, "
            f"n_pos={s['n_pos_any']}/{s['n_sites']}"
        )

    fig.text(
        0.62,
        0.15,
        "\n".join(text_lines),
        fontsize=9,
        family="monospace",
        va="bottom",
    )

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def make_class_heatmap_page(pdf, summaries, target=0.5, top_n=30):
    candidates = choose_candidate_simulations(summaries, target=target, top_n=top_n)

    max_classes = max(s["n_classes"] for s in candidates)
    heat = np.full((len(candidates), max_classes), np.nan)

    for i, s in enumerate(candidates):
        vals = s["prop_pos_by_class"]
        heat[i, : len(vals)] = vals

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)

    im = ax.imshow(heat, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Omega class")
    ax.set_ylabel("Simulation")
    ax.set_title(f"Per-class positive-site fractions for simulations closest to target={target:.2f}")

    ax.set_xticks(range(max_classes))
    ax.set_xticklabels([f"class {j}" for j in range(max_classes)])
    ax.set_yticks(range(len(candidates)))
    ax.set_yticklabels([f"sim {s['sim_id']}" for s in candidates])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction of sites with omega > 1")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def make_mean_omega_page(pdf, summaries, target=0.5, top_n=30):
    candidates = choose_candidate_simulations(summaries, target=target, top_n=top_n)

    max_classes = max(s["n_classes"] for s in candidates)
    heat = np.full((len(candidates), max_classes), np.nan)

    for i, s in enumerate(candidates):
        vals = s["mean_omega_by_class"]
        heat[i, : len(vals)] = vals

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)

    im = ax.imshow(heat, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Omega class")
    ax.set_ylabel("Simulation")
    ax.set_title(f"Mean omega by class for simulations closest to target={target:.2f}")

    ax.set_xticks(range(max_classes))
    ax.set_xticklabels([f"class {j}" for j in range(max_classes)])
    ax.set_yticks(range(len(candidates)))
    ax.set_yticklabels([f"sim {s['sim_id']}" for s in candidates])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean omega")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def make_simulation_page(pdf, summary: dict):
    omega = summary["omega"]
    prop_pos_by_class = summary["prop_pos_by_class"]
    mean_omega_by_class = summary["mean_omega_by_class"]
    max_omega = summary["max_omega"]

    n_sites, n_classes = omega.shape
    class_labels = [f"class {j}" for j in range(n_classes)]

    fig = plt.figure(figsize=(11, 8.5))

    ax1 = fig.add_subplot(221)
    ax1.bar(range(n_classes), prop_pos_by_class)
    ax1.set_xticks(range(n_classes))
    ax1.set_xticklabels(class_labels, rotation=45, ha="right")
    ax1.set_ylabel("Fraction omega > 1")
    ax1.set_title("Per-class positive-site fraction")

    ax2 = fig.add_subplot(222)
    ax2.boxplot([omega[:, j][np.isfinite(omega[:, j])] for j in range(n_classes)], tick_labels=class_labels)
    ax2.set_ylabel("Omega")
    ax2.set_title("Omega distributions by class")
    ax2.tick_params(axis="x", rotation=45)

    ax3 = fig.add_subplot(223)
    sorted_max = np.sort(max_omega)[::-1]
    ax3.plot(np.arange(1, len(sorted_max) + 1), sorted_max)
    ax3.axhline(1.0, linestyle="--")
    ax3.set_xlabel("Site rank")
    ax3.set_ylabel("Max omega across classes")
    ax3.set_title("Sorted sitewise max omega")

    ax4 = fig.add_subplot(224)
    order = np.argsort(max_omega)[::-1]
    # Clip high values for readability in the heatmap
    heat = np.clip(omega[order, :], 0, 5)
    im = ax4.imshow(heat, aspect="auto", interpolation="nearest")
    ax4.set_xlabel("Omega class")
    ax4.set_ylabel("Sites (sorted by max omega)")
    ax4.set_title("Sitewise omega heatmap (clipped at 5)")
    ax4.set_xticks(range(n_classes))
    ax4.set_xticklabels(class_labels, rotation=45, ha="right")
    cbar = fig.colorbar(im, ax=ax4)
    cbar.set_label("Omega")

    fig.suptitle(
        f"Simulation {summary['sim_id']} | "
        f"sites={n_sites} | "
        f"prop_pos_any={summary['prop_pos_any']:.3f} | "
        f"n_pos_any={summary['n_pos_any']}/{n_sites}",
        fontsize=14,
    )

    if summary["source"]:
        fig.text(0.02, 0.02, f"Source: {summary['source']}", fontsize=9)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path, help="Path to omnibus_multi_true_rates.npz")
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Path to omnibus_multi_true_rates.meta.json. Default: replace .npz with .meta.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("omega_class_report.pdf"),
        help="Output PDF path",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.5,
        help="Target fraction of sites with any omega class > 1",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of candidate simulations to include as detailed pages",
    )
    parser.add_argument(
        "--max-overview",
        type=int,
        default=30,
        help="How many top candidates to show in overview heatmaps",
    )
    args = parser.parse_args()

    meta_path = args.meta
    if meta_path is None:
        meta_path = args.npz.with_suffix(".meta.json")

    summaries = build_simulation_summaries(args.npz, meta_path)
    candidates = choose_candidate_simulations(summaries, target=args.target, top_n=args.top)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(args.out) as pdf:
        make_overview_page(pdf, summaries, target=args.target, top_n=args.top)
        make_class_heatmap_page(pdf, summaries, target=args.target, top_n=args.max_overview)
        make_mean_omega_page(pdf, summaries, target=args.target, top_n=args.max_overview)

        for s in candidates:
            make_simulation_page(pdf, s)

    print(f"Wrote {args.out}")
    print("Top candidates:")
    for s in candidates:
        print(
            f"sim_{s['sim_id']}: "
            f"prop_pos_any={s['prop_pos_any']:.3f}, "
            f"n_pos_any={s['n_pos_any']}/{s['n_sites']}"
        )


if __name__ == "__main__":
    main()