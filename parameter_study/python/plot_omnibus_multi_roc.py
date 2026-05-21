#!/usr/bin/env python3
"""
Plot ROC/PR curves and AUC/AUPRC summaries from run_one_omnibus_multi outputs.

This version expects the output files that omnibus_multi already writes, e.g.

    original_BAME_roc.csv
    original_BAME_pr.csv
    original_BAME_site_posteriors.csv
    kernel_stddev_0p5_roc.csv
    kernel_stddev_0p5_pr.csv
    kernel_stddev_0p5_site_posteriors.csv

The ROC files are expected to contain at least:

    threshold,tpr,fpr,tp,fp,fn,tn,method,kernel_stddev

The PR files are expected to contain precision/recall columns, or enough
confusion-count columns to reconstruct them:

    threshold,precision,recall,tp,fp,fn,tn,method,kernel_stddev

The site posterior files are optional for plotting curves, but are summarized
when present:

    site,posterior_prob_positive,bayes_factor,threshold,true_positive,method,kernel_stddev
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROC_GRID = np.linspace(0.0, 1.0, 501)
PR_GRID = np.linspace(0.0, 1.0, 501)


@dataclass(frozen=True)
class CurveFile:
    path: Path
    simulation_id: str
    curve_kind: str  # "roc" or "pr"
    method: str
    kernel_stddev: float | None
    df: pd.DataFrame


@dataclass(frozen=True)
class SitePosteriorFile:
    path: Path
    simulation_id: str
    method: str
    kernel_stddev: float | None
    df: pd.DataFrame


def set_theme() -> None:
    # User requested seaborn darkgrid specifically.
    sns.set_theme(style="darkgrid", context="talk")


def normalize_col(x: object) -> str:
    s = str(x).strip().lower()
    s = s.replace("β", "beta").replace("α", "alpha")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def get_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    by_norm = {normalize_col(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_col(cand)
        if key in by_norm:
            return by_norm[key]
    return None


def first_nonmissing(series: pd.Series) -> object | None:
    vals = series.dropna()
    if vals.empty:
        return None
    for v in vals:
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None


def parse_kernel_token(token: str) -> float:
    # Handles filename tokens like 0p5, 1p0, 8, 2.5.
    token = token.strip().replace("p", ".")
    return float(token)


def parse_metadata_from_name(path: Path) -> tuple[str | None, float | None, str | None]:
    name = path.name
    lower = name.lower()

    if lower.startswith("original_bame_"):
        if lower.endswith("_roc.csv"):
            return "original_BAME", None, "roc"
        if lower.endswith("_pr.csv"):
            return "original_BAME", None, "pr"
        if lower.endswith("_site_posteriors.csv"):
            return "original_BAME", None, "site_posteriors"

    m = re.match(r"kernel_stddev_([^_]+)_(roc|pr|site_posteriors)\.csv$", lower)
    if m:
        return "smoothFLAVOR_BAME", parse_kernel_token(m.group(1)), m.group(2)

    # Generic fallback: any *_roc.csv, *_pr.csv, *_site_posteriors.csv.
    for suffix, kind in [
        ("_site_posteriors.csv", "site_posteriors"),
        ("_roc.csv", "roc"),
        ("_pr.csv", "pr"),
    ]:
        if lower.endswith(suffix):
            method = name[: -len(suffix)]
            return method, None, kind

    return None, None, None


def metadata_from_dataframe_or_filename(df: pd.DataFrame, path: Path) -> tuple[str, float | None, str]:
    file_method, file_kernel, file_kind = parse_metadata_from_name(path)

    method = file_method
    method_col = get_col(df, ["method"])
    if method_col is not None:
        v = first_nonmissing(df[method_col])
        if v is not None:
            method = str(v)

    kernel = file_kernel
    kernel_col = get_col(df, ["kernel_stddev", "kernel", "sigma", "stddev"])
    if kernel_col is not None:
        v = first_nonmissing(df[kernel_col])
        if v is not None:
            try:
                fv = float(v)
                if not math.isnan(fv):
                    kernel = fv
            except Exception:
                pass

    kind = file_kind
    if kind is None:
        raise ValueError(f"Cannot infer file kind from {path}")
    if method is None:
        raise ValueError(f"Cannot infer method from {path}")
    return method, kernel, kind


def simulation_id_from_path(path: Path, outdir: Path) -> str:
    try:
        rel = path.relative_to(outdir)
    except ValueError:
        return path.parent.name
    parts = rel.parts
    if len(parts) <= 1:
        return "root"
    return parts[0]


def find_input_files(outdir: Path) -> tuple[list[Path], list[Path], list[Path]]:
    roc_files: list[Path] = []
    pr_files: list[Path] = []
    site_files: list[Path] = []

    for p in outdir.rglob("*.csv"):
        lower = p.name.lower()
        # Avoid re-reading our own outputs if the script is rerun.
        if "roc_analysis" in {part.lower() for part in p.parts}:
            continue
        if lower.endswith("_roc.csv"):
            roc_files.append(p)
        elif lower.endswith("_pr.csv"):
            pr_files.append(p)
        elif lower.endswith("_site_posteriors.csv"):
            site_files.append(p)

    return sorted(roc_files), sorted(pr_files), sorted(site_files)


def read_curve_file(path: Path, outdir: Path, expected_kind: str) -> CurveFile | None:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
        return None

    try:
        method, kernel, kind = metadata_from_dataframe_or_filename(df, path)
    except Exception as exc:
        print(f"Warning: skipping {path}: {exc}", file=sys.stderr)
        return None

    if kind != expected_kind:
        kind = expected_kind

    simulation_id = simulation_id_from_path(path, outdir)
    return CurveFile(path, simulation_id, kind, method, kernel, df)


def read_site_file(path: Path, outdir: Path) -> SitePosteriorFile | None:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
        return None

    try:
        method, kernel, _kind = metadata_from_dataframe_or_filename(df, path)
    except Exception as exc:
        print(f"Warning: skipping {path}: {exc}", file=sys.stderr)
        return None

    simulation_id = simulation_id_from_path(path, outdir)
    return SitePosteriorFile(path, simulation_id, method, kernel, df)


def method_key(method: str, kernel: float | None) -> tuple[str, str]:
    if kernel is None or (isinstance(kernel, float) and math.isnan(kernel)):
        return method, ""
    return method, f"{kernel:g}"


def method_label(method: str, kernel: float | None) -> str:
    if kernel is None or (isinstance(kernel, float) and math.isnan(kernel)):
        return method
    if method == "smoothFLAVOR_BAME":
        return f"smoothFLAVOR_BAME σ={kernel:g}"
    return f"{method} σ={kernel:g}"


def clean_roc(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    fpr_col = get_col(df, ["fpr", "false_positive_rate"])
    tpr_col = get_col(df, ["tpr", "recall", "sensitivity", "true_positive_rate"])
    if fpr_col is None or tpr_col is None:
        raise ValueError(f"{path} does not have recognizable fpr/tpr columns")

    out = pd.DataFrame({
        "fpr": pd.to_numeric(df[fpr_col], errors="coerce"),
        "tpr": pd.to_numeric(df[tpr_col], errors="coerce"),
    }).dropna()

    out = out[(out["fpr"] >= 0.0) & (out["fpr"] <= 1.0) & (out["tpr"] >= 0.0) & (out["tpr"] <= 1.0)]
    if out.empty:
        raise ValueError(f"{path} has no usable ROC points")

    # For duplicated FPR values, keep the best TPR, then enforce monotonicity.
    out = out.groupby("fpr", as_index=False)["tpr"].max().sort_values("fpr")
    if out["fpr"].iloc[0] > 0.0:
        out = pd.concat([pd.DataFrame({"fpr": [0.0], "tpr": [0.0]}), out], ignore_index=True)
    if out["fpr"].iloc[-1] < 1.0:
        out = pd.concat([out, pd.DataFrame({"fpr": [1.0], "tpr": [1.0]})], ignore_index=True)
    out = out.groupby("fpr", as_index=False)["tpr"].max().sort_values("fpr")
    out["tpr"] = np.maximum.accumulate(out["tpr"].to_numpy())
    return out


def clean_pr(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    recall_col = get_col(df, ["recall", "tpr", "sensitivity"])
    precision_col = get_col(df, ["precision", "ppv"])

    if recall_col is not None and precision_col is not None:
        out = pd.DataFrame({
            "recall": pd.to_numeric(df[recall_col], errors="coerce"),
            "precision": pd.to_numeric(df[precision_col], errors="coerce"),
        })
    else:
        tp_col = get_col(df, ["tp", "true_positive", "true_positives"])
        fp_col = get_col(df, ["fp", "false_positive", "false_positives"])
        fn_col = get_col(df, ["fn", "false_negative", "false_negatives"])
        if tp_col is None or fp_col is None or fn_col is None:
            raise ValueError(f"{path} does not have precision/recall or tp/fp/fn columns")
        tp = pd.to_numeric(df[tp_col], errors="coerce")
        fp = pd.to_numeric(df[fp_col], errors="coerce")
        fn = pd.to_numeric(df[fn_col], errors="coerce")
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        out = pd.DataFrame({"recall": recall, "precision": precision})

    out = out.dropna()
    out = out[(out["recall"] >= 0.0) & (out["recall"] <= 1.0) & (out["precision"] >= 0.0) & (out["precision"] <= 1.0)]
    if out.empty:
        raise ValueError(f"{path} has no usable PR points")

    # For duplicated recall, keep max precision. Sort by recall for AUPRC.
    out = out.groupby("recall", as_index=False)["precision"].max().sort_values("recall")
    if out["recall"].iloc[0] > 0.0:
        out = pd.concat([pd.DataFrame({"recall": [0.0], "precision": [out["precision"].iloc[0]]}), out], ignore_index=True)
    if out["recall"].iloc[-1] < 1.0:
        out = pd.concat([out, pd.DataFrame({"recall": [1.0], "precision": [out["precision"].iloc[-1]]})], ignore_index=True)
    out = out.groupby("recall", as_index=False)["precision"].max().sort_values("recall")
    return out


def trapezoid_auc(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def interpolate_curve(curve: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "roc":
        x = curve["fpr"].to_numpy(dtype=float)
        y = curve["tpr"].to_numpy(dtype=float)
        return np.interp(ROC_GRID, x, y)
    x = curve["recall"].to_numpy(dtype=float)
    y = curve["precision"].to_numpy(dtype=float)
    return np.interp(PR_GRID, x, y)


def collect_curve_stats(curves: list[CurveFile], kind: str) -> tuple[pd.DataFrame, dict[tuple[str, str], list[pd.DataFrame]]]:
    rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[pd.DataFrame]] = {}

    for cf in curves:
        try:
            clean = clean_roc(cf.df, cf.path) if kind == "roc" else clean_pr(cf.df, cf.path)
        except Exception as exc:
            print(f"Warning: skipping {cf.path}: {exc}", file=sys.stderr)
            continue

        if kind == "roc":
            auc = trapezoid_auc(clean["fpr"].to_numpy(), clean["tpr"].to_numpy())
            metric_name = "auc"
        else:
            auc = trapezoid_auc(clean["recall"].to_numpy(), clean["precision"].to_numpy())
            metric_name = "auprc"

        key = method_key(cf.method, cf.kernel_stddev)
        grouped.setdefault(key, []).append(clean)

        rows.append({
            "curve_kind": kind,
            "simulation_id": cf.simulation_id,
            "method": cf.method,
            "kernel_stddev": cf.kernel_stddev,
            "label": method_label(cf.method, cf.kernel_stddev),
            metric_name: auc,
            "n_points": len(clean),
            "path": str(cf.path),
        })

    return pd.DataFrame(rows), grouped


def group_order(stats: pd.DataFrame, metric: str) -> list[str]:
    if stats.empty:
        return []
    tmp = stats.groupby("label", as_index=False)[metric].mean().sort_values(metric, ascending=False)
    return tmp["label"].tolist()


def plot_mean_curves(
    grouped: dict[tuple[str, str], list[pd.DataFrame]],
    stats: pd.DataFrame,
    kind: str,
    output_path: Path,
) -> None:
    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    metric = "auc" if kind == "roc" else "auprc"
    label_to_metric = stats.groupby("label")[metric].mean().to_dict() if not stats.empty else {}

    # Plot higher AUC/AUPRC first in the legend/order.
    ordered_labels = group_order(stats, metric)
    ordered_keys = sorted(
        grouped,
        key=lambda k: ordered_labels.index(method_label(k[0], None if k[1] == "" else float(k[1])))
        if method_label(k[0], None if k[1] == "" else float(k[1])) in ordered_labels else 999,
    )

    for key in ordered_keys:
        method, kernel_str = key
        kernel = None if kernel_str == "" else float(kernel_str)
        label = method_label(method, kernel)
        mats = np.vstack([interpolate_curve(c, kind) for c in grouped[key]])
        mean = mats.mean(axis=0)
        sd = mats.std(axis=0) if mats.shape[0] > 1 else np.zeros_like(mean)

        metric_value = label_to_metric.get(label)
        display_label = f"{label} ({metric.upper()}={metric_value:.3f})" if metric_value is not None else label

        if kind == "roc":
            x = ROC_GRID
            y = mean
            ax.plot(x, y, linewidth=2.5, label=display_label)
            if mats.shape[0] > 1:
                ax.fill_between(x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1), alpha=0.15)
        else:
            x = PR_GRID
            y = mean
            ax.plot(x, y, linewidth=2.5, label=display_label)
            if mats.shape[0] > 1:
                ax.fill_between(x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1), alpha=0.15)

    if kind == "roc":
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="random")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-recall curves")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_metric_bar(stats: pd.DataFrame, metric: str, output_path: Path) -> None:
    if stats.empty or metric not in stats.columns:
        return

    plot_df = stats.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return

    order = group_order(plot_df, metric)
    summary = (
        plot_df.groupby("label", dropna=False)[metric]
        .agg(mean="mean", sd="std", n="count")
        .reindex(order)
        .reset_index()
    )

    x = np.arange(len(summary))
    means = summary["mean"].to_numpy(dtype=float)
    sds = summary["sd"].fillna(0.0).to_numpy(dtype=float)
    yerr = sds if np.any(sds > 0) else None

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(order)), 7))
    ax.bar(x, means, yerr=yerr, capsize=4 if yerr is not None else 0)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"].tolist())
    ax.set_xlabel("")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by method")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def summarize_stats(stats: pd.DataFrame, metric: str) -> pd.DataFrame:
    if stats.empty or metric not in stats.columns:
        return pd.DataFrame()
    return (
        stats.groupby(["curve_kind", "method", "kernel_stddev", "label"], dropna=False)[metric]
        .agg(n_curves="count", mean="mean", sd="std", median="median", min="min", max="max")
        .reset_index()
        .sort_values(["curve_kind", "mean"], ascending=[True, False])
    )


def summarize_site_posteriors(files: list[SitePosteriorFile]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sf in files:
        df = sf.df
        posterior_col = get_col(df, ["posterior_prob_positive", "posterior", "probability", "prob", "score"])
        bf_col = get_col(df, ["bayes_factor", "BayesFactor", "bf"])
        truth_col = get_col(df, ["true_positive", "truth_positive", "is_positive", "truth", "y_true"])
        threshold_col = get_col(df, ["threshold", "detected", "called", "significant"])

        row: dict[str, object] = {
            "simulation_id": sf.simulation_id,
            "method": sf.method,
            "kernel_stddev": sf.kernel_stddev,
            "label": method_label(sf.method, sf.kernel_stddev),
            "n_sites": len(df),
            "path": str(sf.path),
        }

        if posterior_col is not None:
            posterior = pd.to_numeric(df[posterior_col], errors="coerce")
            row.update({
                "posterior_mean": posterior.mean(),
                "posterior_median": posterior.median(),
                "posterior_max": posterior.max(),
            })
        if bf_col is not None:
            bf = pd.to_numeric(df[bf_col], errors="coerce")
            row.update({"bayes_factor_mean": bf.mean(), "bayes_factor_max": bf.max()})
        if truth_col is not None:
            truth = df[truth_col]
            if truth.dtype == bool:
                y = truth.astype(bool)
            else:
                y = truth.astype(str).str.lower().isin(["true", "t", "1", "yes", "y"])
            row.update({"n_true_positive": int(y.sum()), "prevalence": float(y.mean())})
        if threshold_col is not None:
            thr = df[threshold_col]
            if thr.dtype == bool:
                called = thr.astype(bool)
            else:
                called = thr.astype(str).str.lower().isin(["true", "t", "1", "yes", "y"])
            row.update({"n_called_positive": int(called.sum()), "call_rate": float(called.mean())})

        rows.append(row)

    return pd.DataFrame(rows)


def write_all_curve_points(curves: list[CurveFile], kind: str, output_path: Path) -> None:
    frames: list[pd.DataFrame] = []
    for cf in curves:
        try:
            clean = clean_roc(cf.df, cf.path) if kind == "roc" else clean_pr(cf.df, cf.path)
        except Exception:
            continue
        clean = clean.copy()
        clean["curve_kind"] = kind
        clean["simulation_id"] = cf.simulation_id
        clean["method"] = cf.method
        clean["kernel_stddev"] = cf.kernel_stddev
        clean["label"] = method_label(cf.method, cf.kernel_stddev)
        clean["path"] = str(cf.path)
        frames.append(clean)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ROC/PR curves and AUC/AUPRC summaries from existing *_roc.csv, *_pr.csv, and *_site_posteriors.csv files.",
    )
    parser.add_argument("outdir", type=Path, help="Output directory produced by run_one_omnibus_multi / omnibus_multi.")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Directory for generated plots/tables. Default: OUTDIR/roc_analysis.",
    )
    parser.add_argument(
        "--no-pr",
        action="store_true",
        help="Skip PR curves/AUPRC even if *_pr.csv files are present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.expanduser().resolve()
    if not outdir.exists():
        raise FileNotFoundError(outdir)

    analysis_dir = args.analysis_dir.expanduser().resolve() if args.analysis_dir is not None else outdir / "roc_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    set_theme()

    roc_paths, pr_paths, site_paths = find_input_files(outdir)
    if not roc_paths:
        raise RuntimeError(f"No *_roc.csv files found under {outdir}")

    roc_files = [x for x in (read_curve_file(p, outdir, "roc") for p in roc_paths) if x is not None]
    pr_files = [x for x in (read_curve_file(p, outdir, "pr") for p in pr_paths) if x is not None] if not args.no_pr else []
    site_files = [x for x in (read_site_file(p, outdir) for p in site_paths) if x is not None]

    roc_stats, roc_grouped = collect_curve_stats(roc_files, "roc")
    if roc_stats.empty:
        raise RuntimeError("Found *_roc.csv files, but none contained usable fpr/tpr data.")

    roc_stats.to_csv(analysis_dir / "per_file_roc_auc.csv", index=False)
    summarize_stats(roc_stats, "auc").to_csv(analysis_dir / "summary_roc_auc.csv", index=False)
    write_all_curve_points(roc_files, "roc", analysis_dir / "all_roc_points.csv")
    plot_mean_curves(roc_grouped, roc_stats, "roc", analysis_dir / "roc_curves.png")
    plot_metric_bar(roc_stats, "auc", analysis_dir / "auc_bar.png")

    if pr_files:
        pr_stats, pr_grouped = collect_curve_stats(pr_files, "pr")
        if not pr_stats.empty:
            pr_stats.to_csv(analysis_dir / "per_file_pr_auprc.csv", index=False)
            summarize_stats(pr_stats, "auprc").to_csv(analysis_dir / "summary_pr_auprc.csv", index=False)
            write_all_curve_points(pr_files, "pr", analysis_dir / "all_pr_points.csv")
            plot_mean_curves(pr_grouped, pr_stats, "pr", analysis_dir / "pr_curves.png")
            plot_metric_bar(pr_stats, "auprc", analysis_dir / "auprc_bar.png")

    if site_files:
        site_summary = summarize_site_posteriors(site_files)
        if not site_summary.empty:
            site_summary.to_csv(analysis_dir / "site_posterior_summary.csv", index=False)

    print(f"Read {len(roc_files)} ROC file(s), {len(pr_files)} PR file(s), and {len(site_files)} site posterior file(s).")
    print(f"Wrote results to: {analysis_dir}")
    print("Main outputs:")
    print(f"  {analysis_dir / 'roc_curves.png'}")
    print(f"  {analysis_dir / 'auc_bar.png'}")
    if pr_files:
        print(f"  {analysis_dir / 'pr_curves.png'}")
        print(f"  {analysis_dir / 'auprc_bar.png'}")


if __name__ == "__main__":
    main()
