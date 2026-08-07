#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


# These match the full FLAVOR grid currently defined in FLAVOR.jl.
DEFAULT_MU_GRID = (0.01, 16.0, 8)
DEFAULT_ALPHA_GRID = (0.01, 10.0, 8)


def load_meta(meta_path: Path | None) -> dict:
    if meta_path is None or not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def parameter_names_from_meta(meta: dict, n_params: int) -> list[str]:
    names = meta.get("parameter_names")
    if names is not None and len(names) == n_params:
        return [str(x) for x in names]
    return ["alpha"] + [f"beta_group_{i}" for i in range(1, n_params)]


def clean_group_name(name: str, group_index: int, prefix: str) -> str:
    name = str(name)
    name = name.replace("simulator.", "")
    name = name.replace("simulator_", "")
    name = name.replace("beta.", "beta_")
    name = name.replace("omega.", "omega_")
    name = name.replace("rate.", "rate_")

    if name.startswith(("beta_", "omega_", "rate_")):
        suffix = name.split("_", 1)[1]
        return f"{prefix}_{suffix}"

    return f"{prefix}_group_{group_index}"


def get_source_for_sim(meta: dict, sim_id: int) -> str:
    for row in meta.get("shapes", []):
        if int(row.get("sim_id")) == int(sim_id):
            return str(row.get("source", ""))
    return ""


def iter_rate_matrices(npz_path: Path, meta: dict):
    z = np.load(npz_path, allow_pickle=True)
    sim_ids = np.asarray(z["sim_ids"], dtype=int)

    if "rates" in z.files:
        rates = z["rates"]
        for i, sim_id in enumerate(sim_ids):
            yield int(sim_id), get_source_for_sim(meta, int(sim_id)), np.asarray(rates[i], dtype=float)
    else:
        for i, sim_id in enumerate(sim_ids):
            key = f"rates_{i:04d}"
            if key not in z.files:
                raise KeyError(f"Expected key {key!r} in {npz_path}")
            yield int(sim_id), get_source_for_sim(meta, int(sim_id)), np.asarray(z[key], dtype=float)


def finite_values(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def mean_finite(x: np.ndarray) -> float:
    x = finite_values(x)
    return float(np.mean(x)) if x.size else float("nan")


def median_finite(x: np.ndarray) -> float:
    x = finite_values(x)
    return float(np.median(x)) if x.size else float("nan")


def std_finite(x: np.ndarray) -> float:
    x = finite_values(x)
    return float(np.std(x)) if x.size else float("nan")


def fmt_float(x: float, digits: int = 10) -> str:
    if np.isnan(x):
        return "NaN"
    if np.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return f"{x:.{digits}g}"


def flavor_grid(lower: float, upper: float, n_below_one: int) -> np.ndarray:
    """Reproduce FLAVOR's transformed gridsetup function."""
    if lower <= 0 or upper <= 0:
        raise ValueError("FLAVOR grid bounds must be positive")
    if n_below_one < 1:
        raise ValueError("n_below_one must be at least 1")

    tr = lambda x: np.maximum(0.0, 10.0**x - 0.05)
    trinv = lambda x: np.log10(x + 0.05)
    step = (trinv(1.0) - trinv(lower)) / n_below_one

    # Julia's a:step:b includes the endpoint when it falls on the sequence.
    transformed = np.arange(trinv(lower), trinv(upper) + step * 1e-9, step)
    return np.asarray(tr(transformed), dtype=float)


def nearest_grid_indices(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.abs(values[:, None] - grid[None, :]).argmin(axis=1)


def bounded_ramp(x: float, lower_full: float, upper_full: float) -> float:
    """
    Score a prevalence in [0, 1].

    It rises linearly from zero to one before lower_full, remains one through
    upper_full, and declines linearly to zero as prevalence approaches one.
    """
    if not np.isfinite(x) or x <= 0.0 or x >= 1.0:
        return 0.0
    if x < lower_full:
        return float(x / lower_full)
    if x <= upper_full:
        return 1.0
    return float(max(0.0, (1.0 - x) / (1.0 - upper_full)))


def weighted_mean_available(values_and_weights: list[tuple[float, float]]) -> float:
    valid = [(value, weight) for value, weight in values_and_weights if np.isfinite(value)]
    if not valid:
        return float("nan")
    numerator = sum(value * weight for value, weight in valid)
    denominator = sum(weight for _, weight in valid)
    return float(numerator / denominator)


def calculate_suitability_metrics(
    rates: np.ndarray,
    mu_grid: np.ndarray,
    alpha_grid: np.ndarray,
) -> dict[str, float | int]:
    """
    Summarize how well a simulation can expose the strengths of smoothing over
    FLAVOR's (mu, alpha) grid.

    The first rate column is alpha. The remaining columns are beta values for
    branch groups. For each site, omega = beta / alpha and mu is approximated by
    the mean omega over groups.

    The composite score is deliberately transparent:
      45% smoothing opportunity on the (mu, alpha) grid
      25% episodicity among positively selected sites
      15% a non-degenerate prevalence of positive sites
       5% positive-selection strength
      10% coverage by the interior of the FLAVOR grid
    """
    alpha = np.asarray(rates[:, 0], dtype=float)
    beta = np.asarray(rates[:, 1:], dtype=float)

    omega = np.full(beta.shape, np.nan, dtype=float)
    valid_alpha = np.isfinite(alpha) & (alpha > 0.0)
    np.divide(
        beta,
        alpha[:, None],
        out=omega,
        where=valid_alpha[:, None] & np.isfinite(beta),
    )

    finite_omega = np.isfinite(omega)
    n_finite_groups = finite_omega.sum(axis=1)
    valid_site = valid_alpha & (n_finite_groups > 0)

    site_mu = np.full(alpha.shape, np.nan, dtype=float)
    omega_sums = np.where(finite_omega, omega, 0.0).sum(axis=1)
    np.divide(omega_sums, n_finite_groups, out=site_mu, where=n_finite_groups > 0)
    valid_site &= np.isfinite(site_mu) & (site_mu > 0.0)

    n_valid_sites = int(valid_site.sum())
    if n_valid_sites == 0:
        return {
            "n_valid_rate_sites": 0,
            "site_mu_mean": float("nan"),
            "site_mu_median": float("nan"),
            "log_mu_sd": float("nan"),
            "log_alpha_sd": float("nan"),
            "prop_pos_any_class": float("nan"),
            "prop_episodic_sites": float("nan"),
            "episodic_share_of_positive": float("nan"),
            "median_positive_group_fraction": float("nan"),
            "median_positive_omega": float("nan"),
            "mu_alpha_in_grid_fraction": float("nan"),
            "mu_alpha_interior_fraction": float("nan"),
            "mu_alpha_occupied_cells": 0,
            "mu_alpha_effective_cells": float("nan"),
            "mu_alpha_adjacent_mass_fraction": float("nan"),
            "smoothing_opportunity_score": float("nan"),
            "positive_prevalence_score": float("nan"),
            "episodic_score": float("nan"),
            "selection_strength_score": float("nan"),
            "grid_coverage_score": float("nan"),
            "suitability_score": float("nan"),
        }

    positive_counts = ((omega > 1.0) & finite_omega).sum(axis=1)
    positive_group_fraction = np.full(alpha.shape, np.nan, dtype=float)
    np.divide(
        positive_counts,
        n_finite_groups,
        out=positive_group_fraction,
        where=n_finite_groups > 0,
    )

    any_positive = valid_site & (positive_counts > 0)
    episodic = valid_site & (positive_counts > 0) & (positive_counts < n_finite_groups)

    prop_pos = float(any_positive.sum() / n_valid_sites)
    prop_episodic = float(episodic.sum() / n_valid_sites)
    episodic_share = (
        float(episodic.sum() / any_positive.sum())
        if any_positive.any() and beta.shape[1] > 1
        else float("nan")
    )

    positive_omega_values = omega[np.isfinite(omega) & (omega > 1.0)]
    median_positive_omega = median_finite(positive_omega_values)
    median_positive_group_fraction = median_finite(positive_group_fraction[any_positive])

    valid_mu = site_mu[valid_site]
    valid_alpha_values = alpha[valid_site]

    in_grid = (
        (valid_mu >= mu_grid[0])
        & (valid_mu <= mu_grid[-1])
        & (valid_alpha_values >= alpha_grid[0])
        & (valid_alpha_values <= alpha_grid[-1])
    )
    in_grid_fraction = float(np.mean(in_grid))

    occupied_cells = 0
    effective_cells = float("nan")
    adjacent_mass_fraction = 0.0
    interior_fraction = 0.0

    if np.any(in_grid):
        mu_indices = nearest_grid_indices(valid_mu[in_grid], mu_grid)
        alpha_indices = nearest_grid_indices(valid_alpha_values[in_grid], alpha_grid)
        cells = list(zip(mu_indices.tolist(), alpha_indices.tolist()))
        cell_counts = Counter(cells)
        occupied_cells = len(cell_counts)

        counts = np.asarray(list(cell_counts.values()), dtype=float)
        cell_probs = counts / counts.sum()
        effective_cells = float(np.exp(-np.sum(cell_probs * np.log(cell_probs))))

        neighboring_cells = set()
        occupied = set(cell_counts)
        for cell in occupied:
            i, j = cell
            has_distinct_neighbor = any(
                other != cell
                and abs(other[0] - i) <= 1
                and abs(other[1] - j) <= 1
                for other in occupied
            )
            if has_distinct_neighbor:
                neighboring_cells.add(cell)

        adjacent_mass = sum(cell_counts[cell] for cell in neighboring_cells)
        adjacent_mass_fraction = float(adjacent_mass / sum(cell_counts.values()))

        interior = (
            (mu_indices > 0)
            & (mu_indices < len(mu_grid) - 1)
            & (alpha_indices > 0)
            & (alpha_indices < len(alpha_grid) - 1)
        )
        interior_fraction = float(np.mean(interior))

    # A single occupied grid cell does not demonstrate a benefit from smoothing.
    effective_cell_score = (
        float(np.clip((effective_cells - 1.0) / 4.0, 0.0, 1.0))
        if np.isfinite(effective_cells)
        else 0.0
    )
    smoothing_opportunity = adjacent_mass_fraction * effective_cell_score

    # Reward simulations with enough positive sites to measure power, but avoid
    # simulations where almost every site is positive and the task is trivial.
    positive_prevalence_score = bounded_ramp(prop_pos, lower_full=0.05, upper_full=0.50)

    # Episodicity is unavailable with only one beta/rate group; the composite
    # score then automatically reweights over the remaining components.
    episodic_score = episodic_share

    # Rates barely above one provide little signal. The score reaches one at a
    # median positively selected omega of 3 and then plateaus.
    selection_strength_score = (
        float(np.clip(np.log(median_positive_omega) / np.log(3.0), 0.0, 1.0))
        if np.isfinite(median_positive_omega) and median_positive_omega > 1.0
        else 0.0
    )

    # Being inside the grid is essential. Interior cells receive full credit;
    # boundary cells receive half-credit because convolution is edge-sensitive.
    grid_coverage_score = in_grid_fraction * (0.5 + 0.5 * interior_fraction)

    suitability_score = 100.0 * weighted_mean_available(
        [
            (smoothing_opportunity, 0.45),
            (episodic_score, 0.25),
            (positive_prevalence_score, 0.15),
            (selection_strength_score, 0.05),
            (grid_coverage_score, 0.10),
        ]
    )

    return {
        "n_valid_rate_sites": n_valid_sites,
        "site_mu_mean": mean_finite(valid_mu),
        "site_mu_median": median_finite(valid_mu),
        "log_mu_sd": std_finite(np.log(valid_mu)),
        "log_alpha_sd": std_finite(np.log(valid_alpha_values)),
        "prop_pos_any_class": prop_pos,
        "prop_episodic_sites": prop_episodic,
        "episodic_share_of_positive": episodic_share,
        "median_positive_group_fraction": median_positive_group_fraction,
        "median_positive_omega": median_positive_omega,
        "mu_alpha_in_grid_fraction": in_grid_fraction,
        "mu_alpha_interior_fraction": interior_fraction,
        "mu_alpha_occupied_cells": occupied_cells,
        "mu_alpha_effective_cells": effective_cells,
        "mu_alpha_adjacent_mass_fraction": adjacent_mass_fraction,
        "smoothing_opportunity_score": smoothing_opportunity,
        "positive_prevalence_score": positive_prevalence_score,
        "episodic_score": episodic_score,
        "selection_strength_score": selection_strength_score,
        "grid_coverage_score": grid_coverage_score,
        "suitability_score": suitability_score,
    }


def add_fieldnames(fieldnames: list[str], names: list[str]) -> None:
    for name in names:
        if name not in fieldnames:
            fieldnames.append(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path, help="Path to omnibus_multi_true_rates.npz")
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Path to omnibus_multi_true_rates.meta.json. Defaults to replacing .npz with .meta.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("omnibus_multi_true_rate_means.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--include-omega-means",
        action="store_true",
        help="Also include means of beta_group / alpha for each group.",
    )
    parser.add_argument(
        "--sim",
        type=int,
        action="append",
        default=None,
        help="Optional simulation id to include. Can be repeated. Default: include all.",
    )
    parser.add_argument(
        "--max-sims",
        type=int,
        default=None,
        help="Read at most this many selected simulations before ranking.",
    )
    parser.add_argument(
        "--top-sims",
        type=int,
        default=None,
        help="After ranking, write only the top N simulations.",
    )
    parser.add_argument(
        "--preserve-input-order",
        action="store_true",
        help="Do not sort rows by suitability score. Suitability columns are still added.",
    )
    parser.add_argument("--mu-grid-lower", type=float, default=DEFAULT_MU_GRID[0])
    parser.add_argument("--mu-grid-upper", type=float, default=DEFAULT_MU_GRID[1])
    parser.add_argument("--mu-grid-below-one", type=int, default=DEFAULT_MU_GRID[2])
    parser.add_argument("--alpha-grid-lower", type=float, default=DEFAULT_ALPHA_GRID[0])
    parser.add_argument("--alpha-grid-upper", type=float, default=DEFAULT_ALPHA_GRID[1])
    parser.add_argument("--alpha-grid-below-one", type=int, default=DEFAULT_ALPHA_GRID[2])
    args = parser.parse_args()

    if args.top_sims is not None and args.top_sims < 1:
        parser.error("--top-sims must be at least 1")

    meta_path = args.meta
    if meta_path is None:
        meta_path = args.npz.with_suffix(".meta.json")

    meta = load_meta(meta_path)
    wanted_sims = None if args.sim is None else set(args.sim)

    mu_grid = flavor_grid(args.mu_grid_lower, args.mu_grid_upper, args.mu_grid_below_one)
    alpha_grid = flavor_grid(
        args.alpha_grid_lower,
        args.alpha_grid_upper,
        args.alpha_grid_below_one,
    )

    suitability_fields = [
        "suitability_rank",
        "suitability_score",
        "smoothing_opportunity_score",
        "episodic_score",
        "positive_prevalence_score",
        "selection_strength_score",
        "grid_coverage_score",
        "n_valid_rate_sites",
        "site_mu_mean",
        "site_mu_median",
        "log_mu_sd",
        "log_alpha_sd",
        "prop_pos_any_class",
        "prop_episodic_sites",
        "episodic_share_of_positive",
        "median_positive_group_fraction",
        "median_positive_omega",
        "mu_alpha_in_grid_fraction",
        "mu_alpha_interior_fraction",
        "mu_alpha_occupied_cells",
        "mu_alpha_effective_cells",
        "mu_alpha_adjacent_mass_fraction",
    ]

    rows = []
    all_fieldnames = [
        "simulation_id",
        "source",
        "n_sites",
        "n_parameters",
        *suitability_fields,
    ]

    count = 0

    for input_order, (sim_id, source, rates) in enumerate(iter_rate_matrices(args.npz, meta)):
        if wanted_sims is not None and sim_id not in wanted_sims:
            continue
        if args.max_sims is not None and count >= args.max_sims:
            break

        count += 1

        if rates.ndim != 2:
            raise ValueError(f"Simulation {sim_id}: expected 2D sites × parameters matrix, got {rates.shape}")
        if rates.shape[1] < 2:
            raise ValueError(f"Simulation {sim_id}: expected alpha plus at least one beta/rate column, got {rates.shape}")

        n_sites, n_params = rates.shape
        param_names = parameter_names_from_meta(meta, n_params)
        suitability = calculate_suitability_metrics(rates, mu_grid, alpha_grid)

        row = {
            "simulation_id": sim_id,
            "source": source,
            "n_sites": n_sites,
            "n_parameters": n_params,
            "alpha_mean": mean_finite(rates[:, 0]),
            "_input_order": input_order,
            **suitability,
        }

        add_fieldnames(all_fieldnames, ["alpha_mean"])

        for j in range(1, n_params):
            group_index = j
            colname = clean_group_name(param_names[j], group_index, "beta")
            mean_name = f"{colname}_mean"
            row[mean_name] = mean_finite(rates[:, j])
            add_fieldnames(all_fieldnames, [mean_name])

        if args.include_omega_means:
            alpha = rates[:, [0]]
            beta = rates[:, 1:]
            omega = np.full(beta.shape, np.nan, dtype=float)
            np.divide(
                beta,
                alpha,
                out=omega,
                where=np.isfinite(beta) & np.isfinite(alpha) & (alpha != 0),
            )

            any_pos = np.any(omega > 1.0, axis=1)
            row["n_pos_any_class"] = int(np.sum(any_pos))
            add_fieldnames(all_fieldnames, ["n_pos_any_class"])

            for j in range(omega.shape[1]):
                group_index = j + 1
                colname = clean_group_name(param_names[group_index], group_index, "omega")
                mean_name = f"{colname}_mean"
                prop_name = f"prop_pos_{colname}"
                n_name = f"n_pos_{colname}"

                row[mean_name] = mean_finite(omega[:, j])
                row[prop_name] = float(np.mean(omega[:, j] > 1.0))
                row[n_name] = int(np.sum(omega[:, j] > 1.0))
                add_fieldnames(all_fieldnames, [mean_name, prop_name, n_name])

        rows.append(row)

    if not rows:
        raise RuntimeError("No simulations selected.")

    if not args.preserve_input_order:
        rows.sort(
            key=lambda row: (
                -row["suitability_score"] if np.isfinite(row["suitability_score"]) else math.inf,
                row["simulation_id"],
            )
        )

    for rank, row in enumerate(rows, start=1):
        row["suitability_rank"] = rank

    if args.top_sims is not None:
        rows = rows[: args.top_sims]

    # Format floating-point values only after sorting, so ranking uses full precision.
    formatted_rows = []
    for row in rows:
        formatted = {}
        for key, value in row.items():
            if key.startswith("_"):
                continue
            if isinstance(value, (float, np.floating)):
                formatted[key] = fmt_float(float(value))
            elif isinstance(value, (int, np.integer)):
                formatted[key] = int(value)
            else:
                formatted[key] = value
        formatted_rows.append(formatted)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(formatted_rows)

    print(f"Wrote {args.out}")
    print(f"Included {count} simulations")
    if not args.preserve_input_order:
        print("Ordered by suitability_score (highest first)")
    if args.top_sims is not None:
        print(f"Wrote top {len(rows)} simulations")


if __name__ == "__main__":
    main()
