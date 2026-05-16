#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np


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


def mean_finite(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def fmt_float(x: float, digits: int = 10) -> str:
    if np.isnan(x):
        return "NaN"
    if np.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return f"{x:.{digits}g}"


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
    parser.add_argument("--max-sims", type=int, default=None)
    args = parser.parse_args()

    meta_path = args.meta
    if meta_path is None:
        meta_path = args.npz.with_suffix(".meta.json")

    meta = load_meta(meta_path)
    wanted_sims = None if args.sim is None else set(args.sim)

    rows = []
    all_fieldnames = ["simulation_id", "source", "n_sites", "n_parameters"]

    count = 0

    for sim_id, source, rates in iter_rate_matrices(args.npz, meta):
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

        row = {
            "simulation_id": sim_id,
            "source": source,
            "n_sites": n_sites,
            "n_parameters": n_params,
            "alpha_mean": fmt_float(mean_finite(rates[:, 0])),
        }

        if "alpha_mean" not in all_fieldnames:
            all_fieldnames.append("alpha_mean")

        for j in range(1, n_params):
            group_index = j
            colname = clean_group_name(param_names[j], group_index, "beta")
            mean_name = f"{colname}_mean"
            row[mean_name] = fmt_float(mean_finite(rates[:, j]))

            if mean_name not in all_fieldnames:
                all_fieldnames.append(mean_name)

        if args.include_omega_means:
            alpha = rates[:, [0]]
            beta = rates[:, 1:]
            omega = np.full(beta.shape, np.nan, dtype=float)
            np.divide(beta, alpha, out=omega, where=(alpha != 0))

            any_pos = np.any(omega > 1.0, axis=1)
            row["n_pos_any_class"] = int(np.sum(any_pos))
            row["prop_pos_any_class"] = fmt_float(float(np.mean(any_pos)))

            for name in ["n_pos_any_class", "prop_pos_any_class"]:
                if name not in all_fieldnames:
                    all_fieldnames.append(name)

            for j in range(omega.shape[1]):
                group_index = j + 1
                colname = clean_group_name(param_names[group_index], group_index, "omega")
                mean_name = f"{colname}_mean"
                prop_name = f"prop_pos_{colname}"
                n_name = f"n_pos_{colname}"

                row[mean_name] = fmt_float(mean_finite(omega[:, j]))
                row[prop_name] = fmt_float(float(np.mean(omega[:, j] > 1.0)))
                row[n_name] = int(np.sum(omega[:, j] > 1.0))

                for field in [mean_name, prop_name, n_name]:
                    if field not in all_fieldnames:
                        all_fieldnames.append(field)

        rows.append(row)

    if not rows:
        raise RuntimeError("No simulations selected.")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out}")
    print(f"Included {count} simulations")


if __name__ == "__main__":
    main()
