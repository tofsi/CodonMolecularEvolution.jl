#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_meta(meta_path):
    if meta_path is None or not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def get_source_for_sim(meta, sim_id):
    for row in meta.get("shapes", []):
        if int(row.get("sim_id")) == int(sim_id):
            return str(row.get("source", ""))
    return ""


def iter_rate_matrices(npz_path, meta):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--target", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    meta_path = args.meta
    if meta_path is None:
        meta_path = args.npz.with_suffix(".meta.json")

    meta = load_meta(meta_path)

    rows = []

    for sim_id, source, rates in iter_rate_matrices(args.npz, meta):
        if rates.ndim != 2 or rates.shape[1] < 2:
            raise ValueError(f"sim {sim_id}: expected sites × parameters with alpha plus beta groups, got {rates.shape}")

        alpha = rates[:, [0]]
        beta = rates[:, 1:]

        omega = np.full(beta.shape, np.nan, dtype=float)
        np.divide(beta, alpha, out=omega, where=(alpha != 0))

        pos_by_group = omega > 1.0
        pos_any = np.any(pos_by_group, axis=1)

        n_sites = rates.shape[0]
        n_pos_any = int(np.sum(pos_any))
        prop_pos_any = n_pos_any / n_sites

        row = {
            "simulation_id": sim_id,
            "simulation_label": f"sim_{sim_id}",
            "example_replicate_id": f"sim_{sim_id}_replicate_1",
            "tree_file": f"sims.{sim_id}.nwk",
            "source": source,
            "n_sites": n_sites,
            "n_pos_any_group": n_pos_any,
            "prop_pos_any_group": prop_pos_any,
            "distance_to_target": abs(prop_pos_any - args.target),
            "mean_max_omega": float(np.nanmean(np.nanmax(omega, axis=1))),
            "max_omega": float(np.nanmax(omega)),
        }

        for j in range(pos_by_group.shape[1]):
            row[f"prop_pos_group_{j}"] = float(np.mean(pos_by_group[:, j]))

        rows.append(row)

    rows.sort(key=lambda r: r["distance_to_target"])

    top_rows = rows[: args.top]

    fieldnames = list(top_rows[0].keys())

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(top_rows)
        print(f"Wrote {args.out}")

    for r in top_rows:
        print(
            f"{r['example_replicate_id']}: "
            f"prop_pos_any_group={r['prop_pos_any_group']:.3f}, "
            f"n_pos={r['n_pos_any_group']}/{r['n_sites']}, "
            f"distance={r['distance_to_target']:.3f}, "
            f"tree={r['tree_file']}"
        )


if __name__ == "__main__":
    main()