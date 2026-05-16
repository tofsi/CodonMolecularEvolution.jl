#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


def load_meta(meta_path: Path | None) -> dict:
    if meta_path is None or not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def get_parameter_names(meta: dict, n_params: int) -> list[str]:
    names = meta.get("parameter_names")
    if names is not None and len(names) == n_params:
        return [str(x) for x in names]

    return ["alpha"] + [f"rate_{i}" for i in range(1, n_params)]


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


def fmt(x: float) -> str:
    if np.isnan(x):
        return "NaN"
    if np.isinf(x):
        return "Inf"
    return f"{x:.5g}"


def build_sim_table(sim_id: int, source: str, rates: np.ndarray, meta: dict, include_ratios: bool):
    n_sites, n_params = rates.shape
    param_names = get_parameter_names(meta, n_params)

    header = ["site"] + param_names

    rows = [header]

    for site_index in range(n_sites):
        row = [str(site_index + 1)] + [fmt(x) for x in rates[site_index, :]]
        rows.append(row)

    if include_ratios and n_params >= 2:
        ratio_header = ["site"] + [f"{name}/alpha" for name in param_names[1:]] + ["max_ratio", "positive"]
        ratio_rows = [ratio_header]

        alpha = rates[:, [0]]
        ratios = np.full((n_sites, n_params - 1), np.nan, dtype=float)
        np.divide(rates[:, 1:], alpha, out=ratios, where=(alpha != 0))

        for site_index in range(n_sites):
            r = ratios[site_index, :]
            positive = bool(np.any(r > 1.0))
            ratio_rows.append(
                [str(site_index + 1)]
                + [fmt(x) for x in r]
                + [fmt(np.nanmax(r)), str(positive)]
            )

        return rows, ratio_rows

    return rows, None


def make_table(data, font_size=6):
    table = Table(data, repeatRows=1)

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path, help="Path to omnibus_multi_true_rates.npz")
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Path to omnibus_multi_true_rates.meta.json. Defaults to replacing .npz with .meta.json.",
    )
    parser.add_argument("--out", type=Path, default=Path("omnibus_multi_true_rates.pdf"))
    parser.add_argument("--include-ratios", action="store_true")
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

    doc = SimpleDocTemplate(
        str(args.out),
        pagesize=landscape(A4),
        leftMargin=24,
        rightMargin=24,
        topMargin=24,
        bottomMargin=24,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Omnibus-multi true rate parameters", styles["Title"]))
    story.append(Paragraph(f"Source NPZ: {args.npz}", styles["Normal"]))
    if meta_path.exists():
        story.append(Paragraph(f"Metadata: {meta_path}", styles["Normal"]))
    story.append(Spacer(1, 12))

    count = 0

    for sim_id, source, rates in iter_rate_matrices(args.npz, meta):
        if wanted_sims is not None and sim_id not in wanted_sims:
            continue

        if args.max_sims is not None and count >= args.max_sims:
            break

        count += 1

        n_sites, n_params = rates.shape
        story.append(Paragraph(f"Simulation {sim_id}", styles["Heading1"]))
        if source:
            story.append(Paragraph(f"Settings source: {source}", styles["Normal"]))
        story.append(Paragraph(f"Shape: {n_sites} sites x {n_params} parameters", styles["Normal"]))
        story.append(Spacer(1, 6))

        rates_table, ratios_table = build_sim_table(
            sim_id,
            source,
            rates,
            meta,
            include_ratios=args.include_ratios,
        )

        story.append(Paragraph("Raw true rates", styles["Heading2"]))
        story.append(make_table(rates_table, font_size=5.5))

        if ratios_table is not None:
            story.append(Spacer(1, 10))
            story.append(Paragraph("Derived ratios to alpha", styles["Heading2"]))
            story.append(make_table(ratios_table, font_size=5.5))

        story.append(PageBreak())

    if count == 0:
        raise RuntimeError("No simulations selected for output.")

    doc.build(story)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()