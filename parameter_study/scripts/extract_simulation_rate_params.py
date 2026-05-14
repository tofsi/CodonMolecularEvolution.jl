#!/usr/bin/env python3
"""
Extract true site rates from HyPhy Contrast-FEL omnibus / omnibus-multi
settings files.

Input:
  - extracted directory containing sims.XXX.settings files, OR
  - omnibus-multi.tar.bz / .tar.bz2 archive

Output:
  - .npz file with a compact tensor, when all simulations have the same shape:
        rates[simulation_index, site_index, parameter_index]
    otherwise stores one matrix per simulation.
  - sidecar .meta.json with simulation ids, filenames, parameter names, shapes.

Typical use:
  python extract_omnibus_true_rates.py omnibus-multi.tar.bz2 \
      --out omnibus_multi_true_rates.npz \
      --csv omnibus_multi_true_rates.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SETTINGS_RE = re.compile(r"(?:^|/)sims\.(\d+)\.settings$")
NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass
class SimRates:
    sim_id: int
    source: str
    rates: np.ndarray          # shape: sites × parameters
    variables: dict[int, str]  # usually maps beta/omega class index -> HyPhy variable name


def is_scalar(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def intish(x: Any) -> bool:
    try:
        int(str(x))
        return True
    except ValueError:
        return False


def clean_parameter_name(s: str) -> str:
    # "simulator.omega.class0" -> "omega_class0"
    s = s.replace("simulator.", "")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    return s.strip("_") or "rate"


def get_profile(data: dict[str, Any]) -> dict[str, Any] | None:
    """Handle both flat key 'simulator.site.profile' and nested JSON."""
    if "simulator.site.profile" in data:
        return data["simulator.site.profile"]

    cur: Any = data
    for key in ("simulator", "site", "profile"):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur if isinstance(cur, dict) else None


def as_numeric_row(obj: Any) -> list[float]:
    if isinstance(obj, dict):
        items = list(obj.items())
        if all(intish(k) for k, _ in items):
            items = sorted(items, key=lambda kv: int(str(kv[0])))
        return [float(v) for _, v in items]

    if isinstance(obj, (list, tuple)):
        return [float(v) for v in obj]

    if isinstance(obj, str):
        nums = NUM_RE.findall(obj)
        if not nums:
            raise ValueError(f"Could not parse numeric row from string: {obj[:80]!r}")
        return [float(x) for x in nums]

    raise TypeError(f"Cannot coerce {type(obj)} to a numeric row")


def parse_rate_matrix_literal(text: str) -> np.ndarray:
    """
    Parse HyPhy-like matrix literal:

      {
        {0.77, 0, 0, 0, 0}
        {2.98, 0, 0.4, 0, 0.14}
      }

    This is used as a fallback in case the settings file is not strict JSON.
    """
    rows: list[list[float]] = []

    # Match only innermost {...} rows.
    for body in re.findall(r"\{([^{}]+)\}", text):
        nums = NUM_RE.findall(body)
        if len(nums) >= 2:
            rows.append([float(x) for x in nums])

    if not rows:
        raise ValueError("No rate rows found in HyPhy matrix literal.")

    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent row widths in rate matrix: {sorted(widths)}")

    return np.asarray(rows, dtype=np.float64)


def coerce_rate_matrix(obj: Any) -> np.ndarray:
    """Coerce JSON-parsed rates into a sites × parameters matrix."""
    if isinstance(obj, str):
        return parse_rate_matrix_literal(obj)

    if isinstance(obj, dict):
        items = list(obj.items())

        # Single row encoded as {"0": 0.77, "1": 0, ...}
        if items and all(is_scalar(v) for _, v in items):
            if all(intish(k) for k, _ in items):
                items = sorted(items, key=lambda kv: int(str(kv[0])))
            rows = [[float(v) for _, v in items]]
        else:
            if all(intish(k) for k, _ in items):
                items = sorted(items, key=lambda kv: int(str(kv[0])))
            rows = [as_numeric_row(v) for _, v in items]

    elif isinstance(obj, list):
        if not obj:
            raise ValueError("Empty rates list.")
        if all(is_scalar(v) for v in obj):
            rows = [as_numeric_row(obj)]
        else:
            rows = [as_numeric_row(v) for v in obj]

    else:
        raise TypeError(f"Unsupported rates object type: {type(obj)}")

    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent row widths in rate matrix: {sorted(widths)}")

    return np.asarray(rows, dtype=np.float64)


def coerce_variables(obj: Any) -> dict[int, str]:
    if not isinstance(obj, dict):
        return {}
    out: dict[int, str] = {}
    for k, v in obj.items():
        if intish(k):
            out[int(str(k))] = str(v)
    return out


def extract_brace_block_after_key(text: str, key: str, start: int = 0) -> str | None:
    """
    Find a key and return the balanced {...} block after its colon.
    Ignores braces inside quoted strings.
    """
    pos = text.find(f'"{key}"', start)
    if pos < 0:
        pos = text.find(key, start)
    if pos < 0:
        return None

    colon = text.find(":", pos)
    if colon < 0:
        return None

    open_brace = text.find("{", colon)
    if open_brace < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(open_brace, len(text)):
        c = text[i]

        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue

        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace : i + 1]

    return None


def parse_variables_literal(text: str | None) -> dict[int, str]:
    if text is None:
        return {}
    return {int(k): v for k, v in re.findall(r'"(\d+)"\s*:\s*"([^"]+)"', text)}


def parse_settings_text(text: str, source: str) -> tuple[np.ndarray, dict[int, str]]:
    """
    First try strict JSON. If that fails, fall back to extracting the HyPhy-style
    matrix literal shown in the prompt.
    """
    try:
        data = json.loads(text)
        profile = get_profile(data)
        if profile is None:
            raise KeyError("Missing simulator.site.profile")

        rates = coerce_rate_matrix(profile["rates"])
        variables = coerce_variables(profile.get("variables", {}))
        return rates, variables

    except Exception as json_error:
        profile_pos = text.find('"simulator.site.profile"')
        if profile_pos < 0:
            profile_pos = 0

        rates_block = extract_brace_block_after_key(text, "rates", start=profile_pos)
        if rates_block is None:
            raise RuntimeError(
                f"Could not parse rates from {source}. "
                f"Strict JSON error was: {json_error}"
            ) from json_error

        vars_block = extract_brace_block_after_key(text, "variables", start=profile_pos)
        rates = parse_rate_matrix_literal(rates_block)
        variables = parse_variables_literal(vars_block)
        return rates, variables


def iter_settings_inputs(input_path: Path) -> Iterable[tuple[int, str, str]]:
    """
    Yield (sim_id, source_name, text) for every sims.XXX.settings file.
    Excludes sims.XXX.settings.replicate.N files by regex.
    """
    if input_path.is_dir():
        for p in sorted(input_path.rglob("sims.*.settings")):
            rel = p.relative_to(input_path).as_posix()
            m = SETTINGS_RE.search(rel)
            if not m:
                continue
            yield int(m.group(1)), rel, p.read_text(encoding="utf-8")

    else:
        with tarfile.open(input_path, mode="r:*") as tf:
            for member in sorted(tf.getmembers(), key=lambda m: m.name):
                if not member.isfile():
                    continue
                m = SETTINGS_RE.search(member.name)
                if not m:
                    continue

                f = tf.extractfile(member)
                if f is None:
                    continue

                text = f.read().decode("utf-8")
                yield int(m.group(1)), member.name, text


def infer_parameter_names(n_params: int, variables: dict[int, str]) -> list[str]:
    """
    Rates matrix columns are assumed to be:
      column 0: alpha
      columns 1..: branch-set-specific beta/omega/rate variables
    """
    if variables and len(variables) == n_params - 1:
        ordered = [variables[i] for i in sorted(variables)]
        return ["alpha"] + [clean_parameter_name(x) for x in ordered]

    return ["alpha"] + [f"rate_{i}" for i in range(1, n_params)]


def ratio_matrix(rates: np.ndarray) -> np.ndarray:
    """
    Compute rate_i / alpha for columns 1:end.
    Use only if the non-alpha columns are beta-like rates.
    """
    alpha = rates[:, [0]]
    out = np.full((rates.shape[0], rates.shape[1] - 1), np.nan, dtype=rates.dtype)
    np.divide(rates[:, 1:], alpha, out=out, where=(alpha != 0))
    return out


def write_csv(records: list[SimRates], csv_path: Path, parameter_names: list[str]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["simulation_id", "source", "site"] + parameter_names)

        for rec in records:
            for site_idx, row in enumerate(rec.rates, start=1):
                writer.writerow([rec.sim_id, rec.source, site_idx] + list(row))


def write_outputs(
    records: list[SimRates],
    out_npz: Path,
    dtype: np.dtype,
    save_rate_ratios: bool,
    csv_path: Path | None,
) -> None:
    records = sorted(records, key=lambda r: r.sim_id)

    first_n_params = records[0].rates.shape[1]
    parameter_names = infer_parameter_names(first_n_params, records[0].variables)

    shapes = [tuple(r.rates.shape) for r in records]
    rectangular = all(s == shapes[0] for s in shapes)

    arrays: dict[str, np.ndarray] = {
        "sim_ids": np.asarray([r.sim_id for r in records], dtype=np.int64),
        "sources": np.asarray([r.source for r in records], dtype=str),
    }

    if rectangular:
        stacked = np.stack([r.rates.astype(dtype, copy=False) for r in records], axis=0)
        arrays["rates"] = stacked

        if save_rate_ratios:
            alpha = stacked[:, :, [0]]
            ratios = np.full(stacked[:, :, 1:].shape, np.nan, dtype=stacked.dtype)
            np.divide(stacked[:, :, 1:], alpha, out=ratios, where=(alpha != 0))
            arrays["rate_ratios_to_alpha"] = ratios

    else:
        for i, rec in enumerate(records):
            key = f"rates_{i:04d}"
            arrays[key] = rec.rates.astype(dtype, copy=False)
            if save_rate_ratios:
                arrays[f"rate_ratios_to_alpha_{i:04d}"] = ratio_matrix(
                    rec.rates.astype(dtype, copy=False)
                )

    np.savez_compressed(out_npz, **arrays)

    meta = {
        "description": "True simulator.site.profile rates extracted from HyPhy omnibus settings files.",
        "array_layout_if_rectangular": "rates[simulation_index, site_index, parameter_index]",
        "site_indexing": "NPZ arrays are 0-based; CSV site column is 1-based.",
        "rectangular": rectangular,
        "parameter_names": parameter_names,
        "n_simulations": len(records),
        "shapes": [
            {"sim_id": r.sim_id, "source": r.source, "shape_sites_by_parameters": list(r.rates.shape)}
            for r in records
        ],
        "variables_by_simulation": [
            {"sim_id": r.sim_id, "variables": {str(k): v for k, v in sorted(r.variables.items())}}
            for r in records
        ],
        "saved_rate_ratios_to_alpha": save_rate_ratios,
    }

    meta_path = out_npz.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if csv_path is not None:
        write_csv(records, csv_path, parameter_names)

    print(f"Wrote {out_npz}")
    print(f"Wrote {meta_path}")
    if csv_path is not None:
        print(f"Wrote {csv_path}")
    print(f"Extracted {len(records)} simulations.")
    print(f"Rectangular tensor: {rectangular}")
    if rectangular:
        print(f"rates shape: {arrays['rates'].shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        help="Path to extracted omnibus directory or omnibus-multi.tar.bz2 archive.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("omnibus_multi_true_rates.npz"),
        help="Output .npz path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional long-form CSV output path.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Floating-point dtype to store in the .npz.",
    )
    parser.add_argument(
        "--save-rate-ratios-to-alpha",
        action="store_true",
        help=(
            "Also save columns 1:end divided by alpha. "
            "Use only if columns 1:end are beta-like rates and you want omega-like ratios."
        ),
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Skip files that fail to parse instead of aborting.",
    )
    args = parser.parse_args()

    records: list[SimRates] = []
    failures: list[tuple[str, str]] = []

    for sim_id, source, text in iter_settings_inputs(args.input):
        try:
            rates, variables = parse_settings_text(text, source)
            if rates.ndim != 2:
                raise ValueError(f"Expected 2D rates matrix, got shape {rates.shape}")
            if rates.shape[1] < 2:
                raise ValueError(f"Expected alpha plus at least one rate column, got {rates.shape}")
            records.append(SimRates(sim_id=sim_id, source=source, rates=rates, variables=variables))
        except Exception as e:
            if args.keep_going:
                failures.append((source, repr(e)))
            else:
                raise

    if not records:
        raise RuntimeError("No sims.XXX.settings files were extracted.")

    if failures:
        print("Skipped files:")
        for source, err in failures:
            print(f"  {source}: {err}")

    write_outputs(
        records=records,
        out_npz=args.out,
        dtype=np.dtype(args.dtype),
        save_rate_ratios=args.save_rate_ratios_to_alpha,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    main()