"""
Table formatting utilities reproducing Tables 1-9 of the paper in Markdown,
LaTeX, or CSV format.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from .evaluator import (
    COMPACT_TIMESTAMPS_25FPS,
    COMPACT_TIMESTAMPS_60FPS,
    EXTENDED_TIMESTAMPS_25FPS,
)
from .results_loader import (
    compute_multiseed_hmpe,
    compute_suite_hmpe,
    load_benchmark_suite,
)

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "ZeroVel": "ZeroVel",
    "HRI": "HRI",
    "DMGNN": "DMGNN",
    "PVRED": "PVRED",
    "AuxFormer": "AuxFormer",
}

METRIC_DECIMALS: Dict[str, int] = {
    "MPJPE": 1,
    "JAD": 3,
    "PSDS": 4,
    "MAE_XYZ": 3,
    "MAE_ZXY": 3,
}


def _render_table(headers: List[str], rows: List[List[str]], output_format: str = "markdown") -> str:
    fmt = output_format.strip().lower()
    if fmt == "csv":
        lines = [",".join(headers)]
        for r in rows:
            lines.append(",".join(r))
        return "\n".join(lines)
    if fmt == "latex":
        lines = [" & ".join(headers) + r" \\", r"\hline"]
        for r in rows:
            lines.append(" & ".join(r) + r" \\")
        return "\n".join(lines)
    # Default: GitHub-Flavored Markdown table
    col_widths = [max(len(str(x)) for x in [h] + [r[i] for r in rows]) for i, h in enumerate(headers)]
    hdr_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    ]
    return "\n".join([hdr_line, sep_line] + body_lines)


def _format_timestamp_header(t_ms: int, dataset: str) -> str:
    if "60fps" in dataset and t_ms == 17:
        return "16.67 ms"
    return f"{t_ms} ms"


def format_metric_table(
    dataset: str = "amass_25fps",
    metric: str = "MPJPE",
    action: Optional[str] = None,
    timestamps_ms: Optional[List[int]] = None,
    decimals: Optional[int] = None,
    benchmark_root: Optional[Union[str, Path]] = None,
    extra_models: Optional[Dict[str, Union[str, Path]]] = None,
    output_format: str = "markdown",
) -> str:
    """
    Generate a single-run evaluation table (reproducing Tables 1, 2, 3, and 5).

    Args:
        dataset: `"h36m_25fps"`, `"amass_25fps"`, `"amass_60fps"`, or `"amass2h36m_25fps"`.
        metric: `"MPJPE"`, `"JAD"`, `"PSDS"`, `"MAE_XYZ"`, or `"MAE_ZXY"`.
        action: Row name (`"mean"` or one of `H36M_ACTIONS` for H3.6M; `"amass"` for AMASS).
        timestamps_ms: List of timestamps in ms. Defaults to:
            - `[1000]` for `PSDS`,
            - `[40, 80, 120, 160, 200, 320, 400, 600, 800, 1000]` for `MAE_XYZ` / `MAE_ZXY` (Table 1),
            - `[17, 100, 200, 400, 600, 800, 1000]` for 60 fps (Table 5),
            - `[40, 120, 200, 400, 600, 800, 1000]` for 25 fps (Tables 2 & 3).
        decimals: Optional override for number of decimal places.
        benchmark_root: Optional root directory for benchmark CSVs.
        extra_models: Optional `{model_name: csv_dir}` to append custom models.
        output_format: `"markdown"`, `"latex"`, or `"csv"`.
    """
    if action is None:
        action = "mean" if "h36m" in dataset else "amass"
    if timestamps_ms is None:
        if metric == "PSDS":
            timestamps_ms = [1000]
        elif metric in ("MAE_XYZ", "MAE_ZXY"):
            timestamps_ms = EXTENDED_TIMESTAMPS_25FPS
        elif "60fps" in dataset:
            timestamps_ms = COMPACT_TIMESTAMPS_60FPS
        else:
            timestamps_ms = COMPACT_TIMESTAMPS_25FPS

    dec = METRIC_DECIMALS.get(metric, 3) if decimals is None else decimals

    suite = load_benchmark_suite(
        dataset=dataset,
        metrics=(metric,),
        benchmark_root=benchmark_root,
        extra_models=extra_models,
        scale_mpjpe_to_mm=True,
    )

    headers = ["Model"] + [_format_timestamp_header(t, dataset) for t in timestamps_ms]
    rows: List[List[str]] = []
    for model_name, model_data in suite.items():
        if metric not in model_data:
            continue
        m_entry = model_data[metric]
        ts = m_entry["timestamps_ms"]
        vals = m_entry["rows"][action]
        disp_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        row_cells = [disp_name]
        for t in timestamps_ms:
            idx = int(np.where(ts == t)[0][0])
            row_cells.append(f"{vals[idx]:.{dec}f}")
        rows.append(row_cells)

    return _render_table(headers, rows, output_format=output_format)


def format_hmpe_table(
    dataset: str = "amass_25fps",
    benchmark_root: Optional[Union[str, Path]] = None,
    extra_models: Optional[Dict[str, Union[str, Path]]] = None,
    output_format: str = "markdown",
) -> str:
    """
    Generate Table 4 (`dataset="amass_25fps"`) or Table 6 (`dataset="amass_60fps"`):
    `HMPE_400`, `HMPE_1000`, and `Increase [pp]` on the `[0, 100]` scale.

    Args:
        dataset: `"amass_25fps"` (Table 4) or `"amass_60fps"` (Table 6).
        benchmark_root: Optional custom root directory for benchmark CSVs
            (defaults to `<repo>/benchmark_results`).
        extra_models: Optional `{model_name: csv_dir}` mapping to include custom models.
        output_format: `"markdown"`, `"latex"`, or `"csv"`.
    """
    hmpe_400 = compute_suite_hmpe(
        dataset=dataset, t_ms=400, benchmark_root=benchmark_root, extra_models=extra_models
    )
    hmpe_1000 = compute_suite_hmpe(
        dataset=dataset, t_ms=1000, benchmark_root=benchmark_root, extra_models=extra_models
    )

    headers = ["Model", "HMPE_400ms", "HMPE_1000ms", "Increase [pp]"]
    rows: List[List[str]] = []
    for model_name in hmpe_400:
        disp = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        v400 = hmpe_400[model_name]["hmpe"]
        v1000 = hmpe_1000[model_name]["hmpe"]
        inc = round(v1000, 2) - round(v400, 2)
        rows.append([disp, f"{v400:.2f}", f"{v1000:.2f}", f"{inc:.2f}"])

    return _render_table(headers, rows, output_format=output_format)


def format_multiseed_table(
    metric: str = "MPJPE",
    timestamps_ms: Optional[List[int]] = None,
    benchmark_root: Optional[Union[str, Path]] = None,
    output_format: str = "markdown",
) -> str:
    """
    Generate Appendix A multi-seed (12 runs on AMASS 25 fps) tables with `mean ± std`:
    - Table 7: `MPJPE` (1 decimal) and `JAD` (4 decimals)
    - Table 8: `PSDS` at `1000 ms` (4 decimals)

    Args:
        metric: `"MPJPE"`, `"JAD"`, or `"PSDS"`.
        timestamps_ms: Optional list of timestamps in ms.
        benchmark_root: Optional custom root directory (defaults to `<repo>/benchmark_results`).
        output_format: `"markdown"`, `"latex"`, or `"csv"`.
    """
    if timestamps_ms is None:
        timestamps_ms = [1000] if metric == "PSDS" else COMPACT_TIMESTAMPS_25FPS

    dec = 4 if metric in ("JAD", "PSDS") else 1

    suite = load_benchmark_suite(
        dataset="amass_25fps_multiseed",
        metrics=(metric,),
        benchmark_root=benchmark_root,
        scale_mpjpe_to_mm=True,
    )

    headers = ["Model"] + [f"{t} ms" for t in timestamps_ms]
    rows: List[List[str]] = []
    sep = r" \pm " if output_format.strip().lower() == "latex" else " ± "

    for model_name, model_data in suite.items():
        m_entry = model_data[metric]
        ts = m_entry["timestamps_ms"]
        means = m_entry["rows"]["amass"]
        stds = m_entry["rows"].get("std")
        disp = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        row = [disp]
        for t in timestamps_ms:
            idx = int(np.where(ts == t)[0][0])
            if stds is not None:
                cell = f"{means[idx]:.{dec}f}{sep}{stds[idx]:.{dec}f}"
                if output_format.strip().lower() == "latex":
                    cell = f"${cell}$"
            else:
                cell = f"{means[idx]:.{dec}f}"
            row.append(cell)
        rows.append(row)

    return _render_table(headers, rows, output_format=output_format)


def format_multiseed_hmpe_table(
    benchmark_root: Optional[Union[str, Path]] = None,
    output_format: str = "markdown",
) -> str:
    """
    Generate Appendix A Table 9: multi-seed (12 runs on AMASS 25 fps) `HMPE_400`
    and `HMPE_1000` (`mean ± std`) on the `[0, 100]` scale.

    Args:
        benchmark_root: Optional custom root directory (defaults to `<repo>/benchmark_results`).
        output_format: `"markdown"`, `"latex"`, or `"csv"`.
    """
    hmpe_400 = compute_multiseed_hmpe(t_ms=400, benchmark_root=benchmark_root)
    hmpe_1000 = compute_multiseed_hmpe(t_ms=1000, benchmark_root=benchmark_root)

    sep = r" \pm " if output_format.strip().lower() == "latex" else " ± "
    headers = ["Model", "HMPE_400ms", "HMPE_1000ms"]
    rows: List[List[str]] = []

    for model_name in ["ZeroVel", "HRI", "DMGNN", "PVRED", "AuxFormer"]:
        disp = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        h4 = hmpe_400[model_name]
        h10 = hmpe_1000[model_name]

        if model_name == "ZeroVel":
            c_h4 = f"{h4['mean']:.2f}"
            c_h10 = f"{h10['mean']:.2f}"
        else:
            c_h4 = f"{h4['mean']:.2f}{sep}{h4['std']:.2f}"
            c_h10 = f"{h10['mean']:.2f}{sep}{h10['std']:.2f}"
            if output_format.strip().lower() == "latex":
                c_h4, c_h10 = f"${c_h4}$", f"${c_h10}$"

        rows.append([disp, c_h4, c_h10])

    return _render_table(headers, rows, output_format=output_format)
