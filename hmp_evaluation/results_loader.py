"""
Utilities for loading and aggregating evaluation CSV logs and computing
HMPE_400 / HMPE_1000 scores across benchmark suites and multi-seed runs.
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .evaluator import compute_hmpe_score, get_reference_psds

DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parent.parent / "benchmark_results"

# Complete mapping of all 19 CSV files produced by `EvaluationProtocol.save_all_metrics`
METRIC_FILES: Dict[str, str] = {
    # Instantaneous per-frame metrics (used in the paper)
    "MPJPE": "Frame_Positional_Error.csv",
    "JAD": "Frame_Joint_Angle_Diff_Local.csv",
    "JAD_Global": "Frame_Joint_Angle_Diff_Global.csv",
    "MAE_XYZ": "Frame_Euler_Error_XYZ.csv",
    "MAE_ZXY": "Frame_Euler_Error_ZXY.csv",
    "Frame_Default": "Frame_Default.csv",
    # Spectral realism metric (sliding 1-second windows)
    "PSDS": "Wasserstein_Distance.csv",
    # Windowed-mean metrics at standard evaluation timestamps
    "Windowed_MPJPE": "Positional_Error.csv",
    "Windowed_JAD": "Joint_Angle_Diff_Local.csv",
    "Windowed_JAD_Global": "Joint_Angle_Diff_Global.csv",
    "Windowed_MAE_XYZ": "Euler_Error_XYZ.csv",
    "Windowed_MAE_ZXY": "Euler_Error_ZXY.csv",
    "Default_Error": "Default_Error.csv",
    # Cumulative metrics at standard evaluation timestamps
    "Cumulative_MPJPE": "Cumulative_Positional_Error.csv",
    "Cumulative_JAD": "Cumulative_Joint_Angle_Diff_Local.csv",
    "Cumulative_JAD_Global": "Cumulative_Joint_Angle_Diff_Global.csv",
    "Cumulative_MAE_XYZ": "Cumulative_Euler_Error_XYZ.csv",
    "Cumulative_MAE_ZXY": "Cumulative_Euler_Error_ZXY.csv",
    "Cumulative_Default_Error": "Cumulative_Default_Error.csv",
}

SUITE_MODELS: Dict[str, List[str]] = {
    "h36m_25fps": ["ZeroVel", "HRI", "DMGNN", "PVRED"],
    "amass_25fps": ["ZeroVel", "HRI", "DMGNN", "PVRED", "AuxFormer"],
    "amass_60fps": ["ZeroVel", "HRI", "DMGNN", "PVRED", "AuxFormer"],
    "amass2h36m_25fps": ["ZeroVel", "HRI", "DMGNN", "PVRED"],
    "amass_25fps_multiseed": ["ZeroVel", "HRI", "DMGNN", "PVRED", "AuxFormer"],
}


def _extract_ms(header_col: str) -> int:
    m = re.findall(r"\d+", header_col)
    if not m:
        raise ValueError(f"Could not parse timestamp in ms from header '{header_col}'")
    return int(m[0])


def load_metric_csv(
    folder_path: Union[str, Path],
    metric: str = "MPJPE",
    scale_mpjpe_to_mm: bool = True,
) -> dict:
    """
    Load a metric CSV from an evaluation directory.

    Args:
        folder_path: Directory containing the CSV files produced by `EvaluationProtocol`.
        metric: Metric key from `METRIC_FILES` (e.g. `"MPJPE"`, `"JAD"`, `"PSDS"`,
            `"MAE_XYZ"`, `"MAE_ZXY"`) or an explicit CSV filename.
        scale_mpjpe_to_mm: If `True` (default), converts Positional Error values from
            meters to millimeters (`* 1000.0`) as reported in the paper.

    Returns:
        Dictionary with keys:
        - `"timestamps_ms"`: `np.ndarray` of integer timestamps in milliseconds,
        - `"rows"`: `Dict[str, np.ndarray]` mapping each row name (e.g. `"amass"`, `"mean"`,
          `"walking"`, `"std"`) to its 1D array of values across timestamps.
    """
    folder = Path(folder_path)
    file_name = METRIC_FILES.get(metric, metric if metric.endswith(".csv") else f"{metric}.csv")
    csv_path = folder / file_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Metric file '{csv_path}' not found.")

    with open(csv_path, newline="") as fp:
        raw_rows = list(csv.reader(fp))

    if not raw_rows:
        raise ValueError(f"Empty CSV file: {csv_path}")

    timestamps_ms = np.array([_extract_ms(c) for c in raw_rows[0][1:]], dtype=int)
    is_pos_err = "Positional_Error" in file_name
    factor = 1000.0 if (is_pos_err and scale_mpjpe_to_mm) else 1.0

    rows_dict: Dict[str, np.ndarray] = {}
    for r in raw_rows[1:]:
        if not r or not r[0].strip():
            continue
        key = r[0].strip()
        vals = np.array([float(x) * factor for x in r[1:]], dtype=float)
        rows_dict[key] = vals

    return {
        "timestamps_ms": timestamps_ms,
        "rows": rows_dict,
        "file_path": str(csv_path),
    }


def get_suite_directories(
    dataset: str = "amass_25fps",
    benchmark_root: Optional[Union[str, Path]] = None,
    extra_models: Optional[Dict[str, Union[str, Path]]] = None,
) -> Dict[str, Path]:
    """
    Return the ordered dictionary of `{model_name: csv_directory}` for a benchmark suite.

    Args:
        dataset: Benchmark suite key (`"h36m_25fps"`, `"amass_25fps"`, `"amass_60fps"`,
            `"amass2h36m_25fps"`, or `"amass_25fps_multiseed"`).
        benchmark_root: Root directory containing the benchmark suites. Defaults to
            `DEFAULT_BENCHMARK_ROOT` (`<repo>/benchmark_results`).
        extra_models: Optional mapping `{model_name: csv_folder_path}` to append one or
            more custom models for side-by-side comparison without modifying `benchmark_results/`.
    """
    root = Path(benchmark_root) if benchmark_root is not None else DEFAULT_BENCHMARK_ROOT
    if dataset not in SUITE_MODELS:
        raise ValueError(f"Unknown dataset suite '{dataset}'. Available: {list(SUITE_MODELS.keys())}")

    dirs: Dict[str, Path] = {}
    if dataset == "amass_25fps_multiseed":
        dirs["ZeroVel"] = root / "amass_25fps" / "ZeroVel"
        for m in ["HRI", "DMGNN", "PVRED", "AuxFormer"]:
            dirs[m] = root / "amass_25fps_multiseed" / "aggregated" / m
    else:
        for m in SUITE_MODELS[dataset]:
            candidate = root / dataset / m
            if candidate.is_dir():
                dirs[m] = candidate

    if extra_models:
        for m_name, m_path in extra_models.items():
            dirs[m_name] = Path(m_path)

    return dirs


def load_benchmark_suite(
    dataset: str = "amass_25fps",
    metrics: Tuple[str, ...] = ("MPJPE", "JAD", "PSDS", "MAE_XYZ", "MAE_ZXY"),
    benchmark_root: Optional[Union[str, Path]] = None,
    extra_models: Optional[Dict[str, Union[str, Path]]] = None,
    scale_mpjpe_to_mm: bool = True,
) -> Dict[str, Dict[str, dict]]:
    """
    Load multiple metrics for all models in a benchmark dataset suite.

    Args:
        dataset: Benchmark suite key (`"h36m_25fps"`, `"amass_25fps"`, `"amass_60fps"`,
            `"amass2h36m_25fps"`, or `"amass_25fps_multiseed"`).
        metrics: Tuple of metric keys from `METRIC_FILES` to load.
        benchmark_root: Optional custom root directory (defaults to `DEFAULT_BENCHMARK_ROOT`).
        extra_models: Optional `{model_name: csv_folder_path}` mapping to include custom models.
        scale_mpjpe_to_mm: If `True`, scales positional error values from meters to millimeters.

    Returns:
        Nested dictionary `data[model_name][metric_name]`.
    """
    dirs = get_suite_directories(dataset, benchmark_root=benchmark_root, extra_models=extra_models)
    suite_data: Dict[str, Dict[str, dict]] = {}
    for model_name, folder in dirs.items():
        suite_data[model_name] = {}
        for metric in metrics:
            try:
                suite_data[model_name][metric] = load_metric_csv(
                    folder, metric=metric, scale_mpjpe_to_mm=scale_mpjpe_to_mm
                )
            except FileNotFoundError:
                continue
    return suite_data


def compute_suite_hmpe(
    dataset: str = "amass_25fps",
    t_ms: int = 400,
    row_name: Optional[str] = None,
    benchmark_root: Optional[Union[str, Path]] = None,
    extra_models: Optional[Dict[str, Union[str, Path]]] = None,
) -> Dict[str, dict]:
    """
    Compute `HMPE_t` (e.g. `HMPE_400` or `HMPE_1000`), normalized radar values, and
    raw metric values for all models in a benchmark suite.

    Args:
        dataset: `"h36m_25fps"`, `"amass_25fps"`, `"amass_60fps"`, or `"amass2h36m_25fps"`.
        t_ms: Prediction horizon in milliseconds for MPJPE and JAD (`400` or `1000`).
        row_name: Row to evaluate (`"mean"` for H3.6M, `"amass"` for AMASS). Auto-detected if `None`.
        benchmark_root: Path to `benchmark_results`.
        extra_models: Optional dictionary `{model_name: csv_folder_path}` to compare custom models.

    Returns:
        Dictionary `results[model_name]` containing:
        - `"hmpe"`: float `HMPE_t` score in `[0, 100]` (`ZeroVel = 100.0`),
        - `"normalized"`: `(norm_jad, norm_mpjpe, norm_psds)` in `[0, 1]`,
        - `"raw"`: `{"JAD": jad_t, "MPJPE": mpjpe_t_mm, "PSDS": psds_1000}`,
        - `"psds_train_test"`: dataset reference `PSDS(Train, Test)`.
    """
    ref_key = "amass_25fps" if dataset == "amass_25fps_multiseed" else dataset
    psds_tt = get_reference_psds(ref_key, "train", "test")
    if row_name is None:
        row_name = "mean" if "h36m" in dataset else "amass"

    suite = load_benchmark_suite(
        dataset=dataset,
        metrics=("JAD", "MPJPE", "PSDS"),
        benchmark_root=benchmark_root,
        extra_models=extra_models,
        scale_mpjpe_to_mm=True,
    )

    def _get_at(model_dict: dict, metric: str, target_ms: int) -> float:
        m_data = model_dict[metric]
        ts = m_data["timestamps_ms"]
        idx = int(np.where(ts == target_ms)[0][0])
        return float(m_data["rows"][row_name][idx])

    zv_jad = _get_at(suite["ZeroVel"], "JAD", t_ms)
    zv_mpjpe = _get_at(suite["ZeroVel"], "MPJPE", t_ms)

    results: Dict[str, dict] = {}
    for model_name, model_dict in suite.items():
        jad_t = _get_at(model_dict, "JAD", t_ms)
        mpjpe_t = _get_at(model_dict, "MPJPE", t_ms)
        psds_1000 = _get_at(model_dict, "PSDS", 1000)
        hmpe_val, norm_axes = compute_hmpe_score(
            mpjpe_t=mpjpe_t,
            jad_t=jad_t,
            psds_1000=psds_1000,
            zero_vel_mpjpe_t=zv_mpjpe,
            zero_vel_jad_t=zv_jad,
            psds_train_test=psds_tt,
        )
        results[model_name] = {
            "hmpe": hmpe_val,
            "normalized": norm_axes,
            "raw": {"JAD": jad_t, "MPJPE": mpjpe_t, "PSDS": psds_1000},
            "psds_train_test": psds_tt,
        }
    return results


def compute_multiseed_hmpe(
    t_ms: int = 400,
    benchmark_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute the 12-seed mean and sample standard deviation (`ddof=1`) of `HMPE_t`
    on AMASS (25 fps) as reported in Appendix A (Table 9) of the paper.

    Returns:
        Dictionary `results[model_name] = {"mean": float, "std": float, "seeds": List[float]}`.
    """
    root = Path(benchmark_root) if benchmark_root is not None else DEFAULT_BENCHMARK_ROOT
    zv_dir = root / "amass_25fps" / "ZeroVel"
    psds_tt = get_reference_psds("amass_25fps", "train", "test")

    zv_jad_data = load_metric_csv(zv_dir, "JAD")
    zv_mpjpe_data = load_metric_csv(zv_dir, "MPJPE", scale_mpjpe_to_mm=True)
    t_idx = int(np.where(zv_jad_data["timestamps_ms"] == t_ms)[0][0])
    zv_jad = float(zv_jad_data["rows"]["amass"][t_idx])
    zv_mpjpe = float(zv_mpjpe_data["rows"]["amass"][t_idx])

    results: Dict[str, Dict[str, float]] = {
        "ZeroVel": {"mean": 100.0, "std": 0.0, "seeds": [100.0]}
    }

    seeds_root = root / "amass_25fps_multiseed" / "seeds"
    for model in ["HRI", "DMGNN", "PVRED", "AuxFormer"]:
        seed_dirs = sorted((seeds_root / model).glob("seed_*"))
        hmpe_vals: List[float] = []
        for s_dir in seed_dirs:
            jad_d = load_metric_csv(s_dir, "JAD")
            mpjpe_d = load_metric_csv(s_dir, "MPJPE", scale_mpjpe_to_mm=True)
            psds_d = load_metric_csv(s_dir, "PSDS")
            jad_t = float(jad_d["rows"]["amass"][t_idx])
            mpjpe_t = float(mpjpe_d["rows"]["amass"][t_idx])
            psds_1000 = float(psds_d["rows"]["amass"][0])
            hmpe_val, _ = compute_hmpe_score(
                mpjpe_t=mpjpe_t,
                jad_t=jad_t,
                psds_1000=psds_1000,
                zero_vel_mpjpe_t=zv_mpjpe,
                zero_vel_jad_t=zv_jad,
                psds_train_test=psds_tt,
            )
            hmpe_vals.append(hmpe_val)

        arr = np.array(hmpe_vals, dtype=float)
        results[model] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "seeds": hmpe_vals,
        }

    return results


def aggregate_multiseed_dirs(
    seed_dirs: List[Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Tuple[List[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
    """
    Aggregate a list of multi-seed evaluation directories by computing the element-wise
    mean and sample standard deviation (`ddof=1`) across all seeds for each CSV file.

    Args:
        seed_dirs: List of directories (one per random seed) containing evaluation CSVs.
        output_dir: Optional directory where the aggregated CSVs (with mean rows and `'std'`
            rows) will be written.

    Returns:
        Dictionary mapping each CSV filename to `(header, mean_rows_dict, std_rows_dict)`.
    """
    dirs = [Path(d) for d in seed_dirs]
    if not dirs:
        raise ValueError("seed_dirs cannot be empty.")

    csv_names = sorted({f.name for d in dirs for f in d.glob("*.csv")})
    aggregated = {}

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = None

    for file_name in csv_names:
        header: Optional[List[str]] = None
        rows_by_key: Dict[str, List[np.ndarray]] = {}
        for d in dirs:
            f_path = d / file_name
            if not f_path.exists():
                continue
            with open(f_path, newline="") as fp:
                rows = list(csv.reader(fp))
            if header is None:
                header = rows[0]
            for r in rows[1:]:
                if not r or not r[0].strip():
                    continue
                key = r[0].strip()
                vals = np.array([float(x) for x in r[1:]], dtype=float)
                rows_by_key.setdefault(key, []).append(vals)

        assert header is not None
        mean_dict: Dict[str, np.ndarray] = {}
        std_dict: Dict[str, np.ndarray] = {}
        for key, arr_list in rows_by_key.items():
            stacked = np.vstack(arr_list)
            mean_dict[key] = np.mean(stacked, axis=0)
            std_dict[key] = np.std(stacked, axis=0, ddof=1)

        aggregated[file_name] = (header, mean_dict, std_dict)

        if out_path is not None:
            with open(out_path / file_name, "w", newline="") as fp:
                writer = csv.writer(fp, lineterminator="\n")
                writer.writerow(header)
                for key, m_vals in mean_dict.items():
                    writer.writerow([key] + [repr(float(x)) for x in m_vals])
                for key, s_vals in std_dict.items():
                    std_label = "std" if len(std_dict) == 1 else f"{key}_std"
                    writer.writerow([std_label] + [repr(float(x)) for x in s_vals])

    return aggregated
