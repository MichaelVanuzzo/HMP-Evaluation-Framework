"""
Plotting utilities reproducing all 12 figures of the paper 1-to-1:
- Fig. 1: Literature survey of HMP metrics per year (2021-2025)
- Fig. 2: Literature survey of HMP datasets per year (2021-2025)
- Fig. 3: Average MAE XYZ and MAE ZXY on H3.6M (25 fps)
- Fig. 4: Average MPJPE [mm] and JAD [rad] on H3.6M (25 fps)
- Fig. 5: PSDS up to 30 s (log scale) on H3.6M (25 fps)
- Fig. 6: Average MPJPE [mm] and JAD [rad] on AMASS (25 fps)
- Fig. 7: PSDS up to 30 s (log scale) on AMASS (25 fps)
- Fig. 8: HMPE_400 and HMPE_1000 Radar Charts on AMASS (25 fps)
- Fig. 9: Average MPJPE [mm] and JAD [rad] on AMASS (60 fps)
- Fig. 10: PSDS up to 30 s (log scale) on AMASS (60 fps)
- Fig. 11: HMPE_400 and HMPE_1000 Radar Charts on AMASS (60 fps)
- Fig. 12: Multi-seed (12 runs) MPJPE and JAD on AMASS (25 fps) with +-1 std bands
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np

from .evaluator import get_reference_psds
from .results_loader import compute_suite_hmpe, load_benchmark_suite
from .tables import MODEL_DISPLAY_NAMES

MODEL_COLORS: Dict[str, str] = {
    "HRI": "blue",
    "DMGNN": "red",
    "PVRED": "green",
    "AuxFormer": "orange",
    "ZeroVel": "black",
}

MODEL_STYLES: Dict[str, str] = {
    "HRI": "-",
    "DMGNN": "-",
    "PVRED": "-",
    "AuxFormer": "-",
    "ZeroVel": "--",
}

PLOT_MODEL_ORDER = ["HRI", "DMGNN", "PVRED", "AuxFormer", "ZeroVel"]
EXTRA_MODEL_COLORS = ["brown", "cyan", "olive", "pink", "gray"]

SURVEY_YEARS = [2021, 2022, 2023, 2024, 2025]
SURVEY_TOTAL_PAPERS = [10, 22, 29, 37, 35]

SURVEY_METRICS_BY_YEAR: Dict[str, list] = {
    "MAE": [8, 13, 8, 8, 4],
    "MPJPE": [7, 11, 23, 32, 27],
    "PCK": [2, 0, 0, 0, 1],
    "JAD": [1, 0, 0, 0, 1],
    "Others": [1, 3, 3, 5, 9],
}

SURVEY_DATASETS_BY_YEAR: Dict[str, list] = {
    "H36M": [10, 20, 27, 30, 33],
    "CMU": [7, 9, 17, 21, 19],
    "3DPW": [3, 5, 11, 18, 16],
    "AMASS": [2, 0, 5, 9, 12],
    "Other": [2, 4, 4, 9, 8],
}


def _ordered_models(suite: dict) -> list:
    ordered = [m for m in PLOT_MODEL_ORDER if m in suite]
    for m in suite:
        if m not in ordered:
            ordered.append(m)
    return ordered


def _model_color(model_name: str, ordered_models: list) -> str:
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    custom_models = [m for m in ordered_models if m not in MODEL_COLORS]
    idx = custom_models.index(model_name)
    return EXTRA_MODEL_COLORS[idx % len(EXTRA_MODEL_COLORS)]


def _plot_yearly_survey_bars(
    years: list,
    total_papers: list,
    series: Dict[str, list],
) -> plt.Figure:
    """
    Render a seaborn `whitegrid`-style grouped yearly survey bar chart with
    `Total Papers` backdrop, white bar edges, and 4-sided light-gray plot border
    matching Figures 1 and 2 of the paper.
    """
    # Exact 5-color seaborn HLS palette used in the paper
    hls_palette = [
        (0.86, 0.3712, 0.33999999999999997),
        (0.7664, 0.86, 0.33999999999999997),
        (0.33999999999999997, 0.86, 0.5472),
        (0.33999999999999997, 0.5472, 0.86),
        (0.7664, 0.33999999999999997, 0.86),
    ]

    x = np.arange(len(years), dtype=float)
    n_groups = len(series)
    total_bar_width = 0.8
    bar_width = total_bar_width / n_groups

    fig, ax = plt.subplots(figsize=(15.0, 8.0), facecolor="white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", color="#cccccc", linewidth=0.8)
    ax.grid(axis="x", visible=False)

    # Background bars for Total Papers (white edge matching seaborn barplot)
    bg_bars = ax.bar(
        x,
        [p + 0.05 for p in total_papers],
        width=total_bar_width,
        color="gray",
        alpha=0.3,
        edgecolor="white",
        linewidth=1.0,
        label="Total Papers",
    )
    for rect, val in zip(bg_bars, total_papers):
        ax.annotate(
            f"{val:d}",
            (rect.get_x() + rect.get_width() / 2.0, rect.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=10,
            color="#222222",
        )

    # Grouped foreground bars for each category (white edge matching seaborn barplot)
    for idx, (label, counts) in enumerate(series.items()):
        offset = -total_bar_width / 2.0 + (idx + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            counts,
            width=bar_width,
            color=hls_palette[idx % len(hls_palette)],
            edgecolor="white",
            linewidth=1.0,
            label=label,
        )
        for rect, val in zip(bars, counts):
            ax.annotate(
                f"{val:d}",
                (rect.get_x() + rect.get_width() / 2.0, rect.get_height()),
                ha="center",
                va="center",
                xytext=(0, 9),
                textcoords="offset points",
                fontsize=10,
                color="#222222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=11, color="#222222")
    ax.set_xlabel("Year", fontsize=14, color="#222222")
    ax.set_ylabel("Number of Papers", fontsize=14, color="#222222")
    ax.set_ylim(0, 39.5)
    ax.set_xlim(-0.5, len(years) - 0.5)
    ax.tick_params(axis="both", length=0, colors="#222222", labelsize=11)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.8)

    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_metrics_survey_2021_2025(
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Reproduce Fig. 1 of the paper: Metrics used in HMP studies (2021-2025)."""
    fig = _plot_yearly_survey_bars(
        years=SURVEY_YEARS,
        total_papers=SURVEY_TOTAL_PAPERS,
        series=SURVEY_METRICS_BY_YEAR,
    )
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig


def plot_datasets_survey_2021_2025(
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Reproduce Fig. 2 of the paper: Datasets used in HMP studies (2021-2025)."""
    fig = _plot_yearly_survey_bars(
        years=SURVEY_YEARS,
        total_papers=SURVEY_TOTAL_PAPERS,
        series=SURVEY_DATASETS_BY_YEAR,
    )
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig


def plot_literature_survey_2021_2025(
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[Tuple[plt.Figure, plt.Figure]]:
    """
    Reproduce Fig. 1 (Metrics used in HMP studies, 2021-2025) and
    Fig. 2 (Datasets used in HMP studies, 2021-2025) of the paper.
    """
    out = Path(save_dir) if save_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    fig_metrics = plot_metrics_survey_2021_2025(
        save_path=(out / "fig1_metrics_survey_2021_2025.svg") if out is not None else None,
        show=show,
    )
    fig_datasets = plot_datasets_survey_2021_2025(
        save_path=(out / "fig2_datasets_survey_2021_2025.svg") if out is not None else None,
        show=show,
    )
    if show:
        return None
    assert fig_metrics is not None and fig_datasets is not None
    return fig_metrics, fig_datasets


def plot_mae_xyz_vs_zxy(
    dataset: str = "h36m_25fps",
    action: Optional[str] = None,
    max_ms: int = 1000,
    benchmark_root: Optional[Union[str, Path]] = None,
    extra_models: Optional[Dict[str, Union[str, Path]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot `MAE XYZ` vs `MAE ZXY` [rad] up to `max_ms` with matched Y-axis limits
    (reproducing Fig. 3 of the paper for `dataset="h36m_25fps"`).

    Args:
        dataset: Benchmark suite key (`"h36m_25fps"`, `"amass_25fps"`, etc.).
        action: Row name (`"mean"` for H3.6M, `"amass"` for AMASS). Auto-detected if `None`.
        max_ms: Maximum prediction horizon in milliseconds.
        benchmark_root: Optional custom root directory (defaults to `<repo>/benchmark_results`).
        extra_models: Optional `{model_name: csv_dir}` mapping to overlay custom models.
        save_path: Optional file path to save the figure (e.g. `.svg` or `.png`).
        show: If `True`, displays the figure via `plt.show()` and returns `None`;
            if `False`, returns the Matplotlib `Figure`.
    """
    if action is None:
        action = "mean" if "h36m" in dataset else "amass"

    suite = load_benchmark_suite(
        dataset=dataset,
        metrics=("MAE_XYZ", "MAE_ZXY"),
        benchmark_root=benchmark_root,
        extra_models=extra_models,
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 4.0))
    titles = [("MAE_XYZ", "MAE XYZ"), ("MAE_ZXY", "MAE ZXY")]

    ordered = _ordered_models(suite)
    y_max = 0.0
    for ax_idx, (m_key, title) in enumerate(titles):
        ax = axes[ax_idx]
        ax.set_title(title)
        for model_name in ordered:
            if m_key not in suite[model_name]:
                continue
            m_data = suite[model_name][m_key]
            ts = m_data["timestamps_ms"]
            mask = ts <= max_ms
            x = ts[mask]
            y = m_data["rows"][action][mask]
            y_max = max(y_max, float(np.max(y)))
            color = _model_color(model_name, ordered)
            ls = MODEL_STYLES.get(model_name, "-")
            label = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            ax.plot(x, y, color=color, linestyle=ls, label=label)
        ax.grid(True)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("[rad]")
        ax.set_xticks([0, 250, 500, 750, 1000])

    # Match Y-axis scale across both subplots as in Fig. 3
    for ax in axes:
        ax.set_ylim(0.0, y_max * 1.06)

    axes[1].legend(loc="lower right")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig


def plot_mpjpe_and_jad(
    dataset: str = "amass_25fps",
    action: Optional[str] = None,
    max_ms: int = 1000,
    benchmark_root: Optional[Union[str, Path]] = None,
    extra_models: Optional[Dict[str, Union[str, Path]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot `MPJPE` [mm] and `JAD` [rad] over time up to `max_ms`, reproducing:
    - Fig. 4 (`dataset="h36m_25fps"`)
    - Fig. 6 (`dataset="amass_25fps"`)
    - Fig. 9 (`dataset="amass_60fps"`)
    - Fig. 12 (`dataset="amass_25fps_multiseed"`, with `±1 std` shaded bands)

    Args:
        dataset: Benchmark suite key.
        action: Row name (`"mean"` for H3.6M, `"amass"` for AMASS). Auto-detected if `None`.
        max_ms: Maximum prediction horizon in milliseconds.
        benchmark_root: Optional custom root directory (defaults to `<repo>/benchmark_results`).
        extra_models: Optional `{model_name: csv_dir}` mapping to overlay custom models.
        save_path: Optional file path to save the figure (e.g. `.svg` or `.png`).
        show: If `True`, displays the figure via `plt.show()` and returns `None`;
            if `False`, returns the Matplotlib `Figure`.
    """
    if action is None:
        action = "mean" if "h36m" in dataset else "amass"

    suite = load_benchmark_suite(
        dataset=dataset,
        metrics=("MPJPE", "JAD"),
        benchmark_root=benchmark_root,
        extra_models=extra_models,
        scale_mpjpe_to_mm=True,
    )

    is_multiseed = "multiseed" in dataset
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 4.0))
    panels = [("MPJPE", "MPJPE", "[mm]"), ("JAD", "JAD", "[rad]")]

    ordered = _ordered_models(suite)
    for idx, (m_key, title, ylabel) in enumerate(panels):
        ax = axes[idx]
        ax.set_title(title)
        for model_name in ordered:
            if m_key not in suite[model_name]:
                continue
            m_data = suite[model_name][m_key]
            ts = m_data["timestamps_ms"]
            mask = ts <= max_ms
            x = ts[mask]
            y = m_data["rows"][action][mask]
            stds = m_data["rows"].get("std")
            color = _model_color(model_name, ordered)
            ls = MODEL_STYLES.get(model_name, "-")
            label = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            lw = 0.8 if is_multiseed else 1.5

            ax.plot(x, y, color=color, linestyle=ls, linewidth=lw, label=label)
            if stds is not None:
                s = stds[mask]
                ax.fill_between(
                    x,
                    y - s,
                    y + s,
                    color=color,
                    alpha=0.35,
                    edgecolor="none",
                    linewidth=0,
                )

        ax.grid(True)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 250, 500, 750, 1000])

    axes[1].legend(loc="lower right")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig


def plot_psds(
    dataset: str = "amass_25fps",
    action: Optional[str] = None,
    exclude_zero_vel: bool = False,
    show_trend: bool = False,
    benchmark_root: Optional[Union[str, Path]] = None,
    extra_models: Optional[Dict[str, Union[str, Path]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot `PSDS` over prediction horizons up to 30 s on a logarithmic Y-axis
    together with the `Train-Test` reference line, reproducing:
    - Fig. 5 (`dataset="h36m_25fps"`)
    - Fig. 7 (`dataset="amass_25fps"`)
    - Fig. 10 (`dataset="amass_60fps"`)

    Args:
        dataset: Benchmark suite key.
        action: Row name (`"mean"` for H3.6M, `"amass"` for AMASS). Auto-detected if `None`.
        exclude_zero_vel: If `True`, omits the `ZeroVel` curve.
        show_trend: If `True`, overlays 3rd-degree polynomial trend curves.
        benchmark_root: Optional custom root directory (defaults to `<repo>/benchmark_results`).
        extra_models: Optional `{model_name: csv_dir}` mapping to overlay custom models.
        save_path: Optional file path to save the figure (e.g. `.svg` or `.png`).
        show: If `True`, displays the figure via `plt.show()` and returns `None`;
            if `False`, returns the Matplotlib `Figure`.
    """
    if action is None:
        action = "mean" if "h36m" in dataset else "amass"

    ref_key = "amass_25fps" if dataset == "amass_25fps_multiseed" else dataset
    psds_tt = get_reference_psds(ref_key, "train", "test")
    is_60fps = "60fps" in dataset
    step = 24 if is_60fps else 10

    suite = load_benchmark_suite(
        dataset=dataset,
        metrics=("PSDS",),
        benchmark_root=benchmark_root,
        extra_models=extra_models,
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.0))
    ax.set_title("PSDS")

    ordered = _ordered_models(suite)
    for model_name in ordered:
        if exclude_zero_vel and model_name == "ZeroVel":
            continue
        if "PSDS" not in suite[model_name]:
            continue
        m_data = suite[model_name]["PSDS"]
        x_sec = m_data["timestamps_ms"] / 1000.0
        means = m_data["rows"][action]
        stds = m_data["rows"].get("std")
        color = _model_color(model_name, ordered)
        ls = MODEL_STYLES.get(model_name, "-")
        label = MODEL_DISPLAY_NAMES.get(model_name, model_name)

        ax.semilogy(x_sec[::step], means[::step], color=color, linestyle=ls, label=label)
        if stds is not None:
            lower = np.maximum(means - stds, 1e-4)
            upper = means + stds
            ax.fill_between(
                x_sec[::step],
                lower[::step],
                upper[::step],
                color=color,
                alpha=0.25,
                edgecolor="none",
            )

        if show_trend and len(means) >= 4:
            idx_range = np.arange(len(means))
            trend_poly = np.poly1d(np.polyfit(idx_range, means, 3))
            ax.semilogy(x_sec[::step], np.maximum(trend_poly(idx_range)[::step], 1e-4), "--", color="gray", alpha=0.7)

    ax.axhline(psds_tt, color="m", linestyle="-", label="Train-Test")

    if is_60fps:
        yticks = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
        yticklabels = ["0.01", "0.03", "0.1", "0.3", "1", "3", "10", "30"]
        ax.set_ylim([0.01, 35])
    else:
        yticks = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
        yticklabels = ["0.01", "0.03", "0.1", "0.3", "1", "3", "10"]
        ax.set_ylim([0.01, 15])

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim([0, 30])
    ax.set_xlabel("Time [s]")
    ax.grid(True)

    if dataset == "h36m_25fps":
        ax.legend(loc="lower right", frameon=True)
    else:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            frameon=True,
        )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig


def plot_hmpe_radar(
    dataset: str = "amass_25fps",
    timestamps_ms: Tuple[int, ...] = (400, 1000),
    action: Optional[str] = None,
    benchmark_root: Optional[Union[str, Path]] = None,
    extra_models: Optional[Dict[str, Union[str, Path]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot side-by-side polar radar charts of `HMPE_t` (`HMPE_400` and `HMPE_1000`)
    with normalized `[JAD, MPJPE, PSDS]` axes and the magenta `PSDS(Train, Test)`
    marker at normalized radius `0.1`, reproducing:
    - Fig. 8 (`dataset="amass_25fps"`)
    - Fig. 11 (`dataset="amass_60fps"`)

    Args:
        dataset: Benchmark suite key (`"amass_25fps"` or `"amass_60fps"`).
        timestamps_ms: Prediction horizons in ms to render side by side.
        action: Row name (`"mean"` for H3.6M, `"amass"` for AMASS). Auto-detected if `None`.
        benchmark_root: Optional custom root directory (defaults to `<repo>/benchmark_results`).
        extra_models: Optional `{model_name: csv_dir}` mapping to overlay custom models.
        save_path: Optional file path to save the figure (e.g. `.svg` or `.png`).
        show: If `True`, displays the figure via `plt.show()` and returns `None`;
            if `False`, returns the Matplotlib `Figure`.
    """
    categories = ["JAD", "MPJPE", "PSDS"]
    angles = [i * 2.0 * np.pi / len(categories) for i in range(len(categories))]
    angles += angles[:1]

    fig, axs = plt.subplots(
        1,
        len(timestamps_ms),
        figsize=(3.0 * len(timestamps_ms), 4.0),
        subplot_kw=dict(polar=True),
    )
    if len(timestamps_ms) == 1:
        axs = [axs]

    n_models = 0
    for panel_idx, t_ms in enumerate(timestamps_ms):
        ax = axs[panel_idx]
        hmpe_suite = compute_suite_hmpe(
            dataset=dataset,
            t_ms=t_ms,
            row_name=action,
            benchmark_root=benchmark_root,
            extra_models=extra_models,
        )

        ordered = _ordered_models(hmpe_suite)
        n_models = max(n_models, len(ordered))

        for model_name in ordered:
            vals = list(hmpe_suite[model_name]["normalized"])
            closed_vals = vals + [vals[0]]
            color = _model_color(model_name, ordered)
            label = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            ax.fill(angles, closed_vals, color=color, alpha=0.25)
            if panel_idx == 0:
                ax.plot(angles, closed_vals, color=color, label=label)
            else:
                ax.plot(angles, closed_vals, color=color)

        # Reference PSDS(Train, Test) dot at normalized radius 0.1 on the PSDS axis
        ax.plot(angles[2], 0.1, "mo")

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_theta_direction(1)
        ax.set_theta_offset(-np.pi * 3.0 / 2.0 - (1.0 / 3.0) * np.pi)
        ax.set_ylim(0, 1)
        ax.set_title(rf"$\mathrm{{HMPE}}_{{{t_ms}}}$", pad=8)

    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=max(1, n_models),
        frameon=True,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig
