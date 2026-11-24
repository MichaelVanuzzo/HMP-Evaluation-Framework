# HMP-Evaluation-Framework

Official repository for the paper **"Evaluating human motion prediction: a framework for accuracy and realism"**, published in *Neural Computing and Applications* (Springer, 2026).

[![Paper DOI](https://img.shields.io/badge/DOI-10.1007%2Fs00521--026--12517--x-blue)](https://doi.org/10.1007/s00521-026-12517-x)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## Overview

Evaluating 3D Human Motion Prediction (HMP) requires assessing both **pose accuracy** (in Cartesian and angular spaces) and **motion realism** (smoothness and frequency-domain consistency over long horizons), while avoiding dataset-specific and skeleton-specific biases.

This framework builds on [`hmp_utils`](https://github.com/MichaelVanuzzo/hmp_utils) to provide a standardized, body-model-independent evaluation protocol based on four core metrics:

1. **Normalized $\text{MPJPE}_t$ (Mean Per Joint Position Error, $[\text{mm}]$)**: Instantaneous Euclidean joint position error at time $t$ computed on a **normalized skeleton** where the reference left femur bone is scaled to $1\text{ m}$ (`JOINT_TO_NORMALIZE`), ensuring fair cross-dataset comparisons across Human3.6M and AMASS (SMPL/SMPL+H/SMPL-X).
2. **$\text{JAD}_t$ (Joint Angle Difference, $[\text{rad}]$)**: Geodesic angular distance on $SO(3)$ between predicted and ground-truth local joint orientations across the major body joints, avoiding Euler-angle gimbal lock and parameterization ambiguities (`XYZ` vs `ZXY`).
3. **$\text{PSDS}$ (Power Spectral Density Shift)**: Mean Wasserstein distance between the normalized Power Spectral Density (PSD) of predicted motion sequences and the reference training-set PSD over sliding $1\text{ s}$ windows, compared against the intrinsic dataset baseline $\text{PSDS}(\text{Train}, \text{Test})$.
4. **$\text{HMPE}_t$ (Human Motion Prediction Error)**: Unified scalar score in $[0, 100]$ (with $\text{ZeroVel} = 100.00$) representing the normalized area of the radar-chart triangle formed by $\widehat{\text{JAD}}_t$, $\widehat{\text{MPJPE}}_t$ (normalized by the Zero-Velocity baseline $\text{ZeroVel}$), and $\widehat{\text{PSDS}}_{1000}$ (normalized by $10 \times \text{PSDS}(\text{Train}, \text{Test})$):

$$\text{HMPE}_t = 100 \cdot \frac{\widehat{\text{MPJPE}}_t \cdot \widehat{\text{JAD}}_t + \widehat{\text{JAD}}_t \cdot \widehat{\text{PSDS}}_{1000} + \widehat{\text{PSDS}}_{1000} \cdot \widehat{\text{MPJPE}}_t}{3}$$

## Installation

Clone the repository and install the package (which automatically installs [`hmp_utils`](https://github.com/MichaelVanuzzo/hmp_utils) and all required dependencies). Using a dedicated Python virtual environment (`venv` or `conda`) is recommended:

```bash
git clone https://github.com/MichaelVanuzzo/HMP-Evaluation-Framework.git
cd HMP-Evaluation-Framework

# Optional (recommended): create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

pip install -e .

# Optional: register the environment as a Jupyter kernel to run the notebooks
pip install ipykernel
python -m ipykernel install --user --name hmp-eval --display-name "Python (HMP-Evaluation)"
```

## Repository Structure

```text
HMP-Evaluation-Framework/
├── hmp_evaluation/
│   ├── evaluator.py                         # HMPEvaluator protocol wrapper (MPJPE, JAD, PSDS, HMPE)
│   ├── results_loader.py                    # Loader & multi-seed aggregator for benchmark CSV logs
│   ├── tables.py                            # Generators for Tables 1-9 (Markdown, LaTeX, CSV)
│   └── plots.py                             # Generators for Figures 1-12 (including HMPE Radar Charts)
├── benchmark_results/
│   ├── h36m_25fps/                          # H3.6M (25 fps): ZeroVel, HRI, DMGNN, PVRED
│   ├── amass_25fps/                         # AMASS (25 fps): ZeroVel, HRI, DMGNN, PVRED, AuxFormer
│   ├── amass_60fps/                         # AMASS (60 fps): ZeroVel, HRI, DMGNN, PVRED, AuxFormer
│   ├── amass2h36m_25fps/                    # AMASS converted to H3.6M (25 fps)
│   └── amass_25fps_multiseed/               # 12-seed evaluation on AMASS (25 fps): raw seeds + aggregated mean/std
├── scripts/
│   └── regenerate_multiseed_aggregates.py   # Recomputes aggregated/ mean & std CSVs from the 12 raw seeds
└── notebooks/
    ├── 1_evaluate_new_model_demo.ipynb      # Step-by-step guide to evaluating a new HMP model
    ├── 2_reproduce_paper_tables.ipynb       # Reproduces Tables 1-9 of the paper (1-to-1)
    └── 3_reproduce_paper_figures.ipynb      # Reproduces Figures 1-12 of the paper (1-to-1)
```

## Quick Start

### 1. Evaluate a New HMP Model

```python
from hmp_evaluation import HMPEvaluator

# Initialize evaluator (automatically normalizes the left femur to 1 m)
evaluator = HMPEvaluator(dataset="amass", framerate=25)

# Evaluate predicted and ground-truth sequences in angle-axis format: (batch, seq_len, n_joints, 3)
evaluator.evaluate_action(predictions=pred_aa, targets=gt_aa, action="amass")
evaluator.save_results("./my_model_amass25_eval")
```

### 2. Compare Against the Paper's Benchmark Models & Plot Radar Charts

```python
from hmp_evaluation import format_metric_table, plot_hmpe_radar

# Print AMASS (25 fps) MPJPE table including your custom model
print(format_metric_table(
    dataset="amass_25fps",
    metric="MPJPE",
    extra_models={"MyModel": "./my_model_amass25_eval"},
))

# Plot HMPE_400 and HMPE_1000 Radar Charts overlaying your model
plot_hmpe_radar(
    dataset="amass_25fps",
    extra_models={"MyModel": "./my_model_amass25_eval"},
    save_path="hmpe_radar_comparison.svg",
)
```

## Citation

If you use this evaluation framework, the benchmark results, or `hmp_utils` in your research, please cite:

```bibtex
@article{vanuzzo2026evaluating,
  title   = {Evaluating human motion prediction: a framework for accuracy and realism},
  author  = {Vanuzzo, Michael and Casarin, Marco and Guidolin, Mattia and Michieletto, Stefano and Reggiani, Monica},
  journal = {Neural Computing and Applications},
  volume  = {38},
  number  = {902},
  year    = {2026},
  publisher = {Springer},
  doi     = {10.1007/s00521-026-12517-x}
}
```

## Contact

For questions or inquiries, please contact Michael Vanuzzo at [michael.vanuzzo@unipd.it](mailto:michael.vanuzzo@unipd.it) or [michaelvanuzzo@gmail.com](mailto:michaelvanuzzo@gmail.com).

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE) for details.
