#!/usr/bin/env python3
"""
Regenerate the aggregated multi-seed CSVs (`mean` and sample `std`, `ddof=1`)
in `benchmark_results/amass_25fps_multiseed/aggregated/` directly from the
12 raw random-seed evaluation directories in `benchmark_results/amass_25fps_multiseed/seeds/`.

The 12 evaluation seeds used in Appendix A (Tables 7-9 and Fig. 12) were
generated deterministically with:

    rng = np.random.RandomState(42)
    seeds = rng.randint(0, 2**31, size=12)

Usage:
    python scripts/regenerate_multiseed_aggregates.py
"""

from hmp_evaluation import DEFAULT_BENCHMARK_ROOT, aggregate_multiseed_dirs


def main() -> None:
    multiseed_root = DEFAULT_BENCHMARK_ROOT / "amass_25fps_multiseed"
    seeds_root = multiseed_root / "seeds"
    agg_root = multiseed_root / "aggregated"

    for model in ["HRI", "DMGNN", "PVRED", "AuxFormer"]:
        seed_dirs = sorted((seeds_root / model).glob("seed_*"))
        if not seed_dirs:
            raise FileNotFoundError(f"No seed directories found in {seeds_root / model}")
        out_dir = agg_root / model
        aggregated = aggregate_multiseed_dirs(seed_dirs, output_dir=out_dir)
        print(f"Aggregated {len(seed_dirs)} seeds ({len(aggregated)} CSVs) -> {out_dir}")


if __name__ == "__main__":
    main()
