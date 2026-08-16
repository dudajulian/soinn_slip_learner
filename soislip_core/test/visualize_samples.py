#!/usr/bin/env python3

"""Visualize sample slip values as a 2D scatter plot."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


BOUNDARY_DISTANCE = 0.125
REGIONS: tuple[tuple[str, float, float, float, float, float], ...] = (
    # Format: (color_name, alpha, x_min, x_max, y_min, y_max)
    ("yellow", 0.2, -8.0, 8.0, 4.9, 15.7),
    ("red", 0.2, 10.7, 15.7, -5.0, 5.0),
    ("red", 0.15, 7.0, 10.7, -5.0, 5.0),
    ("red", 0.1, 4.175, 7.0, -5.0, 5.0),
    ("green", 0.2, -5.0, 5.0, -15.7, -10.7),
    ("green", 0.15, -5.0, 5.0, -10.7, -7.0),
    ("green", 0.1, -5.0, 5.0, -7.0, -4.175),
    ("blue", 0.2, -15.7, -10.7, -5.0, 5.0),
    ("blue", 0.15, -10.7, -7.0, -5.0, 5.0),
    ("blue", 0.1, -7.0, -4.175, -5.0, 5.0),
    ("grey", 0.2, -4.175, 4.175, -4.175, 4.9),
)

def _parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[4]
    # default_csv = workspace_root / "resources/results/20260814_125520_sim_train_samples.csv"
    default_csv = workspace_root / "resources/results/20260814_135840_sim_test_samples.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Scatter plot of sample positions (x, y) where point color represents "
            "slip in the range [0, 1]."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_csv,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path (e.g. plot.png). If omitted, shows interactive plot.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="rainbow",
        help="Matplotlib colormap name.",
    )
    return parser.parse_args()


def _read_samples(csv_path: Path) -> tuple[list[float], list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    slip_values: list[float] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
                slip = float(row["slip"])
            except (KeyError, TypeError, ValueError):
                # Skip malformed rows instead of failing the entire plot.
                continue

            # Ensure color scaling is strictly in the [0, 1] range.
            slip = max(0.0, min(1.0, slip))

            x_values.append(x)
            y_values.append(y)
            slip_values.append(slip)

    return x_values, y_values, slip_values


def main() -> None:
    args = _parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"CSV file not found: {args.input}")

    x_values, y_values, slip_values = _read_samples(args.input)
    if not x_values:
        raise ValueError("No valid samples were found in the CSV file.")

    plt.figure(figsize=(10, 8))
    for color_name, alpha, x_min, x_max, y_min, y_max in REGIONS:
        rect = Rectangle(
            (x_min + BOUNDARY_DISTANCE, y_min + BOUNDARY_DISTANCE),
            x_max - x_min - 2 * BOUNDARY_DISTANCE,
            y_max - y_min - 2 * BOUNDARY_DISTANCE,
            facecolor=color_name,
            edgecolor=color_name,
            alpha=alpha,
        )
        plt.gca().add_patch(rect)

    scatter = plt.scatter(
        x_values,
        y_values,
        c=slip_values,
        cmap=args.cmap,
        vmin=0.0,
        vmax=1.0,
        s=10,
        alpha=0.7,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, shrink=0.5)
    cbar.set_label("slip")

    # plt.title("Slip Samples in XY Space")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-15, 15)
    plt.ylim(-15, 5)

    
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    x_ticks = list(range(-15, 16, 5))
    y_ticks = list(range(-15, 6, 5))
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=300)
        print(f"Saved plot to: {args.output}")
        return

    plt.show()


if __name__ == "__main__":
    main()
