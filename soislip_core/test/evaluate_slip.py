#!/usr/bin/env python3

"""Evaluate slip statistics and SOINN+ predictions for the simulation sample CSV."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
SOISLIP_CORE_DIR = SCRIPT_DIR.parent
if str(SOISLIP_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(SOISLIP_CORE_DIR))

import data_loader as dl
from soinn_py import SoinnPlus


TEST_CSV = Path("/workspaces/vscode_ros2_workspace/resources/results/20260814_135840_sim_test_samples.csv")
TRAIN_CSV = Path("/workspaces/vscode_ros2_workspace/resources/results/20260814_125520_sim_train_samples.csv")
FEATURE_NAMES = ("a", "b", "slope")
BOUNDARY_DISTANCE = 0.125
PLOT_OUTPUT = Path("/workspaces/vscode_ros2_workspace/resources/results/slip_slope_relationships.png")

REGIONS: tuple[tuple[str, float, float, float, float], ...] = (
    ("yellow", -8.0, 8.0, 4.9, 15.7),
    ("red20", 10.7, 15.7, -5.0, 5.0),
    ("red15", 7.0, 10.7, -5.0, 5.0),
    ("red10", 4.175, 7.0, -5.0, 5.0),
    ("green20", -5.0, 5.0, -15.7, -10.7),
    ("green15", -5.0, 5.0, -10.7, -7.0),
    ("green10", -5.0, 5.0, -7.0, -4.175),
    ("blue20", -15.7, -10.7, -5.0, 5.0),
    ("blue15", -10.7, -7.0, -5.0, 5.0),
    ("blue10", -7.0, -4.175, -5.0, 5.0),
    ("grey", -4.175, 4.175, -4.175, 4.9),
)


def _parse_samples(csv_path: Path) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    with csv_path.open("r", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            try:
                samples.append(
                    {
                        "time": float(row["time"]),
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "a": float(row["a"]),
                        "b": float(row["b"]),
                        "slope": float(row["slope"]),
                        "slip": float(row["slip"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

    if not samples:
        raise ValueError(f"No valid rows were found in {csv_path}")

    samples.sort(key=lambda sample: sample["time"])
    return samples


def _classify_sample(x: float, y: float) -> str | None:
    for class_name, x_min, x_max, y_min, y_max in REGIONS:
        if (
            (x_min + BOUNDARY_DISTANCE) < x < (x_max - BOUNDARY_DISTANCE)
            and (y_min + BOUNDARY_DISTANCE) < y < (y_max - BOUNDARY_DISTANCE)
        ):
            return class_name
    return "edge case"


def load_region_data(csv_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    samples = _parse_samples(csv_path)
    loaded: dict[str, tuple[list[np.ndarray], list[float]]] = {}

    for sample in samples:
        class_name = _classify_sample(sample["x"], sample["y"])
        feature_vector = np.array([sample[name] for name in FEATURE_NAMES], dtype=float)
        feature_batches, label_batches = loaded.setdefault(class_name, ([], []))
        feature_batches.append(feature_vector)
        label_batches.append(float(sample["slip"]))

    if not loaded:
        raise ValueError("No samples matched any region")

    return {
        class_name: (np.vstack(feature_batches), np.asarray(label_batches, dtype=float))
        for class_name, (feature_batches, label_batches) in loaded.items()
    }


def _load_ordered_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    samples = _parse_samples(csv_path)
    features = np.vstack(
        [np.array([sample[name] for name in FEATURE_NAMES], dtype=float) for sample in samples]
    )
    labels = np.asarray([sample["slip"] for sample in samples], dtype=float)
    return features, labels


def _stats(values: np.ndarray) -> tuple[float, float, float, float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values)), float(np.median(values)), float(np.min(values)), float(np.max(values))


def _format_value(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.2f}"


def _print_table(title: str, rows: list[tuple[str, ...]], header: tuple[str, ...]) -> None:
    column_spec = "l|" + "c" * (len(header) - 1)
    print(f"\\begin{{tabular}}{{{column_spec}}}")
    print(" & ".join(header) + " \\\\")
    print("\\midrule")
    for row in rows:
        print(" & ".join(row) + " \\\\")
    print("\\end{tabular}")


def _combined_stats(mu: float, sigma: float) -> str:
    return f"({_format_value(mu)}, {_format_value(sigma)})"


def _predict_soinn(soinn: SoinnPlus, features: np.ndarray) -> np.ndarray:
    predictions = np.full(len(features), np.nan, dtype=float)
    for index, feature in enumerate(features):
        pred, _ = soinn.inference(feature, label_clusters=True)
        if pred is not None:
            predictions[index] = float(pred)
    return predictions


def _flatten_region_data(
    region_data: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    class_order = list(region_data.keys())
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    class_batches: list[np.ndarray] = []

    for class_name in class_order:
        features, labels = region_data[class_name]
        feature_batches.append(features)
        label_batches.append(labels)
        class_batches.append(np.full(len(features), class_name, dtype=object))

    return (
        np.vstack(feature_batches),
        np.concatenate(label_batches),
        np.concatenate(class_batches),
        class_order,
    )


def _stats_rows(
    class_order: list[str],
    class_all: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[tuple[str, ...]], dict[str, tuple[float, float, float, float, float]]]:
    rows: list[tuple[str, ...]] = []
    stats: dict[str, tuple[float, float, float, float, float]] = {}

    for class_name in class_order:
        mask = class_all == class_name
        values = _stats(labels[mask])
        stats[class_name] = values
        mu, sigma, median, min_val, max_val = values
        rows.append(
            (
                class_name,
                str(int(np.sum(mask))),
                _combined_stats(mu, sigma),
                _format_value(median),
                _format_value(min_val),
                _format_value(max_val),
            )
        )

    values = _stats(labels)
    stats["merged"] = values
    mu, sigma, median, min_val, max_val = values
    rows.append(
        (
            "merged",
            str(len(labels)),
            _combined_stats(mu, sigma),
            _format_value(median),
            _format_value(min_val),
            _format_value(max_val),
        )
    )
    return rows, stats


def _error_rows(
    class_order: list[str],
    class_all: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    masks = [(class_name, class_all == class_name) for class_name in class_order]
    masks.append(("merged", np.ones(len(class_all), dtype=bool)))

    for class_name, class_mask in masks:
        valid_mask = class_mask & np.isfinite(y_pred) & np.isfinite(y_true)
        errors = y_pred[valid_mask] - y_true[valid_mask]
        mse = float(np.mean(errors**2)) if errors.size else np.nan
        mae = float(np.mean(np.abs(errors))) if errors.size else np.nan
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((y_true[valid_mask] - np.mean(y_true[valid_mask])) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
        rows.append(
            (
                class_name,
                # str(int(np.sum(valid_mask))),
                _format_value(mse),
                _format_value(mae),
                _format_value(r2),
            )
        )

    return rows


def _slip_slope_correlation_rows(
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_test: np.ndarray,
) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for terrain_name in ("green", "blue"):
        class_mask = np.array(
            [str(class_name).startswith(terrain_name) for class_name in class_test],
            dtype=bool,
        )
        valid_mask = class_mask & np.isfinite(x_test[:, 2]) & np.isfinite(y_test)
        slopes = x_test[valid_mask, 2]
        slips = y_test[valid_mask]
        spearman = spearmanr(slopes, slips)
        rows.append((terrain_name, str(len(slopes)), _format_value(spearman.correlation), _format_value(spearman.pvalue)))
    return rows


def _plot_slip_slope_relationships(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    class_samples: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, terrain_name, color in zip(axes, ("green", "blue"), ("forestgreen", "royalblue")):
        class_mask = np.array(
            [str(class_name).startswith(terrain_name) for class_name in class_samples],
            dtype=bool,
        )
        valid_mask = class_mask & np.isfinite(x_samples[:, 2]) & np.isfinite(y_samples)
        slopes = x_samples[valid_mask, 2]
        slips = y_samples[valid_mask]
        axis.scatter(slopes, slips, s=10, alpha=0.35, color=color, label="samples")

        # if len(slopes) >= 2 and np.ptp(slopes) > 0.0:
        #     slope_grid = np.linspace(np.min(slopes), np.max(slopes), 200)
        #     linear_fit = np.polyval(np.polyfit(slopes, slips, deg=1), slope_grid)
        #     axis.plot(slope_grid, linear_fit, color="black", linewidth=1.5, label="linear fit")
        #     if len(slopes) >= 3:
        #         quadratic_fit = np.polyval(np.polyfit(slopes, slips, deg=2), slope_grid)
        #         axis.plot(
        #             slope_grid,
        #             quadratic_fit,
        #             color="darkorange",
        #             linewidth=1.5,
        #             label="quadratic fit",
        #         )
        #     correlation = float(spearmanr(slopes, slips).statistic)
        #     axis.text(
        #         0.03,
        #         0.97,
        #         f"Spearman rho = {correlation:.2f}\nN = {len(slopes)}",
        #         transform=axis.transAxes,
        #         va="top",
        #         bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        #     )

        axis.set_title(f"{terrain_name.capitalize()} regions")
        axis.set_xlabel("slope")
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axis.legend()
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)

    axes[0].set_ylabel("slip")
    figure.suptitle("Slip versus slope")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"Saved plot to: {output_path}")


def main() -> None:
    for data_path in (TRAIN_CSV, TEST_CSV):
        if not data_path.exists():
            raise FileNotFoundError(f"CSV file not found: {data_path}")
    x_train, y_train = _load_ordered_data(TRAIN_CSV)
    soinn = SoinnPlus(dim=len(FEATURE_NAMES))
    for feature, label in zip(x_train, y_train, strict=True):
        soinn.input_signal(feature, label=label)

    train_region_data = load_region_data(TRAIN_CSV)
    x_train, y_train, class_train, class_order = _flatten_region_data(train_region_data)
    train_rows, _ = _stats_rows(class_order, class_train, y_train)

    test_region_data = load_region_data(TEST_CSV)
    x_test, y_test, class_test, class_order = _flatten_region_data(test_region_data)
    test_rows, true_stats = _stats_rows(class_order, class_test, y_test)

    y_pred = _predict_soinn(soinn, x_test)
    comparison_rows: list[tuple[str, ...]] = []
    error_rows = _error_rows(class_order, class_test, y_test, y_pred)
    correlation_rows = _slip_slope_correlation_rows(np.vstack((x_test, x_train)), np.concatenate((y_test, y_train)), np.concatenate((class_test, class_train)))
    _plot_slip_slope_relationships(
        np.vstack((x_test, x_train)),
        np.concatenate((y_test, y_train)),
        np.concatenate((class_test, class_train)),
        PLOT_OUTPUT,
    )

    for class_name in class_order + ["merged"]:
        mu_true, sigma_true, median_true, min_true, max_true = true_stats[class_name]
        mask = class_test == class_name if class_name != "merged" else np.ones(len(class_test), dtype=bool)
        mu_pred, sigma_pred, median_pred, min_pred, max_pred = _stats(y_pred[mask])
        d = (mu_pred - mu_true) / sigma_true if sigma_true != 0.0 else np.nan
        d_med = (median_pred - median_true)
        r = (sigma_pred**2) / (sigma_true**2) if sigma_true != 0.0 else np.nan
        comparison_rows.append(
            (
                class_name,
                str(int(np.sum(mask))),
                _combined_stats(mu_pred, sigma_pred),
                _format_value(median_pred),
                _format_value(min_pred),
                _format_value(max_pred),
                _format_value(d_med),
                _format_value(d),
                _format_value(r),
            )
        )

    print("Train label statistics")
    _print_table("Train label statistics", train_rows, ("\\textbf{Terrain}", "$N$", "($\\mu_s$, $\\sigma_s$)", "$med$", "$min$", "$max$"))
    print()
    print("Test label statistics")
    _print_table("Test label statistics", test_rows, ("\\textbf{Terrain}", "$N$", "($\\mu_s$, $\\sigma_s$)", "$med$", "$min$", "$max$"))
    print()
    # print("Prediction statistics")
    # _print_table("Prediction statistics", pred_rows, ("\\textbf{Terrain}", "$N$", "($\\mu$, $\\sigma$)", "$med$", "$min$", "$max$"))
    # print()
    print("Prediction comparison")
    _print_table("Prediction comparison", comparison_rows, ("\\textbf{Terrain}", "$N$", "($\\tilde\\mu_s$, $\\tilde\\sigma_s$)", "$med$", "$min$", 
                                                            "$max$", "$\\Delta \\text{med}$", "$d$", "$r$"))
    print()
    print("Prediction errors")
    _print_table("Prediction errors", error_rows, ("\\textbf{Terrain}", "$MSE$", "$MAE$", "$R^2$"))
    print()
    print("Slip-slope correlation")
    _print_table("Slip-slope correlation", correlation_rows, ("\\textbf{Terrain}", "$N$", "$\\rho_{slope}$", "$p_{slope}$"))


if __name__ == "__main__":
    main()
