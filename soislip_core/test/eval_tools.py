from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class EvalModelBase:
    """Model placeholder to be overridden by user implementations."""

    def train(self, x_train: np.ndarray, y_train: np.ndarray, class_train: np.ndarray | None = None) -> None:
        raise NotImplementedError("Override train() in your model implementation")

    def predict(self, x_test: np.ndarray, class_test: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError("Override predict() in your model implementation")


class EvalTools:
    # Shared data container. Fill these in caller scripts before running methods.
    x_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    class_train: np.ndarray | None = None
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    class_test: np.ndarray | None = None

    # User-provided model that implements train/predict.
    model: EvalModelBase | None = None

    @classmethod
    def set_data(
        cls,
        x_train: np.ndarray,
        y_train: np.ndarray,
        class_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        class_test: np.ndarray,
    ) -> None:
        cls.x_train = x_train
        cls.y_train = y_train
        cls.class_train = class_train
        cls.x_test = x_test
        cls.y_test = y_test
        cls.class_test = class_test

    @classmethod
    def set_model(cls, model: EvalModelBase) -> None:
        cls.model = model

    @classmethod
    def train_model(cls) -> None:
        if cls.model is None:
            raise ValueError("EvalTools.model is not set")
        if cls.x_train is None or cls.y_train is None:
            raise ValueError("x_train and y_train must be set before training")
        cls.model.train(cls.x_train, cls.y_train, cls.class_train)

    @classmethod
    def predict_model(cls) -> np.ndarray:
        if cls.model is None:
            raise ValueError("EvalTools.model is not set")
        if cls.x_test is None:
            raise ValueError("x_test must be set before prediction")
        cls.y_pred = np.asarray(cls.model.predict(cls.x_test, cls.class_test), dtype=float)
        return cls.y_pred

    @staticmethod
    def denormalize_labels(labels: np.ndarray, cost_mu: float, cost_sigma: float) -> np.ndarray:
        return labels * cost_sigma + cost_mu

    @staticmethod
    def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        valid_true = y_true[valid_mask]
        valid_pred = y_pred[valid_mask]

        if valid_true.size == 0:
            return {
                "rmse": float("nan"),
                "mae": float("nan"),
                "nrmse": float("nan"),
                "r2": float("nan"),
                "r2_fixed": float("nan"),
                "medae": float("nan"),
                "rmse_mae_ratio": float("nan"),
                "used_samples": 0.0,
            }

        errors = valid_pred - valid_true
        mse = np.mean(errors ** 2)
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(errors)))

        y_range = float(np.max(valid_true) - np.min(valid_true))
        nrmse = float(rmse / y_range) if y_range > 0.0 else float("nan")

        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((valid_true - np.mean(valid_true)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

        r2_fixed = float(1.0 - mse) # assuming normalized labels with variance 1.0
        medae = float(np.median(np.abs(errors)))
        rmse_mae_ratio = float(rmse / mae) if mae > 0.0 else float("nan")

        return {
            "rmse": rmse,
            "mae": mae,
            "nrmse": nrmse,
            "r2": r2,
            "r2_fixed": r2_fixed,
            "medae": medae,
            "rmse_mae_ratio": rmse_mae_ratio,
            "used_samples": float(valid_true.size),
        }

    @classmethod
    def regression_metrics(cls) -> dict[str, float]:
        if cls.y_test is None or cls.y_pred is None:
            raise ValueError("y_test and y_pred must be set before computing regression metrics")
        return cls.compute_regression_metrics(cls.y_test, cls.y_pred)

    @staticmethod
    def compute_abstention_metrics(y_pred: np.ndarray) -> dict[str, float]:
        total_samples = len(y_pred)
        confident_mask = np.isfinite(y_pred)
        confident_samples = int(np.sum(confident_mask))
        abstained_samples = total_samples - confident_samples
        coverage = float(confident_samples / total_samples) if total_samples > 0 else float("nan")
        abstain_rate = float(abstained_samples / total_samples) if total_samples > 0 else float("nan")

        return {
            "total_samples": float(total_samples),
            "confident_samples": float(confident_samples),
            "abstained_samples": float(abstained_samples),
            "coverage": coverage,
            "abstain_rate": abstain_rate,
        }

    @classmethod
    def abstention_metrics(cls) -> dict[str, float]:
        if cls.y_pred is None:
            raise ValueError("y_pred must be set before computing abstention metrics")
        return cls.compute_abstention_metrics(cls.y_pred)

    @staticmethod
    def compute_uncertainty_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sigma: np.ndarray,
    ) -> dict[str, float]:
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(sigma) & (sigma > 0.0)
        valid_true = y_true[valid_mask]
        valid_pred = y_pred[valid_mask]
        valid_sigma = sigma[valid_mask]

        if valid_true.size == 0:
            return {
                "r": float("nan"),
                "pinaw": float("nan"),
                "pinaw_fixed": float("nan"),
                "z_mean": float("nan"),
                "z_std": float("nan"),
                "avg_sigma": float("nan"),
            }

        errors = valid_pred - valid_true
        y_range = float(np.max(valid_true) - np.min(valid_true))
        r = float(np.mean(np.abs(errors) <= 2.0 * valid_sigma))
        pinaw = float(np.mean(4.0 * valid_sigma) / y_range) if y_range > 0.0 else float("nan")
        pinaw_fixed = float(np.mean(4.0 * valid_sigma))
        z_scores = errors / valid_sigma

        return {
            "r": r,
            "pinaw": pinaw,
            "pinaw_fixed": pinaw_fixed,
            "z_mean": float(np.mean(z_scores)),
            "z_std": float(np.std(z_scores)),
            "avg_sigma": float(np.mean(valid_sigma)),
        }

    @staticmethod
    def compute_reference_metrics(
        y_pred: np.ndarray,
        mu_ref: np.ndarray,
        sigma_ref: np.ndarray,
    ) -> dict[str, float]:
        regression = EvalTools.compute_regression_metrics(mu_ref, y_pred)
        uncertainty = EvalTools.compute_uncertainty_metrics(mu_ref, y_pred, sigma_ref)
        return {
            **regression,
            "r": uncertainty["r"],
            "pinaw": uncertainty["pinaw"],
            "pinaw_fixed": uncertainty["pinaw_fixed"],
            "z_mean": uncertainty["z_mean"],
            "z_std": uncertainty["z_std"],
        }

    @staticmethod
    def print_basic_metrics(
        split_name: str,
        metrics: dict[str, float],
        abstention: dict[str, float],
        aux_count: int,
        aux_label: str,
    ) -> None:
        print(f"{split_name} metrics:")
        print(
            "  samples: "
            f"{int(abstention['total_samples'])}, "
            f"confident: {int(abstention['confident_samples'])}, "
            f"abstained: {int(abstention['abstained_samples'])}, "
            f"coverage: {abstention['coverage']:.6f}, "
            f"abstain_rate: {abstention['abstain_rate']:.6f}"
        )
        print(f"  {aux_label}: {aux_count}")
        print(f"  used for regression metrics: {int(metrics['used_samples'])}")
        print(f"  RMSE:  {metrics['rmse']:.6f}")
        print(f"  MAE:   {metrics['mae']:.6f}")
        print(f"  NRMSE: {metrics['nrmse']:.6f}")
        print(f"  R2:    {metrics['r2']:.6f}")
        print(f"  R2 (fixed): {metrics['r2_fixed']:.6f}")
        print(f"  MedAE: {metrics['medae']:.6f}")
        print(f"  RMSE/MAE: {metrics['rmse_mae_ratio']:.6f}")

    @staticmethod
    def print_reference_metrics(
        split_name: str,
        metrics: dict[str, float],
        abstention: dict[str, float],
        aux_count: int,
        aux_label: str,
        avg_sigma: float | None = None,
    ) -> None:
        EvalTools.print_basic_metrics(split_name, metrics, abstention, aux_count, aux_label)
        print(f"  R:     {metrics['r']:.6f}")
        print(f"  PINAW: {metrics['pinaw']:.6f}")
        print(f"  PINAW (fixed): {metrics['pinaw_fixed']:.6f}")
        print(f"  Z-Score Mean: {metrics['z_mean']:.6f}")
        print(f"  Z-Score Std:  {metrics['z_std']:.6f}")
        if avg_sigma is not None:
            print(f"  Average GP sigma: {avg_sigma:.6f}")

    @staticmethod
    def plot_network(soinn: Any, enabled: bool) -> None:
        if not enabled or not soinn.nodes:
            return

        soinn.show(save=False)
        plt.show()

    @staticmethod
    def plot_learning_curve(
        steps: np.ndarray,
        r_values: np.ndarray,
        sections: list[tuple[int, str]],
        enabled: bool,
        output_path: Path | None,
        title: str = "SOINN+ Learning Curve",
    ) -> None:
        if steps.size == 0:
            return
        if not enabled and output_path is None:
            return

        fig, ax = plt.subplots(figsize=(11.0, 3.5))
        ax.plot(steps, r_values, color="#d95f02", linewidth=2.0, label="SOINN+")
        ax.set_title(title)
        ax.set_xlabel("Learning step")
        ax.set_ylabel("R")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(0.0, float(steps[-1]))
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)

        for section_idx, (start_step, section_name) in enumerate(sections):
            next_start = sections[section_idx + 1][0] if section_idx + 1 < len(sections) else int(steps[-1])
            ax.axvline(start_step, color="#b2a38f", linestyle="--", linewidth=0.9, alpha=0.8)
            middle = 0.5 * (start_step + next_start)
            ax.text(
                middle,
                0.03,
                section_name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.legend(loc="upper right")
        fig.tight_layout()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=180)
            print(f"Saved learning curve plot to {output_path}")

        if enabled:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def plot_learning_curve_comparison(
        steps: np.ndarray,
        overall_r: np.ndarray,
        class_r_curves: dict[str, np.ndarray],
        output_path: Path,
        sections: list[tuple[int, str]],
        enabled: bool,
    ) -> None:
        if steps.size == 0:
            return

        fig, ax = plt.subplots(figsize=(11.0, 4.2))
        ax.plot(steps, overall_r, linewidth=2.6, color="#1f1f1f", label="all")

        class_names = sorted(class_r_curves.keys())
        if class_names:
            colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(class_names), 2)))
            for idx, class_name in enumerate(class_names):
                ax.plot(
                    steps,
                    class_r_curves[class_name],
                    linewidth=1.8,
                    color=colors[idx],
                    label=str(class_name),
                )

        ax.set_title("R Curve Comparison - All and Per Class")
        ax.set_xlabel("Learning step")
        ax.set_ylabel("R")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(0.0, float(steps[-1]))
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)

        for section_idx, (start_step, section_name) in enumerate(sections):
            next_start = sections[section_idx + 1][0] if section_idx + 1 < len(sections) else int(steps[-1])
            ax.axvline(start_step, color="#b2a38f", linestyle="--", linewidth=0.8, alpha=0.6)
            middle = 0.5 * (start_step + next_start)
            ax.text(
                middle,
                0.03,
                section_name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
        print(f"Saved learning curve comparison plot to {output_path}")

        if enabled:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def _safe_name(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)

    @staticmethod
    def _axis_edges(axis_values: np.ndarray) -> np.ndarray:
        if axis_values.size == 1:
            center = float(axis_values[0])
            return np.array([center - 0.5, center + 0.5], dtype=float)

        deltas = np.diff(axis_values)
        edges = np.empty(axis_values.size + 1, dtype=float)
        edges[1:-1] = axis_values[:-1] + 0.5 * deltas
        edges[0] = axis_values[0] - 0.5 * deltas[0]
        edges[-1] = axis_values[-1] + 0.5 * deltas[-1]
        return edges

    @classmethod
    def save_class_grid_plots(
        cls,
        run_stamp: str,
        positions: np.ndarray,
        class_labels: np.ndarray,
        gp_predictions: np.ndarray,
        soinn_predictions: np.ndarray,
        output_dir: Path,
        cost_mu: float,
        cost_sigma: float,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        for class_name in sorted(np.unique(class_labels)):
            mask = class_labels == class_name
            class_positions = positions[mask]
            gp_values = cls.denormalize_labels(gp_predictions[mask], cost_mu, cost_sigma)
            soinn_values = cls.denormalize_labels(soinn_predictions[mask], cost_mu, cost_sigma)

            x_unique = np.unique(class_positions[:, 0])
            y_unique = np.unique(class_positions[:, 1])
            x_edges = cls._axis_edges(x_unique)
            y_edges = cls._axis_edges(y_unique)

            x_index = {value: idx for idx, value in enumerate(x_unique)}
            y_index = {value: idx for idx, value in enumerate(y_unique)}

            gp_grid = np.full((y_unique.size, x_unique.size), np.nan, dtype=float)
            soinn_grid = np.full((y_unique.size, x_unique.size), np.nan, dtype=float)
            for point, gp_value, soinn_value in zip(class_positions, gp_values, soinn_values):
                x_idx = x_index[point[0]]
                y_idx = y_index[point[1]]
                gp_grid[y_idx, x_idx] = gp_value
                soinn_grid[y_idx, x_idx] = soinn_value

            finite_gp = gp_values[np.isfinite(gp_values)]
            finite_soinn = soinn_values[np.isfinite(soinn_values)]
            if finite_gp.size == 0 and finite_soinn.size == 0:
                continue

            vmin = 0.0
            vmax = 0.03

            for model_name, value_grid in (("gp", gp_grid), ("soinn+", soinn_grid)):
                if not np.any(np.isfinite(value_grid)):
                    continue

                masked_grid = np.ma.masked_invalid(value_grid)

                fig, ax = plt.subplots(figsize=(4.0, 4.0))
                mesh = ax.pcolormesh(
                    x_edges,
                    y_edges,
                    masked_grid,
                    cmap="jet",
                    vmin=vmin,
                    vmax=vmax,
                    shading="flat",
                    edgecolors="none",
                )
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(f"{model_name.upper()} model - {class_name}")
                ax.set_xlim(x_edges[0], x_edges[-1])
                ax.set_ylim(y_edges[0], y_edges[-1])
                fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="prediction")
                fig.tight_layout()

                file_name = f"{run_stamp}_{cls._safe_name(str(class_name))}_{model_name}_grid.png"
                file_path = output_dir / file_name
                fig.savefig(file_path, dpi=180)
                plt.close(fig)
                print(f"Saved {model_name.upper()} grid plot to {file_path}")

    @staticmethod
    def save_metrics_csv(
        output_path: Path,
        metrics: dict[str, float],
        abstention: dict[str, float],
        fallback_count: int,
        avg_gp_sigma: float,
        curve_step: int,
        curve_steps: np.ndarray,
        curve_r: np.ndarray,
        curve_fallback: np.ndarray,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        final_curve_r = float(curve_r[-1]) if curve_r.size > 0 else float("nan")

        with output_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "rmse",
                "mae",
                "nrmse",
                "r2",
                "r",
                "r2_fixed",
                "medae",
                "rmse_mae_ratio",
                "pinaw",
                "pinaw_fixed",
                "z_mean",
                "z_std",
                "used_samples",
                "total_samples",
                "confident_samples",
                "abstained_samples",
                "coverage",
                "abstain_rate",
                "fallback_predictions",
                "average_gp_sigma",
                "curve_step",
                "num_curve_points",
                "final_curve_r",
            ])
            writer.writerow([
                f"{metrics['rmse']:.10g}",
                f"{metrics['mae']:.10g}",
                f"{metrics['nrmse']:.10g}",
                f"{metrics['r2']:.10g}",
                f"{metrics['r']:.10g}",
                f"{metrics['r2_fixed']:.10g}",
                f"{metrics['medae']:.10g}",
                f"{metrics['rmse_mae_ratio']:.10g}",
                f"{metrics['pinaw']:.10g}",
                f"{metrics['pinaw_fixed']:.10g}",
                f"{metrics['z_mean']:.10g}",
                f"{metrics['z_std']:.10g}",
                int(metrics["used_samples"]),
                int(abstention["total_samples"]),
                int(abstention["confident_samples"]),
                int(abstention["abstained_samples"]),
                f"{abstention['coverage']:.10g}",
                f"{abstention['abstain_rate']:.10g}",
                fallback_count,
                f"{avg_gp_sigma:.10g}",
                curve_step,
                int(curve_steps.size),
                f"{final_curve_r:.10g}",
            ])

            writer.writerow([])
            writer.writerow(["learning_step", "r", "fallback_predictions"])
            for step, r_value, fallback_value in zip(curve_steps, curve_r, curve_fallback):
                writer.writerow([int(step), f"{float(r_value):.10g}", int(fallback_value)])

        print(f"Saved metrics CSV to {output_path}")


# Backward-compatible wrappers for existing scripts.
def denormalize_labels(labels: np.ndarray, cost_mu: float, cost_sigma: float) -> np.ndarray:
    return EvalTools.denormalize_labels(labels, cost_mu, cost_sigma)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return EvalTools.compute_regression_metrics(y_true, y_pred)


def compute_abstention_metrics(y_pred: np.ndarray) -> dict[str, float]:
    return EvalTools.compute_abstention_metrics(y_pred)


def compute_uncertainty_metrics(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray) -> dict[str, float]:
    return EvalTools.compute_uncertainty_metrics(y_true, y_pred, sigma)


def compute_reference_metrics(y_pred: np.ndarray, mu_ref: np.ndarray, sigma_ref: np.ndarray) -> dict[str, float]:
    return EvalTools.compute_reference_metrics(y_pred, mu_ref, sigma_ref)


def print_basic_metrics(
    split_name: str,
    metrics: dict[str, float],
    abstention: dict[str, float],
    aux_count: int,
    aux_label: str,
) -> None:
    EvalTools.print_basic_metrics(split_name, metrics, abstention, aux_count, aux_label)


def print_reference_metrics(
    split_name: str,
    metrics: dict[str, float],
    abstention: dict[str, float],
    aux_count: int,
    aux_label: str,
    avg_sigma: float | None = None,
) -> None:
    EvalTools.print_reference_metrics(split_name, metrics, abstention, aux_count, aux_label, avg_sigma)


def plot_network(soinn: Any, enabled: bool) -> None:
    EvalTools.plot_network(soinn, enabled)


def plot_learning_curve(
    steps: np.ndarray,
    r_values: np.ndarray,
    sections: list[tuple[int, str]],
    enabled: bool,
    output_path: Path | None,
    title: str = "SOINN+ Learning Curve",
) -> None:
    EvalTools.plot_learning_curve(steps, r_values, sections, enabled, output_path, title)


def plot_learning_curve_comparison(
    steps: np.ndarray,
    overall_r: np.ndarray,
    class_r_curves: dict[str, np.ndarray],
    output_path: Path,
    sections: list[tuple[int, str]],
    enabled: bool,
) -> None:
    EvalTools.plot_learning_curve_comparison(steps, overall_r, class_r_curves, output_path, sections, enabled)


def save_class_grid_plots(
    run_stamp: str,
    positions: np.ndarray,
    class_labels: np.ndarray,
    gp_predictions: np.ndarray,
    soinn_predictions: np.ndarray,
    output_dir: Path,
    cost_mu: float,
    cost_sigma: float,
) -> None:
    EvalTools.save_class_grid_plots(
        run_stamp,
        positions,
        class_labels,
        gp_predictions,
        soinn_predictions,
        output_dir,
        cost_mu,
        cost_sigma,
    )


def save_metrics_csv(
    output_path: Path,
    metrics: dict[str, float],
    abstention: dict[str, float],
    fallback_count: int,
    avg_gp_sigma: float,
    curve_step: int,
    curve_steps: np.ndarray,
    curve_r: np.ndarray,
    curve_fallback: np.ndarray,
) -> None:
    EvalTools.save_metrics_csv(
        output_path,
        metrics,
        abstention,
        fallback_count,
        avg_gp_sigma,
        curve_step,
        curve_steps,
        curve_r,
        curve_fallback,
    )