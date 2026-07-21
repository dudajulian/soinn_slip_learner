from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import GPy
import numpy as np

import data_loader as dl
import eval_tools as et
from soinn_py import SoinnPlus


TRAIL_DATA_DIRECTORY = dl.TRAIL_DATA_DIRECTORY
GRID_DATA_DIRECTORY = dl.GRID_DATA_DIRECTORY
RESULTS_DIRECTORY = Path("/workspaces/vscode_ros2_workspace/resources/results/evalutation/")
FEATURE_COUNT = 5


class SoinnEvalModel(et.EvalModelBase):
    def __init__(self, shuffle: bool, seed: int, use_fallback: bool) -> None:
        self.shuffle = shuffle
        self.seed = seed
        self.use_fallback = use_fallback
        self.soinn: SoinnPlus | None = None
        self.fallback_count = 0

    def train(self, x_train: np.ndarray, y_train: np.ndarray, class_train: np.ndarray | None = None) -> None:
        soinn = SoinnPlus(dim=FEATURE_COUNT)
        indices = np.arange(len(x_train))
        if self.shuffle:
            np.random.default_rng(self.seed).shuffle(indices)
        for i in indices:
            soinn.input_signal(x_train[i], label=y_train[i])
        self.soinn = soinn

    def predict(self, x_test: np.ndarray, class_test: np.ndarray | None = None) -> np.ndarray:
        if self.soinn is None:
            raise ValueError("Model is not trained")

        predictions = self.soinn.batch_inference(x_test)
        return predictions[:, 0]  # Return only the predicted means


def _sample_indices(n_samples: int, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    if sample_count <= 0 or sample_count >= n_samples:
        return np.arange(n_samples, dtype=int)
    return np.sort(rng.choice(n_samples, size=sample_count, replace=False)).astype(int)


def load_training_data(data_directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    trail_data = dl.load_trail_data(data_directory)
    x_batches, y_batches, c_batches = [], [], []
    for class_name, (x, y) in sorted(trail_data.items()):
        x_batches.append(x)
        y_batches.append(y)
        c_batches.append(np.full(len(x), class_name, dtype=object))

    return np.vstack(x_batches), np.concatenate(y_batches), np.concatenate(c_batches), list(trail_data.keys())


def load_grid_data(data_directory: Path, sample_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    grid_data = dl.load_grid_data(data_directory)
    rng = np.random.default_rng(seed)
    x_batches, p_batches, c_batches = [], [], []

    for class_name, (x, pos) in sorted(grid_data.items()):
        idx = _sample_indices(len(x), sample_count, rng)

        x_batches.append(x[idx])
        p_batches.append(pos[idx])
        c_batches.append(np.full(len(idx), class_name, dtype=object))

    if not x_batches:
        raise ValueError("No grid samples found")

    return np.vstack(x_batches), np.vstack(p_batches), np.concatenate(c_batches), list(grid_data.keys())


def train_reference_gps(
    x_train: np.ndarray,
    y_train: np.ndarray,
    class_train: np.ndarray,
) -> dict[str, GPy.models.GPRegression]:
    models: dict[str, GPy.models.GPRegression] = {}
    for class_name in np.unique(class_train):
        mask = class_train == class_name
        X = np.asarray(x_train[mask], dtype=float)
        y = np.asarray(y_train[mask], dtype=float)[:, None]
        kernel = GPy.kern.RBF(input_dim=FEATURE_COUNT, ARD=False)
        gp = GPy.models.GPRegression(X, y, kernel=kernel)
        gp.optimize_restarts(num_restarts=3,robust=True,verbose=False,messages=False,max_iters=1000)
        models[str(class_name)] = gp
    return models


def gp_reference_predict(
    gp_models: dict[str, GPy.models.GPRegression],
    x_test: np.ndarray,
    class_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mu = np.full(len(x_test), np.nan, dtype=float)
    sigma = np.full(len(x_test), np.nan, dtype=float)

    for class_name in np.unique(class_test):
        mask = class_test == class_name
        gp = gp_models.get(str(class_name))
        if gp is None:
            continue
        pred_mu, pred_var = gp.predict(np.asarray(x_test[mask], dtype=float))
        mu[mask] = pred_mu[:, 0]
        sigma[mask] = np.sqrt(np.maximum(pred_var[:, 0], 0.0))

    return mu, sigma


def _ordered_training_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    class_train: np.ndarray,
    shuffle: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(x_train))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    return x_train[indices], y_train[indices], class_train[indices]


def _training_sections(classes: np.ndarray) -> list[tuple[int, str]]:
    if len(classes) == 0:
        return []
    sections = [(1, str(classes[0]))]
    for i in range(1, len(classes)):
        if classes[i] != classes[i - 1]:
            sections.append((i + 1, str(classes[i])))
    return sections


def predict_soinn(soinn: SoinnPlus, x_test: np.ndarray, use_fallback: bool) -> tuple[np.ndarray, int]:
    preds = np.full(len(x_test), np.nan, dtype=float)
    fallback_count = 0
    for i, feature in enumerate(x_test):
        pred_mean, _ = soinn.inference(feature, label_clusters=True)
        if pred_mean is None and use_fallback:
            fallback_count += 1
            winner = soinn.find_winner(feature)
            pred_mean, _ = soinn.predictions[winner]
            if pred_mean is None:
                pred_mean = soinn.labels[winner]
                print(pred_mean)
        if pred_mean is not None:
            preds[i] = float(pred_mean)
    return preds, fallback_count



def train_soinn_learning_curve(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    class_test: np.ndarray,
    y_ref: np.ndarray,
    sigma_ref: np.ndarray,
    eval_step: int,
    use_fallback: bool,
) -> tuple[SoinnPlus, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    soinn = SoinnPlus(dim=FEATURE_COUNT)
    steps: list[int] = []
    r_values: list[float] = []
    fallback_values: list[int] = []

    class_names = [str(name) for name in sorted(np.unique(class_test))]
    class_masks = {name: (class_test == name) for name in class_names}
    class_r_history: dict[str, list[float]] = {name: [] for name in class_names}

    for i, (x, y) in enumerate(zip(x_train, y_train), start=1):
        soinn.input_signal(x, label=y)
        if i % eval_step == 0 or i == len(x_train):
            preds, fallback_count = predict_soinn(soinn, x_test, use_fallback)
            metrics = et.compute_reference_metrics(preds, y_ref, sigma_ref)
            for class_name in class_names:
                m = class_masks[class_name]
                class_metrics = et.compute_reference_metrics(preds[m], y_ref[m], sigma_ref[m])
                class_r_history[class_name].append(class_metrics["r"])
            steps.append(i)
            r_values.append(metrics["r"])
            fallback_values.append(fallback_count)

    curves = {name: np.asarray(values, dtype=float) for name, values in class_r_history.items()}
    return (
        soinn,
        np.asarray(steps, dtype=int),
        np.asarray(r_values, dtype=float),
        np.asarray(fallback_values, dtype=int),
        curves,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SOINN+ against per-class GP reference models.")
    parser.add_argument("--trail-dir", type=Path, default=TRAIL_DATA_DIRECTORY, help="Directory containing trail files")
    parser.add_argument("--grid-dir", type=Path, default=GRID_DATA_DIRECTORY, help="Directory containing grid files")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle training samples before feeding them to SOINN+")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Keep file order during training")
    parser.set_defaults(shuffle=False)
    parser.add_argument("--fallback", action="store_true", help="Enable nearest-node fallback when SOINN+ abstains")
    parser.add_argument("--no-fallback", dest="fallback", action="store_false", help="Disable fallback predictions")
    parser.set_defaults(fallback=False)
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when shuffling and sampling grid cells")
    parser.add_argument("--grid-samples-per-file", type=int, default=0, help="Number of grid cells sampled per file (0 = all)")
    parser.add_argument("--curve-step", type=int, default=100, help="Evaluate learning curve every N training samples")
    parser.add_argument("--plot", action="store_true", help="Show SOINN+ network plot")
    parser.add_argument("--plot-curve", action="store_true", help="Show R learning curve plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.curve_step <= 0:
        raise ValueError("--curve-step must be a positive integer")

    run_stamp = datetime.now().strftime("%y%m%d%H%M")
    curve_compare_output_path = RESULTS_DIRECTORY / f"{run_stamp}_rcurve_compare.png"
    metrics_output_path = RESULTS_DIRECTORY / f"{run_stamp}_metrics.csv"

    x_train, y_train, class_train, trail_files = load_training_data(args.trail_dir)
    x_test, positions, class_test, grid_files = load_grid_data(
        args.grid_dir,
        sample_count=args.grid_samples_per_file,
        seed=args.seed,
    )

    x_train, y_train, class_train = _ordered_training_data(
        x_train,
        y_train,
        class_train,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    gp_models = train_reference_gps(x_train, y_train, class_train)
    y_ref, sigma_ref = gp_reference_predict(gp_models, x_test, class_test)

    et.EvalTools.set_data(x_train, y_train, class_train, x_test, y_ref, class_test)
    soinn_model = SoinnEvalModel(shuffle=args.shuffle, seed=args.seed, use_fallback=args.fallback)
    et.EvalTools.set_model(soinn_model)
    et.EvalTools.train_model()
    et.EvalTools.y_pred = et.EvalTools.predict_model()

    soinn_curve, curve_steps, curve_r, curve_fallback, curve_r_by_class = train_soinn_learning_curve(
        x_train,
        y_train,
        x_test,
        class_test,
        y_ref,
        sigma_ref,
        eval_step=args.curve_step,
        use_fallback=args.fallback,
    )

    sections = _training_sections(class_train)
    if args.shuffle:
        print("Shuffle enabled: section markers no longer represent contiguous terrain blocks")
        sections = []

    print(f"Loaded {len(trail_files)} trail files")
    print(f"Loaded trail classes: {trail_files}")
    print(f"Loaded grid classes: {grid_files}")
    print(f"Training samples: {len(x_train)}")
    print(f"Feature shape: {x_train.shape}")
    print(f"Label shape: {y_train.shape}")
    print(f"Trained SOINN+ nodes: {len(soinn_model.soinn.nodes) if soinn_model.soinn is not None else 0}")
    print(f"Trained reference GPs: {len(gp_models)}")
    print("Training strategy: full trail set as train; grid as test; y_test from per-class GP reference")
    print(f"Grid sampling -> {args.grid_samples_per_file} cells per file (0 = all cells)")
    print(f"Prediction mode -> fallback: {args.fallback}")
    print(f"Grid sample count -> {len(x_test)}")
    print(f"Learning curve -> checkpoints: {len(curve_steps)}, eval_step: {args.curve_step}")
    if curve_steps.size > 0:
        print(f"Learning curve final R: {curve_r[-1]:.6f} at step {int(curve_steps[-1])}")
        for class_name, r_curve in curve_r_by_class.items():
            print(f"Learning curve final R for class '{class_name}': {r_curve[-1]:.6f} at step {int(curve_steps[-1])}")

    abstention = et.EvalTools.abstention_metrics()
    metrics = et.compute_reference_metrics(et.EvalTools.y_pred, y_ref, sigma_ref)
    et.print_reference_metrics(
        "Grid benchmark",
        metrics,
        abstention,
        aux_count=soinn_model.fallback_count,
        aux_label="fallback predictions used",
    )

    avg_gp_sigma = float(np.nanmean(sigma_ref))
    print(f"Average GP sigma: {avg_gp_sigma:.6f}")

    et.plot_learning_curve_comparison(
        curve_steps,
        overall_r=curve_r,
        class_r_curves=curve_r_by_class,
        output_path=curve_compare_output_path,
        sections=sections,
        enabled=args.plot_curve,
    )

    et.save_class_grid_plots(
        run_stamp=run_stamp,
        positions=positions,
        class_labels=class_test,
        gp_predictions=y_ref,
        soinn_predictions=et.EvalTools.y_pred,
        output_dir=RESULTS_DIRECTORY,
        cost_mu=dl.COST_MU,
        cost_sigma=dl.COST_SIGMA,
    )

    et.save_metrics_csv(
        output_path=metrics_output_path,
        metrics=metrics,
        abstention=abstention,
        fallback_count=soinn_model.fallback_count,
        avg_gp_sigma=avg_gp_sigma,
        curve_step=args.curve_step,
        curve_steps=curve_steps,
        curve_r=curve_r,
        curve_fallback=curve_fallback,
    )

    if args.plot and soinn_curve.nodes:
        et.plot_network(soinn_curve, True)


if __name__ == "__main__":
    main()
