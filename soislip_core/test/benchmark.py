from __future__ import annotations

import argparse
from datetime import datetime, time
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

CLASS_STATS = {
    "black": (472, 0.02168, 0.00904),
    "cubeblack": (482, 0.02176, 0.00755),
    "cubes": (693, 0.02257, 0.00783),
    "cubeturf": (202, 0.01942, 0.00724),
    "flat": (451, 0.00919, 0.00228),
    "slope": (824, 0.02238, 0.01089),
    "turf": (406, 0.00955, 0.00191),
    "merged": (3530, 0.01891, 0.00961),
}


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

    return (
        np.vstack(x_batches),
        np.concatenate(y_batches),
        np.concatenate(c_batches),
        list(trail_data.keys()),
    )

def order_trail_data(
    trail_data: dict[str, tuple[np.ndarray, np.ndarray]],
    class_order: list[str],
    split_into: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load trail training data and concatenate class batches in a user-defined order.

    Args:
        trail_data: Dictionary of trail data loaded by `dl.load_trail_data`.
        class_order: Desired class order, e.g. ["turf", "black", "cubeblack", "slope"].
        split_into: Number of splits for each class batch.

    Returns:
        x_train, y_train, class_train, used_order
    """
    unknown_classes = [c for c in class_order if c not in trail_data]
    if unknown_classes:
        raise ValueError(f"Classes in class_order not found in dataset: {unknown_classes}")

    used_order = [c for c in class_order if c in trail_data]

    x_batches, y_batches, c_batches = [], [], []
    for split in range(split_into):
        for class_name in used_order:
            x, y = trail_data[class_name]
            if split_into > 1:
                n_samples = len(x)
                start_idx = (n_samples * split) // split_into
                end_idx = (n_samples * (split + 1)) // split_into
                x, y = x[start_idx:end_idx], y[start_idx:end_idx]
            x_batches.append(x)
            y_batches.append(y)
            c_batches.append(np.full(len(x), class_name, dtype=object))

    return (
        np.vstack(x_batches),
        np.concatenate(y_batches),
        np.concatenate(c_batches),
        used_order,
    )

def order_grid_data(grid_data, sample_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
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
        trail_data: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, GPy.models.GPRegression]:
    models: dict[str, GPy.models.GPRegression] = {}
    for class_name, (x_train, y_train) in trail_data.items():
        X = np.asarray(x_train, dtype=float)
        y = np.asarray(y_train, dtype=float)[:, None]
        kernel = GPy.kern.RBF(input_dim=FEATURE_COUNT, ARD=False)
        gp = GPy.models.GPRegression(X, y, kernel=kernel)
        gp.optimize(messages=True, max_iters=200)
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
        avg_gp_mu = float(np.nanmean(mu[mask]))
        avg_gp_sigma = float(np.nanmean(sigma[mask]))
        SIG = 0.01
        MU = 0.02
        avg_gp_mu = avg_gp_mu * SIG + MU# denormalize
        avg_gp_sigma = avg_gp_sigma * SIG #denormalize
        print(f"GP for '{class_name}' (mean(mu), mean(sigma)): ({avg_gp_mu:.5f}, {avg_gp_sigma:.5f}), count: {np.sum(mask)}")

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # BYPASS: Use precomputed class statistics for mu and sigma instead of GP predictions
        # _, class_mu, class_sigma = CLASS_STATS[str(class_name)]
        # class_mu = (class_mu - MU) / SIG
        # class_sigma = class_sigma / SIG

        # errors =  mu[mask] - class_mu
        # r = float(np.mean(np.abs(errors) <= 2.0 * class_sigma))
        # print(f"Bypassed GP: GP R value for class '{class_name}': {r:.5f}, count: {np.sum(mask)}")
        # mu[mask] = class_mu
        # sigma[mask] = class_sigma
        # print(f"Bypassed GP: Using class statistics for '{class_name}' (mu, sigma): ({class_mu:.5f}, {class_sigma:.5f}), count: {np.sum(mask)}")
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    print(f"GP reference statistics: mean(mu): {float(np.nanmean(mu)* SIG + MU):.5f}, mean(sigma): {float(np.nanmean(sigma)* SIG):.5f}")
    return mu, sigma


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
            metrics = et.EvalTools.compute_uncertainty_metrics(y_ref, preds, sigma_ref)
            for class_name in class_names:
                m = class_masks[class_name]
                class_metrics = et.EvalTools.compute_uncertainty_metrics(y_ref[m], preds[m], sigma_ref[m])
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
    parser.add_argument("--curve-step", type=int, default=10, help="Evaluate learning curve every N training samples")
    parser.add_argument("--plot", action="store_true", help="Show SOINN+ network plot")
    parser.add_argument("--plot-curve", action="store_true", help="Show R learning curve plots")
    return parser.parse_args()


def main() -> None:
    T1 = ["black", "cubeblack", "cubes", "cubeturf", "flat", "slope", "turf"]
    T2 = ["flat", "black", "turf", "cubes", "cubeblack", "cubeturf", "slope"]
    T3 = ["flat", "black", "cubeblack", "turf", "cubeturf", "cubes", "slope"]
    # T2 = ["cubes", "cubeblack", "black", "cubeturf", "flat", "slope", "turf"]
    # T1 = ["flat", "cubes", "cubeblack", "cubeturf", "slope", "black", "turf"]
    # T3 = ["turf", "black", "cubeblack", "flat", "cubeturf", "cubes", "slope"]
    trails = [(T1, 1), (T2, 1), (T3, 1), (T1, 3)]


    args = parse_args()
    if args.curve_step <= 0:
        raise ValueError("--curve-step must be a positive integer")

    run_stamp = datetime.now().strftime("%y%m%d%H%M")

    # Load and order grid and trail data
    grid_data = dl.load_grid_data(args.grid_dir)
    x_test, positions, class_test, grid_files = order_grid_data(
        grid_data,
        sample_count=args.grid_samples_per_file,
        seed=args.seed,
    )
    trail_data = dl.load_trail_data(args.trail_dir)
    print(f"Grid sample count -> {len(x_test)}")
    print(f"Grid sampling -> {args.grid_samples_per_file} cells per file (0 = all cells)")
    print("\n--------\n")

    start = datetime.now()
    gp_models = train_reference_gps(trail_data)
    stop = datetime.now()
    gp_training_time = (stop - start).total_seconds()
    print(f"Trained {len(gp_models)} per-class GP reference models in {gp_training_time:.2f} seconds")
    start = datetime.now()
    y_ref, sigma_ref = gp_reference_predict(gp_models, x_test, class_test)
    stop = datetime.now()
    gp_prediction_time = (stop - start).total_seconds()
    print(f"Predicted {len(x_test)} grid samples with GP reference models in {gp_prediction_time:.2f} seconds")
    print("\n--------\n\n")

    trail_number = 0
    for trail in trails:
        print("---------------------------------------------------")
        print(f"Running SOINN+ benchmark for T{trail_number}")
        print("---------------------------------------------------\n")
        trail_number += 1
        class_order, split_into = trail
        x_train, y_train, class_train, used_order = order_trail_data(trail_data, class_order, split_into)
        print(f"Using trail order: {used_order} with {split_into} splits per class")
        print(f"Training samples: {len(x_train)}")
        print(f"Feature shape: {x_train.shape}")
        print(f"Label shape: {y_train.shape}")
        print("\n")



        et.EvalTools.set_data(x_train, y_train, class_train, x_test, y_ref, class_test)
        soinn_model = SoinnEvalModel(shuffle=args.shuffle, seed=args.seed, use_fallback=args.fallback)
        et.EvalTools.set_model(soinn_model)
        start = datetime.now()
        et.EvalTools.train_model()
        stop = datetime.now()
        training_time = (stop - start).total_seconds()
        print(f"Trained SOINN+ model in {training_time:.2f} seconds")
        n_clusters = soinn_model.soinn.count_clusters()
        n_nodes = len(soinn_model.soinn.nodes)
        print(f"SOINN+ model has {n_clusters} clusters and {n_nodes} nodes after training")
        start = datetime.now()
        et.EvalTools.y_pred = et.EvalTools.predict_model()
        stop = datetime.now()
        prediction_time = (stop - start).total_seconds()
        print(f"Predicted {len(x_test)} grid samples in {prediction_time:.2f} seconds")

        print("\n")


        # soinn_curve, curve_steps, curve_r, curve_fallback, curve_r_by_class = train_soinn_learning_curve(
        #     x_train,
        #     y_train,
        #     x_test,
        #     class_test,
        #     y_ref,
        #     sigma_ref,
        #     eval_step=args.curve_step,
        #     use_fallback=args.fallback,
        # )

        # sections = _training_sections(class_train)
        # if args.shuffle:
        #     print("Shuffle enabled: section markers no longer represent contiguous terrain blocks")
        #     sections = []

        # print(f"SOINN+ nodes: {len(soinn_model.soinn.nodes) if soinn_model.soinn is not None else 0}")
        # print(f"Learning curve -> checkpoints: {len(curve_steps)}, eval_step: {args.curve_step}")
        # print("\n")

        # print(f"Learning curve final R for trail {trail_number}")
        # if curve_steps.size > 0:
        #     print(f"merged & {curve_r[-1]:.6f}\\\\")
        #     for class_name, r_curve in curve_r_by_class.items():
        #         print(f"{class_name} & {r_curve[-1]:.6f}\\\\")
        # print("\n")

        # # abstention = et.EvalTools.abstention_metrics()
        # regression_metrics = et.EvalTools.regression_metrics()
        # uncertainty_metrics = et.EvalTools.compute_uncertainty_metrics(y_ref, et.EvalTools.y_pred, sigma_ref)
        # avg_gp_sigma = float(np.nanmean(sigma_ref))

        # combined_metrics = {
        #     **regression_metrics,
        #     **uncertainty_metrics,
        #     # **abstention,
        #     "fallback_predictions_used": float(soinn_model.soinn.fallback_count),
        # }

        # et.EvalTools.print_metrics("Grid benchmark", combined_metrics)

        # curve_compare_output_path = RESULTS_DIRECTORY / f"{run_stamp}_rcurve_compare_T{trail_number}.png"
        # metrics_output_path = RESULTS_DIRECTORY / f"{run_stamp}_metrics_T{trail_number}.csv"

        # et.EvalTools.plot_learning_curve_comparison(
        #     curve_steps,
        #     overall_r=curve_r,
        #     class_r_curves=curve_r_by_class,
        #     output_path=curve_compare_output_path,
        #     sections=sections,
        #     enabled=args.plot_curve,
        # )
        # if trail_number == 1:
        #     et.EvalTools.save_class_grid_plots(
        #         run_stamp=run_stamp,
        #         positions=positions,
        #         class_labels=class_test,
        #         gp_predictions=y_ref,
        #         soinn_predictions=et.EvalTools.y_pred,
        #         output_dir=RESULTS_DIRECTORY,
        #         cost_mu=dl.COST_MU,
        #         cost_sigma=dl.COST_SIGMA,
        #     )

        # et.EvalTools.save_metrics_csv(
        #     output_path=metrics_output_path,
        #     metrics=combined_metrics,
        #     class_r_curves=curve_r_by_class,
        #     curve_r=curve_r,
        # )

        # if args.plot and soinn_curve.nodes:
        #     et.EvalTools.plot_network(soinn_curve, True)


if __name__ == "__main__":
    main()
