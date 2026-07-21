from __future__ import annotations

import argparse
from pathlib import Path

import GPy
import numpy as np

import data_loader as dl
import eval_tools as et
from soinn_py import SoinnPlus


DATA_DIRECTORY = dl.TRAIL_DATA_DIRECTORY
GRID_DATA_DIRECTORY = dl.GRID_DATA_DIRECTORY
FEATURE_COUNT = 5
TEST_RATIO = 0.2


class SoinnEvalModel(et.EvalModelBase):
    def __init__(
        self,
        shuffle: bool,
        seed: int,
        use_fallback: bool,
        unsupervised_features: np.ndarray | None = None,
    ) -> None:
        self.shuffle = shuffle
        self.seed = seed
        self.use_fallback = use_fallback
        self.unsupervised_features = unsupervised_features
        self.soinn: SoinnPlus | None = None
        self.fallback_count = 0

    def train(self, x_train: np.ndarray, y_train: np.ndarray, class_train: np.ndarray | None = None) -> None:
        soinn = SoinnPlus(dim=FEATURE_COUNT)
        indices = np.arange(len(x_train))
        if self.shuffle:
            np.random.default_rng(self.seed).shuffle(indices)
        for i in indices:
            soinn.input_signal(x_train[i], label=y_train[i])

        if self.unsupervised_features is not None:
            unsupervised_indices = np.arange(len(self.unsupervised_features))
            if self.shuffle:
                np.random.default_rng(self.seed).shuffle(unsupervised_indices)
            for i in unsupervised_indices:
                soinn.input_signal(self.unsupervised_features[i])

        self.soinn = soinn

    def predict(self, x_test: np.ndarray, class_test: np.ndarray | None = None) -> np.ndarray:
        if self.soinn is None:
            raise ValueError("Model is not trained")

        predictions = np.full(len(x_test), np.nan, dtype=float)
        self.fallback_count = 0

        for i, feature in enumerate(x_test):
            pred_mean, _ = self.soinn.inference(feature, label_clusters=True)
            if pred_mean is None and self.use_fallback:
                self.fallback_count += 1
                winner, _ = self.soinn.find_nearest_nodes(1, self.soinn._check_signal(feature))
                winner_idx = int(winner[0])
                pred_mean = self.soinn.predictions[winner_idx][0]
                if pred_mean is None:
                    pred_mean = self.soinn.labels[winner_idx]
            if pred_mean is not None:
                predictions[i] = float(pred_mean)

        return predictions


class GPEvalModel(et.EvalModelBase):
    def __init__(self, max_iters: int = 1000) -> None:
        self.max_iters = max_iters
        self.models: dict[str, GPy.models.GPRegression] = {}
        self.missing_class_count = 0

    def train(self, x_train: np.ndarray, y_train: np.ndarray, class_train: np.ndarray | None = None) -> None:
        if class_train is None:
            raise ValueError("class_train is required for GPEvalModel")

        self.models = {}
        for class_name in np.unique(class_train):
            mask = class_train == class_name
            X = np.asarray(x_train[mask], dtype=float)
            y = np.asarray(y_train[mask], dtype=float)[:, None]
            kernel = GPy.kern.RBF(input_dim=FEATURE_COUNT, ARD=False)
            gp = GPy.models.GPRegression(X, y, kernel=kernel)
            gp.optimize_restarts(num_restarts=3, robust=True, verbose=False, messages=False, max_iters=self.max_iters)
            self.models[str(class_name)] = gp

    def predict(self, x_test: np.ndarray, class_test: np.ndarray | None = None) -> np.ndarray:
        if class_test is None:
            raise ValueError("class_test is required for GPEvalModel")

        predictions = np.full(len(x_test), np.nan, dtype=float)
        self.missing_class_count = 0

        for class_name in np.unique(class_test):
            mask = class_test == class_name
            gp = self.models.get(str(class_name))
            if gp is None:
                self.missing_class_count += int(np.sum(mask))
                continue
            mu, _ = gp.predict(np.asarray(x_test[mask], dtype=float))
            predictions[mask] = mu[:, 0]

        return predictions


def load_unsupervised_grid_features(data_directory: Path) -> np.ndarray:
    grid_data = dl.load_grid_data(data_directory)
    feature_batches = [features for features, _ in grid_data.values()]
    return np.vstack(feature_batches)


def _split_counts_per_file(n_samples: int) -> tuple[int, int]:
    if n_samples < 2:
        raise ValueError("Each file needs at least 2 rows for train/test chunks")

    n_train = max(1, int(n_samples * (1.0 - TEST_RATIO)))
    n_test = n_samples - n_train
    if n_test < 1:
        n_train -= 1
        n_test = n_samples - n_train
    return n_train, n_test


def _contiguous_true_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    true_indices = np.flatnonzero(mask)
    if true_indices.size == 0:
        return []
    breaks = np.where(np.diff(true_indices) > 1)[0]
    start_positions = np.concatenate(([0], breaks + 1))
    end_positions = np.concatenate((breaks, [len(true_indices) - 1]))
    return [(int(true_indices[s]), int(true_indices[e]) + 1) for s, e in zip(start_positions, end_positions)]


def _sample_indices_with_chunking(
    available_mask: np.ndarray,
    target_count: int,
    num_chunks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_count == 0:
        return np.array([], dtype=int)
    if target_count > int(np.sum(available_mask)):
        raise ValueError("Requested more samples than available when building split")

    if num_chunks <= 0:
        candidates = np.flatnonzero(available_mask)
        return np.sort(rng.choice(candidates, size=target_count, replace=False)).astype(int)

    chunks = min(num_chunks, target_count)
    lengths = np.full(chunks, target_count // chunks, dtype=int)
    lengths[: target_count % chunks] += 1
    selected_parts: list[np.ndarray] = []

    for length in lengths:
        intervals = [i for i in _contiguous_true_intervals(available_mask) if i[1] - i[0] >= length]
        if not intervals:
            break
        start_options = np.array([end - start - length + 1 for start, end in intervals], dtype=int)
        interval_idx = int(rng.choice(len(intervals), p=start_options / np.sum(start_options)))
        interval_start, interval_end = intervals[interval_idx]
        segment_start = int(rng.integers(interval_start, interval_end - length + 1))
        segment = np.arange(segment_start, segment_start + length, dtype=int)
        available_mask[segment] = False
        selected_parts.append(segment)

    remaining = target_count - int(sum(len(part) for part in selected_parts))
    if remaining > 0:
        candidates = np.flatnonzero(available_mask)
        extra = np.sort(rng.choice(candidates, size=remaining, replace=False)).astype(int)
        available_mask[extra] = False
        selected_parts.append(extra)

    return np.sort(np.concatenate(selected_parts)).astype(int) if selected_parts else np.array([], dtype=int)


def load_training_data(
    data_directory: Path,
    seed: int,
    test_chunks: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], list[str]]:
    trail_data = dl.load_trail_data(data_directory)

    rng = np.random.default_rng(seed)
    train_x, train_y, train_c = [], [], []
    test_x, test_y, test_c = [], [], []

    for class_name, (features, labels) in sorted(trail_data.items()):
        classes = np.full(len(features), class_name, dtype=object)
        _, n_test = _split_counts_per_file(len(features))

        available_mask = np.ones(len(features), dtype=bool)
        test_idx = _sample_indices_with_chunking(available_mask, n_test, test_chunks, rng)
        train_idx = np.flatnonzero(available_mask)

        train_x.append(features[train_idx])
        train_y.append(labels[train_idx])
        train_c.append(classes[train_idx])
        test_x.append(features[test_idx])
        test_y.append(labels[test_idx])
        test_c.append(classes[test_idx])

    return {
        "train": (np.vstack(train_x), np.concatenate(train_y), np.concatenate(train_c)),
        "test": (np.vstack(test_x), np.concatenate(test_y), np.concatenate(test_c)),
    }, list(trail_data.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SOINN+ from combined CSV trail data.")
    parser.add_argument("--model", choices=("soinn", "gp"), default="soinn", help="Model to evaluate")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY, help="Directory containing CSV files")
    parser.add_argument("--grid-dir", type=Path, default=GRID_DATA_DIRECTORY, help="Directory containing grid files")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle training samples instead of time-ordered training")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Keep file order during training")
    parser.set_defaults(shuffle=False)
    parser.add_argument("--fallback", action="store_true", help="Enable nearest-node fallback when SOINN abstains")
    parser.add_argument("--no-fallback", dest="fallback", action="store_false", help="Disable fallback predictions")
    parser.set_defaults(fallback=False)
    parser.add_argument("--unsupervised-grid-train", action="store_true", help="Additionally train SOINN unsupervised on grid features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when shuffling")
    parser.add_argument("--test-chunks", type=int, default=5, help="Test split chunks per file (0 = random single samples)")
    parser.add_argument("--gp-max-iters", type=int, default=1000, help="Max optimization iterations per class GP")
    parser.add_argument("--plot", action="store_true", help="Show the optional SOINN verification plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_data_raw, trail_classes = load_training_data(args.data_dir, seed=args.seed, test_chunks=args.test_chunks)

    x_train, y_train, class_train = split_data_raw["train"]
    x_test, y_test, class_test = split_data_raw["test"]

    unsupervised_grid_features = None
    if args.model == "soinn" and args.unsupervised_grid_train:
        unsupervised_grid_features = load_unsupervised_grid_features(args.grid_dir)

    if args.model == "gp":
        model = GPEvalModel(max_iters=args.gp_max_iters)
    else:
        model = SoinnEvalModel(
            shuffle=args.shuffle,
            seed=args.seed,
            use_fallback=args.fallback,
            unsupervised_features=unsupervised_grid_features,
        )

    et.EvalTools.set_data(x_train, y_train, class_train, x_test, y_test, class_test)
    et.EvalTools.set_model(model)
    et.EvalTools.train_model()

    print(f"Loaded classes: {trail_classes}")
    print(f"Training samples: {len(x_train)}")
    print(f"Feature shape: {x_train.shape}")
    print(f"Label shape: {y_train.shape}")
    if isinstance(model, SoinnEvalModel):
        print(f"Trained nodes: {len(model.soinn.nodes) if model.soinn is not None else 0}")
    else:
        print(f"Trained reference GPs: {len(model.models)}")
    print("Split strategy: per-file randomized chunk split with no overlap")
    print(f"Model: {args.model}")
    print(f"Chunk settings -> test_chunks: {args.test_chunks} (0 = random single samples)")
    if args.model == "soinn":
        print(f"Prediction mode -> fallback: {args.fallback}")
        print(f"Unsupervised grid training: {args.unsupervised_grid_train}")
        if unsupervised_grid_features is not None:
            print(f"Unsupervised grid samples: {len(unsupervised_grid_features)}")
    else:
        print(f"GP optimization -> max_iters: {args.gp_max_iters}")
    print(f"Split sizes -> train: {len(x_train)}, test: {len(x_test)}")

    for split_name, x_split, y_split, class_split in (
        ("Train", x_train, y_train, class_train),
        ("Test", x_test, y_test, class_test),
    ):
        et.EvalTools.x_test = x_split
        et.EvalTools.y_test = y_split
        et.EvalTools.class_test = class_split
        et.EvalTools.y_pred = et.EvalTools.predict_model()

        abstention = et.EvalTools.abstention_metrics()
        metrics = et.EvalTools.regression_metrics()
        aux_count = model.fallback_count if isinstance(model, SoinnEvalModel) else model.missing_class_count
        aux_label = "fallback predictions used" if isinstance(model, SoinnEvalModel) else "missing-class predictions"
        et.EvalTools.print_basic_metrics(
            split_name,
            metrics,
            abstention,
            aux_count=aux_count,
            aux_label=aux_label,
        )

    if isinstance(model, SoinnEvalModel) and model.soinn is not None:
        et.EvalTools.plot_network(model.soinn, args.plot)


if __name__ == "__main__":
    main()
