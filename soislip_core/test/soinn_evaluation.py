from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soinn_py import SoinnPlus


DATA_DIRECTORY = Path("/workspaces/vscode_ros2_workspace/resources/pragr/trails/stab/")
FEATURE_COUNT = 5
LABEL_INDEX = 6

# Normalization constants from the paper.
STRUCTURAL_MU = 0.5
STRUCTURAL_SIGMA = 0.2
APPEARANCE_MU = 0.0
APPEARANCE_SIGMA = 10.0
COST_MU = 0.02
COST_SIGMA = 0.01
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def find_csv_files(data_directory: Path) -> list[Path]:
    return sorted(
        path
        for pattern in ("*.csv", "*.trail")
        for path in data_directory.rglob(pattern)
        if path.is_file()
    )


def load_csv_file(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []

    with csv_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if len(row) <= LABEL_INDEX:
                continue

            try:
                numeric_row = [float(value.strip()) for value in row[: LABEL_INDEX + 1]]
            except ValueError:
                continue

            rows.append(numeric_row)

    if not rows:
        raise ValueError(f"No numeric training rows found in {csv_path}")

    data = np.asarray(rows, dtype=float)
    timestamps = data[:, 0]
    features = data[:, 1 : 1 + FEATURE_COUNT]
    labels = data[:, LABEL_INDEX]

    # Keep each file in temporal order to match robot experience.
    order = np.argsort(timestamps, kind="stable")
    return features[order], labels[order]


def get_class_name(csv_path: Path) -> str:
    return csv_path.name.split(".", maxsplit=1)[0]


def _split_counts_per_file(
    n_samples: int,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[int, int, int]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    if n_samples < 3:
        raise ValueError("Each file needs at least 3 rows for train/validation/test chunks")

    n_train = max(1, int(n_samples * train_ratio))
    n_val = max(1, int(n_samples * val_ratio))
    n_test = n_samples - n_train - n_val

    while n_test < 1:
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError("Cannot create non-empty train/validation/test chunks for a file")
        n_test = n_samples - n_train - n_val

    return n_train, n_val, n_test


def _contiguous_true_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    true_indices = np.flatnonzero(mask)
    if true_indices.size == 0:
        return []

    breaks = np.where(np.diff(true_indices) > 1)[0]
    start_positions = np.concatenate(([0], breaks + 1))
    end_positions = np.concatenate((breaks, [len(true_indices) - 1]))

    intervals = []
    for start_pos, end_pos in zip(start_positions, end_positions):
        start = int(true_indices[start_pos])
        end = int(true_indices[end_pos]) + 1
        intervals.append((start, end))
    return intervals


def _sample_indices_with_chunking(
    available_mask: np.ndarray,
    target_count: int,
    num_chunks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_count == 0:
        return np.array([], dtype=int)

    available_count = int(np.sum(available_mask))
    if target_count > available_count:
        raise ValueError("Requested more samples than available when building split")

    if num_chunks <= 0:
        candidates = np.flatnonzero(available_mask)
        selected = np.sort(rng.choice(candidates, size=target_count, replace=False))
        return selected.astype(int)

    chunks = min(num_chunks, target_count)
    lengths = np.full(chunks, target_count // chunks, dtype=int)
    lengths[: target_count % chunks] += 1

    selected_parts: list[np.ndarray] = []

    for length in lengths:
        intervals = [interval for interval in _contiguous_true_intervals(available_mask) if interval[1] - interval[0] >= length]

        if not intervals:
            break

        start_options = np.array([end - start - length + 1 for start, end in intervals], dtype=int)
        interval_idx = int(rng.choice(len(intervals), p=start_options / np.sum(start_options)))
        interval_start, interval_end = intervals[interval_idx]
        segment_start = int(rng.integers(interval_start, interval_end - length + 1))
        segment = np.arange(segment_start, segment_start + length, dtype=int)

        available_mask[segment] = False
        selected_parts.append(segment)

    selected_count = int(sum(len(part) for part in selected_parts))
    remaining = target_count - selected_count

    if remaining > 0:
        candidates = np.flatnonzero(available_mask)
        extra = np.sort(rng.choice(candidates, size=remaining, replace=False)).astype(int)
        available_mask[extra] = False
        selected_parts.append(extra)

    if not selected_parts:
        return np.array([], dtype=int)

    return np.sort(np.concatenate(selected_parts)).astype(int)


def load_training_data(
    data_directory: Path,
    seed: int,
    val_chunks: int,
    test_chunks: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], list[Path], dict[str, float]]:
    csv_files = find_csv_files(data_directory)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_directory}")

    rng = np.random.default_rng(seed)

    train_features_batches = []
    train_labels_batches = []
    train_class_batches = []
    val_features_batches = []
    val_labels_batches = []
    val_class_batches = []
    test_features_batches = []
    test_labels_batches = []
    test_class_batches = []
    class_label_batches: dict[str, list[np.ndarray]] = {}

    for csv_path in csv_files:
        features, labels = load_csv_file(csv_path)
        class_name = get_class_name(csv_path)
        classes = np.full(len(features), class_name, dtype=object)
        class_label_batches.setdefault(class_name, []).append(labels)
        n_train, n_val, _ = _split_counts_per_file(len(features))
        n_test = len(features) - n_train - n_val

        available_mask = np.ones(len(features), dtype=bool)
        val_idx = _sample_indices_with_chunking(available_mask, n_val, val_chunks, rng)
        test_idx = _sample_indices_with_chunking(available_mask, n_test, test_chunks, rng)
        train_idx = np.flatnonzero(available_mask)

        train_features_batches.append(features[train_idx])
        train_labels_batches.append(labels[train_idx])
        train_class_batches.append(classes[train_idx])
        val_features_batches.append(features[val_idx])
        val_labels_batches.append(labels[val_idx])
        val_class_batches.append(classes[val_idx])
        test_features_batches.append(features[test_idx])
        test_labels_batches.append(labels[test_idx])
        test_class_batches.append(classes[test_idx])

    split_data = {
        "train": (
            np.vstack(train_features_batches),
            np.concatenate(train_labels_batches),
            np.concatenate(train_class_batches),
        ),
        "val": (
            np.vstack(val_features_batches),
            np.concatenate(val_labels_batches),
            np.concatenate(val_class_batches),
        ),
        "test": (
            np.vstack(test_features_batches),
            np.concatenate(test_labels_batches),
            np.concatenate(test_class_batches),
        ),
    }

    # Global per-class thresholds for class_variance_cr from the full dataset (before splitting).
    class_thresholds = {
        class_name: 2.0 * float(np.std(np.concatenate(label_parts)))
        for class_name, label_parts in class_label_batches.items()
    }

    return split_data, csv_files, class_thresholds


def normalize_features_labels(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.array([
        STRUCTURAL_MU,
        STRUCTURAL_MU,
        STRUCTURAL_MU,
        APPEARANCE_MU,
        APPEARANCE_MU,
    ])
    sigmas = np.array([
        STRUCTURAL_SIGMA,
        STRUCTURAL_SIGMA,
        STRUCTURAL_SIGMA,
        APPEARANCE_SIGMA,
        APPEARANCE_SIGMA,
    ])

    normalized_features = (features - means) / sigmas
    normalized_labels = (labels - COST_MU) / COST_SIGMA
    return normalized_features, normalized_labels


def denormalize_labels(labels: np.ndarray) -> np.ndarray:
    return labels * COST_SIGMA + COST_MU


def train_soinn(features: np.ndarray, labels: np.ndarray, shuffle: bool, seed: int) -> SoinnPlus:
    soinn = SoinnPlus(dim=FEATURE_COUNT)
    indices = np.arange(len(features))

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for index in indices:
        soinn.input_signal(features[index], label=labels[index])

    return soinn


def predict_soinn(soinn: SoinnPlus, features: np.ndarray, use_fallback: bool) -> tuple[np.ndarray, int]:
    predictions = np.full(len(features), np.nan)
    fallback_count = 0

    for i, feature in enumerate(features):
        pred_mean, _ = soinn.inference(feature, label_clusters=True)

        if pred_mean is None and use_fallback:
            fallback_count += 1
            winner, _ = soinn.find_nearest_nodes(1, soinn.check_signal(feature))
            winner_idx = int(winner[0])
            pred_mean = soinn.predictions[winner_idx][0]
            if pred_mean is None:
                pred_mean = soinn.labels[winner_idx]

        if pred_mean is not None:
            predictions[i] = float(pred_mean)

    return predictions, fallback_count


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
    class_thresholds: dict[str, float],
) -> dict[str, float]:
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    valid_true = y_true[valid_mask]
    valid_pred = y_pred[valid_mask]
    valid_classes = class_names[valid_mask]

    if valid_true.size == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "nrmse": float("nan"),
            "r2": float("nan"),
            "class_variance_cr": float("nan"),
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

    # Per-sample threshold from globally precomputed class statistics.
    per_sample_thresholds = np.array([class_thresholds[str(class_name)] for class_name in valid_classes])
    class_variance_cr = float(np.mean(np.abs(errors) < per_sample_thresholds))

    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
        "r2": r2,
        "class_variance_cr": class_variance_cr,
        "used_samples": float(valid_true.size),
    }


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


def print_metrics(split_name: str, metrics: dict[str, float], abstention: dict[str, float], fallback_count: int) -> None:
    print(f"{split_name} metrics:")
    print(
        "  samples: "
        f"{int(abstention['total_samples'])}, "
        f"confident: {int(abstention['confident_samples'])}, "
        f"abstained: {int(abstention['abstained_samples'])}, "
        f"coverage: {abstention['coverage']:.6f}, "
        f"abstain_rate: {abstention['abstain_rate']:.6f}"
    )
    print(f"  fallback predictions used: {fallback_count}")
    print(f"  used for regression metrics: {int(metrics['used_samples'])}")
    print(f"  RMSE:  {metrics['rmse']:.6f}")
    print(f"  MAE:   {metrics['mae']:.6f}")
    print(f"  NRMSE: {metrics['nrmse']:.6f}")
    print(f"  R2:    {metrics['r2']:.6f}")
    print(f"  R':    {metrics['class_variance_cr']:.6f}")


def show_training_summary(csv_files: list[Path], features: np.ndarray, labels: np.ndarray, soinn: SoinnPlus) -> None:
    print(f"Loaded {len(csv_files)} CSV files")
    print(f"Training samples: {len(features)}")
    print(f"Feature shape: {features.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Trained nodes: {len(soinn.nodes)}")


def maybe_plot(soinn: SoinnPlus, enabled: bool) -> None:
    if not enabled or not soinn.nodes:
        return

    soinn.show(save=False)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SOINN+ from combined CSV trail data.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY, help="Directory containing CSV files")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle training samples instead of time-ordered training")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Keep file order during training")
    parser.set_defaults(shuffle=False)
    parser.add_argument("--fallback", action="store_true", help="Enable nearest-node fallback when SOINN abstains")
    parser.add_argument("--no-fallback", dest="fallback", action="store_false", help="Disable fallback predictions")
    parser.set_defaults(fallback=False)
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when shuffling")
    parser.add_argument(
        "--val-chunks",
        type=int,
        default=3,
        help="Validation split chunks per file (0 = random single samples)",
    )
    parser.add_argument(
        "--test-chunks",
        type=int,
        default=3,
        help="Test split chunks per file (0 = random single samples)",
    )
    parser.add_argument("--plot", action="store_true", help="Show the optional SOINN verification plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_data_raw, csv_files, class_thresholds = load_training_data(
        args.data_dir,
        seed=args.seed,
        val_chunks=args.val_chunks,
        test_chunks=args.test_chunks,
    )

    train_features_raw, train_labels_raw, train_classes = split_data_raw["train"]
    val_features_raw, val_labels_raw, val_classes = split_data_raw["val"]
    test_features_raw, test_labels_raw, test_classes = split_data_raw["test"]

    train_features, train_labels = normalize_features_labels(train_features_raw, train_labels_raw)
    val_features, val_labels = normalize_features_labels(val_features_raw, val_labels_raw)
    test_features, test_labels = normalize_features_labels(test_features_raw, test_labels_raw)

    soinn = train_soinn(train_features, train_labels, shuffle=args.shuffle, seed=args.seed)
    show_training_summary(csv_files, train_features, train_labels, soinn)
    print("Split strategy: per-file randomized chunk split with no overlap")
    print(
        f"Chunk settings -> val_chunks: {args.val_chunks}, "
        f"test_chunks: {args.test_chunks} (0 = random single samples)"
    )
    print(f"Prediction mode -> fallback: {args.fallback}")
    print(
        f"Split sizes -> train: {len(train_features)}, val: {len(val_features)}, "
        f"test: {len(test_features)}"
    )

    for split_name, split_features, split_labels, split_classes in (
        ("Train", train_features, train_labels, train_classes),
        ("Validation", val_features, val_labels, val_classes),
        ("Test", test_features, test_labels, test_classes),
    ):
        pred_norm, fallback_count = predict_soinn(soinn, split_features, use_fallback=args.fallback)
        true_cost = denormalize_labels(split_labels)
        pred_cost = denormalize_labels(pred_norm)
        abstention = compute_abstention_metrics(pred_cost)
        metrics = compute_metrics(true_cost, pred_cost, split_classes, class_thresholds)
        print_metrics(split_name, metrics, abstention, fallback_count)

    maybe_plot(soinn, args.plot)


if __name__ == "__main__":
    main()