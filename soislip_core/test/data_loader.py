from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np


ClassNameFn = Callable[[Path], str]

TRAIL_DATA_DIRECTORY = Path("/workspaces/vscode_ros2_workspace/resources/pragr/trails/stab/")
GRID_DATA_DIRECTORY = Path("/workspaces/vscode_ros2_workspace/resources/pragr/grids/")

STRUCTURAL_MU = 0.5
STRUCTURAL_SIGMA = 0.2
APPEARANCE_MU = 0.0
APPEARANCE_SIGMA = 10.0
COST_MU = 0.02
COST_SIGMA = 0.01


def _default_class_name(path: Path) -> str:
    stem = path.name.split(".", maxsplit=1)[0]
    return stem.split("_", maxsplit=1)[0]


def normalize_features(features: np.ndarray) -> np.ndarray:
    means = np.array([STRUCTURAL_MU, STRUCTURAL_MU, STRUCTURAL_MU, APPEARANCE_MU, APPEARANCE_MU])
    sigmas = np.array([STRUCTURAL_SIGMA, STRUCTURAL_SIGMA, STRUCTURAL_SIGMA, APPEARANCE_SIGMA, APPEARANCE_SIGMA])
    return (features - means) / sigmas


def normalize_labels(labels: np.ndarray) -> np.ndarray:
    return (labels - COST_MU) / COST_SIGMA

def denormalize_labels(labels: np.ndarray) -> np.ndarray:
    return (labels * COST_SIGMA) + COST_MU


def _load_trail_file(trail_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    with trail_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                rows.append([float(value.strip()) for value in row])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric training rows found in {trail_path}")

    data = np.asarray(rows, dtype=float)
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    labels = data[:, -1]
    features = data[:, 1:-1]
    return normalize_features(features), normalize_labels(labels)


def _load_grid_file(grid_path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with grid_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 8:
                continue
            try:
                rows.append([float(value.strip()) for value in row[3:8]])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric grid rows found in {grid_path}")

    return normalize_features(np.asarray(rows, dtype=float))


def _load_grid_positions(grid_path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with grid_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 3:
                continue
            try:
                rows.append([float(value.strip()) for value in row[:3]])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric grid rows found in {grid_path}")

    return np.asarray(rows, dtype=float)


def _load_grid_cls_labels(cls_path: Path) -> np.ndarray:
    labels: list[str] = []
    with cls_path.open(newline="") as csv_file:
        for row in csv.reader(csv_file):
            if not row:
                continue
            first_col = row[0].strip()
            if first_col.lstrip().startswith("#"):
                continue
            labels.append(first_col)

    if not labels:
        raise ValueError(f"No class labels found in {cls_path}")

    return np.asarray(labels, dtype=object)


def _canonical_grid_cell_class(cell_label: str) -> str | None:
    normalized = str(cell_label).strip()
    if not normalized or normalized.lower() == "unknown":
        return None
    if normalized.endswith("_walk"):
        return normalized[: -len("_walk")]
    return normalized


def load_trail_data(
    data_directory: Path,
    class_name_fn: ClassNameFn | None = None,
    patterns: tuple[str, ...] = ("*.csv", "*.trail"),
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    class_name_fn = class_name_fn or _default_class_name
    loaded: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}

    for pattern in patterns:
        for path in sorted(data_directory.rglob(pattern)):
            if not path.is_file():
                continue
            features, labels = _load_trail_file(path)
            class_name = class_name_fn(path)
            feature_batches, label_batches = loaded.setdefault(class_name, ([], []))
            feature_batches.append(features)
            label_batches.append(labels)

    if not loaded:
        raise FileNotFoundError(f"No trail files found in {data_directory}")

    return {
        class_name: (np.vstack(feature_batches), np.concatenate(label_batches))
        for class_name, (feature_batches, label_batches) in loaded.items()
    }


def load_grid_data(
    data_directory: Path,
    class_name_fn: ClassNameFn | None = None,
    pattern: str = "*.grid",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    class_name_fn = class_name_fn or _default_class_name
    loaded: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}

    for path in sorted(data_directory.rglob(pattern)):
        if not path.is_file():
            continue
        features = _load_grid_file(path)
        positions = _load_grid_positions(path)
        labels = _load_grid_cls_labels(Path(f"{path}.cls"))
        if len(labels) != len(features):
            raise ValueError(
                f"Row count mismatch for {path}: grid has {len(features)} rows but cls has {len(labels)}"
            )
        class_name = class_name_fn(path)
        canonical_labels = np.array([_canonical_grid_cell_class(label) for label in labels], dtype=object)
        valid_mask = np.array([(label is not None) and (str(label) == class_name) for label in canonical_labels], dtype=bool)
        if not np.any(valid_mask):
            continue
        feature_batches, position_batches = loaded.setdefault(class_name, ([], []))
        feature_batches.append(features[valid_mask])
        position_batches.append(positions[valid_mask])

    if not loaded:
        raise FileNotFoundError(f"No grid files found in {data_directory}")

    return {
        class_name: (np.vstack(feature_batches), np.vstack(position_batches))
        for class_name, (feature_batches, position_batches) in loaded.items()
    }


def load_grid_details(
    data_directory: Path,
    class_name_fn: ClassNameFn | None = None,
    pattern: str = "*.grid",
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    class_name_fn = class_name_fn or _default_class_name
    loaded: dict[str, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]] = {}

    for path in sorted(data_directory.rglob(pattern)):
        if not path.is_file():
            continue
        features = _load_grid_file(path)
        positions = _load_grid_positions(path)
        labels = _load_grid_cls_labels(Path(f"{path}.cls"))
        if len(labels) != len(features):
            raise ValueError(
                f"Row count mismatch for {path}: grid has {len(features)} rows but cls has {len(labels)}"
            )

        class_name = class_name_fn(path)
        feature_batches, position_batches, label_batches = loaded.setdefault(class_name, ([], [], []))
        canonical_labels = np.array([_canonical_grid_cell_class(label) for label in labels], dtype=object)
        valid_mask = np.array([(label is not None) and (str(label) == class_name) for label in canonical_labels], dtype=bool)
        if not np.any(valid_mask):
            continue

        feature_batches.append(features[valid_mask])
        position_batches.append(positions[valid_mask])
        label_batches.append(canonical_labels[valid_mask])

    if not loaded:
        raise FileNotFoundError(f"No grid files found in {data_directory}")

    return {
        class_name: (np.vstack(feature_batches), np.vstack(position_batches), np.concatenate(label_batches))
        for class_name, (feature_batches, position_batches, label_batches) in loaded.items()
    }

def main() -> None:
    try:
        trail_data = load_trail_data(TRAIL_DATA_DIRECTORY)
        print(f"Loaded trail data from {TRAIL_DATA_DIRECTORY}")
        for class_name, (features, labels) in trail_data.items():
            denormalized_labels = denormalize_labels(labels)
            print(f"{class_name} & {features.shape[0]} & ({np.mean(denormalized_labels):.5f}, {np.std(denormalized_labels):.5f}) \\\\")

        all_labels = np.concatenate([labels for _, labels in trail_data.values()])
        denormalized_labels = denormalize_labels(all_labels)
        print("\\midrule")
        print(f"merged & {all_labels.shape[0]} & ({np.mean(denormalized_labels):.5f}, {np.std(denormalized_labels):.5f}) \\\\")
    except FileNotFoundError as e:
        print(e)

    try:
        grid_data = load_grid_data(GRID_DATA_DIRECTORY)
        feature_len = 0
        print(f"Loaded grid data from {GRID_DATA_DIRECTORY}")
        for class_name, (features, positions) in grid_data.items():
            print(f"{class_name} & {features.shape[0]} \\\\")
            feature_len += features.shape[0]
        print("\\midrule")
        print(f"merged & {feature_len} \\\\")
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
