from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import GPy

from soinn_py import SoinnPlus


TRAIL_DATA_DIRECTORY = Path("/workspaces/vscode_ros2_workspace/resources/pragr/trails/stab/")
GRID_DATA_DIRECTORY = Path("/workspaces/vscode_ros2_workspace/resources/pragr/grids/")
RESULTS_DIRECTORY = Path("/workspaces/vscode_ros2_workspace/resources/results/evalutation/")
FEATURE_COUNT = 5
LABEL_INDEX = 6

# Normalization constants from the pragr2019benchmarking for feature and label normalization.
STRUCTURAL_MU = 0.5
STRUCTURAL_SIGMA = 0.2
APPEARANCE_MU = 0.0
APPEARANCE_SIGMA = 10.0
COST_MU = 0.02
COST_SIGMA = 0.01


def find_data_files(data_directory: Path, patterns: tuple[str, ...]) -> list[Path]:
	return sorted(
		path
		for pattern in patterns
		for path in data_directory.rglob(pattern)
		if path.is_file()
	)


def load_trail_file(trail_path: Path) -> tuple[np.ndarray, np.ndarray]:
	rows: list[list[float]] = []

	with trail_path.open(newline="") as csv_file:
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
		raise ValueError(f"No numeric training rows found in {trail_path}")

	data = np.asarray(rows, dtype=float)
	timestamps = data[:, 0]
	features = data[:, 1 : 1 + FEATURE_COUNT]
	labels = data[:, LABEL_INDEX]

	order = np.argsort(timestamps, kind="stable")
	return features[order], labels[order]


def load_grid_file(grid_path: Path) -> tuple[np.ndarray, np.ndarray]:
	rows: list[list[float]] = []

	with grid_path.open(newline="") as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
			if not row or row[0].lstrip().startswith("#"):
				continue

			if len(row) < 8:
				continue

			try:
				numeric_row = [float(value.strip()) for value in row[:8]]
			except ValueError:
				continue

			rows.append(numeric_row)

	if not rows:
		raise ValueError(f"No numeric grid rows found in {grid_path}")

	data = np.asarray(rows, dtype=float)
	positions = data[:, :3]
	features = data[:, 3:8]
	return features, positions


def load_grid_cls_file(cls_path: Path) -> np.ndarray:
	labels: list[str] = []

	with cls_path.open(newline="") as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
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
	if not normalized:
		return None
	if normalized.lower() == "unknown":
		return None
	if normalized.endswith("_walk"):
		return normalized[: -len("_walk")]
	return normalized


def get_class_name(data_path: Path) -> str:
	name = data_path.name.split(".", maxsplit=1)[0]
	return name.split("_", maxsplit=1)[0]


def normalize_features(features: np.ndarray) -> np.ndarray:
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

	return (features - means) / sigmas


def normalize_labels(labels: np.ndarray) -> np.ndarray:
	return (labels - COST_MU) / COST_SIGMA


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


def _ordered_training_data(
	features: np.ndarray,
	labels: np.ndarray,
	classes: np.ndarray,
	shuffle: bool,
	seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	indices = np.arange(len(features))
	if shuffle:
		rng = np.random.default_rng(seed)
		rng.shuffle(indices)

	return features[indices], labels[indices], classes[indices]


def _training_sections(classes: np.ndarray) -> list[tuple[int, str]]:
	if len(classes) == 0:
		return []

	sections = [(1, str(classes[0]))]
	for i in range(1, len(classes)):
		if classes[i] != classes[i - 1]:
			sections.append((i + 1, str(classes[i])))

	return sections


def train_soinn_learning_curve(
	train_features: np.ndarray,
	train_labels: np.ndarray,
	grid_features: np.ndarray,
	grid_classes: np.ndarray,
	mu_ref: np.ndarray,
	sigma_ref: np.ndarray,
	eval_step: int,
	use_fallback: bool,
) -> tuple[SoinnPlus, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
	soinn = SoinnPlus(dim=FEATURE_COUNT)
	steps: list[int] = []
	r_values: list[float] = []
	fallback_values: list[int] = []
	class_names = [str(name) for name in sorted(np.unique(grid_classes))]
	class_masks = {name: (grid_classes == name) for name in class_names}
	class_r_history: dict[str, list[float]] = {name: [] for name in class_names}

	for i, (feature, label) in enumerate(zip(train_features, train_labels), start=1):
		soinn.input_signal(feature, label=label)

		if i % eval_step == 0 or i == len(train_features):
			predictions, fallback_count = predict_soinn(soinn, grid_features, use_fallback=use_fallback)
			metrics = compute_reference_metrics(predictions, mu_ref, sigma_ref)
			for class_name in class_names:
				mask = class_masks[class_name]
				class_metrics = compute_reference_metrics(predictions[mask], mu_ref[mask], sigma_ref[mask])
				class_r_history[class_name].append(class_metrics["r"])
			steps.append(i)
			r_values.append(metrics["r"])
			fallback_values.append(fallback_count)

	class_r_curves = {
		name: np.asarray(values, dtype=float)
		for name, values in class_r_history.items()
	}

	return (
		soinn,
		np.asarray(steps, dtype=int),
		np.asarray(r_values, dtype=float),
		np.asarray(fallback_values, dtype=int),
		class_r_curves,
	)


def train_reference_gps(
	train_features: np.ndarray,
	train_labels: np.ndarray,
	train_classes: np.ndarray,
) -> dict[str, GPy.models.GPRegression]:
	reference_models: dict[str, GPy.models.GPRegression] = {}

	for class_name in np.unique(train_classes):
		mask = train_classes == class_name
		X = np.asarray(train_features[mask], dtype=float)
		y = np.asarray(train_labels[mask], dtype=float)[:, None]

		kernel = GPy.kern.RBF(input_dim=FEATURE_COUNT, ARD=True)
		gp = GPy.models.GPRegression(X, y, kernel=kernel)
		gp.optimize(messages=False, max_iters=1000)
		reference_models[str(class_name)] = gp

	return reference_models


def reference_predict(
	reference_models: dict[str, GPy.models.GPRegression],
	features: np.ndarray,
	class_labels: np.ndarray | None = None,
	selection_mode: str = "min-variance",
) -> tuple[np.ndarray, np.ndarray]:
	all_mu = []
	all_sigma = []
	model_names = []

	for class_name, gp in reference_models.items():
		model_names.append(class_name)
		mu, var = gp.predict(np.asarray(features, dtype=float))
		all_mu.append(mu[:, 0])
		all_sigma.append(np.sqrt(np.maximum(var[:, 0], 0.0)))

	if not all_mu:
		raise ValueError("No reference GP models available for prediction")

	all_mu = np.stack(all_mu, axis=0)
	all_sigma = np.stack(all_sigma, axis=0)

	if selection_mode == "same-class":
		if class_labels is None:
			raise ValueError("class_labels are required when selection_mode is 'same-class'")
		if len(class_labels) != features.shape[0]:
			raise ValueError("class_labels length must match number of feature samples")

		class_to_model_idx = {name: idx for idx, name in enumerate(model_names)}
		best_idx = np.argmin(all_sigma, axis=0)
		for i, class_name in enumerate(class_labels):
			model_idx = class_to_model_idx.get(str(class_name))
			if model_idx is not None:
				best_idx[i] = model_idx
	else:
		best_idx = np.argmin(all_sigma, axis=0)

	sample_indices = np.arange(features.shape[0])
	mu_ref = all_mu[best_idx, sample_indices]
	sigma_ref = all_sigma[best_idx, sample_indices]

	return mu_ref, sigma_ref


def _sample_indices(n_samples: int, sample_count: int, rng: np.random.Generator) -> np.ndarray:
	if sample_count <= 0 or sample_count >= n_samples:
		return np.arange(n_samples, dtype=int)

	return np.sort(rng.choice(n_samples, size=sample_count, replace=False)).astype(int)


def load_training_data(
	data_directory: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path]]:
	trail_files = find_data_files(data_directory, ("*.trail",))
	if not trail_files:
		raise FileNotFoundError(f"No trail files found in {data_directory}")

	feature_batches = []
	label_batches = []
	class_batches = []

	for trail_path in trail_files:
		features, labels = load_trail_file(trail_path)
		class_name = get_class_name(trail_path)
		feature_batches.append(normalize_features(features))
		label_batches.append(normalize_labels(labels))
		class_batches.append(np.full(len(features), class_name, dtype=object))

	return (
		np.vstack(feature_batches),
		np.concatenate(label_batches),
		np.concatenate(class_batches),
		trail_files,
	)


def load_grid_data(data_directory: Path, sample_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path]]:
	grid_files = find_data_files(data_directory, ("*.grid",))
	if not grid_files:
		raise FileNotFoundError(f"No grid files found in {data_directory}")

	rng = np.random.default_rng(seed)
	feature_batches = []
	position_batches = []
	class_batches = []

	for grid_path in grid_files:
		features, positions = load_grid_file(grid_path)
		class_name = get_class_name(grid_path)
		cls_path = Path(f"{grid_path}.cls")
		if not cls_path.is_file():
			raise FileNotFoundError(f"Missing class label file for {grid_path}: {cls_path}")
		cell_labels = load_grid_cls_file(cls_path)

		if len(cell_labels) != len(features):
			raise ValueError(
				f"Row count mismatch for {grid_path}: grid has {len(features)} rows but cls has {len(cell_labels)}"
			)

		canonical_labels = np.array([_canonical_grid_cell_class(label) for label in cell_labels], dtype=object)
		valid_mask = np.array(
			[(label is not None) and (str(label) == class_name) for label in canonical_labels],
			dtype=bool,
		)

		if not np.any(valid_mask):
			continue

		filtered_features = features[valid_mask]
		filtered_positions = positions[valid_mask]
		filtered_labels = canonical_labels[valid_mask]
		selected = _sample_indices(len(filtered_features), sample_count, rng)

		feature_batches.append(normalize_features(filtered_features[selected]))
		position_batches.append(filtered_positions[selected])
		class_batches.append(filtered_labels[selected])

	if not feature_batches:
		raise ValueError("No valid grid samples found after applying class labels and unknown filtering")

	return (
		np.vstack(feature_batches),
		np.vstack(position_batches),
		np.concatenate(class_batches),
		grid_files,
	)


def compute_reference_metrics(
	y_pred: np.ndarray,
	mu_ref: np.ndarray,
	sigma_ref: np.ndarray,
) -> dict[str, float]:
	valid_mask = np.isfinite(y_pred) & np.isfinite(mu_ref) & np.isfinite(sigma_ref)
	valid_pred = y_pred[valid_mask]
	valid_mu = mu_ref[valid_mask]
	valid_sigma = sigma_ref[valid_mask]

	if valid_pred.size == 0:
		return {
			"rmse": float("nan"),
			"mae": float("nan"),
			"nrmse": float("nan"),
			"r2": float("nan"),
			"r": float("nan"),
			"used_samples": 0.0,
		}

	errors = valid_pred - valid_mu
	mse = np.mean(errors ** 2)
	rmse = float(np.sqrt(mse))
	mae = float(np.mean(np.abs(errors)))

	y_range = float(np.max(valid_mu) - np.min(valid_mu))
	nrmse = float(rmse / y_range) if y_range > 0.0 else float("nan")

	ss_res = float(np.sum(errors ** 2))
	ss_tot = float(np.sum((valid_mu - np.mean(valid_mu)) ** 2))
	r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

	r = float(np.mean(np.abs(errors) <= 2.0 * valid_sigma))

	return {
		"rmse": rmse,
		"mae": mae,
		"nrmse": nrmse,
		"r2": r2,
		"r": r,
		"used_samples": float(valid_pred.size),
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
	print(f"  R:     {metrics['r']:.6f}")


def show_training_summary(
	trail_files: list[Path],
	grid_files: list[Path],
	train_features: np.ndarray,
	train_labels: np.ndarray,
	soinn: SoinnPlus,
	reference_models: dict[str, GPy.models.GPRegression],
) -> None:
	print(f"Loaded {len(trail_files)} trail files")
	print(f"Loaded {len(grid_files)} grid files")
	print(f"Training samples: {len(train_features)}")
	print(f"Feature shape: {train_features.shape}")
	print(f"Label shape: {train_labels.shape}")
	print(f"Trained SOINN+ nodes: {len(soinn.nodes)}")
	print(f"Trained reference GPs: {len(reference_models)}")


def predict_soinn(soinn: SoinnPlus, features: np.ndarray, use_fallback: bool) -> tuple[np.ndarray, int]:
	predictions = np.full(len(features), np.nan)
	fallback_count = 0

	for i, feature in enumerate(features):
		pred_mean, _ = soinn.inference(feature, label_clusters=(i==0))

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


def plot_network(soinn: SoinnPlus, enabled: bool) -> None:
	if not enabled or not soinn.nodes:
		return

	soinn.show(save=False)
	plt.show()


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


def _safe_name(name: str) -> str:
	return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _axis_edges(axis_values: np.ndarray) -> np.ndarray:
	"""Build cell edges from sorted axis centers for pcolormesh plotting."""
	if axis_values.size == 1:
		center = float(axis_values[0])
		return np.array([center - 0.5, center + 0.5], dtype=float)

	deltas = np.diff(axis_values)
	edges = np.empty(axis_values.size + 1, dtype=float)
	edges[1:-1] = axis_values[:-1] + 0.5 * deltas
	edges[0] = axis_values[0] - 0.5 * deltas[0]
	edges[-1] = axis_values[-1] + 0.5 * deltas[-1]
	return edges


def save_class_grid_plots(
	run_stamp: str,
	positions: np.ndarray,
	class_labels: np.ndarray,
	gp_predictions: np.ndarray,
	soinn_predictions: np.ndarray,
	output_dir: Path,
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	for class_name in sorted(np.unique(class_labels)):
		mask = class_labels == class_name
		class_positions = positions[mask]
		gp_values = denormalize_labels(gp_predictions[mask])
		soinn_values = denormalize_labels(soinn_predictions[mask])

		x_unique = np.unique(class_positions[:, 0])
		y_unique = np.unique(class_positions[:, 1])
		x_edges = _axis_edges(x_unique)
		y_edges = _axis_edges(y_unique)

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

		# Determine the color scale limits for the plots based on the finite values of both GP and SOINN+ predictions.
		# if finite_gp.size > 0 and finite_soinn.size > 0:
		# 	vmin = float(min(np.min(finite_gp), np.min(finite_soinn)))
		# 	vmax = float(max(np.max(finite_gp), np.max(finite_soinn)))
		# elif finite_gp.size > 0:
		# 	vmin = float(np.min(finite_gp))
		# 	vmax = float(np.max(finite_gp))
		# else:
		# 	vmin = float(np.min(finite_soinn))
		# 	vmax = float(np.max(finite_soinn))

		# if np.isclose(vmin, vmax):
		# 	vmin -= 1e-6
		# 	vmax += 1e-6

		# Use Pragr2019Benchmarking normalization constants for cost prediction to set a fixed color scale for the plots.
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

			file_name = f"{run_stamp}_{_safe_name(str(class_name))}_{model_name}_grid.png"
			file_path = output_dir / file_name
			fig.savefig(file_path, dpi=180)
			plt.close(fig)
			print(f"Saved {model_name.upper()} grid plot to {file_path}")


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
	parser.add_argument(
		"--grid-samples-per-file",
		type=int,
		default=0,
		help="Number of grid cells sampled from each file for benchmark evaluation (0 = use all)",
	)
	parser.add_argument("--plot", action="store_true", help="Show the optional SOINN+ verification plot")
	parser.add_argument("--curve-step", type=int, default=100, help="Evaluate learning curve every N training samples")
	parser.add_argument("--plot-curve", action="store_true", help="Show the SOINN+ learning curve plot")
	parser.add_argument(
		"--reference-selection",
		choices=("min-variance", "same-class"),
		default="min-variance",
		help="Reference GP selection mode: choose GP with smallest variance or class-matched GP",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.curve_step <= 0:
		raise ValueError("--curve-step must be a positive integer")
	run_stamp = datetime.now().strftime("%y%m%d%H%M")
	curve_compare_output_path = RESULTS_DIRECTORY / f"{run_stamp}_rcurve_compare.png"
	metrics_output_path = RESULTS_DIRECTORY / f"{run_stamp}_metrics.csv"

	train_features, train_labels, trail_classes, trail_files = load_training_data(args.trail_dir)
	grid_features, grid_positions, grid_classes, grid_files = load_grid_data(
		args.grid_dir,
		sample_count=args.grid_samples_per_file,
		seed=args.seed,
	)
	train_features, train_labels, train_classes = _ordered_training_data(
		train_features,
		train_labels,
		trail_classes,
		shuffle=args.shuffle,
		seed=args.seed,
	)

	reference_models = train_reference_gps(train_features, train_labels, train_classes)
	mu_ref, sigma_ref = reference_predict(
		reference_models,
		grid_features,
		class_labels=grid_classes,
		selection_mode=args.reference_selection,
	)
	soinn, curve_steps, curve_r, curve_fallback, curve_r_by_class = train_soinn_learning_curve(
		train_features,
		train_labels,
		grid_features,
		grid_classes,
		mu_ref,
		sigma_ref,
		eval_step=args.curve_step,
		use_fallback=args.fallback,
	)
	sections = _training_sections(train_classes)
	if args.shuffle:
		print("Shuffle enabled: section markers no longer represent contiguous terrain blocks")
		sections = []

	show_training_summary(trail_files, grid_files, train_features, train_labels, soinn, reference_models)
	print("Training strategy: use all trail data for both SOINN+ and the per-class GP reference")
	print(f"Grid sampling -> {args.grid_samples_per_file} cells per file (0 = all cells)")
	print(f"Prediction mode -> fallback: {args.fallback}")
	print(f"Reference GP selection -> {args.reference_selection}")
	print(f"Grid sample count -> {len(grid_features)}")
	print(f"Learning curve -> checkpoints: {len(curve_steps)}, eval_step: {args.curve_step}")
	if curve_steps.size > 0:
		print(f"Learning curve final R: {curve_r[-1]:.6f} at step {int(curve_steps[-1])}")

	predictions, fallback_count = predict_soinn(soinn, grid_features, use_fallback=args.fallback)
	metrics = compute_reference_metrics(predictions, mu_ref, sigma_ref)
	abstention = compute_abstention_metrics(predictions)
	print_metrics("Grid benchmark", metrics, abstention, fallback_count)
	avg_gp_sigma = float(np.mean(sigma_ref))
	print(f"Average GP sigma: {avg_gp_sigma:.6f}")
	plot_learning_curve_comparison(
		curve_steps,
		overall_r=curve_r,
		class_r_curves=curve_r_by_class,
		output_path=curve_compare_output_path,
		sections=sections,
		enabled=args.plot_curve,
	)
	save_class_grid_plots(
		run_stamp,
		positions=grid_positions,
		class_labels=grid_classes,
		gp_predictions=mu_ref,
		soinn_predictions=predictions,
		output_dir=RESULTS_DIRECTORY,
	)
	save_metrics_csv(
		metrics_output_path,
		metrics=metrics,
		abstention=abstention,
		fallback_count=fallback_count,
		avg_gp_sigma=avg_gp_sigma,
		curve_step=args.curve_step,
		curve_steps=curve_steps,
		curve_r=curve_r,
		curve_fallback=curve_fallback,
	)

	plot_network(soinn, args.plot)


if __name__ == "__main__":
	main()
