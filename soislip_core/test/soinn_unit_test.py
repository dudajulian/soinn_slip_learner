#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
import pytest


# Keep test import stable for direct execution and for colcon/pytest discovery.
PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from soinn_py.soinnplus import SoinnPlus, isoutlier


def _make_small_model(dim=3):
    model = SoinnPlus(dim=dim)
    # Disable periodic pruning to keep tests deterministic.
    model.delete_noise_handler = lambda: None
    model.label_period = 10**9
    return model


def test_check_signal_accepts_list_and_shape():
    model = SoinnPlus(dim=3)
    signal = model._check_signal([1.0, 2.0, 3.0])
    assert signal.shape == (1, 3)
    assert np.allclose(signal, np.array([[1.0, 2.0, 3.0]]))


def test_check_signal_rejects_wrong_length():
    model = SoinnPlus(dim=3)
    with pytest.raises(ValueError):
        model._check_signal([1.0, 2.0])


def test_input_signal_initializes_first_three_nodes():
    model = _make_small_model(dim=3)

    model.input_signal([1.0, 0.0, 0.0])
    model.input_signal([2.0, 1.0, 0.0])
    model.input_signal([3.0, 0.0, 1.0])

    assert len(model.nodes) == 3
    assert model.signal_num == 3
    assert model.adjacency_mat.shape == (3, 3)


def test_training_creates_edges():
    model = _make_small_model(dim=3)
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.1, 0.05, -0.02],
            [0.9, -0.05, 0.03],
            [1.05, 0.02, 0.01],
            [0.95, -0.01, -0.02],
            [1.02, 0.03, -0.01],
        ]
    )

    for point in points:
        model.input_signal(point)

    assert len(model.nodes) >= 3
    assert model.links_created >= 1
    assert model.adjacency_mat.nnz > 0


def test_inference_returns_prediction_and_confidence_for_nearby_signal():
    model = _make_small_model(dim=3)
    points = np.array(
        [
            [2.0, 1.00, 1.00],
            [2.1, 1.05, 0.95],
            [1.9, 0.95, 1.02],
            [2.0, 1.02, 0.98],
            [2.05, 0.98, 1.01],
            [1.95, 1.01, 0.99],
        ]
    )

    for point in points:
        model.input_signal(point)

    prediction, confidence = model.inference([1.0, 1.0])
    assert prediction is not None
    assert isinstance(float(prediction), float)
    assert 0.0 <= confidence <= 1.0


def test_collect_cluster_edge_age_returns_nonempty_for_connected_nodes():
    model = _make_small_model(dim=3)
    for point in ([1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 0.0, 1.0]):
        model.add_node(model._check_signal(point))

    model.add_edge([0, 1])
    model.add_edge([1, 2])
    model.set_edge_age(0, 1, 3)
    model.set_edge_age(1, 2, 5)

    ages = model.collect_cluster_edge_age(1)
    assert sorted(ages.tolist()) == [3, 5]


def test_isoutlier_flags_extreme_value():
    tf, low, high, center = isoutlier(np.array([1.0, 1.1, 0.9, 1.0, 20.0]))
    assert tf[-1]
    assert low < center < high