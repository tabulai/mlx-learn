"""Weighted votes remain deterministic and retain distinctions before rounding."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device._neighbors_votes import class_predictions, class_probabilities


def _reference(indices, targets, n_classes, distances, squared):
    scores = np.zeros((len(indices), n_classes), np.float64)
    if distances is None:
        weights = np.ones(indices.shape, np.float64)
    else:
        d = distances.astype(np.float64)
        if squared:
            d = np.sqrt(d)
        zero = d == 0
        with np.errstate(divide="ignore"):
            weights = np.where(np.any(zero, axis=1, keepdims=True), zero, 1.0 / d)
    for row in range(len(indices)):
        np.add.at(scores[row], targets[indices[row]], weights[row])
    return scores / scores.sum(axis=1, keepdims=True), scores.argmax(axis=1)


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu])
@pytest.mark.parametrize("n_classes,k", [(2, 7), (12, 64), (33, 7), (3, 65), (37, 257), (2, 2048)])
@pytest.mark.parametrize("weighting", ["uniform", "distance", "squared"])
def test_votes_match_independent_float64_reference(device, n_classes, k, weighting):
    rng = np.random.default_rng(133)
    indices = np.broadcast_to(np.arange(k, dtype=np.int32), (9, k)).copy()
    targets = rng.integers(n_classes, size=k, dtype=np.int32)
    distances = None
    if weighting != "uniform":
        distances = np.exp(rng.uniform(-6, 6, size=indices.shape)).astype(np.float32)
        distances[0, :2] = 0
    reference, labels = _reference(indices, targets, n_classes, distances, weighting == "squared")
    with mx.stream(device):
        im, tm = mx.array(indices), mx.array(targets)
        dm = None if distances is None else mx.array(distances)
        actual = class_probabilities(im, tm, n_classes, dm, squared_distances=weighting == "squared")
        winners = class_predictions(im, tm, n_classes, dm, squared_distances=weighting == "squared")
        mx.eval(actual, winners)
    np.testing.assert_allclose(np.asarray(actual), reference, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(np.asarray(winners), labels)


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu])
@pytest.mark.parametrize("n_classes,k", [(2, 4), (33, 4), (2, 68)])
def test_prediction_compares_votes_before_float32_probability_rounding(device, n_classes, k):
    # Class 1 has a strictly larger vote, but both probabilities round to 0.5.
    targets = np.resize(np.array([0, 1], dtype=np.int32), k)
    distances = np.full((1, k), 2, np.float32)
    distances[0, -1] = np.nextafter(np.float32(2), np.float32(0))
    with mx.stream(device):
        indices = mx.arange(k, dtype=mx.int32)[None]
        actual = class_probabilities(indices, mx.array(targets), n_classes, mx.array(distances))
        winners = class_predictions(indices, mx.array(targets), n_classes, mx.array(distances))
        mx.eval(actual, winners)
    assert np.asarray(actual)[0, 0] == np.asarray(actual)[0, 1]
    np.testing.assert_array_equal(np.asarray(winners), [1])


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu])
def test_squared_distance_votes_preserve_a_difference_lost_by_float32_sqrt(device):
    with mx.stream(device):
        indices = mx.array([[0, 1]], mx.int32)
        targets = mx.array([0, 1], mx.int32)
        squared = mx.array([[float(np.nextafter(np.float32(1), np.float32(2))), 1]], mx.float32)
        winners = class_predictions(indices, targets, 2, squared, squared_distances=True)
        mx.eval(winners)
    np.testing.assert_array_equal(np.asarray(winners), [1])


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu])
@pytest.mark.parametrize("n_classes,k", [(2, 32), (33, 32), (2, 2048)])
def test_weighted_votes_are_bitwise_repeatable(device, n_classes, k):
    rng = np.random.default_rng(23)
    with mx.stream(device):
        indices = mx.array(np.broadcast_to(np.arange(k, dtype=np.int32), (16, k)).copy())
        targets = mx.array(rng.integers(n_classes, size=k, dtype=np.int32))
        distances = mx.array(np.exp(rng.uniform(-7, 7, (16, k))).astype(np.float32))
        expected = None
        for _ in range(10):
            result = class_probabilities(indices, targets, n_classes, distances)
            mx.eval(result)
            bits = np.asarray(result).view(np.uint32).copy()
            if expected is None:
                expected = bits
            else:
                np.testing.assert_array_equal(bits, expected)


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu])
def test_squared_weights_do_not_underflow_before_square_root(device):
    squared = np.array([[1e-30, 1e30]], dtype=np.float32)
    indices, targets = np.array([[0, 1]], np.int32), np.array([0, 1], np.int32)
    expected, _ = _reference(indices, targets, 2, squared, True)
    with mx.stream(device):
        actual = class_probabilities(mx.array(indices), mx.array(targets), 2, mx.array(squared),
                                     squared_distances=True)
        mx.eval(actual)
    assert np.asarray(actual)[0, 1] > 0
    np.testing.assert_allclose(np.asarray(actual), expected, atol=0, rtol=1e-6)


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu])
@pytest.mark.parametrize("n_classes,k", [(2, 2), (33, 2), (2, 65)])
def test_infinite_distances_have_zero_weight_and_all_infinite_rows_are_invalid(device, n_classes, k):
    values = np.full((2, k), np.inf, np.float32)
    values[0, 0] = 1
    with mx.stream(device):
        indices = mx.broadcast_to(mx.arange(k, dtype=mx.int32), (2, k))
        targets = mx.arange(k, dtype=mx.int32) % 2
        distances = mx.array(values)
        probabilities = class_probabilities(indices, targets, n_classes, distances)
        predictions = class_predictions(indices, targets, n_classes, distances)
        mx.eval(probabilities, predictions)
    assert np.asarray(probabilities)[0, 0] == 1
    assert np.isnan(np.asarray(probabilities)[1]).all()
    np.testing.assert_array_equal(np.asarray(predictions), [0, -1])
