import numpy as np

from architecture_simulator.gtsu_cycle.geometry_semantics import (
    numpy_fps, numpy_knn, symmetric_coordinate_quantize,
)


def test_numpy_fps_and_knn_are_deterministic_with_duplicate_points():
    points = np.asarray([
        [0, 0, 0], [0, 0, 0], [4, 0, 0], [0, 3, 0], [1, 1, 0],
    ], dtype=np.int64)
    centers = numpy_fps(points, 3, skip_origin=False)
    assert centers.tolist() == [0, 2, 3]
    assert numpy_knn(points, centers, 3).tolist() == [
        [0, 1, 4], [2, 4, 0], [3, 4, 0],
    ]


def test_symmetric_coordinate_quantize_respects_signed_range():
    points = np.asarray([[-2.0, 0.0, 2.0]], dtype=np.float32)
    result = symmetric_coordinate_quantize(points, 8)
    assert result.tolist() == [[-127, 0, 127]]
