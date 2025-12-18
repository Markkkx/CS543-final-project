import unittest
import numpy as np
import cv2 as cv
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stitching.seam_finder import SeamFinder
from stitching.seam_solver import (
    SeamSolverBase, DPSeamSolver, OptimizedDPSeamSolver
)
from stitching.seam_benchmark import SeamBenchmark, BenchmarkResult
from stitching.seam_optimizer import (
    ImagePyramid, compute_cost_vectorized, find_seam_dp_optimized,
    AdaptiveSeamFinder
)


class TestSeamSolverBase(unittest.TestCase):
    def setUp(self):
        self.img1 = np.zeros((100, 150, 3), dtype=np.uint8)
        self.img1[:, :100] = [255, 0, 0]  # Blue region
        self.img1[:, 100:] = [200, 0, 0]  # Lighter blue

        self.img2 = np.zeros((100, 150, 3), dtype=np.uint8)
        self.img2[:, :50] = [0, 200, 0]   # Lighter green
        self.img2[:, 50:] = [0, 255, 0]   # Green region
        self.mask1 = np.ones((100, 150), dtype=np.uint8) * 255
        self.mask2 = np.ones((100, 150), dtype=np.uint8) * 255
        self.corner1 = (0, 0)
        self.corner2 = (100, 0)

    def test_compute_overlap_region(self):
        solver = DPSeamSolver()
        result = solver.compute_overlap_region(
            self.img1, self.mask1, self.corner1,
            self.img2, self.mask2, self.corner2
        )

        self.assertIsNotNone(result)
        overlap_mask, roi1, roi2, global_rect = result
        self.assertEqual(overlap_mask.shape, (100, 50))  # Height x Width of overlap

    def test_compute_color_cost(self):
        solver = DPSeamSolver()
        region1 = np.array([[[255, 0, 0]]], dtype=np.float32)
        region2 = np.array([[[0, 255, 0]]], dtype=np.float32)
        cost = solver.compute_cost(region1, region2)
        expected = 255**2 + 255**2  # R diff + G diff
        self.assertEqual(cost[0, 0], expected)

    def test_compute_gradient_cost(self):
        solver = DPSeamSolver(cost_type="color_grad")
        region1 = np.zeros((10, 10, 3), dtype=np.float32)
        region1[:, 5:] = 255
        region2 = np.zeros((10, 10, 3), dtype=np.float32)
        region2[5:, :] = 255
        cost = solver.compute_cost(region1, region2)
        self.assertTrue(np.any(cost > 0))


class TestDPSeamSolver(unittest.TestCase):
    def setUp(self):
        self.solver = DPSeamSolver()

    def test_find_seam_dp_simple(self):
        cost = np.array([
            [10, 1, 10, 10],
            [10, 1, 10, 10],
            [10, 10, 1, 10],
            [10, 10, 1, 10],
        ], dtype=np.float32)

        mask = np.ones_like(cost, dtype=np.uint8) * 255

        seam = self.solver._find_seam_dp_vectorized(cost, mask)
        self.assertEqual(seam[0], 1)  # First row, column 1
        self.assertEqual(seam[1], 1)  # Second row, column 1
        self.assertEqual(seam[2], 2)  # Third row, column 2
        self.assertEqual(seam[3], 2)  # Fourth row, column 2

    def test_find_seam_with_mask(self):
        cost = np.ones((5, 5), dtype=np.float32) * 10
        cost[:, 2] = 1  # Low cost column

        mask = np.ones_like(cost, dtype=np.uint8) * 255
        mask[:, 2] = 0  # Block the low cost column

        seam = self.solver._find_seam_dp_vectorized(cost, mask)

        # Seam should avoid column 2
        for row_idx, col_idx in enumerate(seam):
            self.assertNotEqual(col_idx, 2)

    def test_find_with_multiple_images(self):
        imgs = [
            np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8),
        ]

        corners = [(0, 0), (100, 0), (200, 0)]

        masks = [
            np.ones((100, 150), dtype=np.uint8) * 255,
            np.ones((100, 150), dtype=np.uint8) * 255,
            np.ones((100, 150), dtype=np.uint8) * 255,
        ]

        result = self.solver.find(imgs, corners, masks)

        self.assertEqual(len(result), 3)
        for mask in result:
            self.assertEqual(mask.shape, (100, 150))


class TestOptimizedDPSeamSolver(unittest.TestCase):
    def test_multiscale_seam_finding(self):
        solver = OptimizedDPSeamSolver(scale_factor=0.5, min_size=20)
        imgs = [
            np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8),
            np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8),
        ]

        corners = [(0, 0), (150, 0)]

        masks = [
            np.ones((200, 300), dtype=np.uint8) * 255,
            np.ones((200, 300), dtype=np.uint8) * 255,
        ]

        result = solver.find(imgs, corners, masks)

        self.assertEqual(len(result), 2)


class TestSeamFinder(unittest.TestCase):
    def test_opencv_solver_initialization(self):
        for solver_name in ["dp_color", "dp_colorgrad", "gc_color", "voronoi"]:
            finder = SeamFinder(solver_name)
            self.assertEqual(finder.finder_name, solver_name)
            self.assertFalse(finder.is_custom)

    def test_custom_solver_initialization(self):
        for solver_name in ["custom_dp_color", "custom_dp_colorgrad",
                           "optimized_dp_color", "optimized_dp_colorgrad"]:
            finder = SeamFinder(solver_name)
            self.assertEqual(finder.finder_name, solver_name)
            self.assertTrue(finder.is_custom)

    def test_invalid_solver_raises_error(self):
        with self.assertRaises(ValueError):
            SeamFinder("invalid_solver")

    def test_custom_dp_produces_masks(self):
        finder = SeamFinder("custom_dp_color")

        imgs = [
            np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8),
        ]
        corners = [(0, 0), (100, 0)]
        masks = [
            np.ones((100, 150), dtype=np.uint8) * 255,
            np.ones((100, 150), dtype=np.uint8) * 255,
        ]

        result = finder.find(imgs, corners, masks)

        self.assertEqual(len(result), 2)
        for mask in result:
            self.assertEqual(mask.shape, (100, 150))
            self.assertEqual(mask.dtype, np.uint8)


class TestSeamBenchmark(unittest.TestCase):
    def setUp(self):
        self.benchmark = SeamBenchmark(
            solvers=["dp_color", "voronoi", "custom_dp_color"],
            include_graphcut=False,
            num_runs=1,
            warmup_runs=0,
        )

    def test_add_test_case(self):
        imgs = [np.zeros((50, 50, 3), dtype=np.uint8)]
        corners = [(0, 0)]
        masks = [np.ones((50, 50), dtype=np.uint8) * 255]

        self.benchmark.add_test_case(imgs, corners, masks, name="test1")

        self.assertEqual(len(self.benchmark.test_cases), 1)
        self.assertEqual(self.benchmark.test_cases[0]["name"], "test1")

    def test_run_single_benchmark(self):
        imgs = [
            np.random.randint(0, 255, (50, 75, 3), dtype=np.uint8),
            np.random.randint(0, 255, (50, 75, 3), dtype=np.uint8),
        ]
        corners = [(0, 0), (50, 0)]
        masks = [
            np.ones((50, 75), dtype=np.uint8) * 255,
            np.ones((50, 75), dtype=np.uint8) * 255,
        ]

        result = self.benchmark.run_single("dp_color", imgs, corners, masks)

        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.solver_name, "dp_color")
        self.assertTrue(result.success)
        self.assertGreater(result.execution_time, 0)


class TestImagePyramid(unittest.TestCase):
    def test_build_pyramid(self):
        pyramid = ImagePyramid(num_levels=3, scale_factor=0.5)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        levels = pyramid.build(img)

        self.assertEqual(len(levels), 3)
        self.assertEqual(levels[0].shape, (100, 100, 3))
        self.assertEqual(levels[1].shape, (50, 50, 3))
        self.assertEqual(levels[2].shape, (25, 25, 3))

    def test_build_mask_pyramid(self):
        pyramid = ImagePyramid(num_levels=2, scale_factor=0.5)
        mask = np.ones((80, 80), dtype=np.uint8) * 255

        levels = pyramid.build_mask_pyramid(mask)

        self.assertEqual(len(levels), 2)
        self.assertEqual(levels[0].shape, (80, 80))
        self.assertEqual(levels[1].shape, (40, 40))


class TestCostComputation(unittest.TestCase):
    def test_compute_cost_vectorized(self):
        img1 = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
        img2 = np.array([[[0, 255, 0], [255, 0, 0]]], dtype=np.uint8)

        cost = compute_cost_vectorized(img1, img2, include_gradient=False)

        self.assertEqual(cost.shape, (1, 2))
        self.assertGreater(cost[0, 0], 0)
        self.assertGreater(cost[0, 1], 0)

    def test_find_seam_dp_optimized(self):
        cost = np.array([
            [10, 1, 10],
            [10, 1, 10],
            [10, 1, 10],
        ], dtype=np.float32)

        mask = np.ones_like(cost, dtype=np.uint8) * 255

        seam = find_seam_dp_optimized(cost, mask)
        np.testing.assert_array_equal(seam, [1, 1, 1])


class TestAdaptiveSeamFinder(unittest.TestCase):
    def test_adaptive_selection(self):
        finder = AdaptiveSeamFinder()
        small_imgs = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
        ]
        small_corners = [(0, 0), (25, 0)]
        small_masks = [
            np.ones((50, 50), dtype=np.uint8) * 255,
            np.ones((50, 50), dtype=np.uint8) * 255,
        ]

        result = finder.find(small_imgs, small_corners, small_masks)
        self.assertEqual(len(result), 2)


class TestNumbaAcceleratedSolver(unittest.TestCase):
    def test_numba_solver_initialization(self):
        """Test Numba solver initialization."""
        from stitching.seam_solver import NumbaAcceleratedDPSolver, NUMBA_AVAILABLE
        solver = NumbaAcceleratedDPSolver()
        self.assertEqual(solver.cost_type, "color")
        self.assertTrue(solver.use_multiscale)

    def test_numba_solver_produces_valid_masks(self):
        """Test that Numba solver produces valid masks."""
        from stitching.seam_solver import NumbaAcceleratedDPSolver
        solver = NumbaAcceleratedDPSolver(use_multiscale=False)

        imgs = [
            np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8),
        ]
        corners = [(0, 0), (100, 0)]
        masks = [
            np.ones((100, 150), dtype=np.uint8) * 255,
            np.ones((100, 150), dtype=np.uint8) * 255,
        ]

        result = solver.find(imgs, corners, masks)

        self.assertEqual(len(result), 2)
        for mask in result:
            self.assertEqual(mask.shape, (100, 150))
            self.assertEqual(mask.dtype, np.uint8)

    def test_numba_solver_via_seam_finder(self):
        """Test Numba solver through SeamFinder interface."""
        finder = SeamFinder("numba_dp_color")
        self.assertTrue(finder.is_custom)

        imgs = [
            np.random.randint(0, 255, (50, 75, 3), dtype=np.uint8),
            np.random.randint(0, 255, (50, 75, 3), dtype=np.uint8),
        ]
        corners = [(0, 0), (50, 0)]
        masks = [
            np.ones((50, 75), dtype=np.uint8) * 255,
            np.ones((50, 75), dtype=np.uint8) * 255,
        ]

        result = finder.find(imgs, corners, masks)
        self.assertEqual(len(result), 2)

    def test_numba_multiscale(self):
        """Test Numba solver with multi-scale processing."""
        from stitching.seam_solver import NumbaAcceleratedDPSolver
        solver = NumbaAcceleratedDPSolver(use_multiscale=True, min_size=50)

        # Create larger images to trigger multi-scale
        imgs = [
            np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8),
            np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8),
        ]
        corners = [(0, 0), (150, 0)]
        masks = [
            np.ones((200, 300), dtype=np.uint8) * 255,
            np.ones((200, 300), dtype=np.uint8) * 255,
        ]

        result = solver.find(imgs, corners, masks)
        self.assertEqual(len(result), 2)


class TestGraphCutSolver(unittest.TestCase):
    """Tests for GraphCut solver (requires PyMaxflow)."""

    @classmethod
    def setUpClass(cls):
        """Check if PyMaxflow is available."""
        try:
            import maxflow
            cls.maxflow_available = True
        except ImportError:
            cls.maxflow_available = False

    def test_graphcut_solver_initialization(self):
        """Test GraphCut solver initialization."""
        if not self.maxflow_available:
            self.skipTest("PyMaxflow not available")

        from stitching.seam_solver import GraphCutSeamSolver
        solver = GraphCutSeamSolver()
        self.assertEqual(solver.cost_type, "color")

    def test_graphcut_seam_finding(self):
        """Test GraphCut seam finding produces valid masks."""
        if not self.maxflow_available:
            self.skipTest("PyMaxflow not available")

        from stitching.seam_solver import GraphCutSeamSolver
        solver = GraphCutSeamSolver()

        imgs = [
            np.random.randint(0, 255, (50, 75, 3), dtype=np.uint8),
            np.random.randint(0, 255, (50, 75, 3), dtype=np.uint8),
        ]
        corners = [(0, 0), (50, 0)]
        masks = [
            np.ones((50, 75), dtype=np.uint8) * 255,
            np.ones((50, 75), dtype=np.uint8) * 255,
        ]

        result = solver.find(imgs, corners, masks)

        self.assertEqual(len(result), 2)


def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()
