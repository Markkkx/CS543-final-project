from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2 as cv

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


@njit(cache=True)
def _compute_color_cost_numba(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Numba JIT-compiled color cost computation.
    """
    h, w = img1.shape[:2]
    cost = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            diff_b = float(img1[i, j, 0]) - float(img2[i, j, 0])
            diff_g = float(img1[i, j, 1]) - float(img2[i, j, 1])
            diff_r = float(img1[i, j, 2]) - float(img2[i, j, 2])
            cost[i, j] = diff_b * diff_b + diff_g * diff_g + diff_r * diff_r

    return cost


@njit(cache=True)
def _find_seam_dp_numba(cost: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Numba JIT-compiled DP seam finding - the core algorithm.
    """
    h, w = cost.shape
    dp = np.full((h, w), np.inf, dtype=np.float32)
    parent = np.zeros((h, w), dtype=np.int32)

    for j in range(w):
        if mask[0, j] > 0:
            dp[0, j] = cost[0, j]

    for i in range(1, h):
        for j in range(w):
            if mask[i, j] == 0:
                continue
            min_cost = np.inf
            min_parent = j
            for dj in range(-1, 2):
                pj = j + dj
                if 0 <= pj < w and dp[i-1, pj] < min_cost:
                    min_cost = dp[i-1, pj]
                    min_parent = pj

            dp[i, j] = cost[i, j] + min_cost
            parent[i, j] = min_parent
    seam = np.zeros(h, dtype=np.int32)
    min_val = np.inf
    min_j = 0

    for j in range(w):
        if mask[h-1, j] > 0 and dp[h-1, j] < min_val:
            min_val = dp[h-1, j]
            min_j = j

    seam[h-1] = min_j

    for i in range(h-2, -1, -1):
        seam[i] = parent[i+1, seam[i+1]]

    return seam


@njit(cache=True)
def _create_seam_mask_numba(seam: np.ndarray, h: int, w: int,
                             valid_mask: np.ndarray) -> np.ndarray:
    """
    Numba JIT-compiled seam mask creation.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        seam_col = seam[i]
        for j in range(w):
            if j < seam_col and valid_mask[i, j] > 0:
                mask[i, j] = 255

    return mask


@njit(cache=True, parallel=True)
def _compute_cost_parallel_numba(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Parallel Numba JIT-compiled cost computation using multiple cores.
    """
    h, w = img1.shape[:2]
    cost = np.zeros((h, w), dtype=np.float32)

    for i in prange(h):
        for j in range(w):
            diff_b = float(img1[i, j, 0]) - float(img2[i, j, 0])
            diff_g = float(img1[i, j, 1]) - float(img2[i, j, 1])
            diff_r = float(img1[i, j, 2]) - float(img2[i, j, 2])
            cost[i, j] = diff_b * diff_b + diff_g * diff_g + diff_r * diff_r

    return cost


class SeamSolverBase(ABC):
    """Abstract base class for seam solvers."""

    def __init__(self, cost_type: str = "color"):
        if cost_type not in ("color", "color_grad"):
            raise ValueError(f"Invalid cost_type: {cost_type}")
        self.cost_type = cost_type

    @abstractmethod
    def find(self, imgs: List[np.ndarray], corners: List[Tuple[int, int]],
             masks: List[np.ndarray]) -> List[np.ndarray]:
        pass

    def compute_overlap_region(self, img1: np.ndarray, mask1: np.ndarray,
                               corner1: Tuple[int, int], img2: np.ndarray,
                               mask2: np.ndarray, corner2: Tuple[int, int]) -> Tuple:
        """Compute the overlapping region between two images."""
        x1, y1 = corner1
        x2, y2 = corner2
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        inter_left = max(x1, x2)
        inter_top = max(y1, y2)
        inter_right = min(x1 + w1, x2 + w2)
        inter_bottom = min(y1 + h1, y2 + h2)

        if inter_left >= inter_right or inter_top >= inter_bottom:
            return None

        roi1 = (inter_left - x1, inter_top - y1, inter_right - x1, inter_bottom - y1)
        roi2 = (inter_left - x2, inter_top - y2, inter_right - x2, inter_bottom - y2)

        overlap1 = mask1[roi1[1]:roi1[3], roi1[0]:roi1[2]]
        overlap2 = mask2[roi2[1]:roi2[3], roi2[0]:roi2[2]]
        combined_overlap = cv.bitwise_and(overlap1, overlap2)

        return (combined_overlap, roi1, roi2, (inter_left, inter_top, inter_right, inter_bottom))

    def compute_cost(self, img1_region: np.ndarray, img2_region: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute cost matrix using vectorized operations."""
        diff = img1_region.astype(np.float32) - img2_region.astype(np.float32)
        cost = np.sum(diff ** 2, axis=2)

        if self.cost_type == "color_grad":
            gray1 = cv.cvtColor(img1_region.astype(np.float32), cv.COLOR_BGR2GRAY)
            gray2 = cv.cvtColor(img2_region.astype(np.float32), cv.COLOR_BGR2GRAY)
            grad1_x = cv.Scharr(gray1, cv.CV_32F, 1, 0)
            grad1_y = cv.Scharr(gray1, cv.CV_32F, 0, 1)
            grad2_x = cv.Scharr(gray2, cv.CV_32F, 1, 0)
            grad2_y = cv.Scharr(gray2, cv.CV_32F, 0, 1)
            cost += (grad1_x - grad2_x) ** 2 + (grad1_y - grad2_y) ** 2

        if mask is not None:
            cost[mask == 0] = np.inf

        return cost


class DPSeamSolver(SeamSolverBase):
    """DP seam solver with NumPy vectorization."""

    def find(self, imgs: List[np.ndarray], corners: List[Tuple[int, int]],
             masks: List[np.ndarray]) -> List[np.ndarray]:
        n = len(imgs)
        seam_masks = [mask.copy() for mask in masks]

        for i in range(n):
            for j in range(i + 1, n):
                self._process_pair(imgs[i], seam_masks[i], corners[i],
                                   imgs[j], seam_masks[j], corners[j])
        return seam_masks

    def _process_pair(self, img1, mask1, corner1, img2, mask2, corner2):
        overlap_result = self.compute_overlap_region(img1, mask1, corner1, img2, mask2, corner2)
        if overlap_result is None:
            return

        combined_overlap, roi1, roi2, _ = overlap_result
        if np.sum(combined_overlap) == 0:
            return

        img1_region = img1[roi1[1]:roi1[3], roi1[0]:roi1[2]]
        img2_region = img2[roi2[1]:roi2[3], roi2[0]:roi2[2]]

        cost = self.compute_cost(img1_region, img2_region, combined_overlap)
        seam = self._find_seam_dp_vectorized(cost, combined_overlap)
        seam_mask = self._create_seam_mask_vectorized(seam, cost.shape, combined_overlap)

        mask1[roi1[1]:roi1[3], roi1[0]:roi1[2]] = cv.bitwise_and(
            mask1[roi1[1]:roi1[3], roi1[0]:roi1[2]], seam_mask)
        mask2[roi2[1]:roi2[3], roi2[0]:roi2[2]] = cv.bitwise_and(
            mask2[roi2[1]:roi2[3], roi2[0]:roi2[2]], 255 - seam_mask)

    def _find_seam_dp_vectorized(self, cost: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Vectorized DP seam finding."""
        h, w = cost.shape
        dp = np.full((h, w), np.inf, dtype=np.float32)
        parent = np.zeros((h, w), dtype=np.int32)

        dp[0] = np.where(mask[0] > 0, cost[0], np.inf)

        for i in range(1, h):
            valid = mask[i] > 0
            left = np.roll(dp[i-1], 1)
            left[0] = np.inf
            center = dp[i-1]
            right = np.roll(dp[i-1], -1)
            right[-1] = np.inf

            predecessors = np.vstack([left, center, right])
            min_idx = np.argmin(predecessors, axis=0)
            min_val = np.min(predecessors, axis=0)

            dp[i] = np.where(valid, cost[i] + min_val, np.inf)
            parent[i] = np.clip(np.arange(w) + (min_idx - 1), 0, w - 1)

        seam = np.zeros(h, dtype=np.int32)
        valid_last = dp[h-1].copy()
        valid_last[mask[h-1] == 0] = np.inf
        seam[h-1] = np.argmin(valid_last)

        for i in range(h-2, -1, -1):
            seam[i] = parent[i+1, seam[i+1]]

        return seam

    def _create_seam_mask_vectorized(self, seam: np.ndarray, shape: Tuple[int, int],
                                      valid_mask: np.ndarray) -> np.ndarray:
        h, w = shape
        col_indices = np.arange(w)
        mask = (col_indices[np.newaxis, :] < seam[:, np.newaxis]).astype(np.uint8) * 255
        return cv.bitwise_and(mask, valid_mask)


class GraphCutSeamSolver(SeamSolverBase):
    """GraphCut seam solver using PyMaxflow."""

    def __init__(self, cost_type: str = "color", terminal_weight: float = 1e6):
        super().__init__(cost_type)
        self.terminal_weight = terminal_weight
        self._maxflow_available = None

    def _check_maxflow(self):
        if self._maxflow_available is None:
            try:
                import maxflow
                self._maxflow_available = True
            except ImportError:
                self._maxflow_available = False
        return self._maxflow_available

    def find(self, imgs, corners, masks):
        if not self._check_maxflow():
            raise ImportError("PyMaxflow required. Install with: pip install PyMaxflow")

        n = len(imgs)
        seam_masks = [mask.copy() for mask in masks]

        for i in range(n):
            for j in range(i + 1, n):
                self._process_pair(imgs[i], seam_masks[i], corners[i],
                                   imgs[j], seam_masks[j], corners[j])
        return seam_masks

    def _process_pair(self, img1, mask1, corner1, img2, mask2, corner2):
        import maxflow

        overlap_result = self.compute_overlap_region(img1, mask1, corner1, img2, mask2, corner2)
        if overlap_result is None:
            return

        combined_overlap, roi1, roi2, _ = overlap_result
        if np.sum(combined_overlap) == 0:
            return

        img1_region = img1[roi1[1]:roi1[3], roi1[0]:roi1[2]]
        img2_region = img2[roi2[1]:roi2[3], roi2[0]:roi2[2]]
        h, w = combined_overlap.shape

        g = maxflow.Graph[float](h * w, h * w * 2)
        nodes = g.add_nodes(h * w)

        cost = self.compute_cost(img1_region, img2_region, combined_overlap)
        valid_cost = cost[combined_overlap > 0]
        if len(valid_cost) > 0:
            finite = valid_cost[np.isfinite(valid_cost)]
            if len(finite) > 0:
                cost = np.clip(cost, 0, np.percentile(finite, 95))

        # Add edges
        for i in range(h):
            for j in range(w - 1):
                if combined_overlap[i, j] > 0 and combined_overlap[i, j + 1] > 0:
                    weight = (cost[i, j] + cost[i, j + 1]) / 2
                    g.add_edge(nodes[i * w + j], nodes[i * w + j + 1], weight, weight)

        for i in range(h - 1):
            for j in range(w):
                if combined_overlap[i, j] > 0 and combined_overlap[i + 1, j] > 0:
                    weight = (cost[i, j] + cost[i + 1, j]) / 2
                    g.add_edge(nodes[i * w + j], nodes[(i + 1) * w + j], weight, weight)

        # Terminal edges
        for i in range(h):
            for j in range(w):
                if combined_overlap[i, j] == 0:
                    continue
                idx = i * w + j
                if j == 0 or combined_overlap[i, j - 1] == 0:
                    g.add_tedge(nodes[idx], self.terminal_weight, 0)
                if j == w - 1 or combined_overlap[i, j + 1] == 0:
                    g.add_tedge(nodes[idx], 0, self.terminal_weight)

        g.maxflow()

        seam_mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                if combined_overlap[i, j] > 0 and g.get_segment(nodes[i * w + j]) == 0:
                    seam_mask[i, j] = 255

        mask1[roi1[1]:roi1[3], roi1[0]:roi1[2]] = cv.bitwise_and(
            mask1[roi1[1]:roi1[3], roi1[0]:roi1[2]],
            cv.bitwise_or(seam_mask, 255 - combined_overlap))
        mask2[roi2[1]:roi2[3], roi2[0]:roi2[2]] = cv.bitwise_and(
            mask2[roi2[1]:roi2[3], roi2[0]:roi2[2]],
            cv.bitwise_or(255 - seam_mask, 255 - combined_overlap))


class OptimizedDPSeamSolver(DPSeamSolver):
    """
    Highly optimized DP seam solver using multi-scale processing.
    """

    def __init__(self, cost_type: str = "color",
                 scale_factor: float = 0.25,
                 min_size: int = 50,
                 use_parallel: bool = True):
        super().__init__(cost_type)
        self.scale_factor = scale_factor
        self.min_size = min_size
        self.use_parallel = use_parallel

    def find(self, imgs: List[np.ndarray], corners: List[Tuple[int, int]],
             masks: List[np.ndarray]) -> List[np.ndarray]:
        """Find seams with optional parallel processing."""
        n = len(imgs)
        seam_masks = [mask.copy() for mask in masks]

        # Collect all pairs to process
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))

        if self.use_parallel and len(pairs) > 1:
            # Process pairs in parallel
            with ThreadPoolExecutor(max_workers=min(4, len(pairs))) as executor:
                futures = []
                for i, j in pairs:
                    future = executor.submit(
                        self._compute_seam_for_pair,
                        imgs[i], masks[i].copy(), corners[i],
                        imgs[j], masks[j].copy(), corners[j]
                    )
                    futures.append((i, j, future))

                # Collect results and apply to masks
                for i, j, future in futures:
                    result = future.result()
                    if result is not None:
                        seam_mask, roi1, roi2 = result
                        seam_masks[i][roi1[1]:roi1[3], roi1[0]:roi1[2]] = cv.bitwise_and(
                            seam_masks[i][roi1[1]:roi1[3], roi1[0]:roi1[2]], seam_mask)
                        seam_masks[j][roi2[1]:roi2[3], roi2[0]:roi2[2]] = cv.bitwise_and(
                            seam_masks[j][roi2[1]:roi2[3], roi2[0]:roi2[2]], 255 - seam_mask)
        else:
            # Sequential processing
            for i in range(n):
                for j in range(i + 1, n):
                    self._process_pair(imgs[i], seam_masks[i], corners[i],
                                       imgs[j], seam_masks[j], corners[j])

        return seam_masks

    def _compute_seam_for_pair(self, img1, mask1, corner1, img2, mask2, corner2):
        """Compute seam for a pair (for parallel processing)."""
        overlap_result = self.compute_overlap_region(img1, mask1, corner1, img2, mask2, corner2)
        if overlap_result is None:
            return None

        combined_overlap, roi1, roi2, _ = overlap_result
        if np.sum(combined_overlap) == 0:
            return None

        img1_region = img1[roi1[1]:roi1[3], roi1[0]:roi1[2]]
        img2_region = img2[roi2[1]:roi2[3], roi2[0]:roi2[2]]

        cost = self.compute_cost(img1_region, img2_region, combined_overlap)
        seam = self._find_seam_multiscale(cost, combined_overlap)
        seam_mask = self._create_seam_mask_vectorized(seam, cost.shape, combined_overlap)

        return seam_mask, roi1, roi2

    def _find_seam_multiscale(self, cost: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Multi-scale seam finding for speedup on large images."""
        h, w = cost.shape

        # For small images, use single-scale
        if h <= self.min_size or w <= self.min_size:
            return self._find_seam_dp_vectorized(cost, mask)

        # Downsample
        small_h = max(self.min_size, int(h * self.scale_factor))
        small_w = max(self.min_size, int(w * self.scale_factor))

        small_cost = cv.resize(cost, (small_w, small_h), interpolation=cv.INTER_AREA)
        small_mask = cv.resize(mask, (small_w, small_h), interpolation=cv.INTER_NEAREST)

        # Find coarse seam
        coarse_seam = self._find_seam_dp_vectorized(small_cost, small_mask)

        # Upsample seam using interpolation
        scale_x = w / small_w
        scale_y = h / small_h

        # Vectorized interpolation
        row_indices = np.arange(h)
        coarse_indices = row_indices / scale_y
        i_low = np.floor(coarse_indices).astype(int)
        i_high = np.minimum(i_low + 1, len(coarse_seam) - 1)
        t = coarse_indices - i_low

        j_interp = coarse_seam[i_low] * (1 - t) + coarse_seam[i_high] * t
        fine_seam = np.clip((j_interp * scale_x).astype(np.int32), 0, w - 1)

        return fine_seam


class NumbaAcceleratedDPSolver(SeamSolverBase):
    """
    Numba JIT-compiled DP seam solver - FASTEST implementation.
    """

    def __init__(self, cost_type: str = "color",
                 use_multiscale: bool = True,
                 scale_factor: float = 0.5,
                 min_size: int = 100):
        super().__init__(cost_type)
        self.use_multiscale = use_multiscale
        self.scale_factor = scale_factor
        self.min_size = min_size

        if not NUMBA_AVAILABLE:
            import warnings
            warnings.warn("Numba not available, falling back to NumPy implementation")

    def find(self, imgs: List[np.ndarray], corners: List[Tuple[int, int]],
             masks: List[np.ndarray]) -> List[np.ndarray]:
        """Find seams using Numba-accelerated DP."""
        n = len(imgs)
        seam_masks = [mask.copy() for mask in masks]

        for i in range(n):
            for j in range(i + 1, n):
                self._process_pair(imgs[i], seam_masks[i], corners[i],
                                   imgs[j], seam_masks[j], corners[j])
        return seam_masks

    def _process_pair(self, img1, mask1, corner1, img2, mask2, corner2):
        """Process a pair of images using Numba-accelerated DP."""
        overlap_result = self.compute_overlap_region(img1, mask1, corner1, img2, mask2, corner2)
        if overlap_result is None:
            return

        combined_overlap, roi1, roi2, _ = overlap_result
        if np.sum(combined_overlap) == 0:
            return

        img1_region = img1[roi1[1]:roi1[3], roi1[0]:roi1[2]]
        img2_region = img2[roi2[1]:roi2[3], roi2[0]:roi2[2]]

        h, w = combined_overlap.shape

        # Decide whether to use multi-scale
        if self.use_multiscale and h > self.min_size and w > self.min_size:
            seam = self._find_seam_multiscale(img1_region, img2_region, combined_overlap)
        else:
            seam = self._find_seam_direct(img1_region, img2_region, combined_overlap)

        # Create seam mask using Numba
        if NUMBA_AVAILABLE:
            seam_mask = _create_seam_mask_numba(seam, h, w, combined_overlap)
        else:
            seam_mask = self._create_seam_mask_numpy(seam, (h, w), combined_overlap)

        mask1[roi1[1]:roi1[3], roi1[0]:roi1[2]] = cv.bitwise_and(
            mask1[roi1[1]:roi1[3], roi1[0]:roi1[2]], seam_mask)
        mask2[roi2[1]:roi2[3], roi2[0]:roi2[2]] = cv.bitwise_and(
            mask2[roi2[1]:roi2[3], roi2[0]:roi2[2]], 255 - seam_mask)

    def _find_seam_direct(self, img1_region, img2_region, mask):
        """Find seam directly without multi-scale."""
        # Ensure contiguous arrays for Numba
        img1_c = np.ascontiguousarray(img1_region)
        img2_c = np.ascontiguousarray(img2_region)
        mask_c = np.ascontiguousarray(mask)

        if NUMBA_AVAILABLE:
            # Use Numba JIT-compiled functions
            cost = _compute_color_cost_numba(img1_c, img2_c)
            cost[mask_c == 0] = np.inf
            return _find_seam_dp_numba(cost, mask_c)
        else:
            # Fallback to NumPy
            cost = self.compute_cost(img1_region, img2_region, mask)
            return self._find_seam_dp_numpy(cost, mask)

    def _find_seam_multiscale(self, img1_region, img2_region, mask):
        """Multi-scale seam finding for large images."""
        h, w = mask.shape

        # Downsample
        new_h = max(self.min_size, int(h * self.scale_factor))
        new_w = max(self.min_size, int(w * self.scale_factor))

        small_img1 = cv.resize(img1_region, (new_w, new_h), interpolation=cv.INTER_AREA)
        small_img2 = cv.resize(img2_region, (new_w, new_h), interpolation=cv.INTER_AREA)
        small_mask = cv.resize(mask, (new_w, new_h), interpolation=cv.INTER_NEAREST)

        # Find coarse seam
        coarse_seam = self._find_seam_direct(small_img1, small_img2, small_mask)

        # Upsample seam
        scale_x = w / new_w
        scale_y = h / new_h

        row_indices = np.arange(h)
        coarse_indices = row_indices / scale_y
        i_low = np.floor(coarse_indices).astype(int)
        i_high = np.minimum(i_low + 1, len(coarse_seam) - 1)
        t = coarse_indices - i_low

        j_interp = coarse_seam[i_low] * (1 - t) + coarse_seam[i_high] * t
        fine_seam = np.clip((j_interp * scale_x).astype(np.int32), 0, w - 1)

        return fine_seam

    def _find_seam_dp_numpy(self, cost, mask):
        """NumPy fallback for DP seam finding."""
        h, w = cost.shape
        dp = np.full((h, w), np.inf, dtype=np.float32)
        parent = np.zeros((h, w), dtype=np.int32)

        dp[0] = np.where(mask[0] > 0, cost[0], np.inf)

        for i in range(1, h):
            valid = mask[i] > 0
            left = np.roll(dp[i-1], 1)
            left[0] = np.inf
            center = dp[i-1]
            right = np.roll(dp[i-1], -1)
            right[-1] = np.inf

            predecessors = np.vstack([left, center, right])
            min_idx = np.argmin(predecessors, axis=0)
            min_val = np.min(predecessors, axis=0)

            dp[i] = np.where(valid, cost[i] + min_val, np.inf)
            parent[i] = np.clip(np.arange(w) + (min_idx - 1), 0, w - 1)

        seam = np.zeros(h, dtype=np.int32)
        valid_last = dp[h-1].copy()
        valid_last[mask[h-1] == 0] = np.inf
        seam[h-1] = np.argmin(valid_last)

        for i in range(h-2, -1, -1):
            seam[i] = parent[i+1, seam[i+1]]

        return seam

    def _create_seam_mask_numpy(self, seam, shape, valid_mask):
        """NumPy fallback for seam mask creation."""
        h, w = shape
        col_indices = np.arange(w)
        mask = (col_indices[np.newaxis, :] < seam[:, np.newaxis]).astype(np.uint8) * 255
        return cv.bitwise_and(mask, valid_mask)


class OptimizedGraphCutSeamSolver(GraphCutSeamSolver):
    """
    Multi-scale optimized GraphCut seam solver.

    Uses downsampling to reduce graph size while maintaining quality.
    """

    def __init__(self, cost_type: str = "color",
                 terminal_weight: float = 1e6,
                 downsample_factor: float = 0.5,
                 min_size: int = 100):
        super().__init__(cost_type, terminal_weight)
        self.downsample_factor = downsample_factor
        self.min_size = min_size

    def _process_pair(self, img1, mask1, corner1, img2, mask2, corner2):
        import maxflow

        overlap_result = self.compute_overlap_region(img1, mask1, corner1, img2, mask2, corner2)
        if overlap_result is None:
            return

        combined_overlap, roi1, roi2, _ = overlap_result
        if np.sum(combined_overlap) == 0:
            return

        h, w = combined_overlap.shape
        img1_region = img1[roi1[1]:roi1[3], roi1[0]:roi1[2]]
        img2_region = img2[roi2[1]:roi2[3], roi2[0]:roi2[2]]

        # Multi-scale for large regions
        if h > self.min_size and w > self.min_size:
            new_h = max(self.min_size, int(h * self.downsample_factor))
            new_w = max(self.min_size, int(w * self.downsample_factor))

            small_overlap = cv.resize(combined_overlap, (new_w, new_h), interpolation=cv.INTER_NEAREST)
            small_img1 = cv.resize(img1_region, (new_w, new_h), interpolation=cv.INTER_AREA)
            small_img2 = cv.resize(img2_region, (new_w, new_h), interpolation=cv.INTER_AREA)

            seam_mask = self._compute_graphcut(small_img1, small_img2, small_overlap, new_h, new_w)
            seam_mask = cv.resize(seam_mask, (w, h), interpolation=cv.INTER_NEAREST)
        else:
            seam_mask = self._compute_graphcut(img1_region, img2_region, combined_overlap, h, w)

        mask1[roi1[1]:roi1[3], roi1[0]:roi1[2]] = cv.bitwise_and(
            mask1[roi1[1]:roi1[3], roi1[0]:roi1[2]],
            cv.bitwise_or(seam_mask, 255 - combined_overlap))
        mask2[roi2[1]:roi2[3], roi2[0]:roi2[2]] = cv.bitwise_and(
            mask2[roi2[1]:roi2[3], roi2[0]:roi2[2]],
            cv.bitwise_or(255 - seam_mask, 255 - combined_overlap))

    def _compute_graphcut(self, img1_region, img2_region, overlap_mask, h, w):
        import maxflow

        g = maxflow.Graph[float](h * w, h * w * 2)
        nodes = g.add_nodes(h * w)

        cost = self.compute_cost(img1_region, img2_region, overlap_mask)
        valid_cost = cost[overlap_mask > 0]
        if len(valid_cost) > 0:
            finite = valid_cost[np.isfinite(valid_cost)]
            if len(finite) > 0:
                cost = np.clip(cost, 0, np.percentile(finite, 95))

        for i in range(h):
            for j in range(w - 1):
                if overlap_mask[i, j] > 0 and overlap_mask[i, j + 1] > 0:
                    weight = (cost[i, j] + cost[i, j + 1]) / 2
                    g.add_edge(nodes[i * w + j], nodes[i * w + j + 1], weight, weight)

        for i in range(h - 1):
            for j in range(w):
                if overlap_mask[i, j] > 0 and overlap_mask[i + 1, j] > 0:
                    weight = (cost[i, j] + cost[i + 1, j]) / 2
                    g.add_edge(nodes[i * w + j], nodes[(i + 1) * w + j], weight, weight)

        for i in range(h):
            for j in range(w):
                if overlap_mask[i, j] == 0:
                    continue
                idx = i * w + j
                if j == 0 or overlap_mask[i, j - 1] == 0:
                    g.add_tedge(nodes[idx], self.terminal_weight, 0)
                if j == w - 1 or overlap_mask[i, j + 1] == 0:
                    g.add_tedge(nodes[idx], 0, self.terminal_weight)

        g.maxflow()

        seam_mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                if overlap_mask[i, j] > 0 and g.get_segment(nodes[i * w + j]) == 0:
                    seam_mask[i, j] = 255

        return seam_mask
