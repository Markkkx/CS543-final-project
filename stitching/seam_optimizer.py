from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import numpy as np
import cv2 as cv


class ImagePyramid:
    def __init__(self, num_levels: int = 3, scale_factor: float = 0.5):
        self.num_levels = num_levels
        self.scale_factor = scale_factor

    def build(self, img: np.ndarray) -> List[np.ndarray]:
        pyramid = [img]
        current = img

        for _ in range(self.num_levels - 1):
            h, w = current.shape[:2]
            new_h = max(1, int(h * self.scale_factor))
            new_w = max(1, int(w * self.scale_factor))

            if new_h < 10 or new_w < 10:
                break

            current = cv.resize(current, (new_w, new_h),
                                interpolation=cv.INTER_AREA)
            pyramid.append(current)

        return pyramid

    def build_mask_pyramid(self, mask: np.ndarray) -> List[np.ndarray]:
        pyramid = [mask]
        current = mask

        for _ in range(self.num_levels - 1):
            h, w = current.shape[:2]
            new_h = max(1, int(h * self.scale_factor))
            new_w = max(1, int(w * self.scale_factor))

            if new_h < 10 or new_w < 10:
                break

            current = cv.resize(current, (new_w, new_h),
                                interpolation=cv.INTER_NEAREST)
            pyramid.append(current)

        return pyramid


class ParallelSeamProcessor:
    def __init__(self, max_workers: Optional[int] = None,
                 use_processes: bool = False):
        self.max_workers = max_workers
        self.use_processes = use_processes

    def process_pairs(self, process_func, pairs: List[Tuple],
                      **kwargs) -> List[Any]:
        Executor = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        with Executor(max_workers=self.max_workers) as executor:
            futures = []
            for pair in pairs:
                future = executor.submit(process_func, *pair, **kwargs)
                futures.append(future)

            results = [f.result() for f in futures]

        return results


class CostMatrixCache:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []

    def _make_key(self, img1: np.ndarray, img2: np.ndarray) -> str:
        key_data = (
            img1.shape, img2.shape,
            img1[0, 0, 0] if img1.size > 0 else 0,
            img2[0, 0, 0] if img2.size > 0 else 0,
            img1.sum() % 10000,
            img2.sum() % 10000,
        )
        return str(hash(key_data))

    def get(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        key = self._make_key(img1, img2)
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, img1: np.ndarray, img2: np.ndarray,
            cost: np.ndarray):
        key = self._make_key(img1, img2)
        while len(self._cache) >= self.max_size:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]

        self._cache[key] = cost
        self._access_order.append(key)

    def clear(self):
        self._cache.clear()
        self._access_order.clear()


class MemoryEfficientSeamFinder:
    def __init__(self, tile_size: int = 1024, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap

    def find_tiled(self, imgs: List[np.ndarray],
                   corners: List[Tuple[int, int]],
                   masks: List[np.ndarray],
                   solver) -> List[np.ndarray]:
        n = len(imgs)
        seam_masks = [mask.copy() for mask in masks]

        for i in range(n):
            for j in range(i + 1, n):
                self._process_pair_tiled(
                    imgs[i], seam_masks[i], corners[i],
                    imgs[j], seam_masks[j], corners[j],
                    solver
                )

        return seam_masks

    def _process_pair_tiled(self, img1, mask1, corner1,
                            img2, mask2, corner2, solver):
        x1, y1 = corner1
        x2, y2 = corner2
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        inter_left = max(x1, x2)
        inter_top = max(y1, y2)
        inter_right = min(x1 + w1, x2 + w2)
        inter_bottom = min(y1 + h1, y2 + h2)

        if inter_left >= inter_right or inter_top >= inter_bottom:
            return

        overlap_w = inter_right - inter_left
        overlap_h = inter_bottom - inter_top

        if overlap_w <= self.tile_size and overlap_h <= self.tile_size:
            solver._process_pair(img1, mask1, corner1, img2, mask2, corner2)
            return

        tiles_x = (overlap_w + self.tile_size - 1) // self.tile_size
        tiles_y = (overlap_h + self.tile_size - 1) // self.tile_size

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                tile_left = inter_left + tx * self.tile_size - self.overlap
                tile_top = inter_top + ty * self.tile_size - self.overlap
                tile_right = min(inter_right, inter_left + (tx + 1) * self.tile_size + self.overlap)
                tile_bottom = min(inter_bottom, inter_top + (ty + 1) * self.tile_size + self.overlap)

                tile_left = max(tile_left, inter_left)
                tile_top = max(tile_top, inter_top)

                # Extract tile regions
                roi1 = (tile_left - x1, tile_top - y1,
                        tile_right - x1, tile_bottom - y1)
                roi2 = (tile_left - x2, tile_top - y2,
                        tile_right - x2, tile_bottom - y2)

                tile_img1 = img1[roi1[1]:roi1[3], roi1[0]:roi1[2]]
                tile_mask1 = mask1[roi1[1]:roi1[3], roi1[0]:roi1[2]]
                tile_img2 = img2[roi2[1]:roi2[3], roi2[0]:roi2[2]]
                tile_mask2 = mask2[roi2[1]:roi2[3], roi2[0]:roi2[2]]

                if tile_img1.size == 0 or tile_img2.size == 0:
                    continue

                # Process tile
                tile_corner1 = (tile_left, tile_top)
                tile_corner2 = (tile_left, tile_top)

                # Copy masks for processing
                tile_mask1_copy = tile_mask1.copy()
                tile_mask2_copy = tile_mask2.copy()

                solver._process_pair(
                    tile_img1, tile_mask1_copy, tile_corner1,
                    tile_img2, tile_mask2_copy, tile_corner2
                )

                # Update original masks from tile results
                # Only update the non-overlapping core of the tile
                core_top = self.overlap if ty > 0 else 0
                core_left = self.overlap if tx > 0 else 0
                core_bottom = tile_mask1_copy.shape[0] - (self.overlap if ty < tiles_y - 1 else 0)
                core_right = tile_mask1_copy.shape[1] - (self.overlap if tx < tiles_x - 1 else 0)

                mask1_region = mask1[roi1[1] + core_top:roi1[1] + core_bottom,
                                     roi1[0] + core_left:roi1[0] + core_right]
                mask2_region = mask2[roi2[1] + core_top:roi2[1] + core_bottom,
                                     roi2[0] + core_left:roi2[0] + core_right]

                mask1_region[:] = tile_mask1_copy[core_top:core_bottom, core_left:core_right]
                mask2_region[:] = tile_mask2_copy[core_top:core_bottom, core_left:core_right]


def compute_cost_vectorized(img1: np.ndarray, img2: np.ndarray,
                            include_gradient: bool = False) -> np.ndarray:
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    cost = np.sum(diff ** 2, axis=2)

    if include_gradient:
        if len(img1.shape) == 3:
            gray1 = cv.cvtColor(img1.astype(np.float32), cv.COLOR_BGR2GRAY)
            gray2 = cv.cvtColor(img2.astype(np.float32), cv.COLOR_BGR2GRAY)
        else:
            gray1 = img1.astype(np.float32)
            gray2 = img2.astype(np.float32)

        grad1_x = cv.Scharr(gray1, cv.CV_32F, 1, 0)
        grad1_y = cv.Scharr(gray1, cv.CV_32F, 0, 1)
        grad2_x = cv.Scharr(gray2, cv.CV_32F, 1, 0)
        grad2_y = cv.Scharr(gray2, cv.CV_32F, 0, 1)
        grad_cost = (grad1_x - grad2_x) ** 2 + (grad1_y - grad2_y) ** 2
        cost += grad_cost

    return cost


def find_seam_dp_optimized(cost: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
    h, w = cost.shape
    dp = np.full((h, w), np.inf, dtype=np.float32)
    parent = np.zeros((h, w), dtype=np.int32)
    dp[0] = np.where(mask[0] > 0, cost[0], np.inf)

    for i in range(1, h):
        valid = mask[i] > 0
        left = np.roll(dp[i-1], 1)
        left[0] = np.inf
        right = np.roll(dp[i-1], -1)
        right[-1] = np.inf
        center = dp[i-1]
        predecessors = np.vstack([left, center, right])
        min_idx = np.argmin(predecessors, axis=0)
        min_val = np.min(predecessors, axis=0)
        dp[i] = np.where(valid, cost[i] + min_val, np.inf)
        parent[i] = np.where(valid, np.arange(w) + min_idx - 1, 0)
        parent[i] = np.clip(parent[i], 0, w - 1)
    seam = np.zeros(h, dtype=np.int32)
    valid_last = dp[h-1].copy()
    valid_last[mask[h-1] == 0] = np.inf
    seam[h-1] = np.argmin(valid_last)

    for i in range(h-2, -1, -1):
        seam[i] = parent[i+1, seam[i+1]]

    return seam


class AdaptiveSeamFinder:
    def __init__(self):
        self.size_thresholds = {
            'small': 500 * 500,      # Use full resolution DP
            'medium': 2000 * 2000,   # Use optimized DP
            'large': float('inf'),   # Use multi-scale + optimized
        }

    def find(self, imgs: List[np.ndarray],
             corners: List[Tuple[int, int]],
             masks: List[np.ndarray]) -> List[np.ndarray]:
        total_pixels = self._estimate_overlap_size(imgs, corners, masks)
        if total_pixels < self.size_thresholds['small']:
            from .seam_solver import DPSeamSolver
            solver = DPSeamSolver()
        elif total_pixels < self.size_thresholds['medium']:
            from .seam_solver import OptimizedDPSeamSolver
            solver = OptimizedDPSeamSolver()
        else:
            from .seam_solver import OptimizedDPSeamSolver
            solver = OptimizedDPSeamSolver(scale_factor=0.25, min_size=50)

        return solver.find(imgs, corners, masks)

    def _estimate_overlap_size(self, imgs, corners, masks) -> int:
        n = len(imgs)
        total = 0

        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = corners[i]
                x2, y2 = corners[j]
                h1, w1 = imgs[i].shape[:2]
                h2, w2 = imgs[j].shape[:2]

                inter_w = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                inter_h = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

                total += inter_w * inter_h

        return total
