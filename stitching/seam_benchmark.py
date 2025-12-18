import time
import tracemalloc
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json
import os

import cv2 as cv
import numpy as np

from .seam_finder import SeamFinder


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    solver_name: str
    execution_time: float  # seconds
    peak_memory: float  # MB
    success: bool
    error_message: Optional[str] = None
    seam_masks: Optional[List[np.ndarray]] = None
    image_count: int = 0
    total_pixels: int = 0
    overlap_pixels: int = 0
    seam_quality: Optional[float] = None  # Lower is better


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results across multiple runs."""
    solver_name: str
    avg_time: float
    std_time: float
    min_time: float
    max_time: float
    avg_memory: float
    std_memory: float
    success_rate: float
    avg_quality: Optional[float] = None
    runs: List[BenchmarkResult] = field(default_factory=list)


class SeamBenchmark:
    OPENCV_SOLVERS = ["dp_color", "dp_colorgrad", "gc_color", "gc_colorgrad", "voronoi"]
    CUSTOM_SOLVERS = ["custom_dp_color", "custom_dp_colorgrad"]
    OPTIMIZED_SOLVERS = ["optimized_dp_color", "optimized_dp_colorgrad"]
    GRAPHCUT_CUSTOM = ["custom_gc_color", "custom_gc_colorgrad"]
    GRAPHCUT_OPTIMIZED = ["optimized_gc_color", "optimized_gc_colorgrad"]

    def __init__(self, solvers: Optional[List[str]] = None,
                 include_graphcut: bool = True,
                 num_runs: int = 3,
                 warmup_runs: int = 1,
                 measure_quality: bool = True):
        if solvers is None:
            self.solvers = (self.OPENCV_SOLVERS + self.CUSTOM_SOLVERS +
                           self.OPTIMIZED_SOLVERS)
            if include_graphcut:
                self.solvers += self.GRAPHCUT_CUSTOM + self.GRAPHCUT_OPTIMIZED
        else:
            self.solvers = solvers

        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.measure_quality = measure_quality
        self.test_cases: List[Dict] = []

    def add_test_case(self, imgs: List[np.ndarray],
                      corners: List[Tuple[int, int]],
                      masks: List[np.ndarray],
                      name: str = "test"):
        total_pixels = sum(img.shape[0] * img.shape[1] for img in imgs)
        overlap_pixels = self._estimate_overlap(imgs, corners, masks)

        self.test_cases.append({
            "imgs": imgs,
            "corners": corners,
            "masks": masks,
            "name": name,
            "image_count": len(imgs),
            "total_pixels": total_pixels,
            "overlap_pixels": overlap_pixels,
        })

    def _estimate_overlap(self, imgs, corners, masks) -> int:
        """Estimate total overlap pixels."""
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

    def measure_seam_quality(self, imgs: List[np.ndarray],
                             corners: List[Tuple[int, int]],
                             seam_masks: List[np.ndarray]) -> float:
        if seam_masks is None:
            return float('inf')

        total_discontinuity = 0
        count = 0

        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                disc = self._compute_pairwise_discontinuity(
                    imgs[i], corners[i], seam_masks[i],
                    imgs[j], corners[j], seam_masks[j]
                )
                if disc is not None:
                    total_discontinuity += disc
                    count += 1

        return total_discontinuity / max(count, 1)

    def _compute_pairwise_discontinuity(self, img1, corner1, mask1,
                                         img2, corner2, mask2) -> Optional[float]:
        x1, y1 = corner1
        x2, y2 = corner2
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Find overlap region
        inter_left = max(x1, x2)
        inter_top = max(y1, y2)
        inter_right = min(x1 + w1, x2 + w2)
        inter_bottom = min(y1 + h1, y2 + h2)

        if inter_left >= inter_right or inter_top >= inter_bottom:
            return None

        # Extract overlapping regions
        roi1 = (inter_left - x1, inter_top - y1,
                inter_right - x1, inter_bottom - y1)
        roi2 = (inter_left - x2, inter_top - y2,
                inter_right - x2, inter_bottom - y2)

        region1 = img1[roi1[1]:roi1[3], roi1[0]:roi1[2]]
        region2 = img2[roi2[1]:roi2[3], roi2[0]:roi2[2]]

        # Handle UMat masks
        m1 = mask1[roi1[1]:roi1[3], roi1[0]:roi1[2]]
        m2 = mask2[roi2[1]:roi2[3], roi2[0]:roi2[2]]

        if hasattr(m1, 'get'):
            m1 = m1.get()
        if hasattr(m2, 'get'):
            m2 = m2.get()

        # Find seam boundary (where masks differ)
        seam_boundary = cv.Canny(m1.astype(np.uint8), 50, 150)

        if np.sum(seam_boundary) == 0:
            return None

        # Compute color difference at boundary
        boundary_points = np.where(seam_boundary > 0)
        if len(boundary_points[0]) == 0:
            return None

        color_diff = 0
        for y, x in zip(boundary_points[0], boundary_points[1]):
            c1 = region1[y, x].astype(float)
            c2 = region2[y, x].astype(float)
            color_diff += np.sqrt(np.sum((c1 - c2) ** 2))

        return color_diff / len(boundary_points[0])

    def run_single(self, solver_name: str,
                   imgs: List[np.ndarray],
                   corners: List[Tuple[int, int]],
                   masks: List[np.ndarray]) -> BenchmarkResult:
        masks_copy = [mask.copy() for mask in masks]

        try:
            finder = SeamFinder(solver_name)

            tracemalloc.start()
            start_time = time.perf_counter()

            seam_masks = finder.find(imgs, corners, masks_copy)

            end_time = time.perf_counter()
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Measure quality if enabled
            quality = None
            if self.measure_quality:
                try:
                    quality = self.measure_seam_quality(imgs, corners, seam_masks)
                except Exception:
                    quality = None  # Skip quality measurement on error

            return BenchmarkResult(
                solver_name=solver_name,
                execution_time=end_time - start_time,
                peak_memory=peak_memory / (1024 * 1024),
                success=True,
                seam_masks=seam_masks,
                image_count=len(imgs),
                total_pixels=sum(img.shape[0] * img.shape[1] for img in imgs),
                seam_quality=quality,
            )

        except Exception as e:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            return BenchmarkResult(
                solver_name=solver_name,
                execution_time=0,
                peak_memory=0,
                success=False,
                error_message=str(e),
                image_count=len(imgs),
            )

    def run_benchmark(self, solver_name: str,
                      test_case: Dict) -> BenchmarkSummary:
        imgs = test_case["imgs"]
        corners = test_case["corners"]
        masks = test_case["masks"]

        # Warmup
        for _ in range(self.warmup_runs):
            self.run_single(solver_name, imgs, corners, masks)

        # Actual runs
        results = []
        for _ in range(self.num_runs):
            result = self.run_single(solver_name, imgs, corners, masks)
            results.append(result)

        successful = [r for r in results if r.success]

        if successful:
            times = [r.execution_time for r in successful]
            memories = [r.peak_memory for r in successful]
            qualities = [r.seam_quality for r in successful if r.seam_quality is not None]

            return BenchmarkSummary(
                solver_name=solver_name,
                avg_time=np.mean(times),
                std_time=np.std(times),
                min_time=np.min(times),
                max_time=np.max(times),
                avg_memory=np.mean(memories),
                std_memory=np.std(memories),
                success_rate=len(successful) / len(results),
                avg_quality=np.mean(qualities) if qualities else None,
                runs=results,
            )
        else:
            return BenchmarkSummary(
                solver_name=solver_name,
                avg_time=0, std_time=0, min_time=0, max_time=0,
                avg_memory=0, std_memory=0,
                success_rate=0,
                runs=results,
            )

    def run_all(self) -> Dict[str, Dict[str, BenchmarkSummary]]:
        results = {}

        for test_case in self.test_cases:
            case_name = test_case["name"]
            results[case_name] = {}

            print(f"\n{'='*70}")
            print(f"Test Case: {case_name}")
            print(f"Images: {test_case['image_count']}, "
                  f"Total Pixels: {test_case['total_pixels']:,}, "
                  f"Overlap: {test_case['overlap_pixels']:,}")
            print(f"{'='*70}")

            for solver_name in self.solvers:
                print(f"  {solver_name:30s}", end=" ", flush=True)
                summary = self.run_benchmark(solver_name, test_case)
                results[case_name][solver_name] = summary

                if summary.success_rate > 0:
                    quality_str = f", Q:{summary.avg_quality:.1f}" if summary.avg_quality else ""
                    print(f"Time: {summary.avg_time:.4f}s (±{summary.std_time:.4f}s), "
                          f"Mem: {summary.avg_memory:.2f}MB{quality_str}")
                else:
                    error = summary.runs[0].error_message if summary.runs else "Unknown"
                    print(f"FAILED: {error[:40]}...")

        return results

    def print_report(self, results: Dict[str, Dict[str, BenchmarkSummary]],
                     sort_by: str = "time"):
        """Print formatted benchmark report with speedup analysis."""
        print("\n" + "=" * 80)
        print("SEAM SOLVER BENCHMARK REPORT")
        print("=" * 80)

        for case_name, case_results in results.items():
            print(f"\n{'─' * 80}")
            print(f"Test Case: {case_name}")
            print(f"{'─' * 80}")
            summaries = list(case_results.items())
            summaries.sort(key=lambda x: x[1].avg_time if x[1].success_rate > 0 else float('inf'))
            opencv_dp_time = case_results.get("dp_color", BenchmarkSummary("", 0, 0, 0, 0, 0, 0, 0)).avg_time
            opencv_gc_time = case_results.get("gc_color", BenchmarkSummary("", 0, 0, 0, 0, 0, 0, 0)).avg_time

            print(f"\n{'Solver':<32} {'Time (s)':<14} {'Memory (MB)':<12} {'Quality':<10} {'vs OpenCV':<12}")
            print(f"{'-'*32} {'-'*14} {'-'*12} {'-'*10} {'-'*12}")

            for solver_name, summary in summaries:
                if summary.success_rate > 0:
                    time_str = f"{summary.avg_time:.4f}"
                    mem_str = f"{summary.avg_memory:.2f}"
                    quality_str = f"{summary.avg_quality:.1f}" if summary.avg_quality else "N/A"
                    if "gc" in solver_name and opencv_gc_time > 0:
                        speedup = opencv_gc_time / summary.avg_time
                        speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "baseline"
                    elif "dp" in solver_name and opencv_dp_time > 0:
                        speedup = opencv_dp_time / summary.avg_time
                        speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "baseline"
                    elif solver_name == "voronoi":
                        speedup_str = "N/A"
                    else:
                        speedup_str = "N/A"

                    print(f"{solver_name:<32} {time_str:<14} {mem_str:<12} {quality_str:<10} {speedup_str:<12}")
                else:
                    print(f"{solver_name:<32} {'FAILED':<14} {'-':<12} {'-':<10} {'-':<12}")
            self._print_analysis(case_name, case_results)

    def _print_analysis(self, case_name: str, case_results: Dict[str, BenchmarkSummary]):
        """Print analysis of optimization benefits."""
        print(f"\n{'─' * 40}")
        print("Optimization Analysis:")
        print(f"{'─' * 40}")

        # DP comparison
        dp_base = case_results.get("dp_color")
        dp_custom = case_results.get("custom_dp_color")
        dp_optimized = case_results.get("optimized_dp_color")

        if dp_base and dp_base.success_rate > 0:
            print(f"\nDP Seam Solvers (baseline: OpenCV dp_color = {dp_base.avg_time:.4f}s):")

            if dp_custom and dp_custom.success_rate > 0:
                ratio = dp_custom.avg_time / dp_base.avg_time
                print(f"  custom_dp_color:    {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}")

            if dp_optimized and dp_optimized.success_rate > 0:
                ratio = dp_optimized.avg_time / dp_base.avg_time
                status = "FASTER" if ratio < 1 else "slower"
                print(f"  optimized_dp_color: {ratio:.2f}x {status}")

                # Show improvement from custom to optimized
                if dp_custom and dp_custom.success_rate > 0:
                    improvement = dp_custom.avg_time / dp_optimized.avg_time
                    print(f"  Optimization speedup (custom -> optimized): {improvement:.2f}x")

        # GraphCut comparison
        gc_base = case_results.get("gc_color")
        gc_custom = case_results.get("custom_gc_color")
        gc_optimized = case_results.get("optimized_gc_color")

        if gc_base and gc_base.success_rate > 0:
            print(f"\nGraphCut Solvers (baseline: OpenCV gc_color = {gc_base.avg_time:.4f}s):")

            if gc_custom and gc_custom.success_rate > 0:
                ratio = gc_custom.avg_time / gc_base.avg_time
                print(f"  custom_gc_color:    {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}")

            if gc_optimized and gc_optimized.success_rate > 0:
                ratio = gc_optimized.avg_time / gc_base.avg_time
                status = "FASTER" if ratio < 1 else "slower"
                print(f"  optimized_gc_color: {ratio:.2f}x {status}")

    def save_results(self, results: Dict[str, Dict[str, BenchmarkSummary]],
                     filepath: str):
        """Save results to JSON."""
        output = {}
        for case_name, case_results in results.items():
            output[case_name] = {}
            for solver_name, summary in case_results.items():
                output[case_name][solver_name] = {
                    "avg_time": summary.avg_time,
                    "std_time": summary.std_time,
                    "min_time": summary.min_time,
                    "max_time": summary.max_time,
                    "avg_memory": summary.avg_memory,
                    "avg_quality": summary.avg_quality,
                    "success_rate": summary.success_rate,
                }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {filepath}")


def create_test_images(height: int, width: int, overlap_ratio: float = 0.3,
                       num_images: int = 2) -> Tuple[List, List, List]:
    """
    Create synthetic test images with controlled overlap.
    """
    imgs = []
    corners = []
    masks = []

    overlap_width = int(width * overlap_ratio)
    step = width - overlap_width

    for i in range(num_images):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        base_color = [(100 + i * 50) % 255, (50 + i * 80) % 255, (150 + i * 30) % 255]
        img[:, :] = base_color
        for x in range(width):
            factor = x / width
            img[:, x] = [int(c * (0.5 + 0.5 * factor)) for c in base_color]
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        imgs.append(img)
        corners.append((i * step, 0))
        masks.append(np.ones((height, width), dtype=np.uint8) * 255)

    return imgs, corners, masks


def run_scalability_benchmark():
    """
    Run benchmark demonstrating scalability benefits.
    """
    print("\n" + "=" * 70)
    print("SEAM OPTIMIZATION BENCHMARK")
    print("Comparing Our Optimized Solver vs OpenStitching Default (OpenCV)")
    print("=" * 70)

    all_results = {}
    print("\n" + "=" * 70)
    print("TEST 1: Scalability with Image Size (2 images)")
    print("=" * 70)

    sizes = [
        (200, 300, "2img_200x300"),
        (400, 600, "2img_400x600"),
        (800, 1200, "2img_800x1200"),
        (1200, 1800, "2img_1200x1800"),
    ]

    solvers = ["dp_color", "optimized_dp_color"]

    for height, width, name in sizes:
        imgs, corners, masks = create_test_images(height, width, overlap_ratio=0.4, num_images=2)
        benchmark = SeamBenchmark(solvers=solvers, include_graphcut=False, num_runs=3, warmup_runs=1, measure_quality=False)
        benchmark.add_test_case(imgs, corners, masks, name=name)
        results = benchmark.run_all()
        all_results[name] = results[name]
    print("\n" + "=" * 70)
    print("TEST 2: Multiple Images (Parallel Processing Advantage)")
    print("=" * 70)

    multi_configs = [
        (300, 400, 3, "3img_300x400"),
        (300, 400, 4, "4img_300x400"),
        (300, 400, 5, "5img_300x400"),
        (300, 400, 6, "6img_300x400"),
    ]

    for height, width, num_imgs, name in multi_configs:
        imgs, corners, masks = create_test_images(height, width, overlap_ratio=0.4, num_images=num_imgs)
        benchmark = SeamBenchmark(solvers=solvers, include_graphcut=False, num_runs=3, warmup_runs=1, measure_quality=False)
        benchmark.add_test_case(imgs, corners, masks, name=name)
        results = benchmark.run_all()
        all_results[name] = results[name]
    print("\n" + "=" * 70)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\n{'Test Case':<20} {'OpenCV DP':<12} {'Optimized':<12} {'Speedup':<10} {'Winner':<10}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    wins_opencv = 0
    wins_optimized = 0
    total_opencv = 0
    total_optimized = 0

    for name in all_results:
        results = all_results[name]
        opencv_time = results.get("dp_color", BenchmarkSummary("", 0, 0, 0, 0, 0, 0, 0)).avg_time
        optimized_time = results.get("optimized_dp_color", BenchmarkSummary("", 0, 0, 0, 0, 0, 0, 0)).avg_time

        total_opencv += opencv_time
        total_optimized += optimized_time

        if optimized_time > 0 and opencv_time > 0:
            if optimized_time < opencv_time:
                speedup = opencv_time / optimized_time
                winner = "OURS"
                wins_optimized += 1
            else:
                speedup = opencv_time / optimized_time
                winner = "OpenCV"
                wins_opencv += 1

            print(f"{name:<20} {opencv_time:<12.4f} {optimized_time:<12.4f} {speedup:<10.2f}x {winner:<10}")

    # Overall statistics
    print(f"\n{'─' * 70}")
    print(f"OVERALL RESULTS:")
    print(f"  - Our Optimized Solver Wins: {wins_optimized} cases")
    print(f"  - OpenCV Wins: {wins_opencv} cases")
    if total_optimized > 0:
        overall = total_opencv / total_optimized
        print(f"  - Total Time Ratio (OpenCV/Ours): {overall:.2f}x")
        if overall > 1:
            print(f"  - Our solver is {overall:.2f}x FASTER overall!")
        else:
            print(f"  - OpenCV is {1/overall:.2f}x faster overall")
    print(f"{'─' * 70}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    return all_results


def run_quick_benchmark():
    """Run a quick benchmark with default settings."""
    print("Running quick benchmark...")

    imgs, corners, masks = create_test_images(300, 400, overlap_ratio=0.4)

    benchmark = SeamBenchmark(
        solvers=[
            "dp_color",
            "gc_color",
            "voronoi",
            "custom_dp_color",
            "optimized_dp_color",
        ],
        include_graphcut=False,
        num_runs=3,
        warmup_runs=1,
    )

    benchmark.add_test_case(imgs, corners, masks, name="quick_test")
    results = benchmark.run_all()
    benchmark.print_report(results)

    return results


if __name__ == "__main__":
    run_scalability_benchmark()
