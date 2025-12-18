import warnings
from collections import OrderedDict
from typing import List, Tuple, Optional, Union

import cv2 as cv
import numpy as np

from .blender import Blender
from .stitching_error import StitchingWarning


class SeamFinder:
    """
    https://docs.opencv.org/4.x/d7/d09/classcv_1_1detail_1_1SeamFinder.html
    """

    # OpenCV built-in seam finders
    OPENCV_SEAM_FINDER_CHOICES = OrderedDict()
    OPENCV_SEAM_FINDER_CHOICES["dp_color"] = lambda: cv.detail_DpSeamFinder("COLOR")
    OPENCV_SEAM_FINDER_CHOICES["dp_colorgrad"] = lambda: cv.detail_DpSeamFinder("COLOR_GRAD")
    OPENCV_SEAM_FINDER_CHOICES["gc_color"] = lambda: cv.detail_GraphCutSeamFinder("COST_COLOR")
    OPENCV_SEAM_FINDER_CHOICES["gc_colorgrad"] = lambda: cv.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    OPENCV_SEAM_FINDER_CHOICES["voronoi"] = lambda: cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM)
    OPENCV_SEAM_FINDER_CHOICES["no"] = lambda: cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)

    # Custom seam finder choices (lazy loaded)
    CUSTOM_SEAM_FINDER_CHOICES = OrderedDict()
    CUSTOM_SEAM_FINDER_CHOICES["custom_dp_color"] = ("DPSeamSolver", "color")
    CUSTOM_SEAM_FINDER_CHOICES["custom_dp_colorgrad"] = ("DPSeamSolver", "color_grad")
    CUSTOM_SEAM_FINDER_CHOICES["custom_gc_color"] = ("GraphCutSeamSolver", "color")
    CUSTOM_SEAM_FINDER_CHOICES["custom_gc_colorgrad"] = ("GraphCutSeamSolver", "color_grad")
    CUSTOM_SEAM_FINDER_CHOICES["optimized_dp_color"] = ("OptimizedDPSeamSolver", "color")
    CUSTOM_SEAM_FINDER_CHOICES["optimized_dp_colorgrad"] = ("OptimizedDPSeamSolver", "color_grad")
    CUSTOM_SEAM_FINDER_CHOICES["optimized_gc_color"] = ("OptimizedGraphCutSeamSolver", "color")
    CUSTOM_SEAM_FINDER_CHOICES["optimized_gc_colorgrad"] = ("OptimizedGraphCutSeamSolver", "color_grad")
    CUSTOM_SEAM_FINDER_CHOICES["numba_dp_color"] = ("NumbaAcceleratedDPSolver", "color")
    CUSTOM_SEAM_FINDER_CHOICES["numba_dp_colorgrad"] = ("NumbaAcceleratedDPSolver", "color_grad")

    # Combined choices for validation
    SEAM_FINDER_CHOICES = OrderedDict()
    SEAM_FINDER_CHOICES.update({k: v for k, v in OPENCV_SEAM_FINDER_CHOICES.items()})
    SEAM_FINDER_CHOICES.update({k: k for k in CUSTOM_SEAM_FINDER_CHOICES.keys()})

    DEFAULT_SEAM_FINDER = "dp_color"

    def __init__(self, finder: str = DEFAULT_SEAM_FINDER):
        """
        Initialize seam finder.

        Args:
            finder: Name of the seam finding algorithm to use
        """
        if finder not in self.SEAM_FINDER_CHOICES:
            raise ValueError(
                f"Invalid seam finder: {finder}. "
                f"Available choices: {list(self.SEAM_FINDER_CHOICES.keys())}"
            )

        self.finder_name = finder
        self.is_custom = finder in self.CUSTOM_SEAM_FINDER_CHOICES

        if self.is_custom:
            self.finder = self._create_custom_finder(finder)
        else:
            self.finder = self.OPENCV_SEAM_FINDER_CHOICES[finder]()

    def _create_custom_finder(self, finder_name: str):
        """Create a custom seam finder instance."""
        from .seam_solver import (
            DPSeamSolver, GraphCutSeamSolver,
            OptimizedDPSeamSolver, OptimizedGraphCutSeamSolver,
            NumbaAcceleratedDPSolver
        )

        solver_map = {
            "DPSeamSolver": DPSeamSolver,
            "GraphCutSeamSolver": GraphCutSeamSolver,
            "OptimizedDPSeamSolver": OptimizedDPSeamSolver,
            "OptimizedGraphCutSeamSolver": OptimizedGraphCutSeamSolver,
            "NumbaAcceleratedDPSolver": NumbaAcceleratedDPSolver,
        }

        solver_name, cost_type = self.CUSTOM_SEAM_FINDER_CHOICES[finder_name]
        return solver_map[solver_name](cost_type=cost_type)

    def find(self, imgs: List[np.ndarray], corners: List[Tuple[int, int]],
             masks: List[np.ndarray]) -> List[np.ndarray]:
        """
        Find seams for the given images.
        """
        imgs_float = [img.astype(np.float32) for img in imgs]

        if self.is_custom:
            return self.finder.find(imgs_float, corners, masks)
        else:
            return self.finder.find(imgs_float, corners, masks)

    @staticmethod
    def resize(seam_mask, mask):
        dilated_mask = cv.dilate(seam_mask, None)
        resized_seam_mask = cv.resize(
            dilated_mask, (mask.shape[1], mask.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT
        )
        return cv.bitwise_and(resized_seam_mask, mask)

    @staticmethod
    def draw_seam_mask(img, seam_mask, color=(0, 0, 0)):
        seam_mask = cv.UMat.get(seam_mask)
        overlaid_img = np.copy(img)
        overlaid_img[seam_mask == 0] = color
        return overlaid_img

    @staticmethod
    def draw_seam_polygons(panorama, blended_seam_masks, alpha=0.5):
        return add_weighted_image(panorama, blended_seam_masks, alpha)

    @staticmethod
    def draw_seam_lines(panorama, blended_seam_masks, linesize=1, color=(0, 0, 255)):
        seam_lines = SeamFinder.extract_seam_lines(blended_seam_masks, linesize)
        panorama_with_seam_lines = panorama.copy()
        panorama_with_seam_lines[seam_lines == 255] = color
        return panorama_with_seam_lines

    @staticmethod
    def extract_seam_lines(blended_seam_masks, linesize=1):
        seam_lines = cv.Canny(np.uint8(blended_seam_masks), 100, 200)
        seam_indices = (seam_lines == 255).nonzero()
        seam_lines = remove_invalid_line_pixels(
            seam_indices, seam_lines, blended_seam_masks
        )
        kernelsize = linesize + linesize - 1
        kernel = np.ones((kernelsize, kernelsize), np.uint8)
        return cv.dilate(seam_lines, kernel)

    @staticmethod
    def blend_seam_masks(
        seam_masks,
        corners,
        sizes,
        colors=(
            (255, 000, 000),  # Red
            (000, 000, 255),  # Blue
            (000, 255, 000),  # Green
            (000, 255, 255),  # Yellow
            (255, 000, 255),  # Purple
            (128, 128, 255),  # Pink
            (128, 128, 128),  # Gray
            (000, 000, 128),  # Dark Blue
            (000, 128, 255),  # Light Blue
        ),
    ):
        imgs = colored_img_generator(sizes, colors)
        blended_seam_masks, _ = Blender.create_panorama(
            imgs, seam_masks, corners, sizes
        )
        return blended_seam_masks


def colored_img_generator(sizes, colors):
    if len(sizes) + 1 > len(colors):
        warnings.warn(
            "Without additional colors, there will be seam masks with identical colors",  # noqa: E501
            StitchingWarning,
        )

    for idx, size in enumerate(sizes):
        yield create_img_by_size(size, colors[idx % len(colors)])


def create_img_by_size(size, color=(0, 0, 0)):
    width, height = size
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = color
    return img


def add_weighted_image(img1, img2, alpha):
    return cv.addWeighted(img1, alpha, img2, (1.0 - alpha), 0.0)


def remove_invalid_line_pixels(indices, lines, mask):
    for x, y in zip(*indices):
        if check_if_pixel_or_neighbor_is_black(mask, x, y):
            lines[x, y] = 0
    return lines


def check_if_pixel_or_neighbor_is_black(img, x, y):
    check = [
        is_pixel_black(img, x, y),
        is_pixel_black(img, x + 1, y),
        is_pixel_black(img, x - 1, y),
        is_pixel_black(img, x, y + 1),
        is_pixel_black(img, x, y - 1),
    ]
    return any(check)


def is_pixel_black(img, x, y):
    return np.all(get_pixel_value(img, x, y) == 0)


def get_pixel_value(img, x, y):
    try:
        return img[x, y]
    except IndexError:
        pass
