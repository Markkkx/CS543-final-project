import time
import tracemalloc
import urllib.request
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "result" / "photometric"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Result dir:", RESULT_DIR)


def _download_if_needed(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    print("Downloading:", url)
    urllib.request.urlretrieve(url, str(out_path))
    return out_path


# Photometric Compensation
def photometric_compensation(
    images: List[np.ndarray], overlap_masks: List[np.ndarray], min_pixels: int = 50
) -> List[np.ndarray]:
    """
    Sequential photometric compensation using linear gain and offset.

    The first image is treated as reference. Each subsequent image is
    compensated with respect to the previous one using their overlap region.

    I_corrected = gain * I + offset

    Args:
        images:
            List of warped images (BGR, uint8).
        overlap_masks:
            overlap_masks[i] is the overlap mask between images[i] and images[i+1].
        min_pixels:
            Minimum number of overlap pixels required for stable estimation.

    Returns:
        compensated_images:
            List of photometrically compensated images.
    """
    if len(images) == 0:
        return []

    compensated = [images[0].copy()]

    for i in range(1, len(images)):
        img_ref = compensated[i - 1].astype(np.float32)
        img_cur = images[i].astype(np.float32)

        mask = overlap_masks[i - 1]
        if mask is None or np.count_nonzero(mask) < min_pixels:
            compensated.append(images[i])
            continue

        mask_bool = mask > 0

        ref_pixels = img_ref[mask_bool]
        cur_pixels = img_cur[mask_bool]

        if ref_pixels.shape[0] < min_pixels:
            compensated.append(images[i])
            continue

        mean_ref = ref_pixels.mean(axis=0)
        mean_cur = cur_pixels.mean(axis=0)

        gain = mean_ref / (mean_cur + 1e-6)
        offset = mean_ref - gain * mean_cur

        corrected = img_cur * gain + offset
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        compensated.append(corrected)

    return compensated


# Multiband Blending
def multiband_blending(
    images: List[np.ndarray], masks: List[np.ndarray], num_levels: int = 5
) -> np.ndarray:
    """
    Multiband blending using Laplacian pyramids.

    Args:
        images:
            List, warped & photometrically compensated images.
        masks:
            List, seam masks(uint8, 0 or 255), same length as images.
        num_levels:
            Number of pyramid levels.

    Returns:
        blended image (uint8).
    """
    assert len(images) == len(masks), "Images and masks should have same length."

    images_f = [img.astype(np.float32) / 255.0 for img in images]
    masks_f = [mask.astype(np.float32) / 255.0 for mask in masks]

    # Gaussian pyramids
    gp_imgs = []
    gp_masks = []

    for img, mask in zip(images_f, masks_f):
        gp_i = [img]
        gp_m = [mask]

        for _ in range(num_levels):
            gp_i.append(cv2.pyrDown(gp_i[-1]))
            gp_m.append(cv2.pyrDown(gp_m[-1]))

        gp_imgs.append(gp_i)
        gp_masks.append(gp_m)

    # Laplacian pyramids
    lp_imgs = []
    for gp in gp_imgs:
        lp = []
        for i in range(num_levels):
            size = (gp[i].shape[1], gp[i].shape[0])
            up = cv2.pyrUp(gp[i + 1], dstsize=size)
            lp.append(gp[i] - up)
        lp.append(gp[num_levels])
        lp_imgs.append(lp)

    # Blend pyramids
    blended_pyramid = []
    for level in range(num_levels + 1):
        blended = np.zeros_like(lp_imgs[0][level])
        for i in range(len(images)):
            m = gp_masks[i][level]
            if m.ndim == 2:
                m = m[..., None]
            blended += lp_imgs[i][level] * m
        blended_pyramid.append(blended)

    result = blended_pyramid[-1]
    for i in range(num_levels - 1, -1, -1):
        size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + blended_pyramid[i]

    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return result


# =========== overlap_metrics =======
def compute_overlap_metrics(img1, img2, overlap_mask):
    """
    caculate PSNR and SSIM on overlap region only.
    """
    mask = overlap_mask > 0
    if np.count_nonzero(mask) < 50:
        return None, None

    # Apply mask
    img1_o = img1[mask]
    img2_o = img2[mask]

    # PSNR
    psnr = peak_signal_noise_ratio(img1_o, img2_o, data_range=255)

    # SSIM needs full images, mask by zeroing outside
    img1_m = img1.copy()
    img2_m = img2.copy()
    img1_m[~mask] = 0
    img2_m[~mask] = 0

    ssim = structural_similarity(img1_m, img2_m, channel_axis=2, data_range=255)

    return psnr, ssim


# Benchmark
def main():
    print("Photometric & blending benchmark started.")

    # 1. read urls from tests/testdata/TEST_IMAGES.txt
    txt_path = PROJECT_ROOT / "tests" / "testdata" / "TEST_IMAGES.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing {txt_path}")

    urls = []
    with open(txt_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("#"):
                continue
            urls.append(line)

    if len(urls) < 2:
        raise ValueError("TEST_IMAGES.txt must contain at least 2 image URLs")

    # 2. download 2 images
    download_dir = PROJECT_ROOT / "tests" / "testdata" / "downloaded"
    url1 = urls[0]
    url2 = urls[1]

    ext1 = Path(url1).suffix if Path(url1).suffix != "" else ".jpg"
    ext2 = Path(url2).suffix if Path(url2).suffix != "" else ".jpg"

    img1_path = _download_if_needed(url1, download_dir / f"img1{ext1}")
    img2_path = _download_if_needed(url2, download_dir / f"img2{ext2}")

    # 3 read images
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        raise ValueError("Failed to read downloaded images )")

    # 4. make a simple overlap mask on common region
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h = min(gray1.shape[0], gray2.shape[0])
    w = min(gray1.shape[1], gray2.shape[1])

    gray1_c = gray1[:h, :w]
    gray2_c = gray2[:h, :w]

    img1_c = img1[:h, :w]
    img2_c = img2[:h, :w]

    overlap = ((gray1_c > 0) & (gray2_c > 0)).astype(np.uint8) * 255

    warped_images = [img1_c, img2_c]
    overlap_masks = [overlap]

    # 5. simple seam masks (50/50 split) so multiband blending can run
    h, w = img1_c.shape[:2]
    seam_mask1 = np.zeros((h, w), dtype=np.uint8)
    seam_mask1[:, : w // 2] = 255
    seam_mask2 = 255 - seam_mask1
    seam_masks = [seam_mask1, seam_mask2]

    tracemalloc.start()
    t0 = time.perf_counter()

    compensated_images = photometric_compensation(warped_images, overlap_masks)
    # Overlap quality metrics
    psnr_before, ssim_before = compute_overlap_metrics(
        warped_images[0], warped_images[1], overlap_masks[0]
    )

    psnr_after, ssim_after = compute_overlap_metrics(
        compensated_images[0], compensated_images[1], overlap_masks[0]
    )

    print("Overlap metrics:")
    print(f"Before compensation: PSNR={psnr_before:.2f}, SSIM={ssim_before:.4f}")
    print(f"After  compensation: PSNR={psnr_after:.2f}, SSIM={ssim_after:.4f}")

    metrics_path = RESULT_DIR / "overlap_metrics.txt"
    with open(RESULT_DIR / "overlap_metrics.txt", "w") as f:
        f.write(f"PSNR_before: {psnr_before:.2f}\n")
        f.write(f"SSIM_before: {ssim_before:.4f}\n")
        f.write(f"PSNR_after: {psnr_after:.2f}\n")
        f.write(f"SSIM_after: {ssim_after:.4f}\n")

    final_panorama = multiband_blending(compensated_images, seam_masks)

    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    out_path = RESULT_DIR / "panorama_photometric.png"
    cv2.imwrite(str(out_path), final_panorama)

    print("Downloaded 1:", img1_path)
    print("Downloaded 2:", img2_path)
    print("Saved:", out_path)
    print(f"Elapsed time: {elapsed:.4f} s")
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
    print("Benchmark finished.")


if __name__ == "__main__":
    main()
