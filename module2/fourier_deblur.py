"""
Fourier blur/deblur pipeline for Module 2.

Usage:
    python -m module2.fourier_deblur --input path/to/image.jpg --output-dir module2/output --json

Steps:
1) Load input, convert to grayscale float in [0,1]
2) Apply Gaussian blur (13x13, sigma=2.4)
3) Build same Gaussian PSF, shift, and compute Wiener deconvolution (K=1e-3)
4) Save blurred, restored, and montage PNGs
5) Report PSNR for blurred and restored versus original
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"

KERNEL_SIZE = 13
SIGMA = 2.4
WIENER_K = 1e-3


def to_gray_float(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    g1d = cv2.getGaussianKernel(ksize=size, sigma=sigma)
    kernel = g1d @ g1d.T
    return kernel


def wiener_deblur(blurred: np.ndarray, kernel: np.ndarray, k: float) -> np.ndarray:
    """
    Wiener deconvolution in the frequency domain.
    blurred, kernel are float32 in [0,1]; kernel is small.
    """
    h, w = blurred.shape
    psf = np.zeros_like(blurred, dtype=np.float32)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    psf[ph : ph + kh, pw : pw + kw] = kernel
    psf = np.fft.ifftshift(psf)

    H = np.fft.fft2(psf)
    G = np.fft.fft2(blurred)

    H_conj = np.conj(H)
    denom = (np.abs(H) ** 2) + k
    F_hat = (H_conj / denom) * G
    f_rec = np.fft.ifft2(F_hat)
    f_rec = np.real(f_rec)
    f_rec = np.clip(f_rec, 0.0, 1.0)
    return f_rec.astype(np.float32)


def psnr(orig: np.ndarray, other: np.ndarray) -> float:
    return float(cv2.PSNR(orig, other))


def make_montage(images: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Stack three grayscale uint8 images horizontally as BGR."""
    imgs_bgr = [cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) for im in images]
    return cv2.hconcat(imgs_bgr)


def process_image(input_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    color = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(f"Failed to read image: {input_path}")

    gray = to_gray_float(color)

    # Blur
    blurred = cv2.GaussianBlur(gray, (KERNEL_SIZE, KERNEL_SIZE), SIGMA)

    # Wiener deblur
    kernel = gaussian_kernel(KERNEL_SIZE, SIGMA)
    restored = wiener_deblur(blurred, kernel, WIENER_K)

    # Convert to 8-bit for saving/PSNR
    orig_u8 = (gray * 255.0).round().astype(np.uint8)
    blur_u8 = (blurred * 255.0).round().astype(np.uint8)
    rest_u8 = (restored * 255.0).round().astype(np.uint8)

    psnr_blur = psnr(orig_u8, blur_u8)
    psnr_restore = psnr(orig_u8, rest_u8)

    stem = input_path.stem
    blur_path = output_dir / f"{stem}_gaussian_blur.png"
    restore_path = output_dir / f"{stem}_fourier_restore.png"
    montage_path = output_dir / f"{stem}_fourier_montage.png"

    cv2.imwrite(str(blur_path), blur_u8)
    cv2.imwrite(str(restore_path), rest_u8)
    montage = make_montage((orig_u8, blur_u8, rest_u8))
    cv2.imwrite(str(montage_path), montage)

    return {
        "blur_path": str(blur_path),
        "restore_path": str(restore_path),
        "montage_path": str(montage_path),
        "psnr_blur": psnr_blur,
        "psnr_restore": psnr_restore,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fourier Gaussian blur and Wiener deblur pipeline.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for results")
    parser.add_argument("--json", action="store_true", help="Print JSON results")
    args = parser.parse_args()

    result = process_image(Path(args.input), Path(args.output_dir))

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Blurred:", result["blur_path"])
        print("Restored:", result["restore_path"])
        print("Montage:", result["montage_path"])
        print(f"PSNR (orig vs blur): {result['psnr_blur']:.2f} dB")
        print(f"PSNR (orig vs restore): {result['psnr_restore']:.2f} dB")


if __name__ == "__main__":
    main()
