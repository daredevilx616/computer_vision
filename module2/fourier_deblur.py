from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from .generate_dataset import BASE_DIR, SCENE_DIR, generate_dataset, METADATA_PATH

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

KERNEL_SIZE = 13
SIGMA = 2.4
WIENER_K = 1e-3  # noise-to-signal ratio hyperparameter


@dataclass
class FourierResult:
    blur_path: Path
    restore_path: Path
    montage_path: Path
    psnr_blur: float
    psnr_restore: float

    def as_json(self) -> Dict[str, object]:
        def rel(path: Path) -> str:
            try:
                return str(path.relative_to(BASE_DIR))
            except ValueError:
                return str(path)

        return {
            "blur_path": rel(self.blur_path),
            "restore_path": rel(self.restore_path),
            "montage_path": rel(self.montage_path),
            "psnr_blur": self.psnr_blur,
            "psnr_restore": self.psnr_restore,
        }


def ensure_scene() -> Path:
    """Guarantee at least one scene exists and return its path."""
    if not METADATA_PATH.exists():
        generate_dataset()
    scene_path = SCENE_DIR / "scene_boardwalk.png"
    if not scene_path.exists():
        generate_dataset()
    return scene_path


def gaussian_blur(image: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def build_psf(shape: Tuple[int, int], ksize: int, sigma: float) -> np.ndarray:
    """Create a padded Gaussian point-spread function."""
    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    psf = kernel_1d @ kernel_1d.T
    psf /= psf.sum()

    pad = np.zeros(shape, dtype=np.float32)
    h, w = psf.shape
    pad[:h, :w] = psf

    # Shift PSF to center for frequency domain multiplication
    pad = np.roll(pad, -h // 2, axis=0)
    pad = np.roll(pad, -w // 2, axis=1)
    return pad


def wiener_deblur(blurred: np.ndarray, psf: np.ndarray, k: float) -> np.ndarray:
    """Perform Wiener filtering in the frequency domain."""
    blurred_fft = np.fft.fft2(blurred)
    psf_fft = np.fft.fft2(psf, s=blurred.shape)
    conj = np.conj(psf_fft)
    magnitude = np.abs(psf_fft) ** 2
    wiener_filter = conj / (magnitude + k)
    restored = np.fft.ifft2(blurred_fft * wiener_filter)
    return np.real(restored)


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = np.mean((reference - estimate) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

def run_fourier_pipeline(
    image_path: Path,
    output_dir: Path = OUTPUT_DIR,
    kernel_size: int = KERNEL_SIZE,
    sigma: float = SIGMA,
    wiener_k: float = WIENER_K,
) -> FourierResult:
    """Apply Gaussian blur and Wiener deblurring to the provided image."""
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")
    original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    blurred = gaussian_blur(original_gray, kernel_size, sigma)
    psf = build_psf(original_gray.shape, kernel_size, sigma)
    restored = wiener_deblur(blurred, psf, wiener_k)

    blurred_u8 = np.clip(blurred, 0, 255).astype(np.uint8)
    restored_u8 = np.clip(restored, 0, 255).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    blur_path = output_dir / f"{stem}_gaussian_blur.png"
    restored_path = output_dir / f"{stem}_fourier_restored.png"
    montage_path = output_dir / f"{stem}_fourier_montage.png"

    cv2.imwrite(str(blur_path), blurred_u8)
    cv2.imwrite(str(restored_path), restored_u8)
    montage = np.hstack([original_gray.astype(np.uint8), blurred_u8, restored_u8])
    cv2.imwrite(str(montage_path), montage)

    return FourierResult(
        blur_path=blur_path,
        restore_path=restored_path,
        montage_path=montage_path,
        psnr_blur=psnr(original_gray, blurred),
        psnr_restore=psnr(original_gray, restored),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gaussian blur + Fourier-domain restoration experiment.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the source image. Defaults to generated scene if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for outputs (default: {OUTPUT_DIR}).",
    )
    parser.add_argument("--kernel-size", type=int, default=KERNEL_SIZE, help="Odd Gaussian kernel size (default 13).")
    parser.add_argument("--sigma", type=float, default=SIGMA, help="Gaussian sigma (default 2.4).")
    parser.add_argument(
        "--wiener-k",
        type=float,
        default=WIENER_K,
        help="Noise-to-signal ratio constant for Wiener filter (default 1e-3).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON payload to stdout.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    image_path = args.input or ensure_scene()
    result = run_fourier_pipeline(image_path, args.output_dir, args.kernel_size, args.sigma, args.wiener_k)

    def rel(path: Path) -> str:
        try:
            return str(path.relative_to(BASE_DIR))
        except ValueError:
            return str(path)

    def emit(message: str) -> None:
        if args.json:
            print(message, file=sys.stderr)
        else:
            print(message)

    emit("Fourier deblurring experiment completed.")
    emit(f"  Original scene: {rel(image_path)}")
    emit(f"  Blurred image -> {rel(result.blur_path)}")
    emit(f"  Restored image -> {rel(result.restore_path)}")
    emit(f"  Montage -> {rel(result.montage_path)}")
    emit(f"  PSNR(original, blurred) = {result.psnr_blur:.2f} dB")
    emit(f"  PSNR(original, restored) = {result.psnr_restore:.2f} dB")

    if args.json:
        json.dump(result.as_json(), sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
