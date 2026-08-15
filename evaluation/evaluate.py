#!/usr/bin/env python
"""
Comprehensive evaluation suite for image restoration models.

Provides:
- PSNR, SSIM, LPIPS metrics
- Batch evaluation
- Per-degradation analysis
- Time profiling
- Report generation

Usage:
    python evaluation/evaluate.py --model v1 --checkpoint checkpoints/v1/best_v1.pth
    python evaluation/evaluate.py --model v2 --checkpoint checkpoints/v2/best_v2.pth --save-results
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_paired_images
from models import UnifiedRestorerV1, UnifiedRestorerV2


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def psnr(
    img1: np.ndarray,
    img2: np.ndarray,
    max_val: float = 1.0
) -> float:
    """
    Calculate PSNR.

    Args:
        img1: Image 1, range [0, max_val]
        img2: Image 2, same shape
        max_val: Maximum value

    Returns:
        PSNR in dB
    """

    mse = float(
        np.mean(
            (img1 - img2) ** 2
        )
    )

    if mse == 0:
        return 100.0

    return float(
        20 *
        np.log10(
            max_val /
            np.sqrt(mse)
        )
    )


def ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    max_val: float = 1.0
) -> float:
    """
    Calculate SSIM.

    Args:
        img1: Image 1
        img2: Image 2
        max_val: Maximum value

    Returns:
        SSIM score
    """

    C1 = (
        0.01 *
        max_val
    ) ** 2

    C2 = (
        0.03 *
        max_val
    ) ** 2

    mean1 = np.mean(img1)
    mean2 = np.mean(img2)

    var1 = np.var(img1)
    var2 = np.var(img2)

    cov = np.mean(
        (img1 - mean1) *
        (img2 - mean2)
    )

    numerator = (
        (2 * mean1 * mean2 + C1) *
        (2 * cov + C2)
    )

    denominator = (
        (mean1 ** 2 + mean2 ** 2 + C1) *
        (var1 + var2 + C2)
    )

    if denominator == 0:
        return 1.0 if np.array_equal(
            img1,
            img2
        ) else 0.0

    return float(
        numerator /
        denominator
    )


def lpips(
    img1: torch.Tensor,
    img2: torch.Tensor
) -> float:
    """
    Calculate LPIPS if available.

    Args:
        img1: Tensor in [-1, 1]
        img2: Tensor in [-1, 1]

    Returns:
        LPIPS distance.
        Returns 0.0 if LPIPS is unavailable.
    """

    try:

        import lpips

        loss_fn = lpips.LPIPS(
            net="vgg",
            verbose=False
        ).eval()

        with torch.no_grad():

            dist = loss_fn(
                img1,
                img2
            )

        return float(
            dist.item()
        )

    except ImportError:

        return 0.0


# ============================================================
# MODEL EVALUATOR
# ============================================================

class ModelEvaluator:
    """Evaluation orchestrator for restoration models."""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        device: Optional[str] = None,
        save_results: bool = False,
        output_dir: Optional[str] = None
    ):
        """
        Initialize evaluator.

        Args:
            model_name: "v1" or "v2"
            checkpoint_path: Path to checkpoint
            device: Device
            save_results: Save restored images
            output_dir: Output directory
        """

        self.model_name = (
            model_name.lower()
        )

        self.checkpoint_path = (
            checkpoint_path
        )

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        self.save_results = (
            save_results
        )

        self.output_dir = Path(
            output_dir
            or
            f"results/{self.model_name}/evaluation"
        )

        if self.save_results:

            self.output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            (
                self.output_dir /
                "restored"
            ).mkdir(
                exist_ok=True
            )

            (
                self.output_dir /
                "input"
            ).mkdir(
                exist_ok=True
            )

            (
                self.output_dir /
                "gt"
            ).mkdir(
                exist_ok=True
            )

        self.model = (
            self._load_model()
        )

        print(
            f"[Evaluator] Parameters: "
            f"{sum(p.numel() for p in self.model.parameters()):,}"
        )

        print(
            f"[Evaluator] Model: "
            f"{self.model_name}"
        )

        print(
            f"[Evaluator] Device: "
            f"{self.device}"
        )

        print(
            f"[Evaluator] Checkpoint: "
            f"{self.checkpoint_path}"
        )

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_model(
        self
    ) -> nn.Module:
        """Load model from checkpoint."""

        if not os.path.exists(
            self.checkpoint_path
        ):

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        if self.model_name == "v1":

            model = UnifiedRestorerV1(
                num_features=64,
                num_blocks=12
            )

        elif self.model_name == "v2":

            model = UnifiedRestorerV2(
                num_features=64,
                num_blocks=16,
                use_hf_branch=True,
                use_edge_branch=True
            )

        else:

            raise ValueError(
                f"Unknown model: "
                f"{self.model_name}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device
        )

        if (
            isinstance(checkpoint, dict)
            and
            "model_state" in checkpoint
        ):

            model.load_state_dict(
                checkpoint["model_state"]
            )

        else:

            model.load_state_dict(
                checkpoint
            )

        model = model.to(
            self.device
        )

        model.eval()

        return model

    # ========================================================
    # INFERENCE
    # ========================================================

    @torch.no_grad()
    def infer(
        self,
        noisy: torch.Tensor,
        degradation_type: Optional[str] = None
    ) -> Tuple[
        torch.Tensor,
        float
    ]:
        """
        Run inference.

        Input:
            NoisyLR: 128x128

        Expected V2 output:
            Restored HR: 256x256
        """

        if noisy.dim() == 2:

            noisy = noisy.unsqueeze(
                0
            ).unsqueeze(
                0
            )

        elif noisy.dim() == 3:

            noisy = noisy.unsqueeze(
                0
            )

        noisy = noisy.to(
            self.device
        )

        start_time = time.time()

        restored = self.model(
            noisy
        )

        inference_time = (
            time.time() -
            start_time
        )

        restored = torch.clamp(
            restored,
            0.0,
            1.0
        )

        return (
            restored,
            inference_time
        )

    # ========================================================
    # SINGLE IMAGE EVALUATION
    # ========================================================

    @torch.no_grad()
    def evaluate_single(
        self,
        noisy: torch.Tensor,
        gt: torch.Tensor,
        image_index: int = 0
    ) -> Dict[str, float]:
        """
        Evaluate one LR/HR image pair.

        Noisy:
            128x128

        GT:
            256x256

        Model output:
            256x256
        """

        # ----------------------------------------------------
        # Prepare input
        # ----------------------------------------------------

        if noisy.dim() == 2:

            noisy = noisy.unsqueeze(
                0
            )

        if gt.dim() == 2:

            gt = gt.unsqueeze(
                0
            )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        restored, inference_time = (
            self.infer(noisy)
        )

        # ----------------------------------------------------
        # Remove batch dimension
        # ----------------------------------------------------

        restored_single = (
            restored[0]
        )

        gt_single = gt

        noisy_single = noisy

        # ----------------------------------------------------
        # Verify output shape
        # ----------------------------------------------------

        if (
            restored_single.shape
            != gt_single.shape
        ):

            raise ValueError(
                "Model output shape does not "
                "match GT shape for image "
                f"{image_index}: "
                f"output={tuple(restored_single.shape)}, "
                f"GT={tuple(gt_single.shape)}"
            )

        # ----------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------

        restored_np = (
            restored_single
            .detach()
            .cpu()
            .numpy()
        )

        gt_np = (
            gt_single
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Input baseline
        #
        # Upsample LR using nearest-neighbor only for
        # baseline comparison.
        # ----------------------------------------------------

        noisy_for_baseline = torch.nn.functional.interpolate(
            noisy_single.unsqueeze(0),
            size=gt_single.shape[-2:],
            mode="bilinear",
            align_corners=False
        )[0]

        noisy_baseline_np = (
            noisy_for_baseline
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        psnr_restored = psnr(
            restored_np,
            gt_np,
            max_val=1.0
        )

        ssim_restored = ssim(
            restored_np,
            gt_np,
            max_val=1.0
        )

        psnr_input = psnr(
            noisy_baseline_np,
            gt_np,
            max_val=1.0
        )

        ssim_input = ssim(
            noisy_baseline_np,
            gt_np,
            max_val=1.0
        )

        metrics = {
            "psnr": float(
                psnr_restored
            ),

            "ssim": float(
                ssim_restored
            ),

            "psnr_input": float(
                psnr_input
            ),

            "ssim_input": float(
                ssim_input
            ),

            "psnr_improvement": float(
                psnr_restored -
                psnr_input
            ),

            "ssim_improvement": float(
                ssim_restored -
                ssim_input
            ),

            "inference_time_ms": float(
                inference_time * 1000
            )
        }

        # ----------------------------------------------------
        # Optional LPIPS
        # ----------------------------------------------------

        lpips_val = lpips(
            restored_single.unsqueeze(0) * 2.0 - 1.0,
            gt_single.unsqueeze(0) * 2.0 - 1.0
        )

        if lpips_val > 0:

            metrics["lpips"] = float(
                lpips_val
            )

        # ----------------------------------------------------
        # Save images
        # ----------------------------------------------------

        if self.save_results:

            self._save_image(
                restored_single,
                self.output_dir /
                "restored" /
                f"{image_index:06d}.png"
            )

            self._save_image(
                noisy_single,
                self.output_dir /
                "input" /
                f"{image_index:06d}.png"
            )

            self._save_image(
                gt_single,
                self.output_dir /
                "gt" /
                f"{image_index:06d}.png"
            )

        return metrics

    # ========================================================
    # IMAGE SAVING
    # ========================================================

    def _save_image(
        self,
        tensor: torch.Tensor,
        path: Path
    ):
        """Save tensor as PNG."""

        from PIL import Image

        image = (
            tensor
            .detach()
            .cpu()
            .numpy()
        )

        if image.ndim == 3:

            image = image[0]

        image = np.clip(
            image,
            0.0,
            1.0
        )

        image = (
            image * 255.0
        ).round().astype(
            np.uint8
        )

        Image.fromarray(
            image,
            mode="L"
        ).save(
            path
        )

    # ========================================================
    # BATCH EVALUATION
    # ========================================================

    @torch.no_grad()
    def evaluate_batch(
        self,
        noisy: torch.Tensor,
        gt: torch.Tensor,
        degradation_type: Optional[str] = None,
        batch_idx: int = 0
    ) -> Dict[str, float]:
        """
        Evaluate a batch.

        This method processes every image individually because
        LR and HR images have different spatial dimensions.
        """

        batch_size = (
            noisy.shape[0]
            if noisy.dim() == 4
            else 1
        )

        metrics_list = []

        for i in range(
            batch_size
        ):

            if noisy.dim() == 4:

                noisy_i = noisy[i]

            else:

                noisy_i = noisy

            if gt.dim() == 4:

                gt_i = gt[i]

            else:

                gt_i = gt

            metrics = (
                self.evaluate_single(
                    noisy_i,
                    gt_i,
                    image_index=(
                        batch_idx + i
                    )
                )
            )

            metrics_list.append(
                metrics
            )

        # Aggregate batch
        aggregated = {}

        numeric_keys = [
            key
            for key in metrics_list[0]
            if isinstance(
                metrics_list[0][key],
                (int, float)
            )
        ]

        for key in numeric_keys:

            values = [
                metric[key]
                for metric in metrics_list
                if key in metric
            ]

            aggregated[key] = float(
                np.mean(values)
            )

        return aggregated

    # ========================================================
    # DATASET EVALUATION
    # ========================================================

    def evaluate_dataset(
        self,
        gt_dir: str,
        noisy_dir: str,
        max_images: Optional[int] = None,
        batch_size: int = 1
    ) -> Dict[str, any]:
        """
        Evaluate TEST split.

        Dataset structure:

            NoisyLR:
                128 x 128

            GT:
                256 x 256

        The model performs:

            128 x 128
                 ↓
                V2
                 ↓
            256 x 256
        """

        # ----------------------------------------------------
        # Load train / validation / test split
        # ----------------------------------------------------

        train_files, val_files, test_files = (
            load_paired_images(
                gt_dir,
                noisy_dir
            )
        )

        # ----------------------------------------------------
        # Construct paths from test IDs
        # ----------------------------------------------------

        gt_paths = [
            os.path.join(
                gt_dir,
                f"{filename}.npy"
            )
            for filename in test_files
        ]

        noisy_paths = [
            os.path.join(
                noisy_dir,
                f"{filename}.npy"
            )
            for filename in test_files
        ]

        # ----------------------------------------------------
        # Limit images
        # ----------------------------------------------------

        if max_images is not None:

            max_images = max(
                0,
                int(max_images)
            )

            gt_paths = (
                gt_paths[:max_images]
            )

            noisy_paths = (
                noisy_paths[:max_images]
            )

        if not gt_paths:

            raise ValueError(
                "No test images found."
            )

        print(
            f"\n[Evaluate] Dataset: "
            f"{len(gt_paths)} test images"
        )

        print(
            f"[Evaluate] Batch size: "
            f"{batch_size}"
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        all_metrics = []

        # ----------------------------------------------------
        # Evaluate images
        # ----------------------------------------------------

        pbar = tqdm(
            range(
                0,
                len(gt_paths),
                batch_size
            ),
            desc="Evaluating"
        )

        for batch_start in pbar:

            batch_end = min(
                batch_start +
                batch_size,
                len(gt_paths)
            )

            for image_index in range(
                batch_start,
                batch_end
            ):

                gt_path = (
                    gt_paths[
                        image_index
                    ]
                )

                noisy_path = (
                    noisy_paths[
                        image_index
                    ]
                )

                # --------------------------------------------
                # Load
                # --------------------------------------------

                gt_img = np.load(
                    gt_path
                ).astype(
                    np.float32
                )

                noisy_img = np.load(
                    noisy_path
                ).astype(
                    np.float32
                )

                # --------------------------------------------
                # Validate dimensions
                # --------------------------------------------

                if gt_img.ndim != 2:

                    raise ValueError(
                        f"GT image must be 2D: "
                        f"{gt_path}, "
                        f"shape={gt_img.shape}"
                    )

                if noisy_img.ndim != 2:

                    raise ValueError(
                        f"NoisyLR image must be 2D: "
                        f"{noisy_path}, "
                        f"shape={noisy_img.shape}"
                    )

                # --------------------------------------------
                # Validate expected 2x scale
                # --------------------------------------------

                expected_height = (
                    noisy_img.shape[0] * 2
                )

                expected_width = (
                    noisy_img.shape[1] * 2
                )

                if (
                    gt_img.shape[0]
                    != expected_height
                    or
                    gt_img.shape[1]
                    != expected_width
                ):

                    raise ValueError(
                        "Unexpected LR/HR scale for "
                        f"{os.path.basename(gt_path)}: "
                        f"Noisy={noisy_img.shape}, "
                        f"GT={gt_img.shape}. "
                        f"Expected GT to be 2x NoisyLR."
                    )

                # --------------------------------------------
                # Convert to tensors
                # --------------------------------------------

                noisy_tensor = (
                    torch.from_numpy(
                        noisy_img
                    )
                    .float()
                )

                gt_tensor = (
                    torch.from_numpy(
                        gt_img
                    )
                    .float()
                )

                # --------------------------------------------
                # Preserve expected [0,1] range
                # --------------------------------------------

                noisy_tensor = torch.clamp(
                    noisy_tensor,
                    0.0,
                    1.0
                )

                gt_tensor = torch.clamp(
                    gt_tensor,
                    0.0,
                    1.0
                )

                # --------------------------------------------
                # Evaluate
                # --------------------------------------------

                metrics = (
                    self.evaluate_single(
                        noisy_tensor,
                        gt_tensor,
                        image_index=image_index
                    )
                )

                # --------------------------------------------
                # Filename
                # --------------------------------------------

                metrics["filename"] = (
                    os.path.basename(
                        gt_path
                    )
                )

                all_metrics.append(
                    metrics
                )

                # --------------------------------------------
                # Progress
                # --------------------------------------------

                pbar.set_postfix(
                    {
                        "PSNR":
                            f"{metrics['psnr']:.2f}",

                        "SSIM":
                            f"{metrics['ssim']:.4f}"
                    }
                )

        # ----------------------------------------------------
        # Check
        # ----------------------------------------------------

        if not all_metrics:

            raise RuntimeError(
                "No evaluation results produced."
            )

        # ----------------------------------------------------
        # Aggregate
        # ----------------------------------------------------

        numeric_keys = set()

        for metric in all_metrics:

            for key, value in metric.items():

                if isinstance(
                    value,
                    (int, float)
                ):

                    numeric_keys.add(
                        key
                    )

        summary = {}

        for key in sorted(
            numeric_keys
        ):

            values = [
                float(
                    metric[key]
                )
                for metric in all_metrics
                if key in metric
            ]

            if not values:
                continue

            summary[
                f"{key}_mean"
            ] = float(
                np.mean(values)
            )

            summary[
                f"{key}_std"
            ] = float(
                np.std(values)
            )

            summary[
                f"{key}_min"
            ] = float(
                np.min(values)
            )

            summary[
                f"{key}_max"
            ] = float(
                np.max(values)
            )

        summary[
            "num_images"
        ] = len(
            all_metrics
        )

        return {
            "summary": summary,
            "per_image": all_metrics
        }

    # ========================================================
    # SAVE REPORT
    # ========================================================

    def save_report(
        self,
        results: Dict,
        output_file: Optional[str] = None
    ):
        """Save evaluation report."""

        if output_file is None:

            output_file = (
                self.output_dir /
                "evaluation_report.json"
            )

        output_file = Path(
            output_file
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                results,
                f,
                indent=2
            )

        print(
            f"\n[Evaluate] Report saved: "
            f"{output_file}"
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary = (
            results["summary"]
        )

        print(
            "\n" +
            "=" * 70
        )

        print(
            "EVALUATION SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            f"Model:             "
            f"{self.model_name}"
        )

        print(
            f"Images:            "
            f"{summary['num_images']}"
        )

        if "psnr_mean" in summary:

            print(
                f"PSNR:              "
                f"{summary['psnr_mean']:.2f} "
                f"± "
                f"{summary['psnr_std']:.2f}"
            )

        if "ssim_mean" in summary:

            print(
                f"SSIM:              "
                f"{summary['ssim_mean']:.4f} "
                f"± "
                f"{summary['ssim_std']:.4f}"
            )

        if "psnr_input_mean" in summary:

            print(
                f"Input PSNR:        "
                f"{summary['psnr_input_mean']:.2f} "
                f"± "
                f"{summary['psnr_input_std']:.2f}"
            )

        if "ssim_input_mean" in summary:

            print(
                f"Input SSIM:        "
                f"{summary['ssim_input_mean']:.4f} "
                f"± "
                f"{summary['ssim_input_std']:.4f}"
            )

        if "psnr_improvement_mean" in summary:

            print(
                f"PSNR Improvement:  "
                f"{summary['psnr_improvement_mean']:.2f} dB"
            )

        if "ssim_improvement_mean" in summary:

            print(
                f"SSIM Improvement:  "
                f"{summary['ssim_improvement_mean']:.4f}"
            )

        if "lpips_mean" in summary:

            print(
                f"LPIPS:             "
                f"{summary['lpips_mean']:.4f} "
                f"± "
                f"{summary['lpips_std']:.4f}"
            )

        if (
            "inference_time_ms_mean"
            in summary
        ):

            print(
                f"Inference (ms):    "
                f"{summary['inference_time_ms_mean']:.2f} "
                f"± "
                f"{summary['inference_time_ms_std']:.2f}"
            )

        print(
            "=" * 70 +
            "\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Image Restoration Models"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "v1",
            "v2"
        ],
        help="Model to evaluate"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--gt-dir",
        type=str,
        default="train/GT",
        help="Path to GT images directory"
    )

    parser.add_argument(
        "--noisy-dir",
        type=str,
        default="train/NoisyLR",
        help="Path to noisy images directory"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[
            "cuda",
            "cpu"
        ],
        help="Device to use"
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save restored images"
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum images to evaluate"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Create evaluator
    # --------------------------------------------------------

    evaluator = ModelEvaluator(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=args.device,
        save_results=args.save_results,
        output_dir=args.output_dir
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluator.evaluate_dataset(
        gt_dir=args.gt_dir,
        noisy_dir=args.noisy_dir,
        max_images=args.max_images,
        batch_size=args.batch_size
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    evaluator.save_report(
        results
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()