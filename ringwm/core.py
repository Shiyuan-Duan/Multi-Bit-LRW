# ringwm/core.py
"""
RingWM core implementation.

This module provides a minimal, paper-ready reference implementation of a
ring-structured multi-bit latent watermark for SDXL-style latent diffusion
pipelines. The watermark is injected in the FFT domain of the initial noise
latent z_T and decoded from an inverted latent ẑ_T obtained via DDIM inversion.

Design notes (aligned with typical paper “Implementation Details” sections):
- SDXL latent shape for 1024×1024 is typically (1, 4, 128, 128).
- All FFT/iFFT operations are performed in float32 for numerical stability.
- The diffusion model and VAE are loaded in float16 for throughput.
- Ring masks are defined in the FFT-shifted frequency domain (centered DC).
- Each bit is redundantly encoded across multiple rings; default is 8 bits
  across 16 rings (2 rings per bit).

This file intentionally avoids experiment orchestration (sweeps, metrics,
dataset loops). It is meant to be imported by evaluation scripts.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from diffusers import (
    DDIMInverseScheduler,
    DDIMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models import AutoencoderKL


LOGGER = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass(frozen=True)
class RingWMConfig:
    """Configuration for RingWatermarker.

    Attributes
    ----------
    num_bits:
        Payload length in bits. This reference implementation supports 8 only.
    num_rings:
        Number of rings in frequency domain. This reference supports 16 only.
    alpha:
        Watermark mixing strength in FFT domain, in [0, 1].
    ring_width:
        Ring thickness in pixels (in latent frequency map coordinates).
    r_in0:
        Inner radius (pixels) of ring 0.
    w_seed:
        Seed used to generate the base key (derived from a fixed latent sample).
    w_channel:
        Latent channel to watermark (0..C-1). SDXL typically has C=4.
    image_size:
        Pixel image size for SDXL base (default 1024).
    num_inference_steps:
        Default DDIM steps for generation and inversion.
    negative_prompt:
        Optional negative prompt used during generation.
    vae_scaling:
        VAE scaling constant used in SDXL latent space (common value: 0.13025).
    compile_unet:
        If True and torch.compile is available, compile UNet for speed.
    """

    num_bits: int = 8
    num_rings: int = 16
    alpha: float = 0.3
    ring_width: int = 3
    r_in0: int = 8
    w_seed: int = 7433
    w_channel: int = 0

    image_size: int = 1024
    num_inference_steps: int = 50
    negative_prompt: str = "monochrome"
    vae_scaling: float = 0.13025

    compile_unet: bool = True


# ==============================================================================
# Utility functions
# ==============================================================================

def _ensure_dir(path: os.PathLike | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _validate_bits(bits: Sequence[int], *, num_bits: int) -> None:
    if len(bits) != num_bits:
        raise ValueError(f"Expected {num_bits} bits, got {len(bits)}.")
    if any((b not in (0, 1)) for b in bits):
        raise ValueError("Bits must be in {0, 1}.")


def int_to_bits8(m: int) -> List[int]:
    """Convert an integer in [0, 255] to 8 bits (LSB-first)."""
    if not (0 <= m < 256):
        raise ValueError("8-bit message must be in [0, 255].")
    return [(m >> i) & 1 for i in range(8)]


def bits8_to_int(bits: Sequence[int]) -> int:
    """Convert 8 bits (LSB-first) back to an integer in [0, 255]."""
    _validate_bits(bits, num_bits=8)
    m = 0
    for i, b in enumerate(bits):
        m |= (int(b) & 1) << i
    return m


def circle_mask(
    size: int,
    radius: int,
    *,
    x_offset: int = 0,
    y_offset: int = 0,
) -> np.ndarray:
    """Return a (size, size) boolean disk mask (True inside radius)."""
    cx = cy = size // 2
    cx += x_offset
    cy += y_offset
    yy, xx = np.ogrid[:size, :size]
    # Match image coordinate convention used by the original code:
    yy = yy[::-1]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2


def ring_mask(
    size: int,
    r_in: int,
    r_out: int,
    *,
    x_offset: int = 0,
    y_offset: int = 0,
) -> np.ndarray:
    """Return a boolean annulus mask."""
    outer = circle_mask(size, r_out, x_offset=x_offset, y_offset=y_offset)
    inner = circle_mask(size, r_in, x_offset=x_offset, y_offset=y_offset)
    return np.logical_and(outer, ~inner)


def transform_image_sdxl(image: Image.Image, *, image_size: int) -> torch.Tensor:
    """Preprocess PIL image to SDXL input tensor in [-1, 1], shape (3,H,W)."""
    tform = T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
    )
    x = tform(image)  # [0,1]
    return 2.0 * x - 1.0


def tensor_stats(x: torch.Tensor) -> Dict[str, Any]:
    """Return lightweight tensor summary for tracing/debugging."""
    x = x.detach()
    if torch.is_complex(x):
        mag = torch.abs(x)
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "abs_mean": float(mag.mean().cpu()),
            "abs_std": float(mag.std().cpu()),
            "abs_min": float(mag.min().cpu()),
            "abs_max": float(mag.max().cpu()),
        }
    xf = x.float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "mean": float(xf.mean().cpu()),
        "std": float(xf.std().cpu()),
        "min": float(xf.min().cpu()),
        "max": float(xf.max().cpu()),
    }


def fft_mag01(fft_complex: torch.Tensor, *, channel: int = 0) -> np.ndarray:
    """Log-magnitude visualization helper.

    Parameters
    ----------
    fft_complex:
        Complex tensor of shape (1, C, H, W), assumed fftshifted.
    channel:
        Channel index to visualize.

    Returns
    -------
    np.ndarray:
        Float32 array (H, W) in [0, 1].
    """
    x = fft_complex[0, channel]
    mag = torch.log1p(torch.abs(x))
    mag = mag - mag.min()
    mag = mag / (mag.max() + 1e-8)
    return mag.detach().float().cpu().numpy()


def save_gray_png(path: os.PathLike | str, arr01: np.ndarray) -> None:
    """Save grayscale array in [0,1] to PNG."""
    path = Path(path)
    _ensure_dir(path.parent)
    arr = np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def save_bar_png(path: os.PathLike | str, values: Sequence[float], *, title: str, xlabel: str) -> None:
    """Save a simple bar plot if matplotlib is available; otherwise dump JSON."""
    path = Path(path)
    _ensure_dir(path.parent)

    try:
        import matplotlib.pyplot as plt  # optional dependency
        plt.figure()
        plt.bar(list(range(len(values))), list(values))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # pragma: no cover
        fallback = path.with_suffix(".json")
        with open(fallback, "w", encoding="utf-8") as f:
            json.dump(
                {"title": title, "xlabel": xlabel, "values": list(values), "matplotlib_error": str(exc)},
                f,
                indent=2,
            )


# ==============================================================================
# RingWatermarker
# ==============================================================================

class RingWatermarker:
    """Ring-structured multi-bit latent watermark for SDXL pipelines.

    Default mapping (fixed in this reference):
    - num_bits = 8, num_rings = 16
    - ring_idx % 8 == bit_idx
    - each bit is encoded in two rings (ring i and ring i+8)

    The watermark is injected into FFT(z_T) on a single latent channel.
    """

    def __init__(
        self,
        config: RingWMConfig = RingWMConfig(),
        *,
        device: Optional[torch.device] = None,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.cfg = config

        if self.cfg.num_bits != 8 or self.cfg.num_rings != 16:
            raise ValueError("This reference implementation assumes num_bits=8 and num_rings=16.")

        if not (0.0 <= self.cfg.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.dtype = torch_dtype

        self._load_pipeline()
        self._init_rings_and_key()

    # --------------------------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """Load SDXL pipeline components and set scheduler."""
        # NOTE: Keep model IDs here for clarity and reproducibility in the paper.
        # If you want to expose these as config fields, do it in RingWMConfig.
        vae_id = "madebyollin/sdxl-vae-fp16-fix"
        unet_id = "mhdang/dpo-sdxl-text2image-v1"
        base_id = "stabilityai/stable-diffusion-xl-base-1.0"

        LOGGER.info("Loading VAE: %s", vae_id)
        vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=self.dtype)

        LOGGER.info("Loading UNet: %s (subfolder=unet)", unet_id)
        unet = UNet2DConditionModel.from_pretrained(unet_id, subfolder="unet", torch_dtype=self.dtype)

        LOGGER.info("Loading SDXL pipeline: %s", base_id)
        pipe = DiffusionPipeline.from_pretrained(
            base_id,
            unet=unet,
            vae=vae,
            torch_dtype=self.dtype,
        ).to(self.device)

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        # Optional compile for speed (Torch 2.x).
        if self.cfg.compile_unet and hasattr(torch, "compile"):
            try:
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
                LOGGER.info("Compiled UNet with torch.compile.")
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("torch.compile failed; continuing without compile. Error: %s", exc)

        self.pipe = pipe
        self.vae = pipe.vae
        self.unet = pipe.unet

    # --------------------------------------------------------------------------
    # Masks + base key
    # --------------------------------------------------------------------------

    def _prepare_template_latents(self, generator: Optional[torch.Generator]) -> torch.Tensor:
        """Prepare SDXL latents with the same utility used by the pipeline."""
        return self.pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=self.pipe.unet.config.in_channels,
            height=self.cfg.image_size,
            width=self.cfg.image_size,
            dtype=self.pipe.unet.dtype,
            device=self.device,
            generator=generator,
            latents=None,
        )

    def _init_rings_and_key(self) -> None:
        """Initialize ring masks and the base key in FFT domain."""
        template = self._prepare_template_latents(generator=None)
        b, c, h, w = template.shape
        if b != 1:
            raise RuntimeError("This implementation expects batch_size=1.")

        if not (0 <= self.cfg.w_channel < c):
            raise ValueError(f"w_channel must be in [0, {c-1}], got {self.cfg.w_channel}.")

        self.latent_shape = (b, c, h, w)

        # Ring ranges (inner/outer radii in pixels).
        ring_ranges: List[Tuple[int, int]] = []
        for i in range(self.cfg.num_rings):
            r_in = self.cfg.r_in0 + i * self.cfg.ring_width
            r_out = r_in + self.cfg.ring_width
            ring_ranges.append((r_in, r_out))
        self.ring_ranges = ring_ranges

        # Build boolean masks: list of (1,C,H,W) bool
        bit_masks: List[torch.Tensor] = []
        for (r_in, r_out) in ring_ranges:
            ring_np = ring_mask(h, r_in=r_in, r_out=r_out)  # (H,W) bool
            ring_mask_hw = torch.as_tensor(ring_np, device=self.device)

            mask = torch.zeros((1, c, h, w), dtype=torch.bool, device=self.device)
            mask[:, self.cfg.w_channel] = ring_mask_hw
            bit_masks.append(mask)
        self.bit_masks = bit_masks

        # Base key: derive from a fixed-seed latent sample in FFT domain.
        g = torch.Generator(device=self.device).manual_seed(self.cfg.w_seed)
        zT_key = self._prepare_template_latents(generator=g)  # float16

        y_key = self._latents_to_fft(zT_key)  # complex64
        base_key = torch.zeros_like(y_key)

        # Representative value selection:
        # We keep your original design for faithful reproduction: pick a single complex
        # coefficient from the seed FFT and tile it across each ring.
        # If you later swap to a ring-specific random phase codebook, it belongs here.
        for ring_idx, (r_in, r_out) in enumerate(ring_ranges):
            ring_hw = bit_masks[ring_idx][0, self.cfg.w_channel]  # (H,W) bool
            r_center = (r_in + r_out) // 2
            val = y_key[0, self.cfg.w_channel, 0, r_center]  # complex scalar
            base_key[0, self.cfg.w_channel, ring_hw] = val

        self.base_key_fft = base_key

    # --------------------------------------------------------------------------
    # FFT helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def _latents_to_fft(latents: torch.Tensor) -> torch.Tensor:
        """Compute fftshift(fft2(latents)) in float32 -> complex64."""
        lat_f32 = latents.to(torch.float32)
        return torch.fft.fftshift(torch.fft.fft2(lat_f32), dim=(-1, -2))

    @staticmethod
    def _fft_to_latents(latents_fft: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        """Compute real(ifft2(ifftshift))) and cast to dtype."""
        lat = torch.fft.ifft2(torch.fft.ifftshift(latents_fft, dim=(-1, -2))).real
        lat = lat.to(dtype)
        # Defensive clamp of infinities (should be rare).
        lat = torch.nan_to_num(lat, nan=0.0, posinf=4.0, neginf=-4.0)
        return lat

    # --------------------------------------------------------------------------
    # Encode / decode
    # --------------------------------------------------------------------------

    def _encode_bits_in_fft(self, y: torch.Tensor, bits: Sequence[int]) -> torch.Tensor:
        """Inject bits into an FFT tensor (complex), returning a new tensor."""
        _validate_bits(bits, num_bits=self.cfg.num_bits)
        y_wm = y.clone()

        for ring_idx in range(self.cfg.num_rings):
            mask = self.bit_masks[ring_idx]
            bit_idx = ring_idx % self.cfg.num_bits
            sign = 1.0 if bits[bit_idx] == 1 else -1.0

            y_wm[mask] = (1.0 - self.cfg.alpha) * y_wm[mask] + self.cfg.alpha * sign * self.base_key_fft[mask]

        return y_wm

    def _decode_bits_from_fft(self, y: torch.Tensor) -> Tuple[List[int], List[float], List[float]]:
        """Decode bits from an FFT tensor (complex).

        Returns
        -------
        bits_out:
            Decoded bits (length num_bits).
        bit_scores:
            Correlation score per bit (sum over rings assigned to that bit).
        ring_scores:
            Correlation score per ring.
        """
        ring_scores: List[float] = []
        for ring_idx in range(self.cfg.num_rings):
            mask = self.bit_masks[ring_idx]
            yy = y[mask]
            kk = self.base_key_fft[mask]
            # Real inner product for complex vectors: Re( y * conj(k) ) == y_r k_r + y_i k_i
            score = (yy.real * kk.real + yy.imag * kk.imag).sum().item()
            ring_scores.append(float(score))

        bit_scores: List[float] = []
        bits_out: List[int] = []
        for bit_idx in range(self.cfg.num_bits):
            idx = [r for r in range(self.cfg.num_rings) if (r % self.cfg.num_bits) == bit_idx]
            s = float(sum(ring_scores[r] for r in idx))
            bit_scores.append(s)
            bits_out.append(1 if s >= 0.0 else 0)

        return bits_out, bit_scores, ring_scores

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt: str,
        msg: int,
        num_inference_steps: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ) -> Image.Image:
        """Generate a watermarked image for an 8-bit message."""
        bits = int_to_bits8(msg)
        steps = int(num_inference_steps or self.cfg.num_inference_steps)
        neg = self.cfg.negative_prompt if negative_prompt is None else negative_prompt

        zT = self._prepare_template_latents(generator=None)
        y = self._latents_to_fft(zT)
        y_wm = self._encode_bits_in_fft(y, bits)
        zT_wm = self._fft_to_latents(y_wm, dtype=zT.dtype)

        out = self.pipe(
            prompt=prompt,
            negative_prompt=neg,
            num_inference_steps=steps,
            latents=zT_wm,
        )
        return out.images[0]

    @torch.inference_mode()
    def decode(
        self,
        *,
        image: Image.Image,
        num_inference_steps: Optional[int] = None,
    ) -> Tuple[int, List[int], List[float], List[float]]:
        """Decode an 8-bit message from a generated image via DDIM inversion."""
        steps = int(num_inference_steps or self.cfg.num_inference_steps)

        # Swap scheduler to inverse for inversion pass.
        curr_scheduler = self.pipe.scheduler
        self.pipe.scheduler = DDIMInverseScheduler.from_config(self.pipe.scheduler.config)

        x = transform_image_sdxl(image, image_size=self.cfg.image_size).unsqueeze(0)
        x = x.to(self.pipe.unet.dtype).to(self.device)

        z0 = self.vae.encode(x).latent_dist.mode() * float(self.cfg.vae_scaling)

        inv = self.pipe(
            prompt="",
            latents=z0,
            guidance_scale=1.0,
            num_inference_steps=steps,
            output_type="latent",
        ).images

        # Restore scheduler.
        self.pipe.scheduler = curr_scheduler

        y_hat = self._latents_to_fft(inv)
        bits_out, bit_scores, ring_scores = self._decode_bits_from_fft(y_hat)
        msg_out = bits8_to_int(bits_out)
        return msg_out, bits_out, bit_scores, ring_scores

    # --------------------------------------------------------------------------
    # Trace API (for paper figures / debugging)
    # --------------------------------------------------------------------------

    @torch.inference_mode()
    def generate_trace(
        self,
        *,
        prompt: str,
        msg: int,
        out_dir: os.PathLike | str,
        tag: str,
        watermark: bool = True,
        num_inference_steps: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Generate an image while dumping intermediate artifacts and metadata."""
        out_dir = Path(out_dir)
        _ensure_dir(out_dir)
        steps = int(num_inference_steps or self.cfg.num_inference_steps)
        neg = self.cfg.negative_prompt if negative_prompt is None else negative_prompt

        trace: Dict[str, Any] = {
            "config": asdict(self.cfg),
            "prompt": prompt,
            "msg": int(msg),
            "bits": int_to_bits8(msg),
            "watermark": bool(watermark),
            "num_inference_steps": steps,
            "latent_shape": list(self.latent_shape),
            "ring_ranges": [list(x) for x in self.ring_ranges],
        }

        zT = self._prepare_template_latents(generator=None)
        trace["init_latents_stats"] = tensor_stats(zT)

        y = self._latents_to_fft(zT)
        trace["fft_before_stats"] = tensor_stats(y)
        save_gray_png(out_dir / f"{tag}_fft_before.png", fft_mag01(y, channel=self.cfg.w_channel))

        if watermark:
            y2 = self._encode_bits_in_fft(y, trace["bits"])
            trace["fft_after_stats"] = tensor_stats(y2)
            save_gray_png(out_dir / f"{tag}_fft_after.png", fft_mag01(y2, channel=self.cfg.w_channel))

            before = fft_mag01(y, channel=self.cfg.w_channel)
            after = fft_mag01(y2, channel=self.cfg.w_channel)
            diff = after - before
            diff = diff - diff.min()
            diff = diff / (diff.max() + 1e-8)
            save_gray_png(out_dir / f"{tag}_fft_diff.png", diff)
        else:
            y2 = y

        zT2 = self._fft_to_latents(y2, dtype=zT.dtype)
        trace["latents_after_ifft_stats"] = tensor_stats(zT2)

        img = self.pipe(
            prompt=prompt,
            negative_prompt=neg,
            num_inference_steps=steps,
            latents=zT2,
        ).images[0]

        img_path = out_dir / f"{tag}_image.png"
        img.save(img_path)
        trace["image_path"] = str(img_path)

        with open(out_dir / f"{tag}_trace.json", "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)

        return img, trace

    @torch.inference_mode()
    def decode_trace(
        self,
        *,
        image: Image.Image,
        out_dir: os.PathLike | str,
        tag: str,
        num_inference_steps: Optional[int] = None,
    ) -> Tuple[int, List[int], List[float], List[float], Dict[str, Any]]:
        """Decode with logging of inversion FFT and correlation scores."""
        out_dir = Path(out_dir)
        _ensure_dir(out_dir)
        steps = int(num_inference_steps or self.cfg.num_inference_steps)

        trace: Dict[str, Any] = {
            "config": asdict(self.cfg),
            "num_inference_steps": steps,
        }

        curr_scheduler = self.pipe.scheduler
        self.pipe.scheduler = DDIMInverseScheduler.from_config(self.pipe.scheduler.config)

        x = transform_image_sdxl(image, image_size=self.cfg.image_size).unsqueeze(0)
        x = x.to(self.pipe.unet.dtype).to(self.device)
        trace["img_tensor_stats"] = tensor_stats(x)

        z0 = self.vae.encode(x).latent_dist.mode() * float(self.cfg.vae_scaling)
        trace["vae_latents_stats"] = tensor_stats(z0)

        zT_hat = self.pipe(
            prompt="",
            latents=z0,
            guidance_scale=1.0,
            num_inference_steps=steps,
            output_type="latent",
        ).images

        self.pipe.scheduler = curr_scheduler
        trace["inverted_latents_stats"] = tensor_stats(zT_hat)

        y_hat = self._latents_to_fft(zT_hat)
        trace["inv_fft_stats"] = tensor_stats(y_hat)
        save_gray_png(out_dir / f"{tag}_inv_fft.png", fft_mag01(y_hat, channel=self.cfg.w_channel))

        bits_out, bit_scores, ring_scores = self._decode_bits_from_fft(y_hat)
        msg_out = bits8_to_int(bits_out)

        trace["decoded_bits"] = bits_out
        trace["decoded_msg"] = int(msg_out)
        trace["bit_scores"] = bit_scores
        trace["ring_scores"] = ring_scores

        save_bar_png(
            out_dir / f"{tag}_ring_scores.png",
            ring_scores,
            title="Ring correlation scores",
            xlabel="ring idx",
        )
        save_bar_png(
            out_dir / f"{tag}_bit_scores.png",
            bit_scores,
            title="Bit scores (sum of rings per bit)",
            xlabel="bit idx",
        )

        with open(out_dir / f"{tag}_decode_trace.json", "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)

        return msg_out, bits_out, bit_scores, ring_scores, trace
