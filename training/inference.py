"""
CoarseFlow inference utilities.

Main usage:
    1. Full-volume inference for moderate-size inputs:
        phase = predict_phase_from_images(...)

    2. Patch-wise inference for very large inputs:
        phase = predict_phase_patchwise_large_volume(...)

Coordinate conventions:
    - Model-native coordinates are z,y,x.
    - phase_order='xyz' returns phase[...,0]=x, phase[...,1]=y, phase[...,2]=z.
    - phase_order='zyx' returns phase[...,0]=z, phase[...,1]=y, phase[...,2]=x.

Input array conventions:
    - mov_order='zyx': mov shape is (K,H,W)
    - mov_order='yxz': mov shape is (H,W,K)
    - ref_order='zyx': ref shape is (D,H,W)
    - ref_order='yxz': ref shape is (H,W,D)
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, torch.Tensor]


# =============================================================================
# Basic utilities
# =============================================================================


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def import_model_class(
    model_module: str = "CoarseFlow.models.SparseGMFlow3D",
    model_class: str = "CoarseMatchingNetV5",
):
    """
    Dynamically import model class.

    Examples:
        model_cls = import_model_class(
            model_module="CoarseFlow.models.SparseGMFlow3D",
            model_class="CoarseMatchingNetV5",
        )

        model_cls = import_model_class(
            model_module="CoarseFlow.models.SparseGMFlow3D_final",
            model_class="CoarseMatchingNetFinal",
        )
    """
    module = importlib.import_module(model_module)
    if not hasattr(module, model_class):
        raise AttributeError(
            f"Module {model_module!r} does not contain class {model_class!r}."
        )
    return getattr(module, model_class)


def to_numpy_float32(x: ArrayLike, copy: bool = False) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).copy() if copy else np.asarray(x, dtype=np.float32)


def convert_volume_to_zyx(arr: ArrayLike, order: str, name: str) -> np.ndarray:
    """
    Convert a 3D volume/sparse stack to (Z-or-K, Y, X).
    """
    arr = to_numpy_float32(arr, copy=False)

    if arr.ndim != 3:
        raise ValueError(f"{name} must be 3D, but got shape {arr.shape}.")

    if order == "zyx":
        return arr
    if order == "yxz":
        return arr.transpose(2, 0, 1)

    raise ValueError(f"{name}_order must be 'zyx' or 'yxz', but got {order!r}.")


def estimate_norm_stats_np(
    x: ArrayLike,
    p_low: float = 0.001,
    p_high: float = 0.999,
    mask_nonzero: bool = True,
    sample_voxels: int = 2_000_000,
    seed: int = 12345,
) -> Tuple[float, float]:
    """
    Estimate robust normalization percentiles from a large array by sampling.

    p_low/p_high are probabilities, e.g. 0.001 and 0.999.
    """
    x_np = to_numpy_float32(x, copy=False)
    vals = x_np[np.isfinite(x_np)]

    if mask_nonzero:
        vals_nz = vals[vals != 0]
        if vals_nz.size > 100:
            vals = vals_nz

    if vals.size == 0:
        return 0.0, 1.0

    if vals.size > sample_voxels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(vals.size, size=sample_voxels, replace=False)
        vals = vals[idx]

    lo = float(np.percentile(vals, p_low * 100.0))
    hi = float(np.percentile(vals, p_high * 100.0))

    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0

    return lo, hi


def normalize_with_stats_np(x: ArrayLike, lo: float, hi: float, eps: float = 1e-6) -> np.ndarray:
    x_np = to_numpy_float32(x, copy=False)
    y = (x_np - lo) / (hi - lo + eps)
    y = np.clip(y, 0.0, 1.0)
    return y.astype(np.float32, copy=False)


def percentile_normalize_01_np(
    x: ArrayLike,
    p_low: float = 0.001,
    p_high: float = 0.999,
    mask_nonzero: bool = True,
    sample_voxels: int = 2_000_000,
    seed: int = 12345,
) -> np.ndarray:
    lo, hi = estimate_norm_stats_np(
        x,
        p_low=p_low,
        p_high=p_high,
        mask_nonzero=mask_nonzero,
        sample_voxels=sample_voxels,
        seed=seed,
    )
    return normalize_with_stats_np(x, lo, hi)


# =============================================================================
# Model loading
# =============================================================================


def load_coarseflow_model_from_pth(
    pth_path: str,
    model_config: Optional[Dict[str, Any]] = None,
    model_module: str = "CoarseFlow.models.SparseGMFlow3D",
    model_class: str = "CoarseMatchingNetV5",
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
    state_key: str = "model",
):
    """
    Load a CoarseFlow model checkpoint.

    Args:
        pth_path:
            Checkpoint path.
        model_config:
            Exact model config. If None, this function tries ckpt['model_config'].
        model_module:
            Python module containing the model class.
        model_class:
            Model class name, e.g. 'CoarseMatchingNetV5', 'CoarseMatchingNetV6'.
        strict:
            Passed to model.load_state_dict.
        state_key:
            Key for state_dict inside checkpoint. Default: 'model'.

    Returns:
        model, model_config_used, checkpoint, load_message
    """
    device = get_device(device)

    ckpt = torch.load(pth_path, map_location=device, weights_only=False)

    if model_config is None:
        if isinstance(ckpt, dict) and "model_config" in ckpt:
            model_config = ckpt["model_config"]
        else:
            raise ValueError(
                "model_config is required because checkpoint does not contain "
                "ckpt['model_config']. Please pass the exact config used in training."
            )

    model_cls = import_model_class(model_module=model_module, model_class=model_class)
    model = model_cls(**model_config).to(device)

    if isinstance(ckpt, dict) and state_key in ckpt:
        state_dict = ckpt[state_key]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    load_msg = model.load_state_dict(state_dict, strict=strict)
    model.eval()

    return model, model_config, ckpt, load_msg


def call_model_forward(
    model: torch.nn.Module,
    mov_t: torch.Tensor,
    ref_t: torch.Tensor,
    z_init_t: torch.Tensor,
    spacing_t: Optional[torch.Tensor] = None,
    return_match_aux: bool = False,
):
    """
    Robust model forward wrapper.

    Different CoarseMatchingNet versions may or may not accept spacing or
    return_match_aux. This wrapper tries the newer signature first, then falls
    back to older ones.
    """
    try:
        return model(
            mov_t,
            ref_t,
            z_init=z_init_t,
            spacing=spacing_t,
            return_match_aux=return_match_aux,
        )
    except TypeError:
        pass

    try:
        return model(
            mov_t,
            ref_t,
            z_init=z_init_t,
            spacing=spacing_t,
        )
    except TypeError:
        pass

    return model(
        mov_t,
        ref_t,
        z_init=z_init_t,
    )


# =============================================================================
# Control-grid to dense phase
# =============================================================================


def control_coords_to_dense_phase_torch(
    pred_coords_zyx: ArrayLike,
    z_init: ArrayLike,
    image_shape_yx: Tuple[int, int],
    control_stride: int = 16,
    phase_order: str = "xyz",
    device: Optional[Union[str, torch.device]] = None,
    padding_mode: str = "border",
) -> np.ndarray:
    """
    Convert sparse control-point coordinates to dense phase.

    Args:
        pred_coords_zyx:
            (K,Hc,Wc,3), predicted reference coordinates at control points.
            Coordinate order: z,y,x.
        z_init:
            (K,), initial raw z index for each moving slice.
        image_shape_yx:
            Original moving image size: (H,W).
        control_stride:
            Control point stride in raw xy pixels.
        phase_order:
            'xyz' or 'zyx'.

    Returns:
        dense_phase:
            (K,H,W,3)
    """
    device = get_device(device)

    pred_coords_zyx_np = to_numpy_float32(pred_coords_zyx, copy=False)
    z_init_np = to_numpy_float32(z_init, copy=False).reshape(-1)

    if pred_coords_zyx_np.ndim != 4 or pred_coords_zyx_np.shape[-1] != 3:
        raise ValueError(
            f"pred_coords_zyx must have shape (K,Hc,Wc,3), "
            f"but got {pred_coords_zyx_np.shape}."
        )

    K, Hc, Wc, _ = pred_coords_zyx_np.shape
    H, W = image_shape_yx

    if z_init_np.shape[0] != K:
        raise ValueError(f"z_init length must equal K={K}, but got {z_init_np.shape}.")

    # Build control-grid initial coords in raw coordinate system.
    y_ctrl = np.arange(Hc, dtype=np.float32) * float(control_stride)
    x_ctrl = np.arange(Wc, dtype=np.float32) * float(control_stride)
    yy_ctrl, xx_ctrl = np.meshgrid(y_ctrl, x_ctrl, indexing="ij")

    coords0_ctrl = np.zeros_like(pred_coords_zyx_np, dtype=np.float32)
    coords0_ctrl[..., 0] = z_init_np[:, None, None]
    coords0_ctrl[..., 1] = yy_ctrl[None, :, :]
    coords0_ctrl[..., 2] = xx_ctrl[None, :, :]

    disp_ctrl = pred_coords_zyx_np - coords0_ctrl

    # (K,Hc,Wc,3) -> (K,3,Hc,Wc)
    disp_t = torch.from_numpy(disp_ctrl).to(device=device, dtype=torch.float32)
    disp_t = disp_t.permute(0, 3, 1, 2).contiguous()

    y_dense = torch.arange(H, device=device, dtype=torch.float32)
    x_dense = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_dense, x_dense, indexing="ij")

    yy_ctrl_idx = yy / float(control_stride)
    xx_ctrl_idx = xx / float(control_stride)

    if Hc > 1:
        yy_norm = 2.0 * yy_ctrl_idx / float(Hc - 1) - 1.0
    else:
        yy_norm = torch.zeros_like(yy_ctrl_idx)

    if Wc > 1:
        xx_norm = 2.0 * xx_ctrl_idx / float(Wc - 1) - 1.0
    else:
        xx_norm = torch.zeros_like(xx_ctrl_idx)

    # grid_sample expects grid[...,0]=x, grid[...,1]=y
    grid = torch.stack([xx_norm, yy_norm], dim=-1)  # (H,W,2)
    grid = grid[None].expand(K, H, W, 2).contiguous()

    dense_disp = F.grid_sample(
        disp_t,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )

    dense_disp = dense_disp.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    yy_np, xx_np = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )

    dense_coords_zyx = np.zeros((K, H, W, 3), dtype=np.float32)
    dense_coords_zyx[..., 0] = z_init_np[:, None, None]
    dense_coords_zyx[..., 1] = yy_np[None, :, :]
    dense_coords_zyx[..., 2] = xx_np[None, :, :]

    dense_coords_zyx = dense_coords_zyx + dense_disp

    if phase_order == "zyx":
        return dense_coords_zyx
    if phase_order == "xyz":
        return dense_coords_zyx[..., [2, 1, 0]]

    raise ValueError("phase_order must be 'xyz' or 'zyx'.")


# =============================================================================
# Full-image inference
# =============================================================================


@torch.no_grad()
def predict_phase_with_preloaded_model(
    model: torch.nn.Module,
    model_config: Dict[str, Any],
    mov_zyx: ArrayLike,
    ref_zyx: ArrayLike,
    z_init: ArrayLike,
    ref_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    device: Optional[Union[str, torch.device]] = None,
    phase_order: str = "xyz",
    use_amp: bool = True,
    return_dict: bool = False,
) -> Union[np.ndarray, Dict[str, Any]]:
    """
    Predict dense phase for one moderate-size volume using a preloaded model.

    Inputs must already be normalized if needed.

    Args:
        mov_zyx: (K,H,W)
        ref_zyx: (D,H,W)
        z_init:  (K,), global or local z-init consistent with ref_zyx

    Returns:
        phase or dict containing phase and control-grid outputs.
    """
    device = get_device(device)
    model.eval()

    mov_np = to_numpy_float32(mov_zyx, copy=False)
    ref_np = to_numpy_float32(ref_zyx, copy=False)
    z_init_np = to_numpy_float32(z_init, copy=False).reshape(-1)

    if mov_np.ndim != 3 or ref_np.ndim != 3:
        raise ValueError(f"mov/ref must be 3D. Got mov={mov_np.shape}, ref={ref_np.shape}.")

    K, H, W = mov_np.shape
    D, Hr, Wr = ref_np.shape
    if (Hr, Wr) != (H, W):
        raise ValueError(f"XY mismatch: mov={mov_np.shape}, ref={ref_np.shape}.")
    if z_init_np.shape != (K,):
        raise ValueError(f"z_init must have shape ({K},), but got {z_init_np.shape}.")

    mov_t = torch.from_numpy(mov_np).to(device=device, dtype=torch.float32)[None, None]
    ref_t = torch.from_numpy(ref_np).to(device=device, dtype=torch.float32)[None, None]
    z_init_t = torch.from_numpy(z_init_np).to(device=device, dtype=torch.float32)[None]
    spacing_t = torch.as_tensor(ref_spacing, dtype=torch.float32, device=device)[None]

    with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
        outputs = call_model_forward(
            model=model,
            mov_t=mov_t,
            ref_t=ref_t,
            z_init_t=z_init_t,
            spacing_t=spacing_t,
            return_match_aux=False,
        )

    if "pred_coords" not in outputs:
        raise KeyError("Model output does not contain 'pred_coords'.")

    pred_coords_ctrl_zyx = outputs["pred_coords"][0].detach().float().cpu().numpy()
    pred_disp_ctrl_zyx = (
        outputs["pred_disp"][0].detach().float().cpu().numpy()
        if "pred_disp" in outputs
        else None
    )

    control_stride = int(model_config.get("control_stride", 16))

    phase = control_coords_to_dense_phase_torch(
        pred_coords_zyx=pred_coords_ctrl_zyx,
        z_init=z_init_np,
        image_shape_yx=(H, W),
        control_stride=control_stride,
        phase_order=phase_order,
        device=device,
    )

    if not return_dict:
        return phase

    result: Dict[str, Any] = {
        "phase": phase,
        "pred_coords_ctrl_zyx": pred_coords_ctrl_zyx,
        "z_init": z_init_np,
        "model_config": model_config,
    }

    if pred_disp_ctrl_zyx is not None:
        result["pred_disp_ctrl_zyx"] = pred_disp_ctrl_zyx

    for key in [
        "pred_coords_coarse",
        "pred_disp_coarse",
        "coord_residual_delta",
        "residual_delta",
    ]:
        if key in outputs:
            result[f"{key}_ctrl_zyx"] = outputs[key][0].detach().float().cpu().numpy()

    return result


@torch.no_grad()
def predict_phase_from_images(
    mov: ArrayLike,
    ref: ArrayLike,
    z_init: ArrayLike,
    ref_spacing: Sequence[float],
    pth_path: str,
    model_config: Optional[Dict[str, Any]] = None,
    model_module: str = "CoarseFlow.models.SparseGMFlow3D",
    model_class: str = "CoarseMatchingNetV5",
    device: Optional[Union[str, torch.device]] = None,
    mov_order: str = "zyx",
    ref_order: str = "zyx",
    normalize: bool = True,
    norm_p_low: float = 0.001,
    norm_p_high: float = 0.999,
    norm_mask_nonzero: bool = True,
    norm_sample_voxels: int = 2_000_000,
    phase_order: str = "xyz",
    use_amp: bool = True,
    strict_load: bool = True,
    return_dict: bool = False,
) -> Union[np.ndarray, Dict[str, Any]]:
    """
    Full-image inference wrapper.

    Use this only when mov/ref are not too large. For large images, use
    predict_phase_patchwise_large_volume.
    """
    device = get_device(device)

    mov_zyx = convert_volume_to_zyx(mov, order=mov_order, name="mov")
    ref_zyx = convert_volume_to_zyx(ref, order=ref_order, name="ref")

    K, H, W = mov_zyx.shape
    D, Hr, Wr = ref_zyx.shape
    if (Hr, Wr) != (H, W):
        raise ValueError(f"XY mismatch: mov={mov_zyx.shape}, ref={ref_zyx.shape}.")

    z_init_np = to_numpy_float32(z_init, copy=False).reshape(-1)
    if z_init_np.shape != (K,):
        raise ValueError(
            f"z_init must have shape ({K},), but got {z_init_np.shape}. "
            "If mov is (H,W,K), use mov_order='yxz'."
        )

    if normalize:
        mov_zyx = percentile_normalize_01_np(
            mov_zyx,
            p_low=norm_p_low,
            p_high=norm_p_high,
            mask_nonzero=norm_mask_nonzero,
            sample_voxels=norm_sample_voxels,
        )
        ref_zyx = percentile_normalize_01_np(
            ref_zyx,
            p_low=norm_p_low,
            p_high=norm_p_high,
            mask_nonzero=norm_mask_nonzero,
            sample_voxels=norm_sample_voxels,
        )

    model, model_config_used, ckpt, load_msg = load_coarseflow_model_from_pth(
        pth_path=pth_path,
        model_config=model_config,
        model_module=model_module,
        model_class=model_class,
        device=device,
        strict=strict_load,
    )

    result = predict_phase_with_preloaded_model(
        model=model,
        model_config=model_config_used,
        mov_zyx=mov_zyx,
        ref_zyx=ref_zyx,
        z_init=z_init_np,
        ref_spacing=ref_spacing,
        device=device,
        phase_order=phase_order,
        use_amp=use_amp,
        return_dict=True,
    )

    result["checkpoint"] = ckpt
    result["load_msg"] = load_msg

    if return_dict:
        return result
    return result["phase"]


# =============================================================================
# Patch-wise inference for large volumes
# =============================================================================


def make_patch_starts(size: int, patch_size: int, overlap: int) -> List[int]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if patch_size >= size:
        return [0]

    step = patch_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than patch_size.")

    starts = list(range(0, size - patch_size + 1, step))
    last = size - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def get_inner_slices_2d(
    y0: int,
    x0: int,
    patch_h: int,
    patch_w: int,
    full_h: int,
    full_w: int,
    inner_crop_ratio: float = 0.10,
) -> Tuple[slice, slice, slice, slice]:
    """
    Get local/global reliable central region.

    Boundary sides are not cropped, otherwise full-image border would be uncovered.
    """
    by = int(round(patch_h * inner_crop_ratio))
    bx = int(round(patch_w * inner_crop_ratio))

    ly0 = 0 if y0 == 0 else by
    lx0 = 0 if x0 == 0 else bx

    gy1 = y0 + patch_h
    gx1 = x0 + patch_w

    ly1 = patch_h if gy1 >= full_h else patch_h - by
    lx1 = patch_w if gx1 >= full_w else patch_w - bx

    lys = slice(ly0, ly1)
    lxs = slice(lx0, lx1)
    gys = slice(y0 + ly0, y0 + ly1)
    gxs = slice(x0 + lx0, x0 + lx1)

    return lys, lxs, gys, gxs


def make_soft_weight_2d(h: int, w: int) -> np.ndarray:
    """
    Smooth weight map: low at patch edge, high at patch center.
    """
    yy = np.linspace(0, 1, h, dtype=np.float32)
    xx = np.linspace(0, 1, w, dtype=np.float32)

    wy = np.minimum(yy, 1.0 - yy)
    wx = np.minimum(xx, 1.0 - xx)

    wy = wy / (wy.max() + 1e-6)
    wx = wx / (wx.max() + 1e-6)

    weight = wy[:, None] * wx[None, :]
    weight = np.clip(weight, 1e-3, 1.0)
    return weight.astype(np.float32)


def build_z_windows_from_z_init(
    z_init: ArrayLike,
    ref_depth: int,
    z_window_size: int = 30,
    z_margin: int = 5,
    z_stride: Optional[int] = None,
    max_k_per_patch: Optional[int] = 7,
) -> List[Dict[str, Any]]:
    """
    Build z windows and assign moving slices according to initial global z.

    A slice k is assigned to a z window if z_init[k] falls into the usable core
    of that window. z_margin protects against motion moving outside the local
    ref volume.
    """
    z_init_np = to_numpy_float32(z_init, copy=False).reshape(-1)

    if z_window_size <= 0:
        raise ValueError("z_window_size must be positive.")
    if z_window_size > ref_depth:
        z_window_size = ref_depth
    if z_margin < 0:
        raise ValueError("z_margin must be non-negative.")
    if 2 * z_margin >= z_window_size:
        raise ValueError("Require 2*z_margin < z_window_size.")

    if z_stride is None:
        z_stride = max(1, z_window_size - 2 * z_margin)

    overlap = z_window_size - z_stride
    overlap = max(0, min(overlap, z_window_size - 1))

    z_starts = make_patch_starts(ref_depth, z_window_size, overlap)
    windows: List[Dict[str, Any]] = []

    for z0 in z_starts:
        z1 = min(z0 + z_window_size, ref_depth)

        core0 = z0 if z0 == 0 else z0 + z_margin
        core1 = z1 if z1 == ref_depth else z1 - z_margin

        k_indices = np.where((z_init_np >= core0) & (z_init_np <= core1))[0]
        if k_indices.size == 0:
            continue

        if max_k_per_patch is not None and k_indices.size > max_k_per_patch:
            for s in range(0, k_indices.size, max_k_per_patch):
                sub = k_indices[s : s + max_k_per_patch]
                windows.append({"z0": int(z0), "z1": int(z1), "k_indices": sub.astype(np.int64)})
        else:
            windows.append({"z0": int(z0), "z1": int(z1), "k_indices": k_indices.astype(np.int64)})

    return windows


@torch.no_grad()
def predict_phase_one_patch_preloaded(
    model: torch.nn.Module,
    model_config: Dict[str, Any],
    mov_patch_zyx: ArrayLike,
    ref_patch_zyx: ArrayLike,
    z_init_local: ArrayLike,
    ref_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    device: Optional[Union[str, torch.device]] = None,
    phase_order: str = "xyz",
    use_amp: bool = True,
) -> np.ndarray:
    """
    Predict local dense phase for one local patch using a preloaded model.
    """
    return predict_phase_with_preloaded_model(
        model=model,
        model_config=model_config,
        mov_zyx=mov_patch_zyx,
        ref_zyx=ref_patch_zyx,
        z_init=z_init_local,
        ref_spacing=ref_spacing,
        device=device,
        phase_order=phase_order,
        use_amp=use_amp,
        return_dict=False,
    )


@torch.no_grad()
def predict_phase_patchwise_large_volume(
    mov: ArrayLike,
    ref: ArrayLike,
    z_init: ArrayLike,
    ref_spacing: Sequence[float],
    pth_path: str,
    model_config: Optional[Dict[str, Any]] = None,
    model_module: str = "CoarseFlow.models.SparseGMFlow3D",
    model_class: str = "CoarseMatchingNetV5",
    device: Optional[Union[str, torch.device]] = None,
    mov_order: str = "zyx",
    ref_order: str = "zyx",
    patch_size_yx: Tuple[int, int] = (256, 256),
    overlap_yx: Tuple[int, int] = (96, 96),
    inner_crop_ratio: float = 0.10,
    z_window_size: int = 30,
    z_margin: int = 5,
    z_stride: Optional[int] = None,
    max_k_per_patch: Optional[int] = 7,
    normalize: bool = True,
    norm_p_low: float = 0.001,
    norm_p_high: float = 0.999,
    norm_mask_nonzero: bool = True,
    norm_sample_voxels: int = 2_000_000,
    phase_order: str = "xyz",
    use_amp: bool = True,
    strict_load: bool = True,
    return_weight: bool = False,
    return_debug: bool = False,
    verbose: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
    """
    Patch-wise dense phase prediction for very large volumes.

    Args:
        mov:
            If mov_order='zyx': shape (K,H,W).
            If mov_order='yxz': shape (H,W,K).
        ref:
            If ref_order='zyx': shape (D,H,W).
            If ref_order='yxz': shape (H,W,D).
        z_init:
            Global initial z index of each moving slice, shape (K,).
        ref_spacing:
            Reference spacing, usually (z,y,x) in the same convention used during training.
        pth_path:
            Model checkpoint path.
        model_module/model_class:
            Dynamic model import location and class name.
        patch_size_yx/overlap_yx:
            XY patching parameters.
        inner_crop_ratio:
            Discard this ratio from non-boundary patch borders before stitching.
        z_window_size/z_margin:
            Z local reference window and safe margin.
        max_k_per_patch:
            Split moving slices if too many slices fall in one z window.

    Returns:
        phase: (K,H,W,3) by default.
        If return_weight=True, returns (phase, weight_acc).
        If return_debug=True, returns dict.
    """
    device = get_device(device)

    mov_zyx = convert_volume_to_zyx(mov, order=mov_order, name="mov")
    ref_zyx = convert_volume_to_zyx(ref, order=ref_order, name="ref")

    K, H, W = mov_zyx.shape
    D, Hr, Wr = ref_zyx.shape

    if (Hr, Wr) != (H, W):
        raise ValueError(f"XY mismatch: mov={mov_zyx.shape}, ref={ref_zyx.shape}.")

    z_init_np = to_numpy_float32(z_init, copy=False).reshape(-1)
    if z_init_np.shape != (K,):
        raise ValueError(
            f"z_init must have shape ({K},), but got {z_init_np.shape}. "
            "If mov is (H,W,K), use mov_order='yxz'."
        )

    # Load once.
    model, model_config_used, ckpt, load_msg = load_coarseflow_model_from_pth(
        pth_path=pth_path,
        model_config=model_config,
        model_module=model_module,
        model_class=model_class,
        device=device,
        strict=strict_load,
    )
    model.eval()

    if verbose:
        print(f"[Model] loaded: {pth_path}")
        print(f"[Model] class: {model_module}.{model_class}")
        print(f"[Shape] mov=(K,H,W)={mov_zyx.shape}, ref=(D,H,W)={ref_zyx.shape}")
        print(f"[Config] control_stride={model_config_used.get('control_stride', 16)}")

    # Estimate global normalization stats once.
    if normalize:
        mov_lo, mov_hi = estimate_norm_stats_np(
            mov_zyx,
            p_low=norm_p_low,
            p_high=norm_p_high,
            mask_nonzero=norm_mask_nonzero,
            sample_voxels=norm_sample_voxels,
        )
        ref_lo, ref_hi = estimate_norm_stats_np(
            ref_zyx,
            p_low=norm_p_low,
            p_high=norm_p_high,
            mask_nonzero=norm_mask_nonzero,
            sample_voxels=norm_sample_voxels,
        )
        if verbose:
            print(f"[Norm] mov lo/hi = {mov_lo:.6g}, {mov_hi:.6g}")
            print(f"[Norm] ref lo/hi = {ref_lo:.6g}, {ref_hi:.6g}")
    else:
        mov_lo, mov_hi, ref_lo, ref_hi = 0.0, 1.0, 0.0, 1.0

    patch_h, patch_w = patch_size_yx
    overlap_y, overlap_x = overlap_yx

    if patch_h > H or patch_w > W:
        raise ValueError(
            f"patch_size_yx={patch_size_yx} larger than image size H,W={(H,W)}."
        )

    y_starts = make_patch_starts(H, patch_h, overlap_y)
    x_starts = make_patch_starts(W, patch_w, overlap_x)

    z_windows = build_z_windows_from_z_init(
        z_init=z_init_np,
        ref_depth=D,
        z_window_size=z_window_size,
        z_margin=z_margin,
        z_stride=z_stride,
        max_k_per_patch=max_k_per_patch,
    )

    if len(z_windows) == 0:
        raise RuntimeError(
            "No z windows were generated. Check z_init, z_window_size, and z_margin."
        )

    if verbose:
        print(f"[Tiling] z_windows={len(z_windows)}")
        print(f"[Tiling] y_patches={len(y_starts)}, x_patches={len(x_starts)}")
        print(f"[Tiling] total_forwards={len(z_windows) * len(y_starts) * len(x_starts)}")

    phase_acc = np.zeros((K, H, W, 3), dtype=np.float32)
    weight_acc = np.zeros((K, H, W), dtype=np.float32)

    soft_weight_full = make_soft_weight_2d(patch_h, patch_w)

    patch_counter = 0
    total_forwards = len(z_windows) * len(y_starts) * len(x_starts)

    for zw in z_windows:
        z0 = int(zw["z0"])
        z1 = int(zw["z1"])
        k_indices = np.asarray(zw["k_indices"], dtype=np.int64)

        if k_indices.size == 0:
            continue

        z_init_local = z_init_np[k_indices] - float(z0)

        for y0 in y_starts:
            y1 = y0 + patch_h

            for x0 in x_starts:
                x1 = x0 + patch_w
                patch_counter += 1

                mov_patch = mov_zyx[k_indices, y0:y1, x0:x1]
                ref_patch = ref_zyx[z0:z1, y0:y1, x0:x1]

                if normalize:
                    mov_patch = normalize_with_stats_np(mov_patch, mov_lo, mov_hi)
                    ref_patch = normalize_with_stats_np(ref_patch, ref_lo, ref_hi)
                else:
                    mov_patch = mov_patch.astype(np.float32, copy=False)
                    ref_patch = ref_patch.astype(np.float32, copy=False)

                phase_patch = predict_phase_one_patch_preloaded(
                    model=model,
                    model_config=model_config_used,
                    mov_patch_zyx=mov_patch,
                    ref_patch_zyx=ref_patch,
                    z_init_local=z_init_local,
                    ref_spacing=ref_spacing,
                    device=device,
                    phase_order=phase_order,
                    use_amp=use_amp,
                )

                # Convert local phase to global phase.
                if phase_order == "xyz":
                    phase_patch[..., 0] += float(x0)
                    phase_patch[..., 1] += float(y0)
                    phase_patch[..., 2] += float(z0)
                elif phase_order == "zyx":
                    phase_patch[..., 0] += float(z0)
                    phase_patch[..., 1] += float(y0)
                    phase_patch[..., 2] += float(x0)
                else:
                    raise ValueError("phase_order must be 'xyz' or 'zyx'.")

                lys, lxs, gys, gxs = get_inner_slices_2d(
                    y0=y0,
                    x0=x0,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    full_h=H,
                    full_w=W,
                    inner_crop_ratio=inner_crop_ratio,
                )

                w_patch = soft_weight_full[lys, lxs]

                for local_i, global_k in enumerate(k_indices):
                    phase_acc[global_k, gys, gxs, :] += (
                        phase_patch[local_i, lys, lxs, :] * w_patch[..., None]
                    )
                    weight_acc[global_k, gys, gxs] += w_patch

                if verbose and (patch_counter % 20 == 0 or patch_counter == total_forwards):
                    coverage = float(np.mean(weight_acc > 0))
                    print(
                        f"[Patch {patch_counter:05d}/{total_forwards:05d}] "
                        f"z=({z0},{z1}), K={len(k_indices)}, "
                        f"y=({y0},{y1}), x=({x0},{x1}), coverage={coverage:.4f}"
                    )

    phase = np.zeros_like(phase_acc, dtype=np.float32)
    valid = weight_acc > 0
    phase[valid] = phase_acc[valid] / weight_acc[valid, None]

    # Fill uncovered region with identity phase.
    if np.any(~valid):
        if verbose:
            print(f"[Warning] uncovered ratio={float(np.mean(~valid)):.6f}")

        yy, xx = np.meshgrid(
            np.arange(H, dtype=np.float32),
            np.arange(W, dtype=np.float32),
            indexing="ij",
        )

        for k in range(K):
            missing = ~valid[k]
            if phase_order == "xyz":
                phase[k, missing, 0] = xx[missing]
                phase[k, missing, 1] = yy[missing]
                phase[k, missing, 2] = z_init_np[k]
            else:
                phase[k, missing, 0] = z_init_np[k]
                phase[k, missing, 1] = yy[missing]
                phase[k, missing, 2] = xx[missing]

    if verbose:
        print(f"[Done] final coverage={float(np.mean(valid)):.6f}")
        print(f"[Done] phase shape={phase.shape}")

    if return_debug:
        return {
            "phase": phase,
            "weight": weight_acc,
            "valid": valid,
            "z_windows": z_windows,
            "y_starts": y_starts,
            "x_starts": x_starts,
            "model_config": model_config_used,
            "checkpoint": ckpt,
            "load_msg": load_msg,
            "norm_stats": {
                "mov_lo": mov_lo,
                "mov_hi": mov_hi,
                "ref_lo": ref_lo,
                "ref_hi": ref_hi,
            },
        }

    if return_weight:
        return phase, weight_acc

    return phase


# =============================================================================
# Optional convenience wrapper
# =============================================================================


@dataclass
class CoarseFlowInferenceConfig:
    pth_path: str
    model_config: Optional[Dict[str, Any]] = None
    model_module: str = "CoarseFlow.models.SparseGMFlow3D"
    model_class: str = "CoarseMatchingNetV5"
    device: Optional[Union[str, torch.device]] = None
    strict_load: bool = True
    use_amp: bool = True


class CoarseFlowPredictor:
    """
    Reusable predictor that loads the model only once.
    """

    def __init__(self, cfg: CoarseFlowInferenceConfig):
        self.cfg = cfg
        self.device = get_device(cfg.device)
        (
            self.model,
            self.model_config,
            self.checkpoint,
            self.load_msg,
        ) = load_coarseflow_model_from_pth(
            pth_path=cfg.pth_path,
            model_config=cfg.model_config,
            model_module=cfg.model_module,
            model_class=cfg.model_class,
            device=self.device,
            strict=cfg.strict_load,
        )
        self.model.eval()

    @torch.no_grad()
    def predict_full(
        self,
        mov: ArrayLike,
        ref: ArrayLike,
        z_init: ArrayLike,
        ref_spacing: Sequence[float],
        mov_order: str = "zyx",
        ref_order: str = "zyx",
        normalize: bool = True,
        phase_order: str = "xyz",
        return_dict: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]:
        mov_zyx = convert_volume_to_zyx(mov, order=mov_order, name="mov")
        ref_zyx = convert_volume_to_zyx(ref, order=ref_order, name="ref")

        if normalize:
            mov_zyx = percentile_normalize_01_np(mov_zyx)
            ref_zyx = percentile_normalize_01_np(ref_zyx)

        return predict_phase_with_preloaded_model(
            model=self.model,
            model_config=self.model_config,
            mov_zyx=mov_zyx,
            ref_zyx=ref_zyx,
            z_init=z_init,
            ref_spacing=ref_spacing,
            device=self.device,
            phase_order=phase_order,
            use_amp=self.cfg.use_amp,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def predict_patchwise(
        self,
        mov: ArrayLike,
        ref: ArrayLike,
        z_init: ArrayLike,
        ref_spacing: Sequence[float],
        mov_order: str = "zyx",
        ref_order: str = "zyx",
        patch_size_yx: Tuple[int, int] = (256, 256),
        overlap_yx: Tuple[int, int] = (96, 96),
        inner_crop_ratio: float = 0.10,
        z_window_size: int = 30,
        z_margin: int = 5,
        z_stride: Optional[int] = None,
        max_k_per_patch: Optional[int] = 7,
        normalize: bool = True,
        phase_order: str = "xyz",
        return_weight: bool = False,
        return_debug: bool = False,
        verbose: bool = True,
    ):
        """
        Patch-wise prediction using the already loaded model.

        This method duplicates the main patchwise logic, but avoids reloading the
        model. Internally it writes a temporary direct loop using self.model.
        """
        device = self.device

        mov_zyx = convert_volume_to_zyx(mov, order=mov_order, name="mov")
        ref_zyx = convert_volume_to_zyx(ref, order=ref_order, name="ref")

        K, H, W = mov_zyx.shape
        D, Hr, Wr = ref_zyx.shape
        if (Hr, Wr) != (H, W):
            raise ValueError(f"XY mismatch: mov={mov_zyx.shape}, ref={ref_zyx.shape}.")

        z_init_np = to_numpy_float32(z_init, copy=False).reshape(-1)
        if z_init_np.shape != (K,):
            raise ValueError(f"z_init must have shape ({K},), but got {z_init_np.shape}.")

        if normalize:
            mov_lo, mov_hi = estimate_norm_stats_np(mov_zyx)
            ref_lo, ref_hi = estimate_norm_stats_np(ref_zyx)
        else:
            mov_lo, mov_hi, ref_lo, ref_hi = 0.0, 1.0, 0.0, 1.0

        patch_h, patch_w = patch_size_yx
        overlap_y, overlap_x = overlap_yx
        y_starts = make_patch_starts(H, patch_h, overlap_y)
        x_starts = make_patch_starts(W, patch_w, overlap_x)
        z_windows = build_z_windows_from_z_init(
            z_init=z_init_np,
            ref_depth=D,
            z_window_size=z_window_size,
            z_margin=z_margin,
            z_stride=z_stride,
            max_k_per_patch=max_k_per_patch,
        )

        if verbose:
            print(f"[Shape] mov={mov_zyx.shape}, ref={ref_zyx.shape}")
            print(f"[Tiling] z_windows={len(z_windows)}, y={len(y_starts)}, x={len(x_starts)}")

        phase_acc = np.zeros((K, H, W, 3), dtype=np.float32)
        weight_acc = np.zeros((K, H, W), dtype=np.float32)
        soft_weight_full = make_soft_weight_2d(patch_h, patch_w)

        total_forwards = len(z_windows) * len(y_starts) * len(x_starts)
        patch_counter = 0

        for zw in z_windows:
            z0, z1 = int(zw["z0"]), int(zw["z1"])
            k_indices = np.asarray(zw["k_indices"], dtype=np.int64)
            z_init_local = z_init_np[k_indices] - float(z0)

            for y0 in y_starts:
                y1 = y0 + patch_h
                for x0 in x_starts:
                    x1 = x0 + patch_w
                    patch_counter += 1

                    mov_patch = mov_zyx[k_indices, y0:y1, x0:x1]
                    ref_patch = ref_zyx[z0:z1, y0:y1, x0:x1]

                    if normalize:
                        mov_patch = normalize_with_stats_np(mov_patch, mov_lo, mov_hi)
                        ref_patch = normalize_with_stats_np(ref_patch, ref_lo, ref_hi)
                    else:
                        mov_patch = mov_patch.astype(np.float32, copy=False)
                        ref_patch = ref_patch.astype(np.float32, copy=False)

                    phase_patch = predict_phase_one_patch_preloaded(
                        model=self.model,
                        model_config=self.model_config,
                        mov_patch_zyx=mov_patch,
                        ref_patch_zyx=ref_patch,
                        z_init_local=z_init_local,
                        ref_spacing=ref_spacing,
                        device=device,
                        phase_order=phase_order,
                        use_amp=self.cfg.use_amp,
                    )

                    if phase_order == "xyz":
                        phase_patch[..., 0] += float(x0)
                        phase_patch[..., 1] += float(y0)
                        phase_patch[..., 2] += float(z0)
                    else:
                        phase_patch[..., 0] += float(z0)
                        phase_patch[..., 1] += float(y0)
                        phase_patch[..., 2] += float(x0)

                    lys, lxs, gys, gxs = get_inner_slices_2d(
                        y0=y0,
                        x0=x0,
                        patch_h=patch_h,
                        patch_w=patch_w,
                        full_h=H,
                        full_w=W,
                        inner_crop_ratio=inner_crop_ratio,
                    )
                    w_patch = soft_weight_full[lys, lxs]

                    for local_i, global_k in enumerate(k_indices):
                        phase_acc[global_k, gys, gxs, :] += (
                            phase_patch[local_i, lys, lxs, :] * w_patch[..., None]
                        )
                        weight_acc[global_k, gys, gxs] += w_patch

                    if verbose and (patch_counter % 20 == 0 or patch_counter == total_forwards):
                        print(
                            f"[Patch {patch_counter:05d}/{total_forwards:05d}] "
                            f"coverage={float(np.mean(weight_acc > 0)):.4f}"
                        )

        phase = np.zeros_like(phase_acc, dtype=np.float32)
        valid = weight_acc > 0
        phase[valid] = phase_acc[valid] / weight_acc[valid, None]

        if np.any(~valid):
            yy, xx = np.meshgrid(
                np.arange(H, dtype=np.float32),
                np.arange(W, dtype=np.float32),
                indexing="ij",
            )
            for k in range(K):
                missing = ~valid[k]
                if phase_order == "xyz":
                    phase[k, missing, 0] = xx[missing]
                    phase[k, missing, 1] = yy[missing]
                    phase[k, missing, 2] = z_init_np[k]
                else:
                    phase[k, missing, 0] = z_init_np[k]
                    phase[k, missing, 1] = yy[missing]
                    phase[k, missing, 2] = xx[missing]

        if return_debug:
            return {
                "phase": phase,
                "weight": weight_acc,
                "valid": valid,
                "z_windows": z_windows,
                "y_starts": y_starts,
                "x_starts": x_starts,
                "model_config": self.model_config,
                "norm_stats": {
                    "mov_lo": mov_lo,
                    "mov_hi": mov_hi,
                    "ref_lo": ref_lo,
                    "ref_hi": ref_hi,
                },
            }

        if return_weight:
            return phase, weight_acc
        return phase


