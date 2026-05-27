# datasets/synthetic_dataset.py

import os
import json
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Sampler


# ============================================================
# Basic utilities
# ============================================================


def set_random_seed(seed: Optional[int] = None) -> None:
    """Set Python / NumPy / PyTorch random seeds if seed is not None."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def percentile_normalize_np(
    x: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize a numpy image to [0, 1] using percentile clipping."""
    x = x.astype(np.float32, copy=False)
    lo, hi = np.percentile(x, [p_low, p_high])
    x = (x - lo) / (hi - lo + eps)
    x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float32, copy=False)


def make_motion_param_list(
    z_ratio: float,
    difficulty: str = "core",
    use_incompressibility: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build motion parameter list for generateMotion_Biophysical.

    Note:
        The returned zRatio is a fallback value. If
        use_volume_z_ratio_for_motion=True in the dataset, this zRatio will be
        overwritten for each sampled volume according to its own ref_spacing.

    difficulty:
        easy  : small / smooth motion
        core  : default production distribution
        hard  : larger / sharper motion
        mixed : broad mixture
    """
    z_ratio = float(z_ratio)

    if difficulty == "easy":
        configs = [
            dict(art_R_xy=160, amp_xy=1),
            dict(art_R_xy=160, amp_xy=2),
            dict(art_R_xy=120, amp_xy=1),
            dict(art_R_xy=120, amp_xy=2),
        ]
    elif difficulty == "core":
        configs = [
            dict(art_R_xy=120, amp_xy=1),
            dict(art_R_xy=120, amp_xy=2),
            dict(art_R_xy=120, amp_xy=4),
            dict(art_R_xy=120, amp_xy=6),
            dict(art_R_xy=120, amp_xy=8),
            dict(art_R_xy=80,  amp_xy=4),
            dict(art_R_xy=160, amp_xy=4),
        ]
    elif difficulty == "hard":
        configs = [
            dict(art_R_xy=80, amp_xy=8),
            dict(art_R_xy=80, amp_xy=10),
            dict(art_R_xy=60, amp_xy=8),
            dict(art_R_xy=60, amp_xy=10),
            dict(art_R_xy=50, amp_xy=6),
            dict(art_R_xy=50, amp_xy=8),
        ]
    elif difficulty == "mixed":
        configs = [
            dict(art_R_xy=160, amp_xy=1),
            dict(art_R_xy=160, amp_xy=2),
            dict(art_R_xy=120, amp_xy=2),
            dict(art_R_xy=120, amp_xy=4),
            dict(art_R_xy=120, amp_xy=6),
            dict(art_R_xy=120, amp_xy=8),
            dict(art_R_xy=80,  amp_xy=4),
            dict(art_R_xy=80,  amp_xy=6),
            dict(art_R_xy=80,  amp_xy=8),
            dict(art_R_xy=60,  amp_xy=6),
            dict(art_R_xy=60,  amp_xy=8),
        ]
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    out = []
    for cfg in configs:
        item = dict(
            art_R_xy=cfg["art_R_xy"],
            amp_xy=cfg["amp_xy"],
            zRatio=z_ratio,
            use_incompressibility=use_incompressibility,
        )
        out.append(item)

    return out


def normalize_volume_records(
    volumes: Sequence[Any],
    default_ref_spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> List[Dict[str, Any]]:
    """
    Normalize input volumes into records.

    Accepts:
        1. Plain numpy arrays:
            volume  # shape (X,Y,Z)

        2. Dict records:
            {
                "name": str,
                "volume": np.ndarray,       # shape (X,Y,Z)
                "ref_spacing": (sx,sy,sz),  # physical spacing of this volume
            }

    Returns:
        list of records:
            {
                "name": str,
                "volume": np.ndarray,
                "ref_spacing": np.ndarray(3,)
            }
    """
    default_ref_spacing = np.asarray(default_ref_spacing, dtype=np.float32)
    if default_ref_spacing.shape != (3,):
        raise ValueError("default_ref_spacing must be length-3: (sx, sy, sz).")

    records: List[Dict[str, Any]] = []

    for i, item in enumerate(volumes):
        if isinstance(item, dict):
            if "volume" not in item:
                raise ValueError("Volume dict must contain key 'volume'.")

            volume = item["volume"]
            name = item.get("name", f"volume_{i}")
            ref_spacing = item.get("ref_spacing", default_ref_spacing)
        else:
            volume = item
            name = f"volume_{i}"
            ref_spacing = default_ref_spacing

        volume = np.asarray(volume, dtype=np.float32)
        ref_spacing = np.asarray(ref_spacing, dtype=np.float32)

        if volume.ndim != 3:
            raise ValueError(
                f"Each volume must have shape (X,Y,Z), got {volume.shape}."
            )

        if ref_spacing.shape != (3,):
            raise ValueError(
                f"ref_spacing must be length-3, got {ref_spacing}."
            )

        records.append(
            {
                "name": str(name),
                "volume": volume,
                "ref_spacing": ref_spacing,
            }
        )

    if len(records) == 0:
        raise ValueError("volumes must contain at least one volume.")

    return records


def summarize_volume_records(volumes: Sequence[Any], default_ref_spacing=(1.0, 1.0, 1.0)) -> List[Dict[str, Any]]:
    """Return a lightweight summary of volume names, shapes and spacing."""
    records = normalize_volume_records(volumes, default_ref_spacing=default_ref_spacing)
    summary = []
    for r in records:
        summary.append(
            {
                "name": r["name"],
                "shape_xyz": list(r["volume"].shape),
                "ref_spacing": [float(v) for v in r["ref_spacing"]],
            }
        )
    return summary


# ============================================================
# Core synthetic dataset
# ============================================================


class SparseZStackSyntheticDataset(Dataset):
    """
    Synthetic dataset for sparse-z-stack to dense reference volume coarse matching.

    Internal raw data:
        ref_raw:  (X, Y, Z)
        mov_raw:  (X, Y, Z)
        motion:   (X, Y, Z, 3), channel order assumed to be (x,y,z)

    Model output format:
        mov:       (1, K, Y, X)
        ref:       (1, Z, Y, X)
        spacing:   (6,), [sx_ref, sy_ref, sz_ref, sx_mov, sy_mov, sz_mov]
        gt_disp:   (K, Hc, Wc, 3), order=(z,y,x)
        gt_coords: (K, Hc, Wc, 3), order=(z,y,x)
        z_init:    (K,), raw reference z indices

    Important:
        In production mode, each volume should carry its own ref_spacing:
            {"name": ..., "volume": array, "ref_spacing": (sx,sy,sz)}

        sparse_step controls how many reference z-indices separate adjacent
        moving slices. For each sampled volume:
            mov_spacing = (sx_ref, sy_ref, sz_ref * sparse_step)

        Example:
            volume ref_spacing=(1,1,3), sparse_step=5
            sample spacing=[1,1,3, 1,1,15]
    """

    def __init__(
        self,
        volumes: Sequence[Any],
        motion_fn,
        warp_fn,
        num_sparse_slices: int = 5,
        control_stride: int = 16,
        ref_spacing: Sequence[float] = (1.0, 1.0, 1.0),
        mov_spacing: Optional[Sequence[float]] = None,
        sparse_step: Optional[int] = None,
        motion_kwargs: Optional[Dict[str, Any]] = None,
        noise_std: float = 0.0,
        normalize: bool = True,
        gt_direction: str = "mov_to_ref",
        num_samples_per_volume: int = 1,
        crop_size_xy: Optional[Tuple[int, int]] = None,
        random_crop: bool = True,
        random_z_start: bool = True,
        use_volume_z_ratio_for_motion: bool = True,
        return_motion: bool = False,
        return_meta: bool = True,
    ):
        self.motion_fn = motion_fn
        self.warp_fn = warp_fn

        self.num_sparse_slices = int(num_sparse_slices)
        self.control_stride = int(control_stride)

        self.default_ref_spacing = np.asarray(ref_spacing, dtype=np.float32)
        if self.default_ref_spacing.shape != (3,):
            raise ValueError("ref_spacing must be length-3: (sx, sy, sz).")

        self.volume_records = normalize_volume_records(
            volumes,
            default_ref_spacing=self.default_ref_spacing,
        )

        self.sparse_step = None if sparse_step is None else int(sparse_step)

        if mov_spacing is None:
            self.default_mov_spacing = None
        else:
            self.default_mov_spacing = np.asarray(mov_spacing, dtype=np.float32)
            if self.default_mov_spacing.shape != (3,):
                raise ValueError("mov_spacing must be length-3: (sx, sy, sz).")

        if self.sparse_step is None and self.default_mov_spacing is None:
            raise ValueError(
                "Either sparse_step or mov_spacing must be provided. "
                "For production training, prefer sparse_step."
            )

        self.motion_kwargs = motion_kwargs or {}
        self.noise_std = float(noise_std) if noise_std is not None else 0.0
        self.normalize = bool(normalize)
        self.gt_direction = gt_direction
        self.num_samples_per_volume = int(num_samples_per_volume)
        self.crop_size_xy = crop_size_xy
        self.random_crop = bool(random_crop)
        self.random_z_start = bool(random_z_start)
        self.use_volume_z_ratio_for_motion = bool(use_volume_z_ratio_for_motion)
        self.return_motion = bool(return_motion)
        self.return_meta = bool(return_meta)

        if self.gt_direction not in ["mov_to_ref", "ref_to_mov"]:
            raise ValueError("gt_direction must be 'mov_to_ref' or 'ref_to_mov'.")

    def __len__(self) -> int:
        return len(self.volume_records) * self.num_samples_per_volume

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return percentile_normalize_np(x, p_low=1.0, p_high=99.0)

    def _get_ref_spacing_for_volume(self, volume_idx: int) -> np.ndarray:
        return self.volume_records[volume_idx]["ref_spacing"].astype(np.float32)

    def _get_mov_spacing_for_volume(self, ref_spacing: np.ndarray) -> np.ndarray:
        """
        Compute moving spacing for the current volume.

        Preferred production rule:
            mov_spacing = (sx_ref, sy_ref, sz_ref * sparse_step)

        Fallback:
            use default_mov_spacing if sparse_step is None.
        """
        ref_spacing = np.asarray(ref_spacing, dtype=np.float32)

        if self.sparse_step is not None:
            mov_spacing = ref_spacing.copy()
            mov_spacing[2] = ref_spacing[2] * float(self.sparse_step)
            return mov_spacing.astype(np.float32)

        if self.default_mov_spacing is not None:
            return self.default_mov_spacing.astype(np.float32)

        raise ValueError("Either sparse_step or mov_spacing must be provided.")

    @staticmethod
    def _z_ratio_from_ref_spacing(ref_spacing: np.ndarray) -> float:
        sx, sy, sz = [float(v) for v in ref_spacing]
        xy = 0.5 * (sx + sy)
        return float(sz / (xy + 1e-8))

    @staticmethod
    def _sparse_step_in_ref_index(ref_spacing: np.ndarray, mov_spacing: np.ndarray) -> int:
        ref_sz = float(ref_spacing[2])
        mov_sz = float(mov_spacing[2])
        step = mov_sz / max(ref_sz, 1e-8)
        step = max(1, int(round(step)))
        return step

    def _sample_sparse_slices(
        self,
        Z: int,
        ref_spacing: np.ndarray,
        mov_spacing: np.ndarray,
    ) -> np.ndarray:
        """
        Sample K sparse z indices from the reference z-index system.

        step = round(mov_spacing_z / ref_spacing_z)

        If random_z_start=True, randomly choose a legal z start so the model
        sees different absolute z locations instead of always starting at 0.
        """
        K = self.num_sparse_slices
        step = self._sparse_step_in_ref_index(ref_spacing, mov_spacing)

        if Z <= 0:
            raise ValueError("Z must be positive.")

        max_start = Z - 1 - (K - 1) * step

        if max_start >= 0:
            if self.random_z_start:
                start = np.random.randint(0, max_start + 1)
            else:
                start = 0
            z_idx = start + np.arange(K, dtype=np.int64) * step
        else:
            # Not enough z depth for K distinct slices at this spacing.
            # Use valid indices and pad with the last valid slice.
            z_idx = np.arange(0, Z, step, dtype=np.int64)
            if len(z_idx) == 0:
                z_idx = np.array([0], dtype=np.int64)
            if len(z_idx) >= K:
                z_idx = z_idx[:K]
            else:
                pad_num = K - len(z_idx)
                pad = np.full(pad_num, z_idx[-1], dtype=np.int64)
                z_idx = np.concatenate([z_idx, pad], axis=0)

        z_idx = np.clip(z_idx, 0, Z - 1).astype(np.int64)
        return z_idx

    def _make_control_grid(self, X: int, Y: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build stride-grid control points.

        Current model convention:
            y = iy * control_stride
            x = ix * control_stride

        Returns:
            grid_x: (Yc, Xc)
            grid_y: (Yc, Xc)
        """
        xs = np.arange(0, X, self.control_stride, dtype=np.int64)
        ys = np.arange(0, Y, self.control_stride, dtype=np.int64)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        return grid_x, grid_y

    def _build_gt(
        self,
        motion: np.ndarray,
        sparse_z_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build GT control-point displacement and coordinates.

        Args:
            motion:
                (X,Y,Z,3), channel order=(x,y,z)
            sparse_z_idx:
                (K,)

        Returns:
            gt_disp:
                (K,Yc,Xc,3), order=(z,y,x)
            gt_coords:
                (K,Yc,Xc,3), order=(z,y,x)
        """
        X, Y, Z, C = motion.shape
        if C != 3:
            raise ValueError(f"motion must have 3 channels, got {C}.")

        grid_x, grid_y = self._make_control_grid(X, Y)
        Hc, Wc = grid_x.shape
        K = len(sparse_z_idx)

        gt_disp = np.zeros((K, Hc, Wc, 3), dtype=np.float32)
        gt_coords = np.zeros((K, Hc, Wc, 3), dtype=np.float32)

        for kk, z in enumerate(sparse_z_idx):
            z_int = int(np.clip(z, 0, Z - 1))

            # motion[x,y,z,:], channel=(x,y,z)
            disp_xyz = motion[grid_x, grid_y, z_int, :].astype(np.float32)

            if self.gt_direction == "ref_to_mov":
                disp_xyz = -disp_xyz

            # channel order (x,y,z) -> (z,y,x)
            disp_zyx = disp_xyz[..., [2, 1, 0]]

            coords_xyz = np.stack(
                [
                    grid_x.astype(np.float32),
                    grid_y.astype(np.float32),
                    np.full_like(grid_x, float(z_int), dtype=np.float32),
                ],
                axis=-1,
            )
            coords_zyx = coords_xyz[..., [2, 1, 0]]

            gt_disp[kk] = disp_zyx
            gt_coords[kk] = coords_zyx + disp_zyx

        return gt_disp, gt_coords

    def _crop_xy(self, volume: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crop XY region from volume.

        Args:
            volume: (X,Y,Z)

        Returns:
            cropped_volume, (x0,y0)
        """
        if self.crop_size_xy is None:
            return volume, (0, 0)

        crop_x, crop_y = self.crop_size_xy
        X, Y, Z = volume.shape

        crop_x = int(min(crop_x, X))
        crop_y = int(min(crop_y, Y))

        if self.random_crop:
            x0 = np.random.randint(0, X - crop_x + 1)
            y0 = np.random.randint(0, Y - crop_y + 1)
        else:
            x0 = (X - crop_x) // 2
            y0 = (Y - crop_y) // 2

        cropped = volume[x0:x0 + crop_x, y0:y0 + crop_y, :]
        return cropped, (int(x0), int(y0))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        volume_idx = idx % len(self.volume_records)
        record = self.volume_records[volume_idx]

        volume_name = record["name"]
        ref_spacing_cur = self._get_ref_spacing_for_volume(volume_idx)
        mov_spacing_cur = self._get_mov_spacing_for_volume(ref_spacing_cur)
        sparse_step_cur = self._sparse_step_in_ref_index(ref_spacing_cur, mov_spacing_cur)

        ref_raw = record["volume"].astype(np.float32, copy=False)
        ref_raw, crop_origin_xy = self._crop_xy(ref_raw)

        if self.normalize:
            ref_raw = self._normalize(ref_raw)

        X, Y, Z = ref_raw.shape

        # 1. Generate motion on the cropped volume.
        motion_kwargs_cur = dict(self.motion_kwargs)
        if self.use_volume_z_ratio_for_motion:
            motion_kwargs_cur["zRatio"] = self._z_ratio_from_ref_spacing(ref_spacing_cur)

        motion = self.motion_fn(
            ref_raw.shape,
            **motion_kwargs_cur,
        ).astype(np.float32)

        # 2. Warp reference to generate moving volume.
        mov_raw = self.warp_fn(ref_raw, motion).astype(np.float32)

        # 3. Add noise to moving volume.
        if self.noise_std is not None and self.noise_std > 0:
            noise = np.random.randn(*mov_raw.shape).astype(np.float32) * np.float32(self.noise_std)
            mov_raw = (mov_raw + noise).astype(np.float32)

        if self.normalize:
            mov_raw = self._normalize(mov_raw)

        # 4. Sparse z sampling.
        sparse_z_idx = self._sample_sparse_slices(
            Z,
            ref_spacing=ref_spacing_cur,
            mov_spacing=mov_spacing_cur,
        )
        mov_sparse = mov_raw[:, :, sparse_z_idx]  # (X,Y,K)

        # 5. Build GT control grid.
        gt_disp, gt_coords = self._build_gt(motion, sparse_z_idx)

        # 6. Per-sample spacing vector:
        # [sx_ref, sy_ref, sz_ref, sx_mov, sy_mov, sz_mov]
        spacing = np.concatenate([ref_spacing_cur, mov_spacing_cur], axis=0).astype(np.float32)

        # 7. Convert layout for PyTorch model.
        # raw ref:        (X,Y,Z) -> model ref: (1,Z,Y,X)
        # raw mov_sparse: (X,Y,K) -> model mov: (1,K,Y,X)
        ref_zyx = np.transpose(ref_raw, (2, 1, 0)).copy()
        mov_kyx = np.transpose(mov_sparse, (2, 1, 0)).copy()

        sample: Dict[str, Any] = {
            "mov": torch.from_numpy(mov_kyx[None]).float(),
            "ref": torch.from_numpy(ref_zyx[None]).float(),
            "spacing": torch.from_numpy(spacing).float(),
            "gt_disp": torch.from_numpy(gt_disp).float(),
            "gt_coords": torch.from_numpy(gt_coords).float(),
            "z_init": torch.from_numpy(sparse_z_idx.astype(np.float32)).float(),
            "sparse_z_idx": torch.from_numpy(sparse_z_idx.astype(np.float32)).float(),
        }

        if self.return_motion:
            # Warning: this can make cached datasets very large.
            sample["motion"] = torch.from_numpy(motion).float()

        if self.return_meta:
            sample["meta"] = {
                "volume_idx": int(volume_idx),
                "volume_name": volume_name,
                "crop_origin_xy": crop_origin_xy,
                "raw_shape_xyz": tuple(int(v) for v in ref_raw.shape),
                "K": int(self.num_sparse_slices),
                "sparse_step": int(sparse_step_cur),
                "ref_spacing": [float(v) for v in ref_spacing_cur],
                "mov_spacing": [float(v) for v in mov_spacing_cur],
                "motion_kwargs": dict(motion_kwargs_cur),
                "noise_std": float(self.noise_std),
                "gt_direction": self.gt_direction,
            }

        return sample


# ============================================================
# Mixed synthetic dataset
# ============================================================


class MixedSyntheticDataset(Dataset):
    """
    Randomly sample from multiple SparseZStackSyntheticDataset instances.

    This class is useful for one fixed K / sparse_step combination with multiple
    motion parameters and noise levels.
    """

    def __init__(
        self,
        volumes: Sequence[Any],
        motion_fn,
        warp_fn,
        motion_param_list: Sequence[Dict[str, Any]],
        num_samples: int = 500,
        num_sparse_slices: int = 5,
        control_stride: int = 16,
        ref_spacing: Sequence[float] = (1.0, 1.0, 1.0),
        mov_spacing: Optional[Sequence[float]] = None,
        sparse_step: Optional[int] = None,
        noise_std_list: Sequence[float] = (0.0,),
        normalize: bool = True,
        gt_direction: str = "mov_to_ref",
        crop_size_xy: Optional[Tuple[int, int]] = None,
        random_crop: bool = True,
        random_z_start: bool = True,
        use_volume_z_ratio_for_motion: bool = True,
        return_motion: bool = False,
        return_meta: bool = True,
    ):
        self.datasets: List[SparseZStackSyntheticDataset] = []

        for motion_kwargs in motion_param_list:
            for noise_std in noise_std_list:
                ds = SparseZStackSyntheticDataset(
                    volumes=volumes,
                    motion_fn=motion_fn,
                    warp_fn=warp_fn,
                    num_sparse_slices=num_sparse_slices,
                    control_stride=control_stride,
                    ref_spacing=ref_spacing,
                    mov_spacing=mov_spacing,
                    sparse_step=sparse_step,
                    num_samples_per_volume=1,
                    motion_kwargs=motion_kwargs,
                    noise_std=noise_std,
                    normalize=normalize,
                    crop_size_xy=crop_size_xy,
                    random_crop=random_crop,
                    random_z_start=random_z_start,
                    use_volume_z_ratio_for_motion=use_volume_z_ratio_for_motion,
                    return_motion=return_motion,
                    return_meta=return_meta,
                    gt_direction=gt_direction,
                )
                self.datasets.append(ds)

        if len(self.datasets) == 0:
            raise ValueError("No internal datasets were created.")

        self.num_samples = int(num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds = random.choice(self.datasets)
        return ds[random.randrange(len(ds))]


# ============================================================
# Cached dataset
# ============================================================


class CachedDataset(Dataset):
    """
    Cache generated synthetic samples into memory and save/load them by torch.save.
    """

    def __init__(
        self,
        base_dataset: Optional[Dataset] = None,
        num_cached: int = 50,
        cache_path: Optional[str] = None,
    ):
        if cache_path is not None:
            self.samples = torch.load(cache_path, map_location="cpu", weights_only=False)
        else:
            if base_dataset is None:
                raise ValueError("base_dataset must be provided when cache_path is None.")
            self.samples = [base_dataset[i] for i in range(int(num_cached))]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def save(self, cache_path: str) -> None:
        dirname = os.path.dirname(cache_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(self.samples, cache_path)

    @classmethod
    def load(cls, cache_path: str) -> "CachedDataset":
        return cls(cache_path=cache_path)


def load_cached_dataset(path: str) -> Dataset:
    """Load a cached dataset saved by CachedDataset.save()."""
    return CachedDataset.load(path)


# ============================================================
# Dataset part / manifest construction
# ============================================================


def build_one_cached_dataset_part(
    volumes: Sequence[Any],
    motion_fn,
    warp_fn,
    save_root: str,
    split: str,
    K: int,
    sparse_step: int,
    z_ratio: float,
    difficulty: str,
    num_samples: int,
    crop_size_xy: Tuple[int, int] = (256, 256),
    noise_std_list: Sequence[float] = (0.0,),
    control_stride: int = 16,
    normalize: bool = True,
    random_crop: bool = True,
    random_z_start: bool = True,
    use_volume_z_ratio_for_motion: bool = True,
    return_motion: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build and save one cached dataset part.

    One part has fixed:
        K, sparse_step, fallback z_ratio, difficulty.

    If volumes are dict records with ref_spacing, the actual sample spacing is
    per-volume. In that case z_ratio is only a fallback for plain ndarray
    volumes and for make_motion_param_list before per-volume overriding.
    """
    os.makedirs(save_root, exist_ok=True)
    set_random_seed(seed)

    default_ref_spacing = (1.0, 1.0, float(z_ratio))

    motion_param_list = make_motion_param_list(
        z_ratio=z_ratio,
        difficulty=difficulty,
        use_incompressibility=True,
    )

    base_dataset = MixedSyntheticDataset(
        volumes=volumes,
        motion_fn=motion_fn,
        warp_fn=warp_fn,
        motion_param_list=motion_param_list,
        num_samples=num_samples,
        num_sparse_slices=K,
        control_stride=control_stride,
        ref_spacing=default_ref_spacing,
        mov_spacing=None,
        sparse_step=sparse_step,
        noise_std_list=noise_std_list,
        normalize=normalize,
        gt_direction="mov_to_ref",
        crop_size_xy=crop_size_xy,
        random_crop=random_crop,
        random_z_start=random_z_start,
        use_volume_z_ratio_for_motion=use_volume_z_ratio_for_motion,
        return_motion=return_motion,
        return_meta=True,
    )

    cached_dataset = CachedDataset(
        base_dataset=base_dataset,
        num_cached=num_samples,
    )

    tag = (
        f"{split}"
        f"_K{int(K)}"
        f"_step{int(sparse_step)}"
        f"_zratioFallback{float(z_ratio):g}"
        f"_{difficulty}"
        f"_N{int(num_samples)}"
    )

    save_path = os.path.join(save_root, tag + ".pt")
    meta_path = os.path.join(save_root, tag + ".json")

    cached_dataset.save(save_path)

    meta = {
        "save_path": save_path,
        "split": split,
        "K": int(K),
        "sparse_step": int(sparse_step),
        "z_ratio_fallback": float(z_ratio),
        "difficulty": difficulty,
        "num_samples": int(num_samples),
        "crop_size_xy": list(crop_size_xy),
        "control_stride": int(control_stride),
        "spacing_mode": "per_volume_ref_spacing",
        "default_ref_spacing": list(default_ref_spacing),
        "mov_spacing_rule": "mov_spacing = (sx_ref, sy_ref, sz_ref * sparse_step)",
        "volume_records": summarize_volume_records(volumes, default_ref_spacing=default_ref_spacing),
        "noise_std_list": [float(v) for v in noise_std_list],
        "normalize": bool(normalize),
        "random_crop": bool(random_crop),
        "random_z_start": bool(random_z_start),
        "use_volume_z_ratio_for_motion": bool(use_volume_z_ratio_for_motion),
        "return_motion": bool(return_motion),
        "seed": seed,
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Saved dataset] {save_path}")
    print(f"[Saved meta   ] {meta_path}")

    return meta


def build_dataset_family(
    volumes: Sequence[Any],
    motion_fn,
    warp_fn,
    save_root: str,
    split: str,
    K_values: Sequence[int] = (3, 4, 5, 6, 7),
    sparse_steps: Sequence[int] = (3, 4, 5, 6, 8),
    z_ratios: Sequence[float] = (3.0,),
    difficulties: Sequence[str] = ("core",),
    num_samples_per_part: int = 120,
    crop_size_xy: Tuple[int, int] = (256, 256),
    noise_std_list: Sequence[float] = (0.0,),
    control_stride: int = 16,
    normalize: bool = True,
    random_crop: bool = True,
    random_z_start: bool = True,
    use_volume_z_ratio_for_motion: bool = True,
    return_motion: bool = False,
    seed_base: int = 1234,
) -> str:
    """
    Build a dataset family and save a manifest JSON file.

    Note:
        If every volume record has its own ref_spacing, z_ratios should usually
        be set to a single fallback value, e.g. z_ratios=(3.0,), to avoid
        creating redundant dataset parts.
    """
    os.makedirs(save_root, exist_ok=True)

    manifest: Dict[str, Any] = {
        "split": split,
        "save_root": save_root,
        "spacing_mode": "per_volume_ref_spacing",
        "parts": [],
    }

    count = 0
    for K in K_values:
        for sparse_step in sparse_steps:
            for z_ratio in z_ratios:
                for difficulty in difficulties:
                    seed = seed_base + count
                    meta = build_one_cached_dataset_part(
                        volumes=volumes,
                        motion_fn=motion_fn,
                        warp_fn=warp_fn,
                        save_root=save_root,
                        split=split,
                        K=int(K),
                        sparse_step=int(sparse_step),
                        z_ratio=float(z_ratio),
                        difficulty=str(difficulty),
                        num_samples=int(num_samples_per_part),
                        crop_size_xy=crop_size_xy,
                        noise_std_list=noise_std_list,
                        control_stride=control_stride,
                        normalize=normalize,
                        random_crop=random_crop,
                        random_z_start=random_z_start,
                        use_volume_z_ratio_for_motion=use_volume_z_ratio_for_motion,
                        return_motion=return_motion,
                        seed=seed,
                    )
                    manifest["parts"].append(meta)
                    count += 1

    manifest_path = os.path.join(save_root, f"{split}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Saved manifest] {manifest_path}")
    print(f"[Total parts] {len(manifest['parts'])}")

    return manifest_path


def load_dataset_from_manifest(manifest_path: str) -> Tuple[ConcatDataset, List[Dict[str, Any]]]:
    """Load all cached parts from a manifest and return ConcatDataset + part_infos."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    datasets: List[Dataset] = []
    part_infos: List[Dict[str, Any]] = []

    offset = 0
    for part in manifest["parts"]:
        ds = load_cached_dataset(part["save_path"])
        datasets.append(ds)

        n = len(ds)
        part_info = dict(part)
        part_info["start"] = offset
        part_info["end"] = offset + n
        part_info["length"] = n
        part_infos.append(part_info)
        offset += n

    concat_dataset = ConcatDataset(datasets)
    return concat_dataset, part_infos

def get_rank_world_size_from_env(
    distributed: bool = False,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Get DDP rank/world_size from torchrun environment.

    This does not require dist.init_process_group().
    It only reads environment variables set by torchrun.
    """
    if rank is not None and world_size is not None:
        return int(rank), int(world_size)

    if distributed or ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return rank, world_size

    return 0, 1


def shard_batches_for_ddp(
    batches: List[List[int]],
    rank: int = 0,
    world_size: int = 1,
    pad_to_equal: bool = True,
) -> List[List[int]]:
    """
    Split a global batch list across DDP ranks.

    Important:
        DDP requires each rank to run the same number of iterations.
        If len(batches) is not divisible by world_size, we pad by repeating
        several batches so that every rank has the same number of steps.
    """
    rank = int(rank)
    world_size = int(world_size)

    if world_size <= 1:
        return batches

    if len(batches) == 0:
        return []

    if pad_to_equal:
        remainder = len(batches) % world_size
        if remainder != 0:
            pad_num = world_size - remainder
            batches = batches + batches[:pad_num]

    return batches[rank::world_size]



# ============================================================
# Same-K batch sampler and dataloader builder
# ============================================================


class SameKBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that groups samples by K.

    Required for variable-K training because default DataLoader cannot stack
    samples with different moving slice numbers.

    It works with ConcatDataset created by load_dataset_from_manifest().
    Each part_info must contain:
        start, end, K
    """

    def __init__(
        self,
        part_infos: Sequence[Dict[str, Any]],
        batch_size: int = 4,
        shuffle: bool = True,
        drop_last: bool = False,
        distributed: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        seed: int = 0,
        pad_to_equal_batches: bool = True,
    ):
        self.part_infos = list(part_infos)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        self.groups: Dict[int, List[int]] = {}
        for info in self.part_infos:
            K = int(info["K"])
            indices = list(range(int(info["start"]), int(info["end"])))
            self.groups.setdefault(K, []).extend(indices)

        self.batches: List[List[int]] = []
        self._build_batches()
        self.distributed = bool(distributed)
        self.rank, self.world_size = get_rank_world_size_from_env(
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        self.seed = int(seed)
        self.epoch = 0
        self.pad_to_equal_batches = bool(pad_to_equal_batches)
        self.global_batches = []

    def _build_batches(self) -> None:
        rng = random.Random(self.seed + self.epoch)

        global_batches = []

        for K, indices in self.groups.items():
            idx = list(indices)

            if self.shuffle:
                rng.shuffle(idx)

            for i in range(0, len(idx), self.batch_size):
                batch = idx[i:i + self.batch_size]

                if len(batch) < self.batch_size and self.drop_last:
                    continue

                global_batches.append(batch)

        if self.shuffle:
            rng.shuffle(global_batches)

        self.global_batches = global_batches

        self.batches = shard_batches_for_ddp(
            global_batches,
            rank=self.rank,
            world_size=self.world_size,
            pad_to_equal=self.pad_to_equal_batches,
        )

    def __iter__(self):
        self._build_batches()
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._build_batches()

def build_sameK_loader(
    manifest_path: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    distributed: bool = False,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    seed: int = 0,
    pad_to_equal_batches: bool = True,
) -> Tuple[DataLoader, ConcatDataset, List[Dict[str, Any]]]:
    """Build a DataLoader whose batches contain only samples with the same K."""
    dataset, part_infos = load_dataset_from_manifest(manifest_path)

    batch_sampler = SameKBatchSampler(
        part_infos=part_infos,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        seed=seed,
        pad_to_equal_batches=pad_to_equal_batches,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader, dataset, part_infos

def build_sameShape_loader(
    manifest_path: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    verbose: bool = True,
    distributed: bool = False,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    seed: int = 0,
    pad_to_equal_batches: bool = True,
) -> Tuple[DataLoader, ConcatDataset, List[Dict[str, Any]]]:
    """
    Build a DataLoader whose batches contain samples with the same:
        K, reference depth D, H, W.

    This avoids default_collate errors when different volumes have different Z depth.
    """
    dataset, part_infos = load_dataset_from_manifest(manifest_path)

    batch_sampler = SameShapeBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        verbose=verbose,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        seed=seed,
        pad_to_equal_batches=pad_to_equal_batches,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader, dataset, part_infos

def summarize_manifest(manifest_path: str) -> None:
    """Print a compact summary of a dataset manifest."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    parts = manifest["parts"]
    total = sum(int(p["num_samples"]) for p in parts)

    print("=" * 80)
    print(f"Manifest: {manifest_path}")
    print(f"split   : {manifest.get('split')}")
    print(f"spacing : {manifest.get('spacing_mode', 'unknown')}")
    print(f"parts   : {len(parts)}")
    print(f"samples : {total}")

    Ks = sorted(set(int(p["K"]) for p in parts))
    steps = sorted(set(int(p["sparse_step"]) for p in parts))
    diffs = sorted(set(str(p["difficulty"]) for p in parts))
    z_fallbacks = sorted(set(float(p.get("z_ratio_fallback", p.get("z_ratio", 0.0))) for p in parts))

    print(f"K values          : {Ks}")
    print(f"sparse steps      : {steps}")
    print(f"z ratio fallbacks : {z_fallbacks}")
    print(f"difficulties      : {diffs}")

    # Print unique per-volume spacing summary from first part.
    if len(parts) > 0 and "volume_records" in parts[0]:
        print("volume records:")
        for v in parts[0]["volume_records"]:
            print(f"  - {v['name']}: shape={v['shape_xyz']}, ref_spacing={v['ref_spacing']}")

    print("=" * 80)

class SameShapeBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that groups cached samples by actual tensor shape.

    This is safer than SameKBatchSampler because samples may have the same K
    but different reference depth D.

    Group key:
        (K, D, H, W)

    Required for default_collate to stack tensors.
    """

    def  __init__(
        self,
        dataset: Dataset,
        batch_size: int = 4,
        shuffle: bool = True,
        drop_last: bool = False,
        verbose: bool = True,
        distributed: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        seed: int = 0,
        pad_to_equal_batches: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.verbose = bool(verbose)

        self.distributed = bool(distributed)
        self.rank, self.world_size = get_rank_world_size_from_env(
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        self.seed = int(seed)
        self.epoch = 0
        self.pad_to_equal_batches = bool(pad_to_equal_batches)

        self.groups = {}
        self._scan_dataset_shapes()

        self.global_batches = []
        self.batches = []
        self._build_batches()

    def _scan_dataset_shapes(self):
        """
        Scan cached dataset and group by actual sample shape.

        This assumes dataset is cached, so dataset[i] is deterministic and cheap.
        """
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]

            mov = sample["mov"]  # (1,K,H,W)
            ref = sample["ref"]  # (1,D,H,W)

            if mov.dim() != 4:
                raise ValueError(f"sample['mov'] should be (1,K,H,W), got {tuple(mov.shape)}")
            if ref.dim() != 4:
                raise ValueError(f"sample['ref'] should be (1,D,H,W), got {tuple(ref.shape)}")

            _, K, H, W = mov.shape
            _, D, Hr, Wr = ref.shape

            if H != Hr or W != Wr:
                raise ValueError(
                    f"Moving/ref XY mismatch at idx={idx}: "
                    f"mov={tuple(mov.shape)}, ref={tuple(ref.shape)}"
                )

            key = (int(K), int(D), int(H), int(W))
            self.groups.setdefault(key, []).append(idx)

        if self.verbose:
            print("[SameShapeBatchSampler] groups:")
            for key, indices in sorted(self.groups.items()):
                print(f"  key=(K,D,H,W)={key}, n={len(indices)}")

    def _build_batches(self):
        rng = random.Random(self.seed + self.epoch)

        global_batches = []

        for key, indices in self.groups.items():
            idx = list(indices)

            if self.shuffle:
                rng.shuffle(idx)

            for i in range(0, len(idx), self.batch_size):
                batch = idx[i:i + self.batch_size]

                if len(batch) < self.batch_size and self.drop_last:
                    continue

                global_batches.append(batch)

        if self.shuffle:
            rng.shuffle(global_batches)

        self.global_batches = global_batches

        self.batches = shard_batches_for_ddp(
            global_batches,
            rank=self.rank,
            world_size=self.world_size,
            pad_to_equal=self.pad_to_equal_batches,
        )

    def __iter__(self):
        self._build_batches()
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)