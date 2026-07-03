import math
import numpy as np
import torch
import torch.nn.functional as F


def _to_tensor_zyx(x, device="cuda", dtype=torch.float32):
    """
    Convert numpy array to torch tensor.

    Expected input:
        x: (Z,H,W) or (K,H,W)
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _make_control_grid_xy(
    H,
    W,
    control_stride=16,
    device="cuda",
    dtype=torch.float32,
):
    """
    Make raw-coordinate control grid.

    Returns:
        yy: (Hc,Wc)
        xx: (Hc,Wc)
    """
    ys = torch.arange(0, H, control_stride, device=device, dtype=dtype)
    xs = torch.arange(0, W, control_stride, device=device, dtype=dtype)

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return yy, xx


def _extract_patch_grid_sample_2d(
    img_2d,
    center_yx,
    patch_size=8,
    padding_mode="border",
):
    """
    Extract many 2D patches from one image using grid_sample.

    img_2d:
        (H,W)

    center_yx:
        (...,2), raw coordinates, order = y,x

    return:
        patches:
            (..., patch_size, patch_size)
    """
    device = img_2d.device
    dtype = img_2d.dtype

    H, W = img_2d.shape
    center_y = center_yx[..., 0]
    center_x = center_yx[..., 1]

    # Local offsets centered around 0.
    # For patch_size=8, offsets are [-3.5, ..., 3.5].
    half = (patch_size - 1) / 2.0
    offs = torch.arange(patch_size, device=device, dtype=dtype) - half

    dy, dx = torch.meshgrid(offs, offs, indexing="ij")  # (P,P)

    y = center_y[..., None, None] + dy
    x = center_x[..., None, None] + dx

    x_norm = 2.0 * x / max(W - 1, 1) - 1.0
    y_norm = 2.0 * y / max(H - 1, 1) - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1)  # (...,P,P,2)

    flat_shape = center_y.shape
    n = int(np.prod(flat_shape))

    grid_flat = grid.reshape(n, patch_size, patch_size, 2)

    img = img_2d[None, None]  # (1,1,H,W)
    img = img.expand(n, 1, H, W)

    patches = F.grid_sample(
        img,
        grid_flat,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )

    patches = patches[:, 0].reshape(*flat_shape, patch_size, patch_size)

    return patches


def _zncc_score(mov_patch, ref_patches, eps=1e-6):
    """
    Compute ZNCC between one/many moving patches and reference candidate patches.

    mov_patch:
        (..., P, P)

    ref_patches:
        (..., M, P, P)

    return:
        scores:
            (..., M)
    """
    mov = mov_patch
    ref = ref_patches

    mov_mean = mov.mean(dim=(-2, -1), keepdim=True)
    mov_c = mov - mov_mean

    ref_mean = ref.mean(dim=(-2, -1), keepdim=True)
    ref_c = ref - ref_mean

    numerator = (mov_c.unsqueeze(-3) * ref_c).mean(dim=(-2, -1))

    mov_std = torch.sqrt((mov_c ** 2).mean(dim=(-2, -1)) + eps)
    ref_std = torch.sqrt((ref_c ** 2).mean(dim=(-2, -1)) + eps)

    scores = numerator / (mov_std.unsqueeze(-1) * ref_std + eps)

    return scores.clamp(-1.0, 1.0)


def _cosine_score(mov_patch, ref_patches, eps=1e-6):
    """
    Compute cosine similarity between moving patches and reference candidate patches.

    Unlike ZNCC, cosine similarity does NOT subtract the mean, so it is not
    invariant to brightness offset. However it is still scale-invariant and can
    serve as a lightweight alternative.

    mov_patch:
        (..., P, P)

    ref_patches:
        (..., M, P, P)

    return:
        scores:
            (..., M), range roughly [-1, 1]
    """
    mov = mov_patch
    ref = ref_patches

    # Flatten spatial dims
    mov_flat = mov.reshape(*mov.shape[:-2], -1)   # (..., P*P)
    ref_flat = ref.reshape(*ref.shape[:-2], -1)   # (..., M, P*P)

    mov_norm = torch.sqrt((mov_flat ** 2).sum(dim=-1) + eps)        # (...)
    ref_norm = torch.sqrt((ref_flat ** 2).sum(dim=-1) + eps)        # (..., M)

    numerator = (mov_flat.unsqueeze(-2) * ref_flat).sum(dim=-1)     # (..., M)

    scores = numerator / (mov_norm.unsqueeze(-1) * ref_norm + eps)

    return scores.clamp(-1.0, 1.0)


def _ssd_score(mov_patch, ref_patches):
    """
    Negative mean squared error as similarity score (higher = better match).

    Unlike ZNCC/cosine, SSD is a difference-based metric — simpler and more
    direct, but sensitive to intensity offset and scale.

    mov_patch:
        (..., P, P)

    ref_patches:
        (..., M, P, P)

    return:
        scores:
            (..., M), higher is better (negative MSE)
    """
    mov = mov_patch
    ref = ref_patches

    diff_sq = (mov.unsqueeze(-3) - ref) ** 2   # (..., M, P, P)
    mse = diff_sq.mean(dim=(-2, -1))            # (..., M)
    return -mse  # negate so higher = better


def _make_local_candidate_offsets(
    radius=(4, 3, 3),
    xy_step=8,
    z_step=1,
    device="cuda",
    dtype=torch.float32,
):
    """
    Make local candidate offsets.

    radius:
        (rz, ry, rx)

    xy_step:
        raw-pixel step for xy offsets.
        For /8 feature baseline, use xy_step=8.

    z_step:
        reference z-index step.

    return:
        offsets_zyx: (M,3), order z,y,x
    """
    rz, ry, rx = radius

    dz = torch.arange(-rz, rz + 1, device=device, dtype=dtype) * float(z_step)
    dy = torch.arange(-ry, ry + 1, device=device, dtype=dtype) * float(xy_step)
    dx = torch.arange(-rx, rx + 1, device=device, dtype=dtype) * float(xy_step)

    zz, yy, xx = torch.meshgrid(dz, dy, dx, indexing="ij")

    offsets = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)
    return offsets


@torch.no_grad()
def ncc_local_coarse_match_single(
    mov_zyx,
    ref_zyx,
    z_init,
    gt_coords_zyx=None,
    valid_mask=None,
    radius=(4, 3, 3),
    patch_size=8,
    control_stride=16,
    xy_step=8,
    z_step=1,
    temperature=0.05,
    topk=(1, 3, 5),
    device="cuda",
    normalize_input=True,
    chunk_queries=2048,
    return_scores=False,
    match_mode="argmax",
    similarity="zncc",
    topk_for_expectation=50,
):
    """
    Raw-pixel local NCC coarse matcher.

    Inputs
    ------
    mov_zyx:
        moving sparse stack, shape (K,H,W)

    ref_zyx:
        reference dense stack, shape (Z,H,W)

    z_init:
        initial z position for each moving slice, shape (K,)

    gt_coords_zyx:
        optional GT coordinates on the control grid,
        shape (K,Hc,Wc,3), order z,y,x.
        If provided, compute CE/top-k.

    valid_mask:
        optional valid mask on control grid,
        shape (K,Hc,Wc).

    match_mode:
        "argmax"  — hard argmax over candidates
        "expectation" — softmax-weighted average of all candidates
        "topk_expectation" — softmax over top-k scores only, then expectation

    similarity:
        "zncc" or "cosine"

    topk_for_expectation:
        number of top scores to keep when match_mode="topk_expectation"

    Outputs
    -------
    out:
        dict with:
            pred_coords_zyx: (K,Hc,Wc,3)
            prob: optional
            scores: optional
            candidate_coords_zyx: optional
            ce_loss
            topk metrics
    """
    device = torch.device(device)

    mov = _to_tensor_zyx(mov_zyx, device=device)
    ref = _to_tensor_zyx(ref_zyx, device=device)

    if mov.ndim != 3:
        raise ValueError(f"mov_zyx should be (K,H,W), got {tuple(mov.shape)}")
    if ref.ndim != 3:
        raise ValueError(f"ref_zyx should be (Z,H,W), got {tuple(ref.shape)}")

    K, H, W = mov.shape
    Z, Hr, Wr = ref.shape

    if (H, W) != (Hr, Wr):
        raise ValueError(f"XY mismatch: mov={(H,W)}, ref={(Hr,Wr)}")

    z_init_t = torch.as_tensor(
        z_init,
        dtype=torch.float32,
        device=device,
    ).reshape(-1)

    if z_init_t.numel() != K:
        raise ValueError(f"z_init length {z_init_t.numel()} != K={K}")

    # Optional robust normalization for the NCC baseline.
    # Since NCC itself is local normalized, this is less critical,
    # but it helps with extreme outliers and dtype stability.
    if normalize_input:
        mov = _robust_norm_torch(mov)
        ref = _robust_norm_torch(ref)

    yy, xx = _make_control_grid_xy(
        H=H,
        W=W,
        control_stride=control_stride,
        device=device,
        dtype=torch.float32,
    )

    Hc, Wc = yy.shape
    N_per_slice = Hc * Wc
    N = K * Hc * Wc

    # Query centers: (K,Hc,Wc,3), z,y,x
    base_coords = torch.zeros((K, Hc, Wc, 3), device=device, dtype=torch.float32)
    base_coords[..., 0] = z_init_t[:, None, None]
    base_coords[..., 1] = yy[None]
    base_coords[..., 2] = xx[None]

    base_flat = base_coords.reshape(N, 3)

    offsets = _make_local_candidate_offsets(
        radius=radius,
        xy_step=xy_step,
        z_step=z_step,
        device=device,
        dtype=torch.float32,
    )  # (M,3)

    M = offsets.shape[0]

    pred_coords_flat = torch.empty((N, 3), device=device, dtype=torch.float32)

    all_scores = [] if return_scores else None
    all_candidate_coords = [] if return_scores else None

    if gt_coords_zyx is not None:
        gt = torch.as_tensor(gt_coords_zyx, device=device, dtype=torch.float32)
        if tuple(gt.shape) != (K, Hc, Wc, 3):
            raise ValueError(
                f"gt_coords_zyx should be {(K,Hc,Wc,3)}, got {tuple(gt.shape)}"
            )
        gt_flat = gt.reshape(N, 3)
    else:
        gt_flat = None

    if valid_mask is not None:
        valid = torch.as_tensor(valid_mask, device=device).bool()
        if tuple(valid.shape) != (K, Hc, Wc):
            raise ValueError(
                f"valid_mask should be {(K,Hc,Wc)}, got {tuple(valid.shape)}"
            )
        valid_flat = valid.reshape(N)
    else:
        valid_flat = torch.ones((N,), device=device, dtype=torch.bool)

    ce_losses = []
    topk_hits = {int(k): [] for k in topk}
    inside_hits = []
    target_dist_mins = []
    target_inside_hits = []
    # Flatten query identity: slice k and xy center
    k_ids = torch.arange(K, device=device).view(K, 1, 1).expand(K, Hc, Wc).reshape(N)
    y_centers = base_flat[:, 1]
    x_centers = base_flat[:, 2]

    for q0 in range(0, N, chunk_queries):
        q1 = min(q0 + chunk_queries, N)
        Q = q1 - q0

        base_q = base_flat[q0:q1]              # (Q,3)
        k_q = k_ids[q0:q1]                     # (Q,)

        # Candidate coords: (Q,M,3), z,y,x
        cand = base_q[:, None, :] + offsets[None, :, :]

        z = cand[..., 0].round().long()
        y = cand[..., 1]
        x = cand[..., 2]

        valid_cand = (
            (z >= 0) & (z < Z) &
            (y >= 0) & (y <= H - 1) &
            (x >= 0) & (x <= W - 1)
        )

        z_clamped = z.clamp(0, Z - 1)

        # Moving patches: one patch per query.
        mov_centers_yx = torch.stack(
            [y_centers[q0:q1], x_centers[q0:q1]],
            dim=-1,
        )  # (Q,2)

        mov_patches = []
        for kk in range(K):
            mask_k = (k_q == kk)
            if mask_k.any():
                patches_k = _extract_patch_grid_sample_2d(
                    mov[kk],
                    mov_centers_yx[mask_k],
                    patch_size=patch_size,
                )
                mov_patches.append((mask_k, patches_k))

        mov_patch_q = torch.empty(
            (Q, patch_size, patch_size),
            device=device,
            dtype=torch.float32,
        )

        for mask_k, patches_k in mov_patches:
            mov_patch_q[mask_k] = patches_k

        # Reference candidate patches.
        ref_patches = torch.empty(
            (Q, M, patch_size, patch_size),
            device=device,
            dtype=torch.float32,
        )

        # Group by z for efficiency.
        for zz in torch.unique(z_clamped):
            mask_z = (z_clamped == zz)
            if mask_z.any():
                centers_yx = torch.stack(
                    [y[mask_z], x[mask_z]],
                    dim=-1,
                )
                patches_z = _extract_patch_grid_sample_2d(
                    ref[int(zz.item())],
                    centers_yx,
                    patch_size=patch_size,
                )
                ref_patches[mask_z] = patches_z

        if similarity == "zncc":
            scores = _zncc_score(
                mov_patch=mov_patch_q,
                ref_patches=ref_patches,
            )  # (Q,M)
        elif similarity == "cosine":
            scores = _cosine_score(
                mov_patch=mov_patch_q,
                ref_patches=ref_patches,
            )  # (Q,M)
        elif similarity == "ssd":
            scores = _ssd_score(
                mov_patch=mov_patch_q,
                ref_patches=ref_patches,
            )  # (Q,M)
        else:
            raise ValueError(f"Unknown similarity: {similarity!r}")

        scores = scores.masked_fill(~valid_cand, -1e4)

        # ------------------------------------------------------------
        # Matching: argmax / expectation / topk_expectation
        # ------------------------------------------------------------
        logits = scores / float(temperature)

        if match_mode == "argmax":
            best_idx = torch.argmax(logits, dim=-1)  # (Q,)
            pred = cand[
                torch.arange(Q, device=device),
                best_idx,
            ]  # (Q, 3), order = z,y,x
        elif match_mode == "expectation":
            prob = torch.softmax(logits, dim=-1)
            pred = torch.sum(prob[..., None] * cand, dim=1)  # (Q, 3)
        elif match_mode == "topk_expectation":
            prob = torch.softmax(logits, dim=-1)
            # Keep only top-K scores, zero out the rest, renormalize
            topk_vals, topk_idx = torch.topk(logits, k=min(topk_for_expectation, M), dim=-1)
            mask = torch.zeros_like(prob)
            mask.scatter_(-1, topk_idx, 1.0)
            prob = prob * mask
            prob = prob / (prob.sum(dim=-1, keepdim=True) + 1e-12)
            pred = torch.sum(prob[..., None] * cand, dim=1)  # (Q, 3)
        else:
            raise ValueError(f"Unknown match_mode: {match_mode!r}")

        pred_coords_flat[q0:q1] = pred

        if return_scores:
            all_scores.append(scores.detach().cpu())
            all_candidate_coords.append(cand.detach().cpu())

        if gt_flat is not None:
            gt_q = gt_flat[q0:q1]  # (Q,3)

            # nearest candidate to GT as hard target
            dist = torch.linalg.norm(cand - gt_q[:, None, :], dim=-1)
            dist = dist.masked_fill(~valid_cand, float("inf"))

            target_idx = torch.argmin(dist, dim=-1)  # (Q,)
            inside = torch.isfinite(dist[torch.arange(Q, device=device), target_idx])

            target_dist_min = dist[torch.arange(Q, device=device), target_idx]

            # 这个 threshold 只是用来诊断 GT 是否足够接近 candidate set
            inside_threshold = 8.0
            target_inside = target_dist_min <= inside_threshold

            # query_valid 表示这个 query 是否参与评估
            # 第一版可以只要求 valid_mask 且存在有效 candidate
            query_valid = valid_flat[q0:q1] & inside

            if query_valid.any():
                ce = F.cross_entropy(
                    logits[query_valid],
                    target_idx[query_valid],
                    reduction="none",
                )
                ce_losses.append(ce.detach())

                rank = torch.argsort(logits, dim=-1, descending=True)

                for kk in topk_hits:
                    hit = (rank[:, :kk] == target_idx[:, None]).any(dim=1)
                    topk_hits[kk].append(hit[query_valid].float().detach())

                inside_hits.append(inside[valid_flat[q0:q1]].float().detach())

                target_dist_mins.append(target_dist_min[query_valid].detach())
                target_inside_hits.append(target_inside[query_valid].float().detach())
    pred_coords = pred_coords_flat.reshape(K, Hc, Wc, 3)

    out = {
        "pred_coords_zyx": pred_coords.detach().cpu().numpy(),
        "base_coords_zyx": base_coords.detach().cpu().numpy(),
        "radius": radius,
        "patch_size": patch_size,
        "control_stride": control_stride,
        "xy_step": xy_step,
        "z_step": z_step,
        "temperature": temperature,
    }

    if gt_flat is not None:
        if len(ce_losses) > 0:
            ce_all = torch.cat(ce_losses)
            out["ce_loss"] = float(ce_all.mean().cpu())
        else:
            out["ce_loss"] = np.nan

        for kk in topk_hits:
            if len(topk_hits[kk]) > 0:
                hits = torch.cat(topk_hits[kk])
                out[f"top{kk}"] = float(hits.mean().cpu())
            else:
                out[f"top{kk}"] = np.nan

        if len(inside_hits) > 0:
            inside_all = torch.cat(inside_hits)
            out["inside_valid"] = float(inside_all.mean().cpu())
        else:
            out["inside_valid"] = np.nan

    if return_scores:
        out["scores"] = torch.cat(all_scores, dim=0).numpy().reshape(K, Hc, Wc, M)
        out["candidate_coords_zyx"] = (
            torch.cat(all_candidate_coords, dim=0)
            .numpy()
            .reshape(K, Hc, Wc, M, 3)
        )
    if len(target_dist_mins) > 0:
        target_dist_all = torch.cat(target_dist_mins)
        out["target_dist_min_mean"] = float(target_dist_all.mean().cpu())
        out["target_dist_min_p95"] = float(torch.quantile(target_dist_all, 0.95).cpu())
    else:
        out["target_dist_min_mean"] = np.nan
        out["target_dist_min_p95"] = np.nan

    if len(target_inside_hits) > 0:
        target_inside_all = torch.cat(target_inside_hits)
        out["gt_inside_threshold"] = float(target_inside_all.mean().cpu())
    else:
        out["gt_inside_threshold"] = np.nan
    return out


def _robust_norm_torch(x, p_low=0.001, p_high=0.999, eps=1e-6):
    """
    Robust percentile normalization to [0,1].
    """
    vals = x[torch.isfinite(x)]
    vals_nz = vals[vals != 0]

    if vals_nz.numel() > 100:
        vals = vals_nz

    if vals.numel() == 0:
        return torch.zeros_like(x)

    lo = torch.quantile(vals, p_low)
    hi = torch.quantile(vals, p_high)

    if (hi - lo).abs() < eps:
        return torch.zeros_like(x)

    return torch.clamp((x - lo) / (hi - lo + eps), 0.0, 1.0)

def evaluate_ncc_baseline_on_batch(
    batch,
    radius=(4, 3, 3),
    patch_size=8,
    control_stride=16,
    xy_step=8,
    z_step=1,
    temperature=0.05,
    topk=(1, 3, 5),
    device="cuda",
    max_samples=None,
    match_mode="argmax",
    similarity="zncc",
    topk_for_expectation=50,
):
    """
    Evaluate raw-pixel NCC coarse matcher on one simulation batch.

    Returns:
        list of dict rows.
    """
    rows = []

    mov_b = batch["mov"]  # (B,1,K,H,W)
    ref_b = batch["ref"]  # (B,1,Z,H,W)

    if "z_init" in batch:
        z_init_b = batch["z_init"]
    elif "sparse_z_idx" in batch:
        z_init_b = batch["sparse_z_idx"]
    else:
        raise KeyError("batch must contain 'z_init' or 'sparse_z_idx'.")

    gt_b = batch.get("gt_coords", None)
    if gt_b is None:
        gt_b = batch.get("gt_coords_ctrl", None)

    valid_b = batch.get("valid_mask", None)
    if valid_b is None:
        valid_b = batch.get("valid", None)

    B = mov_b.shape[0]

    if max_samples is not None:
        B = min(B, max_samples)

    for b in range(B):
        mov_zyx = mov_b[b, 0].detach().cpu().numpy().astype(np.float32)
        ref_zyx = ref_b[b, 0].detach().cpu().numpy().astype(np.float32)
        z_init = z_init_b[b].detach().cpu().numpy().astype(np.float32)

        gt_coords = None
        if gt_b is not None:
            gt_coords = gt_b[b].detach().cpu().numpy().astype(np.float32)

        valid_mask = None
        if valid_b is not None:
            valid_mask = valid_b[b].detach().cpu().numpy().astype(bool)

        out = ncc_local_coarse_match_single(
            mov_zyx=mov_zyx,
            ref_zyx=ref_zyx,
            z_init=z_init,
            gt_coords_zyx=gt_coords,
            valid_mask=valid_mask,
            radius=radius,
            patch_size=patch_size,
            control_stride=control_stride,
            xy_step=xy_step,
            z_step=z_step,
            temperature=temperature,
            topk=topk,
            device=device,
            normalize_input=True,
            chunk_queries=1024,
            return_scores=False,
            match_mode=match_mode,
            similarity=similarity,
            topk_for_expectation=topk_for_expectation,
        )

        row = {
            "sample": b,
            "ce_loss": out.get("ce_loss", np.nan),
            "inside_valid": out.get("inside_valid", np.nan),

            # debug metrics
            "target_dist_min_mean": out.get("target_dist_min_mean", np.nan),
            "target_dist_min_p95": out.get("target_dist_min_p95", np.nan),
            "gt_inside_threshold": out.get("gt_inside_threshold", np.nan),
        }

        for k in topk:
            row[f"top{k}"] = out.get(f"top{k}", np.nan)

        rows.append(row)

    return rows

import pandas as pd

def evaluate_ncc_baseline_on_loader(
    loader,
    dataset_name="val",
    max_batches=None,
    device="cuda",
    radius=(4, 3, 3),
    patch_size=8,
    control_stride=16,
    xy_step=8,
    z_step=1,
    temperature=0.05,
    topk=(1, 3, 5),
    match_mode="argmax",
    similarity="zncc",
    topk_for_expectation=50,
):
    rows = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch_rows = evaluate_ncc_baseline_on_batch(
            batch=batch,
            radius=radius,
            patch_size=patch_size,
            control_stride=control_stride,
            xy_step=xy_step,
            z_step=z_step,
            temperature=temperature,
            topk=topk,
            device=device,
            match_mode=match_mode,
            similarity=similarity,
            topk_for_expectation=topk_for_expectation,
        )

        for r in batch_rows:
            r["dataset"] = dataset_name
            r["batch_idx"] = batch_idx
            rows.append(r)

        if batch_idx % 5 == 0:
            df_tmp = pd.DataFrame(rows)
            print(
                f"[{dataset_name}] batch={batch_idx} | "
                f"CE={df_tmp['ce_loss'].mean():.4f} | "
                f"top1={df_tmp['top1'].mean():.4f} | "
                f"top5={df_tmp['top5'].mean():.4f}"
            )

    return pd.DataFrame(rows)