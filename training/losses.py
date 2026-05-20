# training/losses.py

import torch
import torch.nn.functional as F


def disp_l1_loss(pred_disp, gt_disp, valid_mask=None):
    """
    pred_disp:  (B,K,H,W,3), order = (z,y,x), raw voxel displacement
    gt_disp:    (B,K,H,W,3), order = (z,y,x), raw voxel displacement
    valid_mask: (B,K,H,W), optional
    """
    loss = (pred_disp - gt_disp).abs().sum(dim=-1)

    if valid_mask is not None:
        valid_mask = valid_mask.to(loss.device).float()
        return (loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    return loss.mean()


def coord_l1_loss(pred_coords, gt_coords, valid_mask=None):
    """
    pred_coords: (B,K,H,W,3), order = (z,y,x), raw reference voxel coords
    gt_coords:   (B,K,H,W,3), order = (z,y,x), raw reference voxel coords
    valid_mask:  (B,K,H,W), optional
    """
    loss = (pred_coords - gt_coords).abs().sum(dim=-1)

    if valid_mask is not None:
        valid_mask = valid_mask.to(loss.device).float()
        return (loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    return loss.mean()


def smoothness_loss(pred_disp):
    """
    pred_disp: (B,K,H,W,3), order = (z,y,x), raw voxel displacement
    """
    loss = pred_disp.sum() * 0.0

    if pred_disp.shape[1] > 1:
        loss = loss + (
            pred_disp[:, 1:, :, :, :] - pred_disp[:, :-1, :, :, :]
        ).abs().mean()

    if pred_disp.shape[2] > 1:
        loss = loss + (
            pred_disp[:, :, 1:, :, :] - pred_disp[:, :, :-1, :, :]
        ).abs().mean()

    if pred_disp.shape[3] > 1:
        loss = loss + (
            pred_disp[:, :, :, 1:, :] - pred_disp[:, :, :, :-1, :]
        ).abs().mean()

    return loss



def disp_magnitude_loss(pred_disp, gt_disp, valid_mask=None, eps=1e-6):
    """
    Penalize displacement magnitude mismatch.

    pred_disp:  (B,K,H,W,3), order = (z,y,x)
    gt_disp:    (B,K,H,W,3), order = (z,y,x)
    valid_mask: (B,K,H,W), optional
    """
    pred_mag = torch.linalg.norm(pred_disp, dim=-1)
    gt_mag = torch.linalg.norm(gt_disp, dim=-1)

    loss = (pred_mag - gt_mag).abs()

    if valid_mask is not None:
        valid_mask = valid_mask.to(loss.device).float()
        return (loss * valid_mask).sum() / (valid_mask.sum() + eps)

    return loss.mean()



def z_spacing_consistency_loss_from_coords(pred_coords, spacing):
    """
    pred_coords: (B,K,H,W,3), order = (z,y,x), raw reference voxel coords
    spacing:     (B,6)
                 [sx_ref, sy_ref, sz_ref, sx_mov, sy_mov, sz_mov]
    """
    B, K, H, W, _ = pred_coords.shape

    if K <= 1:
        return pred_coords.sum() * 0.0

    spacing = spacing.to(pred_coords.device).float()

    ref_sz = spacing[:, 2]
    mov_sz = spacing[:, 5]

    expected_step = mov_sz / (ref_sz + 1e-8)  # (B,)

    z_pred = pred_coords[..., 0]  # (B,K,H,W)

    z_step = z_pred[:, 1:, :, :] - z_pred[:, :-1, :, :]

    expected_step = expected_step[:, None, None, None]

    return (z_step - expected_step).abs().mean()


def z_spacing_consistency_loss_from_disp(pred_disp, z_init, spacing):
    """
    Fallback version.

    pred_disp: (B,K,H,W,3), order = (z,y,x), raw voxel displacement
    z_init:    (B,K) or (K,), float or int initial z position in raw ref voxel coords
    spacing:   (B,6)
    """
    B, K, H, W, _ = pred_disp.shape

    if K <= 1:
        return pred_disp.sum() * 0.0

    if z_init is None:
        raise ValueError("z_init is required when computing z-spacing loss from pred_disp.")

    if z_init.dim() == 1:
        z_init = z_init.unsqueeze(0).repeat(B, 1)

    z_init = z_init.to(pred_disp.device).float()
    spacing = spacing.to(pred_disp.device).float()

    ref_sz = spacing[:, 2]
    mov_sz = spacing[:, 5]

    expected_step = mov_sz / (ref_sz + 1e-8)

    z_base = z_init[:, :, None, None]
    z_pred = z_base + pred_disp[..., 0]

    z_step = z_pred[:, 1:, :, :] - z_pred[:, :-1, :, :]

    expected_step = expected_step[:, None, None, None]

    return (z_step - expected_step).abs().mean()

def get_candidate_valid_mask(outputs, scores):
    """
    Get candidate-level valid mask.

    Returns:
        candidate_valid:
            (B,N,M), bool.
            True means this candidate is inside reference feature volume.
    """
    candidate_valid = outputs.get("candidate_valid_mask", None)

    if candidate_valid is None:
        candidate_valid = torch.ones_like(scores, dtype=torch.bool)
    else:
        candidate_valid = candidate_valid.to(
            device=scores.device,
            dtype=torch.bool,
        )

    return candidate_valid

def local_match_kl_loss(
    outputs,
    gt_coords,
    valid_mask=None,
    sigma=(0.75, 1.5, 1.5),
    inside_threshold=4.0,
):
    """
    Supervise local matching scores using a Gaussian soft target.

    outputs["scores"]:
        (B,N,M), local matching scores.

    outputs["candidate_coords_feat"]:
        (B,N,M,3), candidate coordinates in feature space, order=(z,y,x).

    outputs["coord_scale"]:
        (3,), scale from feature coords to raw coords, order=(z,y,x).

    gt_coords:
        (B,K,H,W,3), raw reference coordinates, order=(z,y,x).

    inside_threshold:
        threshold on normalized squared distance. Points whose GT is too far
        from all local candidates are ignored.
    """
    if outputs.get("scores", None) is None:
        raise ValueError("local_match_kl_loss requires outputs['scores'].")

    if outputs.get("candidate_coords_feat", None) is None:
        raise ValueError("local_match_kl_loss requires outputs['candidate_coords_feat'].")

    if outputs.get("coord_scale", None) is None:
        raise ValueError("local_match_kl_loss requires outputs['coord_scale'].")

    scores = outputs["scores"]                          # (B,N,M)
    candidate_coords = outputs["candidate_coords_feat"]  # (B,N,M,3)
    coord_scale = outputs["coord_scale"].to(scores.device).float()  # (3,)

    B, K, H, W, _ = gt_coords.shape
    N = K * H * W

    gt_coords = gt_coords.to(scores.device).float()

    # raw coords -> feature coords
    gt_feat = gt_coords / coord_scale.view(1, 1, 1, 1, 3)
    gt_feat = gt_feat.view(B, N, 1, 3)

    sigma = torch.tensor(
        sigma,
        device=scores.device,
        dtype=scores.dtype,
    ).view(1, 1, 1, 3)

    # normalized distance to every candidate
    dist2 = (((candidate_coords - gt_feat) / sigma) ** 2).sum(dim=-1)  # (B,N,M)

    # ----------------------------------------------------
    # Candidate-level valid mask
    # Invalid candidates are out-of-bound candidates before clamp.
    # They should not be selected as targets and should not receive soft probability.
    # ----------------------------------------------------
    candidate_valid = get_candidate_valid_mask(outputs, scores)  # (B,N,M)

    large = torch.finfo(dist2.dtype).max / 100.0
    dist2_valid = dist2.masked_fill(~candidate_valid, large)

    min_dist2 = dist2_valid.min(dim=-1).values  # (B,N)
    inside_mask = min_dist2 < inside_threshold

    # Gaussian soft target over valid candidates only
    target_prob = torch.exp(-0.5 * dist2_valid)
    target_prob = target_prob.masked_fill(~candidate_valid, 0.0)
    target_prob = target_prob / (target_prob.sum(dim=-1, keepdim=True) + 1e-8)
    target_prob = target_prob.detach()

    # Use scores after model-side masking.
    # If the model already did masked_fill(-1e4), this is redundant but safe.
    scores_for_loss = scores.masked_fill(~candidate_valid, -1e4)
    log_prob = F.log_softmax(scores_for_loss, dim=-1)

    loss_per_query = F.kl_div(
        log_prob,
        target_prob,
        reduction="none",
    ).sum(dim=-1)  # (B,N)

    if valid_mask is not None:
        valid = valid_mask.view(B, N).to(scores.device).bool()
        inside_mask = inside_mask & valid

    denom = inside_mask.float().sum() + 1e-6
    loss = (loss_per_query * inside_mask.float()).sum() / denom

    with torch.no_grad():
        target_idx = dist2_valid.argmin(dim=-1)    # (B,N)
        pred_idx = scores_for_loss.argmax(dim=-1)  # (B,N)

        top1 = ((pred_idx == target_idx) & inside_mask).float().sum() / denom

        topk = min(5, scores.shape[-1])
        pred_topk = scores.topk(k=topk, dim=-1).indices
        top5 = (
            (pred_topk == target_idx.unsqueeze(-1)).any(dim=-1)
            & inside_mask
        ).float().sum() / denom

        prob = torch.softmax(scores_for_loss, dim=-1)
        entropy = -(prob * (prob + 1e-8).log()).sum(dim=-1)

        if inside_mask.any():
            prob_max = prob.max(dim=-1).values[inside_mask].mean()
            entropy_mean = entropy[inside_mask].mean()
            min_dist_mean = min_dist2[inside_mask].mean()
        else:
            prob_max = scores.new_tensor(0.0)
            entropy_mean = scores.new_tensor(0.0)
            min_dist_mean = scores.new_tensor(0.0)

        inside_metrics = compute_inside_metrics(outputs, gt_coords, valid_mask)

    metrics = {
        "loss_match_kl": loss.detach(),
        "match_top1": top1.detach(),
        "match_top5": top5.detach(),
        "match_prob_max": prob_max.detach(),
        "match_entropy": entropy_mean.detach(),
        "match_inside_ratio": inside_metrics["inside_valid"],
        "match_inside_valid": inside_metrics["inside_valid"],
        "match_inside_all": inside_metrics["inside_all"],
        "match_valid_and_inside_all": inside_metrics["valid_and_inside_all"],
        "match_min_dist2": min_dist_mean.detach(),
        "candidate_valid_ratio": candidate_valid.float().mean().detach(),
    }


    return loss, metrics

def local_match_ce_loss(
    outputs,
    gt_coords,
    valid_mask=None,
    sigma=(0.75, 1.5, 1.5),
    inside_threshold=4.0,
):
    """
    Hard classification loss for local matching.

    For each query, find the candidate closest to GT coord,
    then use cross entropy to force that candidate to have highest score.
    """
    if outputs.get("scores", None) is None:
        raise ValueError("local_match_ce_loss requires outputs['scores'].")

    if outputs.get("candidate_coords_feat", None) is None:
        raise ValueError("local_match_ce_loss requires outputs['candidate_coords_feat'].")

    if outputs.get("coord_scale", None) is None:
        raise ValueError("local_match_ce_loss requires outputs['coord_scale'].")

    scores = outputs["scores"]                          # (B,N,M)
    candidate_coords = outputs["candidate_coords_feat"]  # (B,N,M,3)
    coord_scale = outputs["coord_scale"].to(scores.device).float()

    B, K, H, W, _ = gt_coords.shape
    N = K * H * W

    gt_coords = gt_coords.to(scores.device).float()

    # raw coords -> feature coords
    gt_feat = gt_coords / coord_scale.view(1, 1, 1, 1, 3)
    gt_feat = gt_feat.view(B, N, 1, 3)

    sigma = torch.tensor(
        sigma,
        device=scores.device,
        dtype=scores.dtype,
    ).view(1, 1, 1, 3)

    # normalized distance to each candidate
    dist2 = (((candidate_coords - gt_feat) / sigma) ** 2).sum(dim=-1)  # (B,N,M)

    # ----------------------------------------------------
    # Candidate-level valid mask
    # ----------------------------------------------------
    candidate_valid = get_candidate_valid_mask(outputs, scores)  # (B,N,M)

    large = torch.finfo(dist2.dtype).max / 100.0
    dist2_valid = dist2.masked_fill(~candidate_valid, large)

    target_idx = dist2_valid.argmin(dim=-1)       # (B,N)
    min_dist2 = dist2_valid.min(dim=-1).values    # (B,N)

    inside_mask = min_dist2 < inside_threshold

    scores_for_loss = scores.masked_fill(~candidate_valid, -1e4)
    if valid_mask is not None:
        valid = valid_mask.view(B, N).to(scores.device).bool()
        inside_mask = inside_mask & valid

    # CE per query
    ce = F.cross_entropy(
        scores_for_loss.reshape(B * N, -1),
        target_idx.reshape(B * N),
        reduction="none",
    ).reshape(B, N)

    denom = inside_mask.float().sum() + 1e-6
    loss = (ce * inside_mask.float()).sum() / denom

    with torch.no_grad():
        pred_idx = scores_for_loss.argmax(dim=-1)
        top1 = ((pred_idx == target_idx) & inside_mask).float().sum() / denom

        topk = min(5, scores.shape[-1])
        pred_topk = scores.topk(k=topk, dim=-1).indices        
        top5 = (
            (pred_topk == target_idx.unsqueeze(-1)).any(dim=-1)
            & inside_mask
        ).float().sum() / denom

        prob = torch.softmax(scores, dim=-1)
        entropy = -(prob * (prob + 1e-8).log()).sum(dim=-1)

        if inside_mask.any():
            prob_max = prob.max(dim=-1).values[inside_mask].mean()
            entropy_mean = entropy[inside_mask].mean()
            min_dist_mean = min_dist2[inside_mask].mean()
        else:
            prob_max = scores.new_tensor(0.0)
            entropy_mean = scores.new_tensor(0.0)
            min_dist_mean = scores.new_tensor(0.0)

        inside_metrics = compute_inside_metrics(outputs, gt_coords, valid_mask)

    metrics = {
        "loss_match_ce": loss.detach(),
        "match_top1": top1.detach(),
        "match_top5": top5.detach(),
        "match_prob_max": prob_max.detach(),
        "match_entropy": entropy_mean.detach(),
        "match_inside_ratio": inside_metrics["inside_valid"],
        "match_inside_valid": inside_metrics["inside_valid"],
        "match_inside_all": inside_metrics["inside_all"],
        "match_valid_and_inside_all": inside_metrics["valid_and_inside_all"],
        "match_min_dist2": min_dist_mean.detach(),
        "candidate_valid_ratio": candidate_valid.float().mean().detach(),
    }

    return loss, metrics


def compute_inside_metrics(outputs, gt_coords, valid_mask=None, eps=1e-6):
    """
    Compute whether GT coordinates are inside candidate coordinates.

    Args:
        outputs:
            model outputs. Must contain:
                candidate_coords_feat: (B, N, M, 3)
                coord_scale: (3,)

        gt_coords:
            (B, K, Hc, Wc, 3), raw coordinates, order=(z,y,x)

        valid_mask:
            (B, K, Hc, Wc), bool or float

    Returns:
        metrics dict:
            inside_all:
                fraction of all control points whose GT lies inside candidate region.

            inside_valid:
                fraction among valid control points only.

            valid_and_inside_all:
                fraction of all points that are both valid and inside.
    """
    candidate_coords = outputs["candidate_coords_feat"]  # (B,N,M,3), feature coords
    coord_scale = outputs["coord_scale"].to(
        device=candidate_coords.device,
        dtype=candidate_coords.dtype,
    )

    B, N, M, _ = candidate_coords.shape

    gt_coords = gt_coords.to(
        device=candidate_coords.device,
        dtype=candidate_coords.dtype,
    )

    # raw coords -> feature coords
    gt_feat = gt_coords.clone()
    gt_feat[..., 0] = gt_feat[..., 0] / coord_scale[0]
    gt_feat[..., 1] = gt_feat[..., 1] / coord_scale[1]
    gt_feat[..., 2] = gt_feat[..., 2] / coord_scale[2]

    gt_feat = gt_feat.reshape(B, N, 3)

    # Because candidate_coords are already clamped to valid feature volume,
    # this checks both radius coverage and boundary validity.
    candidate_valid = outputs.get("candidate_valid_mask", None)

    if candidate_valid is not None:
        candidate_valid = candidate_valid.to(
            device=candidate_coords.device,
            dtype=torch.bool,
        )

        cand_for_min = candidate_coords.masked_fill(
            ~candidate_valid[..., None],
            float("inf"),
        )
        cand_for_max = candidate_coords.masked_fill(
            ~candidate_valid[..., None],
            float("-inf"),
        )

        cand_min = cand_for_min.min(dim=2).values
        cand_max = cand_for_max.max(dim=2).values

        has_valid_candidate = candidate_valid.any(dim=2)  # (B,N)
    else:
        cand_min = candidate_coords.min(dim=2).values
        cand_max = candidate_coords.max(dim=2).values
        has_valid_candidate = torch.ones(
            (B, N),
            device=candidate_coords.device,
            dtype=torch.bool,
        )

    inside = (
        (gt_feat[..., 0] >= cand_min[..., 0] - eps) &
        (gt_feat[..., 0] <= cand_max[..., 0] + eps) &
        (gt_feat[..., 1] >= cand_min[..., 1] - eps) &
        (gt_feat[..., 1] <= cand_max[..., 1] + eps) &
        (gt_feat[..., 2] >= cand_min[..., 2] - eps) &
        (gt_feat[..., 2] <= cand_max[..., 2] + eps) &
        has_valid_candidate
        )  # (B,N)

    inside_all = inside.float().mean()

    if valid_mask is not None:
        valid = valid_mask.to(device=inside.device).bool().reshape(B, N)

        valid_count = valid.float().sum().clamp_min(1.0)

        inside_valid = ((inside & valid).float().sum() / valid_count)
        valid_and_inside_all = (inside & valid).float().mean()
        valid_ratio = valid.float().mean()
    else:
        inside_valid = inside_all
        valid_and_inside_all = inside_all
        valid_ratio = torch.ones_like(inside_all)

    if candidate_valid is not None:
        candidate_valid_ratio = candidate_valid.float().mean()
    else:
        candidate_valid_ratio = torch.ones_like(inside_all)

    return {
        "inside_all": inside_all,
        "inside_valid": inside_valid,
        "valid_and_inside_all": valid_and_inside_all,
        "valid_ratio": valid_ratio,
        "candidate_valid_ratio": candidate_valid_ratio,
    }

def total_coarse_loss(
    outputs,
    gt_disp=None,
    gt_coords=None,
    z_init=None,
    sparse_z_idx=None,
    spacing=None,
    valid_mask=None,
    lambda_match=1.0,
    lambda_match_kl=0.5,
    lambda_match_ce=1.0,
    lambda_coord=0.05,
    lambda_disp=0.0,

    # new
    lambda_disp_mag=0.0,

    lambda_smooth=0.005,
    lambda_z_spacing=0.005,

    use_coord_loss=True,
    loss_mode="match",
    match_sigma=(0.5, 1.0, 1.0),
    match_inside_threshold=4.0,
):
    """
    Coarse matching loss.

    Supports two matching-loss paths:

    1. Chunked path:
        outputs contains:
            loss_match_kl_chunked
            loss_match_ce_chunked

        This is the recommended training path.
        It does NOT require full outputs["scores"] / outputs["candidate_coords_feat"].

    2. Full-aux fallback path:
        outputs contains:
            scores
            candidate_coords_feat
            candidate_valid_mask
            coord_scale

        This is useful for debugging / inference diagnostics, but costs much more memory.
    """
    pred_disp = outputs["pred_disp"]
    pred_coords = outputs.get("pred_coords", None)

    if z_init is None:
        z_init = sparse_z_idx

    total = pred_disp.sum() * 0.0
    loss_dict = {}

    # ======================================================
    # 1. Matching loss: prefer chunked CE/KL if available
    # ======================================================
    if loss_mode == "match" or lambda_match > 0:
        if gt_coords is None:
            raise ValueError("match loss requires gt_coords.")

        has_chunked_match_loss = (
            outputs.get("loss_match_kl_chunked", None) is not None
            and outputs.get("loss_match_ce_chunked", None) is not None
        )

        if has_chunked_match_loss:
            # --------------------------------------------------
            # Recommended path:
            # CE/KL already computed inside model chunk-by-chunk.
            # This avoids storing full scores/prob/candidates.
            # --------------------------------------------------
            loss_match_kl = outputs["loss_match_kl_chunked"]
            loss_match_ce = outputs["loss_match_ce_chunked"]

            # Use chunked metrics if model returned them.
            zero_metric = loss_match_ce.detach() * 0.0

            match_metrics = {
                "loss_match_kl": loss_match_kl.detach(),
                "loss_match_ce": loss_match_ce.detach(),

                "match_top1": outputs.get(
                    "match_top1_chunked",
                    zero_metric,
                ).detach(),

                "match_top5": outputs.get(
                    "match_top5_chunked",
                    zero_metric,
                ).detach(),

                "match_prob_max": outputs.get(
                    "match_prob_max_chunked",
                    zero_metric,
                ).detach(),

                "match_entropy": outputs.get(
                    "match_entropy_chunked",
                    zero_metric,
                ).detach(),

                "match_inside_ratio": outputs.get(
                    "match_inside_valid_chunked",
                    zero_metric,
                ).detach(),

                "match_inside_valid": outputs.get(
                    "match_inside_valid_chunked",
                    zero_metric,
                ).detach(),

                "match_inside_all": outputs.get(
                    "match_inside_all_chunked",
                    zero_metric,
                ).detach(),

                "match_valid_and_inside_all": outputs.get(
                    "match_valid_and_inside_all_chunked",
                    zero_metric,
                ).detach(),

                "candidate_valid_ratio": outputs.get(
                    "candidate_valid_ratio_chunked",
                    zero_metric,
                ).detach(),

                # This is not always computed in chunked mode.
                "match_min_dist2": outputs.get(
                    "match_min_dist2_chunked",
                    zero_metric,
                ).detach(),
            }

        else:
            # --------------------------------------------------
            # Fallback path:
            # full scores/candidates must exist.
            # This is memory-heavy, so do NOT use it in normal training.
            # --------------------------------------------------
            loss_match_kl, kl_metrics = local_match_kl_loss(
                outputs=outputs,
                gt_coords=gt_coords,
                valid_mask=valid_mask,
                sigma=match_sigma,
                inside_threshold=match_inside_threshold,
            )

            loss_match_ce, ce_metrics = local_match_ce_loss(
                outputs=outputs,
                gt_coords=gt_coords,
                valid_mask=valid_mask,
                sigma=match_sigma,
                inside_threshold=match_inside_threshold,
            )

            match_metrics = {}
            match_metrics.update(kl_metrics)
            match_metrics.update(ce_metrics)

        loss_match = (
            lambda_match_kl * loss_match_kl
            + lambda_match_ce * loss_match_ce
        )

        total = total + lambda_match * loss_match

        with torch.no_grad():
            if gt_disp is not None:
                pred_mag_map = torch.linalg.norm(pred_disp, dim=-1)
                gt_mag_map = torch.linalg.norm(gt_disp.to(pred_disp.device).float(), dim=-1)

                if valid_mask is not None:
                    valid = valid_mask.to(pred_disp.device).float()
                    denom = valid.sum() + 1e-6
                    pred_mag_mean = (pred_mag_map * valid).sum() / denom
                    gt_mag_mean = (gt_mag_map * valid).sum() / denom
                else:
                    pred_mag_mean = pred_mag_map.mean()
                    gt_mag_mean = gt_mag_map.mean()

                pred_gt_mag_ratio = pred_mag_mean / (gt_mag_mean + 1e-6)
            else:
                pred_mag_mean = pred_disp.sum() * 0.0
                gt_mag_mean = pred_disp.sum() * 0.0
                pred_gt_mag_ratio = pred_disp.sum() * 0.0


        loss_dict.update(match_metrics)
        loss_dict["loss_match"] = loss_match.detach()
        loss_dict["loss_match_kl"] = loss_match_kl.detach()
        loss_dict["loss_match_ce"] = loss_match_ce.detach()

    else:
        loss_match = pred_disp.sum() * 0.0
        loss_dict["loss_match"] = loss_match.detach()
        loss_dict["loss_match_kl"] = loss_match.detach()
        loss_dict["loss_match_ce"] = loss_match.detach()

    # ======================================================
    # 2. Coordinate loss
    # ======================================================
    if use_coord_loss and pred_coords is not None and gt_coords is not None:
        loss_coord = coord_l1_loss(pred_coords, gt_coords, valid_mask)
    else:
        loss_coord = pred_disp.sum() * 0.0

    if loss_mode == "coord":
        total = total + loss_coord
    else:
        total = total + lambda_coord * loss_coord

    # ======================================================
    # 3. Displacement vector loss
    # ======================================================
    if gt_disp is not None:
        loss_disp = disp_l1_loss(pred_disp, gt_disp, valid_mask)
    else:
        loss_disp = pred_disp.sum() * 0.0

    if loss_mode == "disp":
        total = total + loss_disp
    else:
        total = total + lambda_disp * loss_disp


    # ======================================================
    # 3b. Displacement magnitude losses
    #     These terms prevent the conservative shortcut:
    #         pred_mag << gt_mag
    # ======================================================
    if gt_disp is not None:
        loss_disp_mag = disp_magnitude_loss(
            pred_disp=pred_disp,
            gt_disp=gt_disp,
            valid_mask=valid_mask,
        )

    else:
        loss_disp_mag = pred_disp.sum() * 0.0
        loss_disp_under_mag = pred_disp.sum() * 0.0

    total = total + lambda_disp_mag * loss_disp_mag
    # ======================================================
    # 4. Smoothness loss
    # ======================================================
    loss_smooth = smoothness_loss(pred_disp)
    total = total + lambda_smooth * loss_smooth

    # ======================================================
    # 5. Z-spacing consistency
    # ======================================================
    if spacing is None:
        loss_z_spacing = pred_disp.sum() * 0.0
    else:
        if pred_coords is not None:
            loss_z_spacing = z_spacing_consistency_loss_from_coords(
                pred_coords=pred_coords,
                spacing=spacing,
            )
        else:
            loss_z_spacing = z_spacing_consistency_loss_from_disp(
                pred_disp=pred_disp,
                z_init=z_init,
                spacing=spacing,
            )

    total = total + lambda_z_spacing * loss_z_spacing

    # ======================================================
    # 6. Logging
    # ======================================================
    if loss_mode == "match":
        loss_main = loss_match
    elif loss_mode == "coord":
        loss_main = loss_coord
    elif loss_mode == "disp":
        loss_main = loss_disp
    else:
        loss_main = total

    loss_dict.update(
        {
            "loss_total": total.detach(),
            "loss_main": loss_main.detach(),
            "loss_coord": loss_coord.detach(),
            "loss_disp": loss_disp.detach(),

            # new
            "loss_disp_mag": loss_disp_mag.detach(),
            "pred_mag_mean": pred_mag_mean.detach(),
            "gt_mag_mean": gt_mag_mean.detach(),
            "pred_gt_mag_ratio": pred_gt_mag_ratio.detach(),

            "loss_smooth": loss_smooth.detach(),
            "loss_z_spacing": loss_z_spacing.detach(),
        }
    )
    return total, loss_dict