import torch
import pandas as pd
from torch.cuda.amp import autocast
from .train import move_batch_to_device, build_reliable_control_mask, build_gt_in_bounds_mask

@torch.no_grad()
def diagnose_matching_and_coord(
    outputs,
    gt_coords,
    valid_mask=None,
    sigma=(0.5, 1.0, 1.0),
    inside_threshold=4.0,
    topk=5,
    pmax_threshold=0.6,
):
    """
    Diagnose why coord loss is hard to reduce.

    outputs should contain:
        pred_coords:             (B,K,Hc,Wc,3), raw coord, zyx
        scores:                  (B,N,M)
        prob:                    (B,N,M), optional
        candidate_coords_feat:   (B,N,M,3), feature coord, zyx
        coord_scale:             (3,), raw / feature scale, zyx
        candidate_valid_mask:    (B,N,M), optional

    gt_coords:
        (B,K,Hc,Wc,3), raw coord, zyx

    valid_mask:
        (B,K,Hc,Wc)
    """

    device = outputs["scores"].device

    scores = outputs["scores"].float()
    candidate_coords_feat = outputs["candidate_coords_feat"].float()
    pred_coords = outputs["pred_coords"].float()
    coord_scale = outputs["coord_scale"].to(device).float()

    gt_coords = gt_coords.to(device).float()

    B, K, Hc, Wc, _ = gt_coords.shape
    N = K * Hc * Wc
    M = scores.shape[-1]

    pred_coords_flat = pred_coords.reshape(B, N, 3)
    gt_coords_flat = gt_coords.reshape(B, N, 3)

    # raw GT -> feature GT
    gt_feat = gt_coords / coord_scale.view(1, 1, 1, 1, 3)
    gt_feat = gt_feat.reshape(B, N, 1, 3)

    sigma_t = torch.tensor(
        sigma,
        device=device,
        dtype=torch.float32,
    ).view(1, 1, 1, 3)

    # distance from each candidate to GT, in normalized feature space
    dist2 = (((candidate_coords_feat - gt_feat) / sigma_t) ** 2).sum(dim=-1)

    candidate_valid = outputs.get("candidate_valid_mask", None)
    if candidate_valid is None:
        candidate_valid = torch.ones_like(scores, dtype=torch.bool)
    else:
        candidate_valid = candidate_valid.to(device).bool()

    large = torch.finfo(dist2.dtype).max / 100.0
    dist2_valid = dist2.masked_fill(~candidate_valid, large)

    target_idx = dist2_valid.argmin(dim=-1)       # nearest candidate to GT
    min_dist2 = dist2_valid.min(dim=-1).values
    inside_mask = min_dist2 < inside_threshold

    if valid_mask is not None:
        valid = valid_mask.to(device).bool().reshape(B, N)
        eval_mask = inside_mask & valid
    else:
        valid = torch.ones((B, N), device=device, dtype=torch.bool)
        eval_mask = inside_mask

    scores_for_eval = scores.masked_fill(~candidate_valid, -1e4)

    pred_idx = scores_for_eval.argmax(dim=-1)
    k = min(topk, M)
    pred_topk = scores_for_eval.topk(k=k, dim=-1).indices

    top1_correct = pred_idx == target_idx
    topk_correct = (pred_topk == target_idx.unsqueeze(-1)).any(dim=-1)

    prob = outputs.get("prob", None)
    if prob is None:
        prob = torch.softmax(scores_for_eval, dim=-1)
    else:
        prob = prob.to(device).float()

    pmax = prob.max(dim=-1).values
    entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=-1)

    # ------------------------------------------------------------
    # 1) soft predicted coord error
    # 注意：这里用 sum(abs zyx)，和你现在 coord_l1_loss 的定义一致
    # ------------------------------------------------------------
    soft_abs_axis = (pred_coords_flat - gt_coords_flat).abs()  # (B,N,3)
    soft_l1 = soft_abs_axis.sum(dim=-1)

    # ------------------------------------------------------------
    # 2) argmax candidate coord error
    # 如果 argmax candidate error 明显小于 soft pred coord error，
    # 说明 soft expectation / 多峰分布在拖累 coord。
    # ------------------------------------------------------------
    gather_idx = pred_idx.view(B, N, 1, 1).expand(B, N, 1, 3)
    argmax_coord_feat = candidate_coords_feat.gather(dim=2, index=gather_idx).squeeze(2)
    argmax_coord_raw = argmax_coord_feat * coord_scale.view(1, 1, 3)

    argmax_abs_axis = (argmax_coord_raw - gt_coords_flat).abs()
    argmax_l1 = argmax_abs_axis.sum(dim=-1)

    # ------------------------------------------------------------
    # 3) target candidate coord error
    # 这是 candidate grid 本身能达到的理论下限之一。
    # 如果 target candidate error 都不低，说明候选离 GT 本身就有量化误差。
    # ------------------------------------------------------------
    target_gather_idx = target_idx.view(B, N, 1, 1).expand(B, N, 1, 3)
    target_coord_feat = candidate_coords_feat.gather(dim=2, index=target_gather_idx).squeeze(2)
    target_coord_raw = target_coord_feat * coord_scale.view(1, 1, 3)

    target_abs_axis = (target_coord_raw - gt_coords_flat).abs()
    target_l1 = target_abs_axis.sum(dim=-1)

    def masked_mean(x, mask):
        mask = mask.to(x.device).bool()
        if mask.sum() == 0:
            return torch.tensor(float("nan"), device=x.device)
        return x[mask].float().mean()

    def masked_ratio(mask, base):
        base = base.bool()
        if base.sum() == 0:
            return torch.tensor(float("nan"), device=device)
        return (mask.bool() & base).float().sum() / base.float().sum()

    def axis_mean(abs_axis, mask):
        mask = mask.bool()
        if mask.sum() == 0:
            return [float("nan"), float("nan"), float("nan")]
        v = abs_axis[mask].mean(dim=0)
        return [v[0].item(), v[1].item(), v[2].item()]

    top1_mask = eval_mask & top1_correct
    topk_not_top1_mask = eval_mask & (~top1_correct) & topk_correct
    topk_wrong_mask = eval_mask & (~topk_correct)

    high_conf_mask = eval_mask & (pmax >= pmax_threshold)
    low_conf_mask = eval_mask & (pmax < pmax_threshold)

    result = {}

    # basic ratios
    result["num_eval_points"] = int(eval_mask.sum().item())
    result["valid_ratio_all"] = valid.float().mean().item()
    result["inside_ratio_valid"] = masked_ratio(inside_mask, valid).item()
    result["top1"] = masked_ratio(top1_correct, eval_mask).item()
    result[f"top{k}"] = masked_ratio(topk_correct, eval_mask).item()

    # probability sharpness
    result["pmax_mean"] = masked_mean(pmax, eval_mask).item()
    result["entropy_mean"] = masked_mean(entropy, eval_mask).item()
    result["pmax_top1_correct"] = masked_mean(pmax, top1_mask).item()
    result["pmax_top1_wrong"] = masked_mean(pmax, eval_mask & (~top1_correct)).item()
    result["entropy_top1_correct"] = masked_mean(entropy, top1_mask).item()
    result["entropy_top1_wrong"] = masked_mean(entropy, eval_mask & (~top1_correct)).item()

    # coord error by groups
    result["soft_l1_all"] = masked_mean(soft_l1, eval_mask).item()
    result["soft_l1_top1_correct"] = masked_mean(soft_l1, top1_mask).item()
    result["soft_l1_topk_not_top1"] = masked_mean(soft_l1, topk_not_top1_mask).item()
    result["soft_l1_topk_wrong"] = masked_mean(soft_l1, topk_wrong_mask).item()

    result["argmax_l1_all"] = masked_mean(argmax_l1, eval_mask).item()
    result["argmax_l1_top1_correct"] = masked_mean(argmax_l1, top1_mask).item()
    result["argmax_l1_topk_not_top1"] = masked_mean(argmax_l1, topk_not_top1_mask).item()
    result["argmax_l1_topk_wrong"] = masked_mean(argmax_l1, topk_wrong_mask).item()

    result["target_candidate_l1_all"] = masked_mean(target_l1, eval_mask).item()
    result["target_candidate_l1_top1_correct"] = masked_mean(target_l1, top1_mask).item()

    # z/y/x axis error
    result["soft_axis_l1_all_zyx"] = axis_mean(soft_abs_axis, eval_mask)
    result["soft_axis_l1_top1_correct_zyx"] = axis_mean(soft_abs_axis, top1_mask)
    result["argmax_axis_l1_all_zyx"] = axis_mean(argmax_abs_axis, eval_mask)
    result["target_axis_l1_all_zyx"] = axis_mean(target_abs_axis, eval_mask)

    # confidence groups
    result["soft_l1_high_conf"] = masked_mean(soft_l1, high_conf_mask).item()
    result["soft_l1_low_conf"] = masked_mean(soft_l1, low_conf_mask).item()
    result["high_conf_ratio"] = masked_ratio(pmax >= pmax_threshold, eval_mask).item()

    # candidate geometry
    result["min_dist2_mean"] = masked_mean(min_dist2, eval_mask).item()
    result["min_dist2_top1_correct"] = masked_mean(min_dist2, top1_mask).item()
    result["min_dist2_top1_wrong"] = masked_mean(min_dist2, eval_mask & (~top1_correct)).item()
    result["candidate_valid_ratio"] = candidate_valid.float().mean().item()

    return result

@torch.no_grad()
def run_coord_diagnosis(
    model,
    loader,
    device,
    max_batches=20,
    use_amp=True,
    sigma=(0.5, 1.0, 1.0),
    inside_threshold=4.0,
):
    model.eval()

    rows = []

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break

        batch = move_batch_to_device(batch, device)

        mov = batch["mov"]
        ref = batch["ref"]
        gt_coords = batch["gt_coords"]
        z_init = batch.get("z_init", None)
        spacing = batch.get("spacing", None)

        if z_init is None:
            z_init = batch.get("sparse_z_idx", None)

        valid_mask = batch.get("valid_mask", None)

        if valid_mask is None:
            valid_mask = build_reliable_control_mask(
                mov=mov,
                ref=ref,
                z_init=z_init,
                control_stride=model.control_stride,
                intensity_quantile=0.40,
                grad_quantile=0.40,
                smooth_kernel=5,
                use_ref_check=True,
            )

        # 和你 validate_one_epoch 保持一致：排除 GT 出界点
        gt_in_bounds = build_gt_in_bounds_mask(gt_coords, ref)
        valid_mask = valid_mask.bool() & gt_in_bounds.bool()

        with autocast(enabled=use_amp):
            outputs = model(
                mov,
                ref,
                z_init=z_init,
                spacing=spacing,
                return_match_aux=True,
                compute_chunk_match_loss=False,
            )

        diag = diagnose_matching_and_coord(
            outputs=outputs,
            gt_coords=gt_coords,
            valid_mask=valid_mask,
            sigma=sigma,
            inside_threshold=inside_threshold,
            topk=5,
            pmax_threshold=0.6,
        )

        diag["batch_id"] = bi
        rows.append(diag)

    df = pd.DataFrame(rows)

    # 只对数值列求均值
    mean_row = {}
    for c in df.columns:
        if c == "batch_id":
            continue

        if isinstance(df[c].iloc[0], list):
            continue

        mean_row[c] = df[c].mean()

    print("\n===== Diagnosis mean over batches =====")
    for k, v in mean_row.items():
        print(f"{k:35s}: {v:.6f}")

    print("\n===== Axis errors, mean over batches =====")
    for c in [
        "soft_axis_l1_all_zyx",
        "soft_axis_l1_top1_correct_zyx",
        "argmax_axis_l1_all_zyx",
        "target_axis_l1_all_zyx",
    ]:
        if c in df.columns:
            arr = torch.tensor(df[c].tolist(), dtype=torch.float32)
            m = arr.nanmean(dim=0)
            print(f"{c:35s}: z={m[0]:.4f}, y={m[1]:.4f}, x={m[2]:.4f}")

    return df