# training/train.py

import os
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from CoarseFlow.models.SparseGMFlow3D import (
    CoarseMatchingNetV3,
    CoarseMatchingNetV4,
    CoarseMatchingNetV5,
)
from CoarseFlow.models.SparseGMFlow3D_v2 import CoarseMatchingNetV6,count_parameters_by_module,count_parameters
from CoarseFlow.training.losses import total_coarse_loss
import logging
import sys
from scipy.interpolate import RegularGridInterpolator

def setup_logger(save_dir, filename="train.log", mode="a"):
    """
    Create a logger that writes both to terminal and to a log file.

    Args:
        save_dir: checkpoint/log directory
        filename: log filename
        mode: "a" for append, "w" for overwrite
    """
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, filename)

    logger = logging.getLogger(f"coarse_matching_train_{os.path.abspath(save_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Important for notebook / repeated runs:
    # avoid duplicated log lines
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"[Logger] log file: {log_path}")
    return logger

def move_batch_to_device(batch, device):
    out = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            v = v.to(device, non_blocking=True)

            if k in [
                "mov",
                "ref",
                "spacing",
                "gt_disp",
                "gt_coords",
                "valid_mask",
                "z_init",
                "sparse_z_idx",
            ]:
                v = v.float()

            out[k] = v
        else:
            out[k] = v

    return out

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    scaler=None,
    use_amp=True,
    grad_clip=1.0,
    log_interval=10,
    loss_mode="match",
    lambda_match=1.0,
    lambda_match_kl=0.5,
    lambda_match_ce=1.0,
    lambda_coord=0.05,
    lambda_disp=0.0,
    lambda_smooth=0.005,
    lambda_z_spacing=0.005,
    lambda_disp_mag=0.0,
    freeze_encoders=False,
    compute_chunk_match_loss=True,
    match_sigma=(0.5, 0.75, 0.75),
    match_inside_threshold=4.0,
    logger=None,
):
    model.train()
    if freeze_encoders:
        keep_frozen_encoders_eval(model)
    running_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        # Expected:
        # mov: (B,1,K,H,W)
        # ref: (B,1,D,H,W)
        mov = batch["mov"]
        ref = batch["ref"]
        spacing = batch.get("spacing", None)

        gt_disp = batch.get("gt_disp", None)
        gt_coords = batch.get("gt_coords", None)
        sparse_z_idx = batch.get("sparse_z_idx", None)
        z_init = batch.get("z_init", None)

        if z_init is None:
            z_init = sparse_z_idx

        if z_init is None:
            raise ValueError(
                "z_init or sparse_z_idx must be provided. "
                "The new model no longer falls back to torch.arange(K)."
            )

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
        if gt_coords is not None:
            gt_in_bounds = build_gt_in_bounds_mask(gt_coords, ref)

            if valid_mask is None:
                valid_mask = gt_in_bounds
            else:
                valid_mask = valid_mask.bool() & gt_in_bounds.bool()
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            need_chunk_match_loss = (
                compute_chunk_match_loss
                and (loss_mode == "match" or lambda_match > 0)
                and (lambda_match_kl > 0 or lambda_match_ce > 0)
            )

            outputs = model(
                mov,
                ref,
                z_init=z_init,
                spacing=spacing,

                return_match_aux=False,
                compute_chunk_match_loss=need_chunk_match_loss,
                gt_coords=gt_coords,
                valid_mask=valid_mask,
                match_sigma=match_sigma,
                match_inside_threshold=match_inside_threshold,
            )
            loss, loss_dict = total_coarse_loss(
                outputs=outputs,
                gt_disp=gt_disp,
                gt_coords=gt_coords,
                sparse_z_idx=sparse_z_idx,
                spacing=spacing,
                valid_mask=valid_mask,
                loss_mode=loss_mode,
                lambda_match=lambda_match,
                lambda_coord=lambda_coord,
                lambda_disp=lambda_disp,
                lambda_smooth=lambda_smooth,
                lambda_match_kl=lambda_match_kl,
                lambda_match_ce=lambda_match_ce,
                lambda_z_spacing=lambda_z_spacing,
                lambda_disp_mag=lambda_disp_mag,
            )

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        running_loss += loss.item()

        if step % log_interval == 0:
            valid_ratio = valid_mask.float().mean().item() if valid_mask is not None else 1.0
            msg = (
                f"[Epoch {epoch:03d}] "
                f"step {step:04d}/{len(loader)} | "
                f"loss={loss.item():.4f} | "
                f"match={loss_dict.get('loss_match', torch.tensor(0.0)).item():.4f} | "
                f"kl={loss_dict.get('loss_match_kl', torch.tensor(0.0)).item():.4f} | "
                f"ce={loss_dict.get('loss_match_ce', torch.tensor(0.0)).item():.4f} | "
                f"coord={loss_dict['loss_coord'].item():.4f} | "
                f"disp={loss_dict['loss_disp'].item():.4f} | "
                f"smooth={loss_dict['loss_smooth'].item():.4f} | "
                f"z_spacing={loss_dict['loss_z_spacing'].item():.4f} | "
                f"top1={loss_dict.get('match_top1', torch.tensor(0.0)).item():.3f} | "
                f"top5={loss_dict.get('match_top5', torch.tensor(0.0)).item():.3f} | "
                f"pmax={loss_dict.get('match_prob_max', torch.tensor(0.0)).item():.3f} | "
                f"ent={loss_dict.get('match_entropy', torch.tensor(0.0)).item():.3f} | "
                f"inside_valid={loss_dict.get('match_inside_valid', torch.tensor(0.0, device=device)).item():.3f} | "
                f"cand_valid={loss_dict.get('candidate_valid_ratio', torch.tensor(0.0, device=device)).item():.3f} | "
                f"inside_all={loss_dict.get('match_inside_all', torch.tensor(0.0, device=device)).item():.3f} | "
                f"valid_inside_all={loss_dict.get('match_valid_and_inside_all', torch.tensor(0.0, device=device)).item():.3f} | "
                f"time={time.time() - t0:.1f}s | "
                f"valid={valid_ratio:.3f} | "
            )
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)

    return running_loss / max(len(loader), 1)

def build_gt_in_bounds_mask(gt_coords, ref, eps=1e-6):
    """
    Build mask for GT coordinates that are inside the reference volume.

    Args:
        gt_coords:
            (B, K, Hc, Wc, 3), order=(z,y,x), raw coordinates.

        ref:
            (B, 1, D, H, W)

    Returns:
        in_bounds:
            (B, K, Hc, Wc), bool
    """
    if gt_coords is None:
        raise ValueError("gt_coords is required to build GT in-bounds mask.")

    _, _, D, H, W = ref.shape

    in_bounds = (
        (gt_coords[..., 0] >= -eps) & (gt_coords[..., 0] <= (D - 1 + eps)) &
        (gt_coords[..., 1] >= -eps) & (gt_coords[..., 1] <= (H - 1 + eps)) &
        (gt_coords[..., 2] >= -eps) & (gt_coords[..., 2] <= (W - 1 + eps))
    )

    return in_bounds

@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    device,
    epoch,
    use_amp=True,
    loss_mode="match",
    lambda_match=1.0,
    lambda_coord=0.05,
    lambda_disp=0.0,
    lambda_match_kl=0.5,
    lambda_match_ce=1.0,
    lambda_smooth=0.005,
    lambda_z_spacing=0.005,
    lambda_disp_mag=0.0,
    compute_chunk_match_loss=True,
    match_sigma=(0.5, 0.75, 0.75),
    match_inside_threshold=4.0,
    logger=None,
):
    model.eval()

    metric_sums = {}
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        mov = batch["mov"]
        ref = batch["ref"]
        spacing = batch.get("spacing", None)

        gt_disp = batch.get("gt_disp", None)
        gt_coords = batch.get("gt_coords", None)
        z_init = batch.get("z_init", None)

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

        # ----------------------------------------------------
        # Exclude GT coordinates outside the reference volume
        # ----------------------------------------------------
        if gt_coords is not None:
            gt_in_bounds = build_gt_in_bounds_mask(gt_coords, ref)

            if valid_mask is None:
                valid_mask = gt_in_bounds
            else:
                valid_mask = valid_mask.bool() & gt_in_bounds.bool()

        with autocast(enabled=use_amp):
            need_chunk_match_loss = (
                compute_chunk_match_loss
                and (loss_mode == "match" or lambda_match > 0)
                and (lambda_match_kl > 0 or lambda_match_ce > 0)
            )

            outputs = model(
                mov,
                ref,
                z_init=z_init,
                spacing=spacing,

                return_match_aux=False,
                compute_chunk_match_loss=need_chunk_match_loss,
                gt_coords=gt_coords,
                valid_mask=valid_mask,
                match_sigma=match_sigma,
                match_inside_threshold=match_inside_threshold,
            )

            loss, loss_dict = total_coarse_loss(
                outputs=outputs,
                gt_disp=gt_disp,
                gt_coords=gt_coords,
                z_init=z_init,
                spacing=spacing,
                valid_mask=valid_mask,
                loss_mode=loss_mode,
                lambda_match=lambda_match,
                lambda_coord=lambda_coord,
                lambda_disp=lambda_disp,
                lambda_match_kl=lambda_match_kl,
                lambda_match_ce=lambda_match_ce,
                lambda_smooth=lambda_smooth,
                lambda_z_spacing=lambda_z_spacing,
                lambda_disp_mag=lambda_disp_mag,
            )

        # -------------------------
        # collect metrics
        # -------------------------
        batch_metrics = {
            "loss_total": loss.detach(),
            "valid_ratio": valid_mask.float().mean().detach(),
        }

        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                batch_metrics[k] = v.detach()

        # optional: amplitude diagnostics
        with torch.no_grad():
            if gt_disp is not None and outputs.get("pred_disp", None) is not None:
                pred_disp = outputs["pred_disp"]

                pred_mag = torch.linalg.norm(pred_disp, dim=-1)
                gt_mag = torch.linalg.norm(gt_disp, dim=-1)

                batch_metrics["pred_disp_mag"] = pred_mag.mean()
                batch_metrics["gt_disp_mag"] = gt_mag.mean()

                if valid_mask is not None:
                    valid = valid_mask.float()
                    denom = valid.sum() + 1e-6

                    batch_metrics["pred_disp_mag_valid"] = (
                        pred_mag * valid
                    ).sum() / denom

                    batch_metrics["gt_disp_mag_valid"] = (
                        gt_mag * valid
                    ).sum() / denom

                    disp_err = torch.abs(pred_disp - gt_disp).mean(dim=-1)

                    batch_metrics["disp_mae_valid_mean_component"] = (
                        disp_err * valid
                    ).sum() / denom

        for k, v in batch_metrics.items():
            if torch.is_tensor(v):
                v = v.float().mean().item()
            else:
                v = float(v)

            metric_sums[k] = metric_sums.get(k, 0.0) + v

        num_batches += 1

    avg_metrics = {
        k: v / max(num_batches, 1)
        for k, v in metric_sums.items()
    }

    msg = (
        f"[Val Epoch {epoch:03d}] "
        f"loss={avg_metrics.get('loss_total', 0.0):.4f} | "
        f"match={avg_metrics.get('loss_match', 0.0):.4f} | "
        f"kl={avg_metrics.get('loss_match_kl', 0.0):.4f} | "
        f"ce={avg_metrics.get('loss_match_ce', 0.0):.4f} | "
        f"coord={avg_metrics.get('loss_coord', 0.0):.4f} | "
        f"disp={avg_metrics.get('loss_disp', 0.0):.4f} | "
        f"smooth={avg_metrics.get('loss_smooth', 0.0):.4f} | "
        f"z_spacing={avg_metrics.get('loss_z_spacing', 0.0):.4f} | "
        f"top1={avg_metrics.get('match_top1', 0.0):.3f} | "
        f"top5={avg_metrics.get('match_top5', 0.0):.3f} | "
        f"pmax={avg_metrics.get('match_prob_max', 0.0):.3f} | "
        f"ent={avg_metrics.get('match_entropy', 0.0):.3f} | "
        f"inside_valid={avg_metrics.get('match_inside_valid', 0.0):.3f} | "
        f"cand_valid={avg_metrics.get('candidate_valid_ratio', 0.0):.3f} | "
        f"inside_all={avg_metrics.get('match_inside_all', 0.0):.3f} | "
        f"valid_inside_all={avg_metrics.get('match_valid_and_inside_all', 0.0):.3f} | "
        f"valid={avg_metrics.get('valid_ratio', 0.0):.3f} | "
        f"pred_mag={avg_metrics.get('pred_disp_mag_valid', 0.0):.3f} | "
        f"gt_mag={avg_metrics.get('gt_disp_mag_valid', 0.0):.3f}"
        
    )

    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

    return avg_metrics.get("loss_total", 0.0)

def save_checkpoint(
    save_path,
    model,
    optimizer,
    epoch,
    best_val_loss=None,
    model_config=None,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "model_config": model_config,
    }

    torch.save(ckpt, save_path)

def train_coarse_matching_model(
    train_dataset,
    val_dataset=None,
    train_loader=None,
    val_loader=None,
    save_dir="./checkpoints/sparse_gmflow_3d",
    batch_size=1,
    num_workers=0,
    num_epochs=100,
    lr=1e-4,
    weight_decay=1e-4,
    use_amp=True,
    device=None,
    loss_mode="match",
    lambda_match=1.0,
    lambda_coord=0.05,
    lambda_match_kl=0.5,
    lambda_match_ce=1.0,
    lambda_disp=0.0,
    lambda_smooth=0.005,
    lambda_z_spacing=0.005,
    lambda_disp_mag=0.0,
    
    # model config
    dim=96,
    radius=(4, 3, 3),
    temperature=0.05,
    use_learned_matching=True,
    control_stride=8,
    num_refine_iters=1,
    encoder_stride=8,
    query_chunk_size=512,
    matcher_mode="hybrid",

    
    moving_base_channels=(24, 48, 96),
    moving_num_blocks=(1, 2, 1),
    moving_mlp_ratio=2.0,
    moving_window_attn_layers=1,
    moving_window_size=8,
    moving_attn_num_heads=4,
    moving_slice_fusion_blocks=1,


    ref_base_channels=(24, 48, 96),
    ref_num_blocks=(1, 2, 2),
    ref_refine_blocks=1,
    ref_mlp_ratio=2.0,
    ref_attn_layers=6,
    ref_attn_num_heads=4,
    ref_attn_window_size=(4, 8, 8),
    ref_attn_mlp_ratio=2.0,

    use_coord_embed=True,

    # residual coordinate refinement config
    use_coord_residual=False,
    residual_type="mlp",
    residual_num_blocks=3,
    residual_use_3d=True,
    residual_hidden_dim=128,
    residual_max_delta=(0.4, 0.8, 0.8),
    residual_use_disp=True,
    residual_detach_coarse=True,
    residual_detach_features=True,

    # matcher V5 config
    use_offset_encoding=True,
    use_offset_bias=True,
    use_local_cross_attn=True,
    local_attn_temperature=0.20,
    matcher_cross_attn_layers=3,
    matcher_cross_attn_heads=4,
    matcher_ffn_ratio=2.0,
    matcher_attn_drop=0.0,
    matcher_proj_drop=0.0,
    matcher_init_gamma=1e-3,

    # checkpoint
    resume_path=None,
    resume_optimizer=False,
    resume_best_val_loss=False,
    strict_load=False,

    #logging
    log_filename="train.log",
    log_mode="a",
    use_spacing_embed=True,

    compute_chunk_match_loss=True,
    match_sigma=(0.5, 0.75, 0.75),
    match_inside_threshold=4.0,

    train_only_residual=False,
    freeze_encoder=False,
    residual_lr=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    # ----------------------------------------------------
    # Build dataloaders only if they are not provided.
    # This allows same-K batch sampler / custom dataloader.
    # ----------------------------------------------------
    if train_loader is None:
        if train_dataset is None:
            raise ValueError(
                "Either train_dataset or train_loader must be provided."
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    if val_loader is None:
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )

    model_config = dict(
        dim=dim,
        radius=radius,
        temperature=temperature,
        use_learned_matching=use_learned_matching,
        matcher_mode=matcher_mode,
        control_stride=control_stride,
        num_refine_iters=num_refine_iters,
        encoder_stride=encoder_stride,
        query_chunk_size=query_chunk_size,

        # moving encoder config
        moving_base_channels=moving_base_channels,
        moving_num_blocks=moving_num_blocks,
        moving_mlp_ratio=moving_mlp_ratio,
        moving_window_attn_layers=moving_window_attn_layers,
        moving_window_size=moving_window_size,
        moving_attn_num_heads=moving_attn_num_heads,
        moving_slice_fusion_blocks=moving_slice_fusion_blocks,

        # reference encoder config
        ref_base_channels=ref_base_channels,
        ref_num_blocks=ref_num_blocks,
        ref_refine_blocks=ref_refine_blocks,
        ref_mlp_ratio=ref_mlp_ratio,
        ref_attn_layers=ref_attn_layers,
        ref_attn_num_heads=ref_attn_num_heads,
        ref_attn_window_size=ref_attn_window_size,
        ref_attn_mlp_ratio=ref_attn_mlp_ratio,

        # embeddings
        use_coord_embed=use_coord_embed,
        use_spacing_embed=use_spacing_embed,

        # matcher config
        use_offset_encoding=use_offset_encoding,
        use_offset_bias=use_offset_bias,
        use_local_cross_attn=use_local_cross_attn,
        local_attn_temperature=local_attn_temperature,
        matcher_cross_attn_layers=matcher_cross_attn_layers,
        matcher_cross_attn_heads=matcher_cross_attn_heads,
        matcher_ffn_ratio=matcher_ffn_ratio,
        matcher_attn_drop=matcher_attn_drop,
        matcher_proj_drop=matcher_proj_drop,
        matcher_init_gamma=matcher_init_gamma,

        # residual coordinate refinement
        use_coord_residual=use_coord_residual,
        residual_type=residual_type,
        residual_num_blocks=residual_num_blocks,
        residual_use_3d=residual_use_3d,
        residual_hidden_dim=residual_hidden_dim,
        residual_max_delta=residual_max_delta,
        residual_use_disp=residual_use_disp,
        residual_detach_coarse=residual_detach_coarse,
        residual_detach_features=residual_detach_features,
    )
    
    model = CoarseMatchingNetV6(**model_config).to(device)
    count_parameters(model)
    count_parameters_by_module(model, max_depth=1)
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    start_epoch = 1

    logger = setup_logger(
        save_dir=save_dir,
        filename=log_filename,
        mode=log_mode,
    )
    if resume_path is not None:
        logger.info(f"[Resume] Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)

        load_msg = model.load_state_dict(
            ckpt["model"],
            strict=strict_load,
        )

        if not strict_load:
            logger.info(msg=f"[Resume] load_state_dict message: {load_msg}")


        if resume_best_val_loss and "best_val_loss" in ckpt and ckpt["best_val_loss"] is not None:
            best_val_loss = ckpt["best_val_loss"]
        else:
            best_val_loss = float("inf")

        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1

        logger.info(f"[Resume] start_epoch = {start_epoch}")
    else:
        best_val_loss = float("inf")
    # ----------------------------------------------------
    # Freeze / trainable parameter control
    # ----------------------------------------------------
    if train_only_residual:
        if not use_coord_residual:
            raise ValueError(
                "train_only_residual=True requires use_coord_residual=True."
            )

        freeze_all_except_coord_residual(
            model,
            verbose=True,
            logger=logger,
        )

    elif freeze_encoder:
        freeze_coarseflow_encoders(
            model,
            freeze_moving_encoder=True,
            freeze_reference_encoder=True,
            verbose=False,
        )

        logger.info("[Freeze encoders] moving_encoder and reference_encoder are frozen.")

    # ----------------------------------------------------
    # Build optimizer after loading checkpoint and freezing parameters
    # ----------------------------------------------------
    if train_only_residual:
        lr_res = lr if residual_lr is None else residual_lr

        trainable_params = [
            p for p in model.parameters()
            if p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr_res,
            weight_decay=weight_decay,
        )

        logger.info(f"[Optimizer] residual-only optimizer, lr={lr_res}")

    elif residual_lr is not None and hasattr(model, "coord_residual_refiner") and model.coord_residual_refiner is not None:
        residual_param_ids = {
            id(p) for p in model.coord_residual_refiner.parameters()
            if p.requires_grad
        }

        residual_params = []
        other_params = []

        for p in model.parameters():
            if not p.requires_grad:
                continue

            if id(p) in residual_param_ids:
                residual_params.append(p)
            else:
                other_params.append(p)

        param_groups = []

        if len(other_params) > 0:
            param_groups.append(
                {
                    "params": other_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )

        if len(residual_params) > 0:
            param_groups.append(
                {
                    "params": residual_params,
                    "lr": residual_lr,
                    "weight_decay": weight_decay,
                }
            )

        optimizer = torch.optim.AdamW(param_groups)

        logger.info(
            f"[Optimizer] two lr groups: base lr={lr}, residual lr={residual_lr}"
        )

    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )

        logger.info(f"[Optimizer] normal optimizer, lr={lr}")

    if resume_path is not None and resume_optimizer:
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            logger.info("[Resume] optimizer state loaded.")
        else:
            logger.info("[Resume] checkpoint has no optimizer state; skip optimizer resume.")

    scaler = GradScaler(enabled=use_amp)
    
    end_epoch = start_epoch + num_epochs - 1

    for epoch in range(start_epoch, end_epoch + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            use_amp=use_amp,
            loss_mode=loss_mode,
            lambda_match=lambda_match,
            lambda_match_kl=lambda_match_kl,
            lambda_match_ce=lambda_match_ce,
            lambda_coord=lambda_coord,
            lambda_disp=lambda_disp,
            lambda_smooth=lambda_smooth,
            lambda_z_spacing=lambda_z_spacing,
            lambda_disp_mag=lambda_disp_mag,
            freeze_encoders = freeze_encoder,
            logger=logger,
            compute_chunk_match_loss=compute_chunk_match_loss,
            match_sigma=match_sigma,
            match_inside_threshold=match_inside_threshold,
        )

        logger.info(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}")

        save_checkpoint(
            save_path=os.path.join(save_dir, "latest.pth"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            model_config=model_config,
        )

        if val_loader is not None:
            val_loss = validate_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                epoch=epoch,
                use_amp=use_amp,
                loss_mode=loss_mode,
                lambda_match=lambda_match,
                lambda_coord=lambda_coord,
                lambda_match_kl=lambda_match_kl,
                lambda_match_ce=lambda_match_ce,
                lambda_disp=lambda_disp,
                lambda_smooth=lambda_smooth,
                lambda_z_spacing=lambda_z_spacing,
                lambda_disp_mag=lambda_disp_mag,
                logger=logger,
                compute_chunk_match_loss=compute_chunk_match_loss,
                match_sigma=match_sigma,
                match_inside_threshold=match_inside_threshold,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    save_path=os.path.join(save_dir, "best.pth"),
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    model_config=model_config,
                )

    return model

def clear_gpu_cache(*objs):
    """
    Delete given objects and release PyTorch CUDA cache.
    Use after training / validation / failed runs.
    """
    import gc
    import torch

    for obj in objs:
        try:
            del obj
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

        print("[GPU memory after cleanup]")
        print(f"allocated: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
        print(f"reserved : {torch.cuda.memory_reserved() / 1024**3:.3f} GB")

def overfit_one_sample(
    dataset,
    sample_idx=0,
    steps=2000,
    lr=1e-4,
    weight_decay=0.0,
    use_amp=True,
    device=None,
    dim=64,
    radius=(2, 4, 4),
    temperature=0.05,
    use_learned_matching=False,
    num_refine_iters=1,
    control_stride=16,
    encoder_stride=4,
    query_chunk_size=1024,
    loss_mode="match",
    lambda_match=1.0,
    lambda_coord=0.0,
    lambda_disp=0.0,
    lambda_smooth=0.0,
    lambda_z_spacing=0.0,
    lambda_disp_mag=0.0,
    grad_clip=1.0,
    log_interval=50,
    matcher_mode="hybrid",
    moving_num_convnext_blocks=3,
    ref_base_channels=(32, 64, 96),
    ref_num_blocks=(2, 4, 4),
    ref_refine_blocks=2,
    use_coord_embed=True,
    use_offset_encoding=True,
    use_offset_bias=True,
    use_local_cross_attn=True,
    local_attn_temperature=0.20,
):
    """
    Overfit a single synthetic sample.

    Recommended debug settings:
        loss_mode="match"
        lambda_match=1.0
        lambda_coord=0.0
        lambda_smooth=0.0
        lambda_z_spacing=0.0

    This checks whether the model can learn local correspondence scores
    on one fixed moving/reference pair.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    print(f"[Overfit] device = {device}")

    # -------------------------
    # 1. Load one sample
    # -------------------------
    batch = dataset[sample_idx]

    batch = {
        k: v.unsqueeze(0) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }

    batch = move_batch_to_device(batch, device)

    mov = batch["mov"]                  # (B,1,K,H,W)
    ref = batch["ref"]                  # (B,1,D,H,W)
    spacing = batch.get("spacing", None)
    gt_disp = batch.get("gt_disp", None)
    gt_coords = batch.get("gt_coords", None)
    sparse_z_idx = batch.get("sparse_z_idx", None)
    z_init = batch.get("z_init", None)
    valid_mask = batch.get("valid_mask", None)
    if valid_mask is None:
        valid_mask = build_reliable_control_mask(
            mov=mov,
            ref=ref,
            z_init=z_init,
            control_stride=control_stride,
            intensity_quantile=0.40,
            grad_quantile=0.40,
            smooth_kernel=5,
            use_ref_check=True,
        )

    print("[Overfit] valid_mask:", valid_mask.shape)
    print("[Overfit] valid ratio:", valid_mask.float().mean().item())

    if z_init is None:
        z_init = sparse_z_idx

    print("[Overfit] mov:", mov.shape)
    print("[Overfit] ref:", ref.shape)

    if gt_disp is not None:
        print("[Overfit] gt_disp:", gt_disp.shape)
    if gt_coords is not None:
        print("[Overfit] gt_coords:", gt_coords.shape)
    if z_init is not None:
        print("[Overfit] z_init:", z_init)
    if spacing is not None:
        print("[Overfit] spacing:", spacing)

    # -------------------------
    # 2. Build model
    # -------------------------
    model = CoarseMatchingNetV6(
        dim=dim,
        radius=radius,
        temperature=temperature,
        use_learned_matching=use_learned_matching,
        matcher_mode=matcher_mode,
        num_refine_iters=num_refine_iters,
        control_stride=control_stride,
        encoder_stride=encoder_stride,
        query_chunk_size=query_chunk_size,

        moving_num_convnext_blocks=moving_num_convnext_blocks,
        ref_base_channels=ref_base_channels,
        ref_num_blocks=ref_num_blocks,
        ref_refine_blocks=ref_refine_blocks,
        use_coord_embed=use_coord_embed,

        use_offset_encoding=use_offset_encoding,
        use_offset_bias=use_offset_bias,
        use_local_cross_attn=use_local_cross_attn,
        local_attn_temperature=local_attn_temperature,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scaler = GradScaler(enabled=use_amp)

    # -------------------------
    # 3. Training loop
    # -------------------------
    for step in range(1, steps + 1):
        model.train()

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model(
                mov,
                ref,
                z_init=z_init, spacing = spacing
            )

            loss, loss_dict = total_coarse_loss(
                outputs=outputs,
                gt_disp=gt_disp,
                gt_coords=gt_coords,
                z_init=z_init,
                sparse_z_idx=sparse_z_idx,
                spacing=spacing,
                valid_mask=valid_mask,
                loss_mode=loss_mode,
                lambda_match=lambda_match,
                lambda_coord=lambda_coord,
                lambda_disp=lambda_disp,
                lambda_smooth=lambda_smooth,
                lambda_z_spacing=lambda_z_spacing,
                lambda_disp_mag=lambda_disp_mag,
            )

        if use_amp:
            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        # -------------------------
        # 4. Logging
        # -------------------------
        if step % log_interval == 0 or step == 1:
            with torch.no_grad():
                pred_disp = outputs["pred_disp"]

                if gt_disp is not None:
                    disp_mae = (pred_disp - gt_disp).abs().mean().item()
                    z_disp_mae = (
                        pred_disp[..., 0] - gt_disp[..., 0]
                    ).abs().mean().item()
                else:
                    disp_mae = float("nan")
                    z_disp_mae = float("nan")

                if outputs.get("pred_coords", None) is not None and gt_coords is not None:
                    pred_coords = outputs["pred_coords"]
                    coord_mae = (pred_coords - gt_coords).abs().mean().item()
                    z_coord_mae = (
                        pred_coords[..., 0] - gt_coords[..., 0]
                    ).abs().mean().item()
                else:
                    coord_mae = float("nan")
                    z_coord_mae = float("nan")
            valid_ratio = valid_mask.float().mean().item() if valid_mask is not None else 1.0
            print(
                f"[Overfit] step {step:05d} | "
                f"loss={loss.item():.4f} | "
                f"match={loss_dict.get('loss_match', torch.tensor(0.0)).item():.4f} | "
                f"kl={loss_dict.get('loss_match_kl', torch.tensor(0.0)).item():.4f} | "
                f"ce={loss_dict.get('loss_match_ce', torch.tensor(0.0)).item():.4f} | "
                f"coord={loss_dict.get('loss_coord', torch.tensor(0.0)).item():.4f} | "
                f"disp={loss_dict.get('loss_disp', torch.tensor(0.0)).item():.4f} | "
                f"smooth={loss_dict.get('loss_smooth', torch.tensor(0.0)).item():.4f} | "
                f"z_spacing={loss_dict.get('loss_z_spacing', torch.tensor(0.0)).item():.4f} | "
                f"top1={loss_dict.get('match_top1', torch.tensor(0.0)).item():.3f} | "
                f"top5={loss_dict.get('match_top5', torch.tensor(0.0)).item():.3f} | "
                f"pmax={loss_dict.get('match_prob_max', torch.tensor(0.0)).item():.3f} | "
                f"ent={loss_dict.get('match_entropy', torch.tensor(0.0)).item():.3f} | "
                f"inside={loss_dict.get('match_inside_ratio', torch.tensor(0.0)).item():.3f} | "
                f"disp_mae={disp_mae:.4f} | "
                f"z_disp_mae={z_disp_mae:.4f} | "
                f"coord_mae={coord_mae:.4f} | "
                f"z_coord_mae={z_coord_mae:.4f}|"
                f"valid={valid_ratio:.3f} "
            )

    return model, batch

@torch.no_grad()
def build_reliable_control_mask(
    mov,
    ref=None,
    z_init=None,
    control_stride=16,
    intensity_quantile=0.50,
    grad_quantile=0.50,
    smooth_kernel=5,
    use_ref_check=True,
):
    """
    Build valid_mask for control-point loss.

    Args:
        mov:
            (B,1,K,H,W), moving sparse stack.

        ref:
            (B,1,D,H,W), reference volume. Optional.

        z_init:
            (B,K) or (K,), initial z indices in raw ref coordinates.

        control_stride:
            control point spacing in raw xy pixels.

    Returns:
        valid_mask:
            (B,K,Hc,Wc), bool.
            True means this control point is reliable and participates in loss.
    """

    device = mov.device
    B, _, K, H, W = mov.shape

    Hc = (H + control_stride - 1) // control_stride
    Wc = (W + control_stride - 1) // control_stride

    # -------------------------
    # 1. Moving image reliability
    # -------------------------
    img = mov[:, 0]  # (B,K,H,W)

    # spatial gradient magnitude on each moving slice
    gx = torch.zeros_like(img)
    gy = torch.zeros_like(img)

    gx[..., :, 1:] = img[..., :, 1:] - img[..., :, :-1]
    gy[..., 1:, :] = img[..., 1:, :] - img[..., :-1, :]

    grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    # local smoothing, approximate your gaussian/median filtering
    if smooth_kernel is not None and smooth_kernel > 1:
        pad = smooth_kernel // 2
        grad_2d = grad.view(B * K, 1, H, W)
        grad_2d = torch.nn.functional.avg_pool2d(
            grad_2d,
            kernel_size=smooth_kernel,
            stride=1,
            padding=pad,
        )
        grad = grad_2d.view(B, K, H, W)

        img_2d = img.view(B * K, 1, H, W)
        img_2d = torch.nn.functional.avg_pool2d(
            img_2d,
            kernel_size=smooth_kernel,
            stride=1,
            padding=pad,
        )
        img_smooth = img_2d.view(B, K, H, W)
    else:
        img_smooth = img

    # Per-sample, per-slice thresholds
    img_flat = img_smooth.view(B, K, -1)
    grad_flat = grad.view(B, K, -1)

    img_th = torch.quantile(
        img_flat,
        intensity_quantile,
        dim=-1,
        keepdim=True,
    ).view(B, K, 1, 1)

    grad_th = torch.quantile(
        grad_flat,
        grad_quantile,
        dim=-1,
        keepdim=True,
    ).view(B, K, 1, 1)

    valid_mov = (img_smooth > img_th) & (grad > grad_th)

    # Downsample pixel mask to control grid.
    # A control point is valid if its local cell contains enough valid pixels.
    valid_mov_float = valid_mov.float().view(B * K, 1, H, W)

    valid_ctrl = torch.nn.functional.adaptive_avg_pool2d(
        valid_mov_float,
        output_size=(Hc, Wc),
    ).view(B, K, Hc, Wc)

    valid_mask = valid_ctrl > 0.05

    # -------------------------
    # 2. Optional reference check at z_init
    # -------------------------
    if use_ref_check and ref is not None and z_init is not None:
        _, _, D, Hr, Wr = ref.shape

        if z_init.dim() == 1:
            z_init_use = z_init[None, :].repeat(B, 1)
        else:
            z_init_use = z_init

        z_init_use = z_init_use.to(device).long().clamp(0, D - 1)

        ref_slices = []
        for b in range(B):
            ref_slices.append(ref[b, 0, z_init_use[b]])  # (K,H,W)

        ref_sparse = torch.stack(ref_slices, dim=0)  # (B,K,H,W)

        rgx = torch.zeros_like(ref_sparse)
        rgy = torch.zeros_like(ref_sparse)

        rgx[..., :, 1:] = ref_sparse[..., :, 1:] - ref_sparse[..., :, :-1]
        rgy[..., 1:, :] = ref_sparse[..., 1:, :] - ref_sparse[..., :-1, :]

        ref_grad = torch.sqrt(rgx ** 2 + rgy ** 2 + 1e-8)

        if smooth_kernel is not None and smooth_kernel > 1:
            pad = smooth_kernel // 2

            ref_grad_2d = ref_grad.view(B * K, 1, H, W)
            ref_grad_2d = torch.nn.functional.avg_pool2d(
                ref_grad_2d,
                kernel_size=smooth_kernel,
                stride=1,
                padding=pad,
            )
            ref_grad = ref_grad_2d.view(B, K, H, W)

            ref_img_2d = ref_sparse.view(B * K, 1, H, W)
            ref_img_2d = torch.nn.functional.avg_pool2d(
                ref_img_2d,
                kernel_size=smooth_kernel,
                stride=1,
                padding=pad,
            )
            ref_smooth = ref_img_2d.view(B, K, H, W)
        else:
            ref_smooth = ref_sparse

        ref_img_flat = ref_smooth.view(B, K, -1)
        ref_grad_flat = ref_grad.view(B, K, -1)

        ref_img_th = torch.quantile(
            ref_img_flat,
            intensity_quantile,
            dim=-1,
            keepdim=True,
        ).view(B, K, 1, 1)

        ref_grad_th = torch.quantile(
            ref_grad_flat,
            grad_quantile,
            dim=-1,
            keepdim=True,
        ).view(B, K, 1, 1)

        valid_ref = (ref_smooth > ref_img_th) & (ref_grad > ref_grad_th)

        valid_ref_float = valid_ref.float().view(B * K, 1, H, W)

        valid_ref_ctrl = torch.nn.functional.adaptive_avg_pool2d(
            valid_ref_float,
            output_size=(Hc, Wc),
        ).view(B, K, Hc, Wc)

        valid_ref_mask = valid_ref_ctrl > 0.10

        valid_mask = valid_mask & valid_ref_mask

    return valid_mask

def freeze_coarseflow_encoders(
    model,
    freeze_moving_encoder=True,
    freeze_reference_encoder=True,
    verbose=True,
):
    """
    Freeze image encoders and keep matcher/refiner trainable.
    """

    frozen_prefixes = []

    if freeze_moving_encoder and hasattr(model, "moving_encoder"):
        frozen_prefixes.append("moving_encoder")

    if freeze_reference_encoder and hasattr(model, "reference_encoder"):
        frozen_prefixes.append("reference_encoder")

    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in frozen_prefixes):
            p.requires_grad_(False)

    # set frozen modules to eval mode
    if freeze_moving_encoder and hasattr(model, "moving_encoder"):
        model.moving_encoder.eval()

    if freeze_reference_encoder and hasattr(model, "reference_encoder"):
        model.reference_encoder.eval()

    if verbose:
        total = 0
        trainable = 0
        frozen = 0

        for name, p in model.named_parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
            else:
                frozen += n

        print("[Freeze encoders]")
        print(f"  frozen prefixes : {frozen_prefixes}")
        print(f"  total params    : {total / 1e6:.2f} M")
        print(f"  trainable params: {trainable / 1e6:.2f} M")
        print(f"  frozen params   : {frozen / 1e6:.2f} M")

        print("\n[Trainable modules]")
        for name, p in model.named_parameters():
            if p.requires_grad:
                print("  ", name)

def freeze_all_except_coord_residual(model, verbose=True, logger=None):
    """
    Freeze all parameters except coord_residual_refiner.
    Used for residual-only coordinate refinement.
    """
    if not hasattr(model, "coord_residual_refiner") or model.coord_residual_refiner is None:
        raise ValueError(
            "train_only_residual=True requires model.coord_residual_refiner. "
            "Please set use_coord_residual=True in model_config."
        )

    for name, p in model.named_parameters():
        p.requires_grad_(False)

    for name, p in model.coord_residual_refiner.named_parameters():
        p.requires_grad_(True)

    total = 0
    trainable = 0
    frozen = 0

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        else:
            frozen += n

    msg = (
        "[Freeze all except residual]\n"
        f"  total params    : {total / 1e6:.3f} M\n"
        f"  trainable params: {trainable / 1e6:.3f} M\n"
        f"  frozen params   : {frozen / 1e6:.3f} M"
    )

    if logger is not None:
        logger.info(msg)
        logger.info("[Trainable residual parameters]")
        for name, p in model.named_parameters():
            if p.requires_grad:
                logger.info(f"  {name}")
    elif verbose:
        print(msg)
        print("[Trainable residual parameters]")
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(" ", name)

def keep_frozen_encoders_eval(model):
    """
    Call this after model.train() if encoders are frozen.
    """
    if hasattr(model, "moving_encoder"):
        frozen = all(not p.requires_grad for p in model.moving_encoder.parameters())
        if frozen:
            model.moving_encoder.eval()

    if hasattr(model, "reference_encoder"):
        frozen = all(not p.requires_grad for p in model.reference_encoder.parameters())
        if frozen:
            model.reference_encoder.eval()
#############################    inference  ##############################################
@torch.no_grad()
def inference(
    model,
    sample_or_batch,
    device=None,
    use_amp=True,
    build_valid_mask=True,
    control_stride=None,
    return_cpu=True,
    decode_mode="model",
    decode_topk=5,
    decode_temperature=1.0,
):
    """
    Run coarse matching inference on one sample or one batch.

    Args:
        model:
            trained CoarseMatchingNetV3.

        sample_or_batch:
            Either:
                sample from dataset:
                    mov: (1,K,H,W)
                    ref: (1,D,H,W)
                    gt_disp: optional (K,Hc,Wc,3)
                    gt_coords: optional (K,Hc,Wc,3)
                    z_init/sparse_z_idx: (K,)
                or batch:
                    mov: (B,1,K,H,W)
                    ref: (B,1,D,H,W)

        device:
            cuda/cpu.

        build_valid_mask:
            If True, dynamically compute reliable valid_mask
            using build_reliable_control_mask.

        control_stride:
            If None, use model.control_stride.

        return_cpu:
            If True, move outputs to CPU.

    Returns:
        result dict.
    """

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    # -------------------------
    # 1. Make batch dimension
    # -------------------------
    batch = {}

    for k, v in sample_or_batch.items():
        if torch.is_tensor(v):
            # dataset single sample:
            # mov: (1,K,H,W), ref: (1,D,H,W), gt: (K,Hc,Wc,3)
            if k in ["mov", "ref"]:
                if v.dim() == 4:
                    v = v.unsqueeze(0)  # -> (B,1,K,H,W) or (B,1,D,H,W)
            elif k in ["gt_disp", "gt_coords", "valid_mask"]:
                if v.dim() == 4:
                    v = v.unsqueeze(0)  # -> (B,K,Hc,Wc,3) or (B,K,Hc,Wc)
            elif k in ["z_init", "sparse_z_idx","spacing"]:
                if v.dim() == 1:
                    v = v.unsqueeze(0)  # -> (B,K)

            batch[k] = v.to(device)
        else:
            batch[k] = v

    mov = batch["mov"].float()
    ref = batch["ref"].float()
    spacing = batch.get("spacing", None)
    z_init = batch.get("z_init", None)
    sparse_z_idx = batch.get("sparse_z_idx", None)

    if z_init is None:
        z_init = sparse_z_idx

    if z_init is not None:
        z_init = z_init.to(device).float()

    gt_disp = batch.get("gt_disp", None)
    gt_coords = batch.get("gt_coords", None)

    if gt_disp is not None:
        gt_disp = gt_disp.to(device).float()
    if gt_coords is not None:
        gt_coords = gt_coords.to(device).float()

    # -------------------------
    # 2. Forward
    # -------------------------
    with autocast(enabled=use_amp):
        outputs = model(
            mov,
            ref,
            z_init=z_init,
            spacing=spacing,
            return_match_aux=(decode_mode != "model"),
        )

    if outputs.get("scores", None) is not None:
        print("scores shape:", outputs["scores"].shape)
        print("candidate M:", outputs["scores"].shape[-1])
    if decode_mode == "model":
        pred_coords = outputs["pred_coords"]
    else:
        _, K, Hc, Wc, _ = outputs["pred_coords"].shape

        pred_coords = decode_coords_from_scores(
            outputs=outputs,
            K=K,
            Hc=Hc,
            Wc=Wc,
            mode=decode_mode,
            topk=decode_topk,
            temperature=decode_temperature,
        )

    pred_disp = pred_coords - outputs["coords0"]

    # -------------------------
    # 3. Reliable mask
    # -------------------------
    valid_mask = batch.get("valid_mask", None)

    if build_valid_mask and valid_mask is None:
        if control_stride is None:
            control_stride = model.control_stride

        valid_mask = build_reliable_control_mask(
            mov=mov,
            ref=ref,
            z_init=z_init,
            control_stride=control_stride,
            intensity_quantile=0.50,
            grad_quantile=0.50,
            smooth_kernel=5,
            use_ref_check=True,
        )

    if valid_mask is not None:
        valid_mask = valid_mask.to(device).bool()

    # -------------------------
    # 4. Metrics if GT exists
    # -------------------------
    metrics = {}

    if gt_disp is not None:
        disp_err = (pred_disp - gt_disp).abs().mean(dim=-1)  # (B,K,Hc,Wc)

        metrics["disp_mae_all"] = disp_err.mean()

        if valid_mask is not None:
            metrics["disp_mae_valid"] = (
                disp_err * valid_mask.float()
            ).sum() / (valid_mask.float().sum() + 1e-6)

        z_err = (pred_disp[..., 0] - gt_disp[..., 0]).abs()
        metrics["z_disp_mae_all"] = z_err.mean()

        if valid_mask is not None:
            metrics["z_disp_mae_valid"] = (
                z_err * valid_mask.float()
            ).sum() / (valid_mask.float().sum() + 1e-6)

    if gt_coords is not None:
        coord_err = (pred_coords - gt_coords).abs().mean(dim=-1)

        metrics["coord_mae_all"] = coord_err.mean()

        if valid_mask is not None:
            metrics["coord_mae_valid"] = (
                coord_err * valid_mask.float()
            ).sum() / (valid_mask.float().sum() + 1e-6)

        z_coord_err = (pred_coords[..., 0] - gt_coords[..., 0]).abs()
        metrics["z_coord_mae_all"] = z_coord_err.mean()

        if valid_mask is not None:
            metrics["z_coord_mae_valid"] = (
                z_coord_err * valid_mask.float()
            ).sum() / (valid_mask.float().sum() + 1e-6)

    if valid_mask is not None:
        metrics["valid_ratio"] = valid_mask.float().mean()

    # Matching diagnostics
    if outputs.get("scores", None) is not None:
        scores = outputs["scores"]
        prob = torch.softmax(scores, dim=-1)

        metrics["prob_max"] = prob.max(dim=-1).values.mean()
        metrics["entropy"] = (
            -(prob * (prob + 1e-8).log()).sum(dim=-1).mean()
        )

    # -------------------------
    # 5. Package result
    # -------------------------
    result = {
        "pred_disp": pred_disp,
        "pred_coords": pred_coords,
        "coords0": outputs.get("coords0", None),
        "pred_coords_feat": outputs.get("pred_coords_feat", None),
        "coords0_feat": outputs.get("coords0_feat", None),
        "valid_mask": valid_mask,
        "metrics": metrics,
        "raw_outputs": outputs,
    }

    if return_cpu:
        result_cpu = {}

        for k, v in result.items():
            if torch.is_tensor(v):
                result_cpu[k] = v.detach().cpu()
            elif isinstance(v, dict):
                result_cpu[k] = {
                    kk: vv.detach().cpu() if torch.is_tensor(vv) else vv
                    for kk, vv in v.items()
                }
            elif isinstance(v, dict) is False:
                result_cpu[k] = v

        result = result_cpu

    return result


################################## debug functions ##################################
@torch.no_grad()
def estimate_inside_by_radius(batch, model, radii=((3,5,5), (5,5,5), (5,7,7), (7,9,9)), device="cuda"):
    """
    Estimate whether gt_coords is inside local search window
    for different radius settings.

    This does not require training. It only checks geometry.

    Requires:
        batch["mov"]: (B,1,K,H,W)
        batch["ref"]: (B,1,D,H,W)
        batch["gt_coords"]: (B,K,Hc,Wc,3)
        batch["z_init"] or batch["sparse_z_idx"]
    """
    batch = move_batch_to_device(batch, device)

    mov = batch["mov"]
    ref = batch["ref"]
    gt_coords = batch["gt_coords"]
    spacing = batch.get("spacing", None)
    z_init = batch.get("z_init", None)
    if z_init is None:
        z_init = batch.get("sparse_z_idx", None)

    if z_init is None:
        raise ValueError("z_init or sparse_z_idx is required.")

    B, _, K, H, W = mov.shape

    # forward once only to get F_ref scale and coords0 shape
    model.eval()
    outputs = model(mov, ref, z_init=z_init, spacing = spacing)

    coords0_raw = outputs["coords0"]          # (B,K,Hc,Wc,3)
    scale = outputs["coord_scale"]            # (3,)

    coords0_feat = outputs["coords0_feat"]    # (B,K,Hc,Wc,3)

    # convert gt raw coords to feature coords
    scale_z, scale_y, scale_x = scale
    gt_feat = gt_coords.clone()
    gt_feat[..., 0] = gt_feat[..., 0] / scale_z
    gt_feat[..., 1] = gt_feat[..., 1] / scale_y
    gt_feat[..., 2] = gt_feat[..., 2] / scale_x

    diff = (gt_feat - coords0_feat).abs()     # feature-space distance

    print("mean abs diff feat z/y/x:", diff[..., 0].mean().item(), diff[..., 1].mean().item(), diff[..., 2].mean().item())
    print("p95 abs diff feat z/y/x :", torch.quantile(diff[..., 0], 0.95).item(),
                                      torch.quantile(diff[..., 1], 0.95).item(),
                                      torch.quantile(diff[..., 2], 0.95).item())

    for r in radii:
        rz, ry, rx = r
        inside = (
            (diff[..., 0] <= rz) &
            (diff[..., 1] <= ry) &
            (diff[..., 2] <= rx)
        )
        print(f"radius={r}, inside={inside.float().mean().item():.4f}")

@torch.no_grad()
def debug_inside_by_axis(model, batch, device="cuda"):
    model.eval()
    batch = move_batch_to_device(batch, device)

    mov = batch["mov"]
    ref = batch["ref"]
    gt_coords = batch["gt_coords"]

    z_init = batch.get("z_init", None)
    if z_init is None:
        z_init = batch.get("sparse_z_idx", None)

    valid_mask = batch.get("valid_mask", None)

    outputs = model(mov, ref, z_init=z_init)

    coords0_feat = outputs["coords0_feat"]  # (B,K,Hc,Wc,3)
    scale = outputs["coord_scale"]

    gt_feat = gt_coords.clone()
    gt_feat[..., 0] = gt_feat[..., 0] / scale[0]
    gt_feat[..., 1] = gt_feat[..., 1] / scale[1]
    gt_feat[..., 2] = gt_feat[..., 2] / scale[2]

    diff = (gt_feat - coords0_feat).abs()

    if valid_mask is not None:
        valid = valid_mask.bool()
    else:
        valid = torch.ones(diff.shape[:-1], device=diff.device, dtype=torch.bool)

    def ratio(mask):
        return ((mask & valid).float().sum() / (valid.float().sum() + 1e-6)).item()

    print("===== feature-space diff statistics =====")
    for i, name in enumerate(["z", "y", "x"]):
        d = diff[..., i][valid]
        print(
            name,
            "mean=", d.mean().item(),
            "p90=", torch.quantile(d, 0.90).item(),
            "p95=", torch.quantile(d, 0.95).item(),
            "p99=", torch.quantile(d, 0.99).item(),
            "max=", d.max().item(),
        )

    print("\n===== inside by axis =====")
    for radius in [(3,5,5), (4,12,12), (5,15,15), (8,15,15), (10,15,15)]:
        rz, ry, rx = radius

        z_in = diff[..., 0] <= rz
        y_in = diff[..., 1] <= ry
        x_in = diff[..., 2] <= rx
        all_in = z_in & y_in & x_in

        print(f"\nradius={radius}")
        print("z_inside   =", ratio(z_in))
        print("y_inside   =", ratio(y_in))
        print("x_inside   =", ratio(x_in))
        print("all_inside =", ratio(all_in))

@torch.no_grad()
def check_gt_consistency(model, batch, device="cuda"):
    """
    Check whether dataset labels satisfy:

        gt_coords = coords0 + gt_disp

    Coordinate order should be:
        (z, y, x)

    This checks label / coordinate consistency, not model prediction quality.
    """
    model.eval()
    batch = move_batch_to_device(batch, device)

    mov = batch["mov"]
    ref = batch["ref"]

    gt_coords = batch.get("gt_coords", None)
    gt_disp = batch.get("gt_disp", None)

    if gt_coords is None:
        raise ValueError("batch does not contain gt_coords.")
    if gt_disp is None:
        raise ValueError("batch does not contain gt_disp.")

    z_init = batch.get("z_init", None)
    if z_init is None:
        z_init = batch.get("sparse_z_idx", None)

    if z_init is None:
        raise ValueError("batch must contain z_init or sparse_z_idx.")

    outputs = model(
        mov,
        ref,
        z_init=z_init,
    )

    coords0 = outputs["coords0"]

    err_plus = (gt_coords - (coords0 + gt_disp)).abs()
    err_minus = (gt_coords - (coords0 - gt_disp)).abs()

    print("========== Shape ==========")
    print("mov      :", tuple(mov.shape))
    print("ref      :", tuple(ref.shape))
    print("coords0  :", tuple(coords0.shape))
    print("gt_disp  :", tuple(gt_disp.shape))
    print("gt_coords:", tuple(gt_coords.shape))

    print("\n========== Check: gt_coords ?= coords0 + gt_disp ==========")
    print("mean error z/y/x:",
          err_plus[..., 0].mean().item(),
          err_plus[..., 1].mean().item(),
          err_plus[..., 2].mean().item())
    print("max  error z/y/x:",
          err_plus[..., 0].max().item(),
          err_plus[..., 1].max().item(),
          err_plus[..., 2].max().item())

    print("\n========== Check opposite direction: gt_coords ?= coords0 - gt_disp ==========")
    print("mean error z/y/x:",
          err_minus[..., 0].mean().item(),
          err_minus[..., 1].mean().item(),
          err_minus[..., 2].mean().item())
    print("max  error z/y/x:",
          err_minus[..., 0].max().item(),
          err_minus[..., 1].max().item(),
          err_minus[..., 2].max().item())

    print("\n========== Example point ==========")
    print("coords0[0,0,0,0]        :", coords0[0, 0, 0, 0])
    print("gt_disp[0,0,0,0]        :", gt_disp[0, 0, 0, 0])
    print("coords0 + gt_disp       :", coords0[0, 0, 0, 0] + gt_disp[0, 0, 0, 0])
    print("coords0 - gt_disp       :", coords0[0, 0, 0, 0] - gt_disp[0, 0, 0, 0])
    print("gt_coords[0,0,0,0]      :", gt_coords[0, 0, 0, 0])

    return {
        "coords0": coords0,
        "gt_disp": gt_disp,
        "gt_coords": gt_coords,
        "err_plus": err_plus,
        "err_minus": err_minus,
        "outputs": outputs,
    }

@torch.no_grad()
def inspect_gt_disp_distribution(batch, device="cuda", z_ratio=3.0):
    batch = move_batch_to_device(batch, device)

    gt_disp = batch["gt_disp"]  # (B,K,Hc,Wc,3), z,y,x
    valid_mask = batch.get("valid_mask", None)

    if valid_mask is not None:
        valid = valid_mask.bool()
    else:
        valid = torch.ones(gt_disp.shape[:-1], device=gt_disp.device, dtype=torch.bool)

    dz = gt_disp[..., 0][valid].abs()
    dy = gt_disp[..., 1][valid].abs()
    dx = gt_disp[..., 2][valid].abs()
    xy = torch.sqrt(gt_disp[..., 1][valid] ** 2 + gt_disp[..., 2][valid] ** 2)

    def stat(name, arr):
        print(
            name,
            "mean=", arr.mean().item(),
            "p50=", torch.quantile(arr, 0.50).item(),
            "p90=", torch.quantile(arr, 0.90).item(),
            "p95=", torch.quantile(arr, 0.95).item(),
            "p99=", torch.quantile(arr, 0.99).item(),
            "max=", arr.max().item(),
        )

    print("===== raw displacement statistics =====")
    stat("|dz| raw z-slice", dz)
    stat("|dy| raw pixel", dy)
    stat("|dx| raw pixel", dx)
    stat("xy magnitude raw pixel", xy)

    print("\n===== approximate physical z displacement =====")
    stat("|dz| * z_ratio", dz * z_ratio)

@torch.no_grad()
def decode_coords_from_scores(
    outputs,
    K,
    Hc,
    Wc,
    mode="soft",
    topk=5,
    temperature=1.0,
):
    """
    Decode raw-space pred_coords from model local matching scores.

    Args:
        outputs:
            raw model outputs, must contain:
                scores: (B,N,M)
                candidate_coords_feat: (B,N,M,3)
                coord_scale: (3,)

        K, Hc, Wc:
            output control grid shape.

        mode:
            "soft"       : full-window soft expectation
            "argmax"     : highest-score candidate
            "topk_soft"  : soft expectation over top-k candidates only
            "temp_soft"  : full-window soft expectation with temperature

        topk:
            used only for mode="topk_soft".

        temperature:
            used for mode="soft", "topk_soft", "temp_soft".
            Smaller value makes probability sharper.

    Returns:
        pred_coords_raw:
            (B,K,Hc,Wc,3), order z,y,x, raw-space.
    """

    scores = outputs["scores"]                         # (B,N,M)
    cand_feat = outputs["candidate_coords_feat"]        # (B,N,M,3)
    coord_scale = outputs["coord_scale"].to(scores.device).float()

    B, N, M = scores.shape

    if mode == "argmax":
        idx = scores.argmax(dim=-1)  # (B,N)

        pred_feat = cand_feat.gather(
            dim=2,
            index=idx[..., None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)  # (B,N,3)

    elif mode in ["soft", "temp_soft"]:
        prob = torch.softmax(scores / temperature, dim=-1)  # (B,N,M)
        pred_feat = (prob[..., None] * cand_feat).sum(dim=2)

    elif mode == "topk_soft":
        k = min(topk, M)

        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)  # (B,N,k)

        top_cand = cand_feat.gather(
            dim=2,
            index=top_idx[..., None].expand(-1, -1, -1, 3),
        )  # (B,N,k,3)

        prob = torch.softmax(top_scores / temperature, dim=-1)  # (B,N,k)
        pred_feat = (prob[..., None] * top_cand).sum(dim=2)

    else:
        raise ValueError(
            "mode must be one of: 'soft', 'argmax', 'topk_soft', 'temp_soft'"
        )

    pred_raw = pred_feat * coord_scale.view(1, 1, 3)
    pred_raw = pred_raw.view(B, K, Hc, Wc, 3)

    return pred_raw

@torch.no_grad()
def debug_inside_with_bounds(model, batch, device="cuda"):
    model.eval()
    batch = move_batch_to_device(batch, device)

    mov = batch["mov"]
    ref = batch["ref"]
    gt_coords = batch["gt_coords"]

    z_init = batch.get("z_init", None)
    if z_init is None:
        z_init = batch.get("sparse_z_idx", None)

    valid_mask = batch.get("valid_mask", None)

    outputs = model(mov, ref, z_init=z_init)

    coords0_feat = outputs["coords0_feat"]
    scale = outputs["coord_scale"]

    B, _, D, H, W = ref.shape
    _, _, Dr, Hr, Wr = outputs["F_ref"].shape

    gt_feat = gt_coords.clone()
    gt_feat[..., 0] = gt_feat[..., 0] / scale[0]
    gt_feat[..., 1] = gt_feat[..., 1] / scale[1]
    gt_feat[..., 2] = gt_feat[..., 2] / scale[2]

    diff = (gt_feat - coords0_feat).abs()

    if valid_mask is not None:
        valid = valid_mask.bool()
    else:
        valid = torch.ones(diff.shape[:-1], device=diff.device, dtype=torch.bool)

    in_ref_raw = (
        (gt_coords[..., 0] >= 0) & (gt_coords[..., 0] <= D - 1) &
        (gt_coords[..., 1] >= 0) & (gt_coords[..., 1] <= H - 1) &
        (gt_coords[..., 2] >= 0) & (gt_coords[..., 2] <= W - 1)
    )

    in_ref_feat = (
        (gt_feat[..., 0] >= 0) & (gt_feat[..., 0] <= Dr - 1) &
        (gt_feat[..., 1] >= 0) & (gt_feat[..., 1] <= Hr - 1) &
        (gt_feat[..., 2] >= 0) & (gt_feat[..., 2] <= Wr - 1)
    )

    def ratio(mask):
        return ((mask & valid).float().sum() / (valid.float().sum() + 1e-6)).item()

    print("===== GT coordinate validity =====")
    print("in_ref_raw :", ratio(in_ref_raw))
    print("in_ref_feat:", ratio(in_ref_feat))

    print("\n===== out of bound by axis =====")
    print("z_low :", ratio(gt_coords[..., 0] < 0))
    print("z_high:", ratio(gt_coords[..., 0] > D - 1))
    print("y_low :", ratio(gt_coords[..., 1] < 0))
    print("y_high:", ratio(gt_coords[..., 1] > H - 1))
    print("x_low :", ratio(gt_coords[..., 2] < 0))
    print("x_high:", ratio(gt_coords[..., 2] > W - 1))

    print("\n===== radius inside with and without bounds =====")
    for radius in [(3,5,5), (4,12,12), (5,15,15)]:
        rz, ry, rx = radius

        motion_inside = (
            (diff[..., 0] <= rz) &
            (diff[..., 1] <= ry) &
            (diff[..., 2] <= rx)
        )

        valid_candidate_inside = motion_inside & in_ref_feat

        print(f"\nradius={radius}")
        print("motion_inside          =", ratio(motion_inside))
        print("valid_candidate_inside =", ratio(valid_candidate_inside))



################################# Model Predict Initial Coordinate #############################
import torch
import numpy as np

def _to_tensor_5d_mov(mov, device):
    """
    Convert moving image to (B,1,K,H,W).

    Accept:
        mov: numpy or torch
             shape can be (K,H,W), (1,K,H,W), or (B,1,K,H,W)
    """
    if isinstance(mov, np.ndarray):
        mov = torch.from_numpy(mov)

    mov = mov.float()

    if mov.dim() == 3:
        # (K,H,W) -> (1,1,K,H,W)
        mov = mov.unsqueeze(0).unsqueeze(0)
    elif mov.dim() == 4:
        # (1,K,H,W) -> (1,1,K,H,W)
        mov = mov.unsqueeze(0)
    elif mov.dim() == 5:
        pass
    else:
        raise ValueError(f"mov must have shape (K,H,W), (1,K,H,W), or (B,1,K,H,W), got {mov.shape}")

    return mov.to(device)
def _to_tensor_5d_ref(ref, device):
    """
    Convert reference image to (B,1,D,H,W).

    Accept:
        ref: numpy or torch
             shape can be (D,H,W), (1,D,H,W), or (B,1,D,H,W)
    """
    if isinstance(ref, np.ndarray):
        ref = torch.from_numpy(ref)

    ref = ref.float()

    if ref.dim() == 3:
        # (D,H,W) -> (1,1,D,H,W)
        ref = ref.unsqueeze(0).unsqueeze(0)
    elif ref.dim() == 4:
        # (1,D,H,W) -> (1,1,D,H,W)
        ref = ref.unsqueeze(0)
    elif ref.dim() == 5:
        pass
    else:
        raise ValueError(f"ref must have shape (D,H,W), (1,D,H,W), or (B,1,D,H,W), got {ref.shape}")

    return ref.to(device)
def _normalize_image_tensor(x, eps=1e-6):
    """
    Simple percentile normalization to [0,1].
    Use only if your training data used normalize=True.
    """
    q_low = torch.quantile(x, 0.01)
    q_high = torch.quantile(x, 0.995)

    x = (x - q_low) / (q_high - q_low + eps)
    x = x.clamp(0.0, 1.0)

    return x

def _build_spacing_tensor(
    ref_spacing,
    z_init,
    device,
    mov_spacing=None,
):
    """
    Build spacing tensor with order:
        [sx_ref, sy_ref, sz_ref, sx_mov, sy_mov, sz_mov]

    Args:
        ref_spacing:
            tuple/list, (sx_ref, sy_ref, sz_ref)

        z_init:
            array-like, initial z indices in reference coordinate.
            Used to infer moving z spacing if mov_spacing is None.

        mov_spacing:
            optional tuple/list, (sx_mov, sy_mov, sz_mov)

    If mov_spacing is None:
        assume moving xy spacing equals reference xy spacing,
        and moving z spacing = median(diff(z_init)) * sz_ref.
    """
    sx_ref, sy_ref, sz_ref = [float(v) for v in ref_spacing]

    z_init_np = np.asarray(z_init, dtype=np.float32)

    if mov_spacing is None:
        if len(z_init_np) >= 2:
            sparse_step = float(np.median(np.diff(z_init_np)))
        else:
            sparse_step = 1.0

        sx_mov = sx_ref
        sy_mov = sy_ref
        sz_mov = sz_ref * sparse_step
    else:
        sx_mov, sy_mov, sz_mov = [float(v) for v in mov_spacing]

    spacing = torch.tensor(
        [sx_ref, sy_ref, sz_ref, sx_mov, sy_mov, sz_mov],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    return spacing

def load_coarse_model_from_pth(
    pth_path,
    device,
    model_config=None,
    strict=True,
):
    """
    Load CoarseMatchingNetV5 from checkpoint.

    If checkpoint contains "model_config", use it.
    Otherwise use provided model_config.
    """
    ckpt = torch.load(
        pth_path,
        map_location=device,
        weights_only=False,
    )

    if model_config is None:
        model_config = ckpt.get("model_config", None)
        if model_config is None:
            model_config = dict(
                dim=96,
                radius=(4, 3, 3),
                temperature=0.05,
                use_learned_matching=True,
                matcher_mode="hybrid",
                control_stride=8,
                encoder_stride=8,
                num_refine_iters=1,
                query_chunk_size=512,

                moving_base_channels=(24, 48, 96),
                moving_num_blocks=(1, 2, 1),
                moving_mlp_ratio=2.0,
                moving_window_attn_layers=1,
                moving_window_size=8,
                moving_attn_num_heads=4,
                moving_slice_fusion_blocks=1,

                ref_base_channels=(24, 48, 96),
                ref_num_blocks=(1, 2, 2),
                ref_refine_blocks=1,
                ref_mlp_ratio=2.0,
                ref_attn_layers=6,
                ref_attn_num_heads=4,
                ref_attn_window_size=(4, 8, 8),
                ref_attn_mlp_ratio=2.0,

                use_coord_embed=True,
                use_spacing_embed=True,

                use_offset_encoding=True,
                use_offset_bias=True,
                use_local_cross_attn=True,
                local_attn_temperature=0.20,

                use_coord_residual=False,
            )

    model = CoarseMatchingNetV6(**model_config).to(device)

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    load_msg = model.load_state_dict(
        state_dict,
        strict=strict,
    )

    model.eval()

    return model, ckpt, load_msg

def percentile_normalize_01_tensor(
    x,
    p_low=0.01,
    p_high=0.99,
    eps=1e-8,
    mask_nonzero=False,
    max_quantile_samples=1_000_000,
):
    """
    Percentile normalize image tensor to [0,1].

    Args:
        x:
            Tensor with shape:
                (B,1,D,H,W) for reference
                (B,1,K,H,W) for moving

        p_low / p_high:
            Quantile values, not percentages.
            p_low=0.01 means 1% percentile.
            p_high=0.99 means 99% percentile.

        max_quantile_samples:
            Maximum number of pixels used to estimate quantile.
            This avoids torch.quantile failure on very large tensors.
    """
    x = x.float()
    out = torch.empty_like(x)

    B = x.shape[0]

    for b in range(B):
        xb = x[b]
        vals = xb.reshape(-1)

        # Remove NaN / Inf if any
        vals = vals[torch.isfinite(vals)]

        if mask_nonzero:
            valid_vals = vals[vals > 0]
            if valid_vals.numel() > 100:
                vals = valid_vals

        if vals.numel() == 0:
            out[b] = torch.zeros_like(xb)
            continue

        # ----------------------------------------------------
        # Key fix:
        # If image is too large, use strided subsampling
        # to estimate percentile.
        # ----------------------------------------------------
        if vals.numel() > max_quantile_samples:
            step = int((vals.numel() + max_quantile_samples - 1) // max_quantile_samples)
            vals_q = vals[::step].contiguous()
        else:
            vals_q = vals.contiguous()

        lo = torch.quantile(vals_q, p_low)
        hi = torch.quantile(vals_q, p_high)

        denom = hi - lo

        if denom.abs() < eps:
            out[b] = torch.zeros_like(xb)
        else:
            out[b] = (xb - lo) / (denom + eps)

    out = out.clamp(0.0, 1.0)

    return out

def control_coords_to_dense_coords(
    pred_coords,
    z_init,
    image_shape_yx,
    control_stride=16,
    input_order="zyx",
    output_order="xyz",
    extrapolate=True,
):
    """
    Convert sparse control-point predicted coords to dense per-pixel coords.

    Args:
        pred_coords:
            (K,Hc,Wc,3), predicted reference coordinates at control points.
            Default order: z,y,x.

        z_init:
            (K,), initial raw z index for each moving slice.

        image_shape_yx:
            (Y,X), original moving slice size.

        control_stride:
            control point stride in raw xy pixels.

        input_order:
            "zyx" or "xyz".

        output_order:
            "zyx" or "xyz".

    Returns:
        dense_coords:
            (K,Y,X,3), dense reference coordinates.
    """

    pred_coords = np.asarray(pred_coords, dtype=np.float32)

    if input_order == "xyz":
        pred_coords_zyx = pred_coords[..., [2, 1, 0]]
    elif input_order == "zyx":
        pred_coords_zyx = pred_coords
    else:
        raise ValueError("input_order must be 'zyx' or 'xyz'.")

    K, Hc, Wc, _ = pred_coords_zyx.shape
    Y, X = image_shape_yx

    z_init = np.asarray(z_init, dtype=np.float32)
    assert z_init.shape[0] == K

    # -------------------------
    # 1. Build control-grid initial coords
    # -------------------------
    y_ctrl = np.arange(Hc, dtype=np.float32) * control_stride
    x_ctrl = np.arange(Wc, dtype=np.float32) * control_stride

    yy_ctrl, xx_ctrl = np.meshgrid(y_ctrl, x_ctrl, indexing="ij")

    coords0_ctrl = np.zeros_like(pred_coords_zyx, dtype=np.float32)
    coords0_ctrl[..., 0] = z_init[:, None, None]
    coords0_ctrl[..., 1] = yy_ctrl[None, :, :]
    coords0_ctrl[..., 2] = xx_ctrl[None, :, :]

    # Interpolate displacement instead of absolute coords
    pred_disp_ctrl = pred_coords_zyx - coords0_ctrl

    # -------------------------
    # 2. Dense pixel grid
    # -------------------------
    y_dense = np.arange(Y, dtype=np.float32)
    x_dense = np.arange(X, dtype=np.float32)
    yy_dense, xx_dense = np.meshgrid(y_dense, x_dense, indexing="ij")

    query_points = np.stack(
        [yy_dense.ravel(), xx_dense.ravel()],
        axis=-1,
    )

    dense_disp = np.zeros((K, Y, X, 3), dtype=np.float32)

    bounds_error = False
    fill_value = None if extrapolate else 0.0

    # -------------------------
    # 3. Interpolate each slice and component
    # -------------------------
    for k in range(K):
        for c in range(3):
            interp = RegularGridInterpolator(
                points=(y_ctrl, x_ctrl),
                values=pred_disp_ctrl[k, :, :, c],
                method="linear",
                bounds_error=bounds_error,
                fill_value=fill_value,
            )

            dense_disp[k, :, :, c] = interp(query_points).reshape(Y, X)

    # -------------------------
    # 4. Add dense initial coords
    # -------------------------
    dense_coords_zyx = np.zeros((K, Y, X, 3), dtype=np.float32)
    dense_coords_zyx[..., 0] = z_init[:, None, None]
    dense_coords_zyx[..., 1] = yy_dense[None, :, :]
    dense_coords_zyx[..., 2] = xx_dense[None, :, :]

    dense_coords_zyx = dense_coords_zyx + dense_disp

    if output_order == "zyx":
        return dense_coords_zyx
    elif output_order == "xyz":
        return dense_coords_zyx[..., [2, 1, 0]]
    else:
        raise ValueError("output_order must be 'zyx' or 'xyz'.")
    
@torch.no_grad()
def predict_initial_coords(
    ref,
    mov,
    z_init,
    ref_spacing,
    pth_path,
    mov_spacing=None,
    model_config=None,
    device=None,
    normalize=True,
    norm_p_low=0.01,
    norm_p_high=0.99,
    norm_mask_nonzero=False,
    use_amp=True,
    strict_load=True,
    return_cpu=True,
    return_dense=True,
    control_stride=None,
):
    """
    Predict coarse initial coordinates from moving sparse stack to reference volume.

    Args:
        ref:
            reference volume.
            Shape:
                (D,H,W), or (1,D,H,W), or (B,1,D,H,W)

        mov:
            moving sparse stack.
            Shape:
                (K,H,W), or (1,K,H,W), or (B,1,K,H,W)

        z_init:
            initial z index of each moving slice in reference coordinate.
            Shape:
                (K,) or (B,K)

        ref_spacing:
            physical spacing of reference volume:
                (sx_ref, sy_ref, sz_ref)

            Example:
                if zRatio=3 and xy spacing=1:
                    ref_spacing=(1.0, 1.0, 3.0)

        mov_spacing:
            optional physical spacing of moving sparse stack:
                (sx_mov, sy_mov, sz_mov)

            If None, infer:
                sz_mov = median(diff(z_init)) * sz_ref

        pth_path:
            path to model checkpoint.

        model_config:
            required if pth does not contain "model_config".

        normalize:
            whether to normalize ref/mov to [0,1].

    Returns:
        result:
            dict containing:
                pred_coords: (K,Hc,Wc,3), z,y,x raw reference coordinates
                pred_disp:   (K,Hc,Wc,3), z,y,x raw displacement
                coords0:     (K,Hc,Wc,3), z,y,x initial coordinates
                confidence:  (K,Hc,Wc)
                entropy:     (K,Hc,Wc)
                raw_outputs: raw model outputs
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    # -------------------------
    # 1. Prepare tensors
    # -------------------------
    ref_t = _to_tensor_5d_ref(ref, device)
    mov_t = _to_tensor_5d_mov(mov, device)

    if normalize:
        ref_t = percentile_normalize_01_tensor(
            ref_t,
            p_low=norm_p_low,
            p_high=norm_p_high,
            mask_nonzero=norm_mask_nonzero,
        )

        mov_t = percentile_normalize_01_tensor(
            mov_t,
            p_low=norm_p_low,
            p_high=norm_p_high,
            mask_nonzero=norm_mask_nonzero,
        )

    z_init_t = torch.as_tensor(
        z_init,
        dtype=torch.float32,
        device=device,
    )

    if z_init_t.dim() == 1:
        z_init_t = z_init_t.unsqueeze(0)

    spacing_t = _build_spacing_tensor(
        ref_spacing=ref_spacing,
        mov_spacing=mov_spacing,
        z_init=z_init,
        device=device,
    )

    # -------------------------
    # 2. Load model
    # -------------------------
    model, ckpt, load_msg = load_coarse_model_from_pth(
        pth_path=pth_path,
        device=device,
        model_config=model_config,
        strict=strict_load,
    )

    # -------------------------
    # 3. Forward
    # -------------------------
    with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
        outputs = model(
            mov_t,
            ref_t,
            z_init=z_init_t,
            spacing=spacing_t,
        )

    pred_coords_ctrl_zyx = outputs["pred_coords"][0]
    pred_disp_ctrl_zyx = outputs["pred_disp"][0]
    coords0_ctrl_zyx = outputs["coords0"][0]

    confidence = outputs.get("confidence", None)
    entropy = outputs.get("entropy", None)

    if confidence is not None:
        K = mov_t.shape[2]
        Hc = pred_coords_ctrl_zyx.shape[1]
        Wc = pred_coords_ctrl_zyx.shape[2]
        confidence = confidence[0].reshape(K, Hc, Wc)

    if entropy is not None:
        K = mov_t.shape[2]
        Hc = pred_coords_ctrl_zyx.shape[1]
        Wc = pred_coords_ctrl_zyx.shape[2]
        entropy = entropy[0].reshape(K, Hc, Wc)

    result = {
        "pred_coords_ctrl_zyx": pred_coords_ctrl_zyx,
        "pred_disp_ctrl_zyx": pred_disp_ctrl_zyx,
        "coords0_ctrl_zyx": coords0_ctrl_zyx,
        "confidence": confidence,
        "entropy": entropy,
        "spacing": spacing_t,
        "checkpoint": ckpt,
        "load_msg": load_msg,
    }

    # backward compatibility
    result["pred_coords"] = pred_coords_ctrl_zyx
    result["pred_disp"] = pred_disp_ctrl_zyx
    result["coords0"] = coords0_ctrl_zyx

    if return_dense:
        if control_stride is None:
            control_stride_use = model.control_stride
        else:
            control_stride_use = control_stride

        _, _, K, Y, X = mov_t.shape
        z_init_np = z_init_t[0].detach().cpu().numpy().astype(int)

        pred_coords_ctrl_np = pred_coords_ctrl_zyx.detach().cpu().numpy()
        coords0_ctrl_np = coords0_ctrl_zyx.detach().cpu().numpy()

        pred_coords_dense_kyx_xyz = control_coords_to_dense_coords(
            pred_coords=pred_coords_ctrl_np,
            z_init=z_init_np,
            image_shape_yx=(Y, X),
            control_stride=control_stride_use,
            input_order="zyx",
            output_order="xyz",
        )

        coords0_dense_kyx_xyz = control_coords_to_dense_coords(
            pred_coords=coords0_ctrl_np,
            z_init=z_init_np,
            image_shape_yx=(Y, X),
            control_stride=control_stride_use,
            input_order="zyx",
            output_order="xyz",
        )

        pred_disp_dense_kyx_xyz = pred_coords_dense_kyx_xyz - coords0_dense_kyx_xyz
        pred_phase_xyz = pred_coords_dense_kyx_xyz.transpose(2, 1, 0, 3)

        result["pred_coords_dense_kyx_xyz"] = pred_coords_dense_kyx_xyz
        result["coords0_dense_kyx_xyz"] = coords0_dense_kyx_xyz
        result["pred_disp_dense_kyx_xyz"] = pred_disp_dense_kyx_xyz
        result["pred_phase_xyz"] = pred_phase_xyz

    if return_cpu:
        result_cpu = {}
        for k, v in result.items():
            if torch.is_tensor(v):
                result_cpu[k] = v.detach().cpu()
            elif isinstance(v, dict):
                result_cpu[k] = {
                    kk: vv.detach().cpu() if torch.is_tensor(vv) else vv
                    for kk, vv in v.items()
                }
            else:
                result_cpu[k] = v
        result = result_cpu

    return result