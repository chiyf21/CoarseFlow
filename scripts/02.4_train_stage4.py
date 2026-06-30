# scripts/train_stage4_spatial_residual_npu.py

import os
import sys
import json
from pathlib import Path

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from training.train import train_coarse_matching_model
from datasets.synthetic_dataset import (
    build_sameShape_loader,
    summarize_manifest,
)


def get_ddp_info():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_dist = world_size > 1
    return rank, local_rank, world_size, is_dist


def main():
    rank, local_rank, world_size, is_dist = get_ddp_info()
    is_main = rank == 0

    # ------------------------------------------------------------
    # Basic information
    # ------------------------------------------------------------
    if is_main:
        print("=" * 80)
        print("[Stage 4 Spatial Residual Refinement NPU Training]")
        print(f"PROJECT_ROOT = {PROJECT_ROOT}")
        print(f"rank         = {rank}")
        print(f"local_rank   = {local_rank}")
        print(f"world_size   = {world_size}")
        print(f"is_dist      = {is_dist}")
        print("=" * 80)

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------
    manifest_summary_path = PROJECT_ROOT / "cached_datasets/coarseflow_v6/manifest_summary.json"

    with open(manifest_summary_path, "r") as f:
        manifest_summary = json.load(f)

    if is_main:
        for k, v in manifest_summary.items():
            print(k, "->", v)

        for name, path in manifest_summary.items():
            print("\n" + "=" * 100)
            print(name)
            summarize_manifest(path)


    # ------------------------------------------------------------
    # Stage-4 residual-refinement settings
    # ------------------------------------------------------------
    # Keep the coarse V6 matcher at one refinement iteration to avoid
    # the large memory increase from iterative matching. The new spatial
    # residual head is trained on top of the existing coarse checkpoint.
    PER_NPU_BATCH_SIZE = 4
    VAL_BATCH_SIZE = 1
    TRAIN_ONLY_RESIDUAL = True

    # ------------------------------------------------------------
    # Model config
    # ------------------------------------------------------------
    model_config_v6 = dict(
        # =====================================================
        # Core
        # =====================================================
        dim=96,
        radius=(4, 3, 3),
        temperature=0.05,
        use_learned_matching=True,
        matcher_mode="hybrid",

        control_stride=16,
        encoder_stride=8,

        num_refine_iters=1,
        query_chunk_size=256,

        # =====================================================
        # Moving encoder
        # =====================================================
        moving_base_channels=(24, 48, 96),
        moving_num_blocks=(2, 4, 4),
        moving_mlp_ratio=2.0,

        moving_window_attn_layers=6,
        moving_window_size=8,
        moving_attn_num_heads=4,
        moving_slice_fusion_blocks=1,

        # =====================================================
        # Reference encoder
        # =====================================================
        ref_base_channels=(24, 48, 96),
        ref_num_blocks=(2, 4, 4),
        ref_refine_blocks=1,
        ref_mlp_ratio=2.0,

        ref_attn_layers=6,
        ref_attn_num_heads=4,
        ref_attn_window_size=(4, 8, 8),
        ref_attn_mlp_ratio=2.0,

        # =====================================================
        # Embeddings
        # =====================================================
        use_coord_embed=True,
        use_spacing_embed=True,

        # =====================================================
        # Matcher
        # =====================================================
        use_offset_encoding=True,
        use_offset_bias=True,
        use_local_cross_attn=True,
        local_attn_temperature=0.20,
        matcher_cross_attn_layers=3,
        matcher_cross_attn_heads=4,
        matcher_ffn_ratio=2.0,
        matcher_attn_drop=0.0,
        matcher_proj_drop=0.0,
        matcher_init_gamma=1e-2,

        # =====================================================
        # Spatial coordinate residual refinement: ON for Stage 4
        # =====================================================
        # Keep num_refine_iters=1 above. This avoids the memory increase
        # from running the full local matcher 2-3 times. The residual head
        # learns a small continuous correction after the /8 coarse match.
        use_coord_residual=True,
        residual_type="spatial",

        # 128 hidden dim + 3 blocks is deliberately lighter than the
        # previous 256 dim + 5 blocks setting, and is safer for inference.
        residual_hidden_dim=128,
        residual_num_blocks=3,

        # V6 matches on /8 XY features, so allow several raw-pixel XY
        # correction pixels to compensate for coarse-grid quantization.
        residual_max_delta=(2.0, 8.0, 8.0),

        residual_use_disp=True,

        # False uses kernel=(1,3,3) inside the spatial residual head.
        # This reduces memory/compute compared with full 3D residual convs.
        residual_use_3d=False,

        # Detach coarse outputs/features so the residual stage behaves like
        # a lightweight calibrator on top of the trained coarse V6 model.
        residual_detach_coarse=True,
        residual_detach_features=True,
    )

    model_config_stage1 = dict(model_config_v6)

    # ------------------------------------------------------------
    # DataLoader
    # Important:
    #   batch_size here is PER NPU.
    #   global batch size = batch_size * world_size.
    # ------------------------------------------------------------
    train_loader_stage3, _, _ = build_sameShape_loader(
        manifest_summary["stage3.1_train"],
        batch_size=PER_NPU_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        verbose=is_main,

        # DDP-related
        distributed=is_dist,
        rank=rank,
        world_size=world_size,
        seed=1234,
        pad_to_equal_batches=True,
    )

    # Validation only on rank 0.
    if is_main:
        val_loader_stage3, _, _ = build_sameShape_loader(
            manifest_summary["stage3.1_val"],
            batch_size=VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            verbose=True,

            distributed=False,
        )
    else:
        val_loader_stage3 = None
    # if is_main:
    #     print(
    #         f"[DEBUG] len(train_loader_stage3) = {len(train_loader_stage3)}",
    #         flush=True,
    #     )
    #     print(
    #         f"[DEBUG] len(val_loader_stage3) = {len(val_loader_stage3)}",
    #         flush=True,
    #     )
    # else:
    #     print(
    #         f"[DEBUG] rank={rank}, len(train_loader_stage3) = {len(train_loader_stage3)}, val_loader=None",
    #         flush=True,
    #     )
    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------
    model_stage1 = train_coarse_matching_model(
        train_dataset=None,
        val_dataset=None,
        train_loader=train_loader_stage3,
        val_loader=val_loader_stage3,

        save_dir="checkpoints/coarseflow_v7_stage4_spatial_residual",

        num_epochs=600,
        lr=1e-4,
        weight_decay=1e-4,
        batch_size=PER_NPU_BATCH_SIZE,
        num_workers=0,

        use_amp=False,

        **model_config_stage1,

        # Residual stage should be driven mainly by coordinate/displacement
        # supervision, not by re-training the coarse match distribution.
        # If your train.py only supports loss_mode="match", keep this line
        # unchanged and rely on the lambda values below.
        loss_mode="coord",

        lambda_match=0.0,
        lambda_match_kl=0.0,
        lambda_match_ce=0.0,

        lambda_coord=1.0,
        lambda_disp=0.2,

        lambda_smooth=0.01,
        lambda_z_spacing=0.00,
        lambda_disp_mag=0.0,

        compute_chunk_match_loss=False,
        match_sigma=(0.4, 0.6, 0.6),
        match_inside_threshold=4.0,

        resume_path="checkpoints/coarseflow_v7_stage3.1_iter3_sharpen/best.pth",
        resume_optimizer=False,
        resume_best_val_loss=False,
        strict_load=False,

        # This requires train_coarse_matching_model to support freezing all
        # non-residual parameters. If your train.py does not have this option,
        # remove this line and freeze modules manually in train.py.
        train_only_residual=TRAIN_ONLY_RESIDUAL,

        log_filename="train.log",
        log_mode="a",
    )
    
    if is_main:
        print("[Done] Stage 4 spatial residual refinement finished.")


if __name__ == "__main__":
    main()