# Stage 3: More difficult data + sharpen — harder data, resume from Stage 2

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
        print("[Stage 3] More difficult data + sharpen — resume from Stage 2
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
    # Model config
    # ------------------------------------------------------------
    model_config = dict(
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
        query_chunk_size=512,

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
        # Residual refinement: OFF for Stage 1
        # =====================================================
        use_coord_residual=False,
        residual_type="spatial",
        residual_hidden_dim=256,
        residual_num_blocks=5,
        residual_max_delta=(1.5, 3.0, 3.0),
        residual_use_disp=True,
        residual_use_3d=True,
        residual_detach_coarse=True,
        residual_detach_features=True,
    )

    model_config = dict(model_config)

    # ------------------------------------------------------------
    # DataLoader
    # Important:
    #   batch_size here is PER NPU.
    #   global batch size = batch_size * world_size.
    # ------------------------------------------------------------
    train_loader_stage3, _, _ = build_sameShape_loader(
        manifest_summary["stage3_train"],
        batch_size=3,
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
            manifest_summary["stage3_val"],
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            verbose=True,

            distributed=False,
        )
    else:
        val_loader_stage3 = None
    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------
    model = train_coarse_matching_model(
        train_dataset=None,
        val_dataset=None,
        train_loader=train_loader_stage3,
        val_loader=val_loader_stage3,

        save_dir="checkpoints/coarseflow_v7_stage3_More_difficult_data_sharpen",

        num_epochs=600,
        lr=1e-5,
        weight_decay=1e-4,
        batch_size=3,
        num_workers=0,

        use_amp=False,

        **model_config,

        loss_mode="match",

        lambda_match=1.0,
        lambda_match_kl=0.15,
        lambda_match_ce=0.85,

        lambda_coord=0.6,
        lambda_disp=0.0,

        lambda_smooth=0.005,
        lambda_z_spacing=0.00,
        lambda_disp_mag=0.1,

        compute_chunk_match_loss=True,
        match_sigma=(0.4, 0.6, 0.6),
        match_inside_threshold=4.0,

        resume_path="checkpoints/coarseflow_v7_stage2_K5_sharpen/best.pth",
        resume_optimizer=False,
        resume_best_val_loss=False,
        strict_load=True,

        log_filename="train.log",
        log_mode="a",
    )
    if is_main:
        print("[Done] Stage 3 training finished.")


if __name__ == "__main__":
    main()