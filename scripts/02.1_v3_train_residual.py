# v3 Stage 3:
# Residual-only continuous coordinate refinement
#
# Initialization:
#   Stage 2 best checkpoint
#
# Training strategy:
#   - Keep the entire Stage-2 coarse matcher frozen
#   - Enable SpatialCoordResidualHead
#   - Train ONLY the residual head
#   - Optimize final continuous coordinate L1
#
# Coarse model:
#   Swin3D + multi-scale fusion
#   two-pass iterative local matching
#
# Residual formulation:
#
#   x_final = x_coarse + delta_x
#
# where delta_x is predicted in RAW coordinate space.
#
# Recommended launch:
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   torchrun --standalone --nproc_per_node=8 \
#   scripts/04_train_v3_stage3_residual.py
#
# Dataset:
#   Same 3455 train / 1000 val combined dataset
#
# Effective batch size with 8 GPUs:
#
#   4 samples / GPU * 8 GPUs = 32


import os
import sys
import json
from collections import Counter
from pathlib import Path

import torch

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    torch_npu = None


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


# ============================================================
# Stage 3 training configuration
# ============================================================

# Number of ADDITIONAL epochs after loading Stage-2 best.pth
NUM_EPOCHS = 200

# Residual head is newly initialized, therefore use a larger
# learning rate than Stage-2 fine-tuning.
BASE_LR = 1e-4

WEIGHT_DECAY = 1e-4

# First 20 Stage-3 epochs:
#     constant 1e-4
#
# Remaining 100 epochs:
#     cosine 1e-4 -> 1e-6
COSINE_START_EPOCH = 20
COSINE_ETA_MIN = 1e-6


# ============================================================
# Multi-GPU batch configuration
# ============================================================

BATCH_SIZE_PER_DEVICE = 4

NUM_WORKERS = 4

SEED = 1234


# ============================================================
# Checkpoints
# ============================================================

STAGE2_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "coarseflow_swin3d_v3_refine2_sharpen"
    / "best.pth"
)

STAGE3_SAVE_DIR = (
    "checkpoints/"
    "coarseflow_swin3d_v3_stage3_residual"
)


# ============================================================
# DDP
# ============================================================

def get_ddp_info():
    rank = int(
        os.environ.get(
            "RANK",
            "0",
        )
    )

    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
    )

    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )

    is_dist = world_size > 1

    return (
        rank,
        local_rank,
        world_size,
        is_dist,
    )


# ============================================================
# Manifest diagnostics
# ============================================================

def summarize_difficulty_distribution(
    manifest_path,
):
    """
    Print the actual number of samples in each difficulty group.

    This is diagnostic only.
    The dataset distribution is NOT changed.
    """

    with open(
        manifest_path,
        "r",
    ) as f:
        manifest = json.load(f)

    counts = Counter()

    for part in manifest["parts"]:

        difficulty = str(
            part.get(
                "difficulty",
                "unknown",
            )
        )

        num_samples = int(
            part.get(
                "num_samples",
                part.get(
                    "length",
                    0,
                ),
            )
        )

        counts[difficulty] += num_samples

    total = sum(
        counts.values()
    )

    print("-" * 80)

    print(
        "[Difficulty distribution]"
    )

    for difficulty, count in sorted(
        counts.items()
    ):
        ratio = (
            count
            / max(
                total,
                1,
            )
        )

        print(
            f"  {difficulty:12s}: "
            f"{count:5d} samples "
            f"({ratio * 100:6.2f}%)"
        )

    print(
        f"  {'TOTAL':12s}: "
        f"{total:5d} samples"
    )

    if len(counts) >= 2:

        nonzero_counts = [
            value
            for value in counts.values()
            if value > 0
        ]

        if nonzero_counts:

            imbalance_ratio = (
                max(nonzero_counts)
                / min(nonzero_counts)
            )

            print(
                f"  max/min ratio : "
                f"{imbalance_ratio:.2f}"
            )

    print("-" * 80)


# ============================================================
# Main
# ============================================================

def main():

    (
        rank,
        local_rank,
        world_size,
        is_dist,
    ) = get_ddp_info()

    is_main = (
        rank == 0
    )

    global_batch_size = (
        BATCH_SIZE_PER_DEVICE
        * world_size
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    if is_main:

        print(
            "=" * 80
        )

        print(
            "[v3 Stage 3] "
            "Residual-only continuous coordinate refinement"
        )

        print(
            f"PROJECT_ROOT = "
            f"{PROJECT_ROOT}"
        )

        print(
            f"Stage-2 checkpoint = "
            f"{STAGE2_CHECKPOINT}"
        )

        print(
            f"rank={rank}, "
            f"local_rank={local_rank}, "
            f"world_size={world_size}"
        )

        print(
            f"batch/device = "
            f"{BATCH_SIZE_PER_DEVICE}"
        )

        print(
            f"global batch = "
            f"{global_batch_size}"
        )

        print(
            f"Stage-3 epochs = "
            f"{NUM_EPOCHS}"
        )

        print(
            f"Residual LR = "
            f"{BASE_LR}"
        )

        print(
            "=" * 80
        )

    # --------------------------------------------------------
    # Check DDP
    # --------------------------------------------------------

    if world_size != 8 and is_main:

        print(
            "[Warning] This script is designed "
            "for 8-device training."
        )

        print(
            f"[Warning] Current world_size = "
            f"{world_size}"
        )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not STAGE2_CHECKPOINT.exists():

        raise FileNotFoundError(
            "Stage-2 checkpoint not found:\n"
            f"{STAGE2_CHECKPOINT}"
        )

    # ========================================================
    # Dataset manifests
    # ========================================================

    COMBINED_TRAIN = (
        PROJECT_ROOT
        / "cached_datasets"
        / "coarseflow_v6"
        / "combined_train_manifest.json"
    )

    COMBINED_VAL = (
        PROJECT_ROOT
        / "cached_datasets"
        / "coarseflow_v6"
        / "combined_val_manifest.json"
    )

    if is_main:

        print(
            "\n[Data] Combined train manifest:"
        )

        summarize_manifest(
            str(
                COMBINED_TRAIN
            )
        )

        summarize_difficulty_distribution(
            COMBINED_TRAIN
        )

        print(
            "\n[Data] Combined val manifest:"
        )

        summarize_manifest(
            str(
                COMBINED_VAL
            )
        )

        summarize_difficulty_distribution(
            COMBINED_VAL
        )

    # ========================================================
    # Model configuration
    #
    # IMPORTANT:
    #
    # Everything before the residual head is intentionally
    # identical to Stage 2.
    # ========================================================

    model_config = dict(

        # ----------------------------------------------------
        # Core matching
        # ----------------------------------------------------

        dim=96,

        radius=(
            4,
            3,
            3,
        ),

        temperature=0.05,

        coord_temperature=0.5,

        use_learned_matching=True,

        matcher_mode="hybrid",

        control_stride=16,

        encoder_stride=8,

        # Keep Stage-2 iterative matching
        num_refine_iters=2,

        query_chunk_size=512,

        query_downsample_mode="learned",

        # ----------------------------------------------------
        # Legacy encoder configuration
        #
        # Unused for encoder_type="swin3d".
        # Kept for checkpoint compatibility.
        # ----------------------------------------------------

        moving_base_channels=(
            24,
            48,
            96,
        ),

        moving_num_blocks=(
            2,
            4,
            4,
        ),

        moving_mlp_ratio=2.0,

        moving_window_attn_layers=6,
        moving_window_size=8,
        moving_attn_num_heads=4,

        moving_slice_fusion_blocks=1,

        ref_base_channels=(
            24,
            48,
            96,
        ),

        ref_num_blocks=(
            2,
            4,
            4,
        ),

        ref_refine_blocks=1,

        ref_mlp_ratio=2.0,

        ref_attn_layers=6,

        ref_attn_num_heads=4,

        ref_attn_window_size=(
            4,
            8,
            8,
        ),

        ref_attn_mlp_ratio=2.0,

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        use_coord_embed=True,

        use_spacing_embed=True,

        # ----------------------------------------------------
        # Local matcher
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Swin3D
        #
        # IDENTICAL to Stage 2
        # ----------------------------------------------------

        encoder_type="swin3d",

        swin_patch_size=(
            1,
            2,
            2,
        ),

        swin_embed_dim=24,

        swin_depths=(
            2,
            2,
            10,
        ),

        swin_num_heads=(
            3,
            6,
            12,
        ),

        moving_swin_window_sizes=(
            (2, 4, 4),
            (2, 4, 4),
            (2, 4, 4),
        ),

        ref_swin_window_sizes=(
            (2, 4, 4),
            (2, 4, 4),
            (4, 4, 4),
        ),

        swin_mlp_ratio=4.0,

        swin_qkv_bias=True,

        swin_drop_rate=0.0,

        swin_attn_drop_rate=0.0,

        swin_drop_path_rate=0.1,

        swin_patch_norm=True,

        swin_use_checkpoint=True,

        # Stage-2 multi-scale feature fusion
        swin_use_fusion=True,

        swin_fuse_dim=32,

        # ====================================================
        # Stage 3:
        # Continuous coordinate residual refinement
        # ====================================================

        use_coord_residual=True,

        # Spatial residual is preferred over independent MLP
        # because motion residuals are spatially correlated.
        residual_type="spatial",

        residual_hidden_dim=256,

        residual_num_blocks=5,

        # Maximum correction in RAW coordinate space:
        #
        #   dz <= 1.5
        #   dy <= 3.0
        #   dx <= 3.0
        #
        residual_max_delta=(
            1.5,
            3.0,
            3.0,
        ),

        residual_use_disp=True,

        residual_use_3d=True,

        # IMPORTANT:
        #
        # Keep coarse prediction fixed.
        residual_detach_coarse=True,

        # Do not propagate residual gradients into
        # Stage-2 feature extraction.
        residual_detach_features=True,
    )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    if is_main:

        print(
            "\n"
            + "=" * 80
        )

        print(
            "[Config] Stage-3 model configuration"
        )

        for key, value in model_config.items():

            print(
                f"  {key}: "
                f"{value}"
            )

        print(
            "=" * 80
        )

    # ========================================================
    # Train loader
    #
    # Eight-card DDP:
    #
    # each device:
    #     batch = 4
    #
    # total:
    #     batch = 32
    # ========================================================

    train_loader, _, _ = (
        build_sameShape_loader(

            str(
                COMBINED_TRAIN
            ),

            batch_size=(
                BATCH_SIZE_PER_DEVICE
            ),

            shuffle=True,

            num_workers=(
                NUM_WORKERS
            ),

            pin_memory=True,

            drop_last=False,

            verbose=is_main,

            distributed=is_dist,

            rank=rank,

            world_size=world_size,

            seed=SEED,

            pad_to_equal_batches=True,
        )
    )

    # ========================================================
    # Validation loader
    #
    # Validation is intentionally run only on rank 0,
    # consistent with the current training pipeline.
    # ========================================================

    if is_main:

        val_loader, _, _ = (
            build_sameShape_loader(

                str(
                    COMBINED_VAL
                ),

                batch_size=1,

                shuffle=False,

                num_workers=(
                    NUM_WORKERS
                ),

                pin_memory=True,

                drop_last=False,

                verbose=True,

                distributed=False,
            )
        )

    else:

        val_loader = None

    # ========================================================
    # Train Stage 3
    # ========================================================

    model = train_coarse_matching_model(

        train_dataset=None,

        val_dataset=None,

        train_loader=train_loader,

        val_loader=val_loader,

        # ----------------------------------------------------
        # New Stage-3 output directory
        # ----------------------------------------------------

        save_dir=(
            STAGE3_SAVE_DIR
        ),

        log_filename=(
            "train.log"
        ),

        log_mode="w",

        # ====================================================
        # Optimization
        # ====================================================

        # This is 120 ADDITIONAL epochs.
        num_epochs=(
            NUM_EPOCHS
        ),

        # Residual head starts from zero initialization.
        # A new 1e-4 LR is appropriate because the entire
        # coarse network is frozen.
        lr=(
            BASE_LR
        ),

        weight_decay=(
            WEIGHT_DECAY
        ),

        batch_size=(
            BATCH_SIZE_PER_DEVICE
        ),

        num_workers=(
            NUM_WORKERS
        ),

        # Keep numerics consistent with previous stages.
        # Can be changed to True later if AMP is verified.
        use_amp=False,

        # ====================================================
        # LR scheduler
        #
        # local Stage-3 epoch:
        #
        #   1-20:
        #       1e-4
        #
        #   21-120:
        #       cosine -> 1e-6
        # ====================================================

        use_cosine_lr=True,

        cosine_start_epoch=(
            COSINE_START_EPOCH
        ),

        cosine_eta_min=(
            COSINE_ETA_MIN
        ),

        # ====================================================
        # Model
        # ====================================================

        **model_config,

        # ====================================================
        # Stage-3 objective
        #
        # ONLY optimize continuous coordinate accuracy.
        #
        # x_final =
        #     x_coarse + residual_delta
        #
        # L =
        #     |x_final - x_gt|_1
        # ====================================================

        loss_mode="coord",

        # ----------------------------------------------------
        # Disable matcher supervision
        #
        # Matcher is frozen and its job has already been
        # completed by Stage 1 + Stage 2.
        # ----------------------------------------------------

        lambda_match=0.0,

        lambda_match_kl=0.0,

        lambda_match_ce=0.0,

        # With loss_mode="coord", coord loss is the main
        # objective. Keep this at 1 for clarity.
        lambda_coord=1.0,

        # ----------------------------------------------------
        # No duplicated displacement supervision
        # ----------------------------------------------------

        lambda_disp=0.0,

        # ----------------------------------------------------
        # First residual experiment should be as clean as
        # possible: optimize coordinate precision only.
        #
        # Smoothness can be introduced later if residual
        # fields are visibly noisy.
        # ----------------------------------------------------

        lambda_smooth=0.0,

        lambda_z_spacing=0.0,

        lambda_disp_mag=0.0,

        # ----------------------------------------------------
        # Match loss no longer needed
        # ----------------------------------------------------

        compute_chunk_match_loss=False,

        # These arguments are unused when matching loss is
        # disabled, but kept explicit for configuration
        # consistency.
        match_sigma=(
            0.4,
            0.6,
            0.6,
        ),

        match_inside_threshold=4.0,

        # ====================================================
        # Residual-only training
        #
        # train.py will freeze all parameters except:
        #
        #     coord_residual_refiner.*
        #
        # ====================================================

        train_only_residual=True,

        freeze_encoder=False,

        # train_only_residual=True already uses lr above.
        residual_lr=None,

        # ====================================================
        # Load Stage-2 model
        # ====================================================

        resume_path=str(
            STAGE2_CHECKPOINT
        ),

        # New optimizer for new residual head.
        resume_optimizer=False,

        # Stage-2 loss and Stage-3 loss are different,
        # therefore do NOT reuse previous best_val_loss.
        resume_best_val_loss=False,

        # IMPORTANT:
        #
        # Stage-2 checkpoint does not contain:
        #
        #     coord_residual_refiner.*
        #
        # because residual was disabled.
        #
        # Therefore strict loading MUST be False.
        strict_load=False,
    )

    if is_main:

        print(
            "=" * 80
        )

        print(
            "[Done] v3 Stage-3 "
            "residual training finished."
        )

        print(
            f"Outputs saved to: "
            f"{STAGE3_SAVE_DIR}"
        )

        print(
            "=" * 80
        )


if __name__ == "__main__":
    main()