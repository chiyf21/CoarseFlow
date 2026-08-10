# v3: Single-stage joint training on ALL combined data
#
# Design:
#   - ALL train data participates from epoch 1
#   - fixed loss objective for the whole run
#   - LR: 1e-4 for first 300 epochs, then cosine -> 1e-6
#   - anisotropic Swin3D + residual multi-scale fusion
#   - learned /8 -> /16 moving-query downsampling
#
# Dataset:
#   3455 train / 1000 val

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
# Global training configuration
# ============================================================

NUM_EPOCHS = 600

BASE_LR = 1e-4
WEIGHT_DECAY = 1e-4

COSINE_START_EPOCH = 300
COSINE_ETA_MIN = 1e-6

BATCH_SIZE_PER_DEVICE = 4
NUM_WORKERS = 4

SEED = 1234


# ============================================================
# DDP
# ============================================================

def get_ddp_info():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    is_dist = world_size > 1

    return rank, local_rank, world_size, is_dist


# ============================================================
# Manifest diagnostics
# ============================================================

def summarize_difficulty_distribution(manifest_path):
    """
    Print actual sample count for each difficulty group.

    This does NOT rebalance the dataset.
    It only verifies how much each difficulty contributes
    to the merged training / validation set.
    """

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    counts = Counter()

    for part in manifest["parts"]:
        difficulty = str(
            part.get("difficulty", "unknown")
        )

        num_samples = int(
            part.get(
                "num_samples",
                part.get("length", 0),
            )
        )

        counts[difficulty] += num_samples

    total = sum(counts.values())

    print("-" * 80)
    print("[Difficulty distribution]")

    for difficulty, count in sorted(counts.items()):
        ratio = count / max(total, 1)

        print(
            f"  {difficulty:12s}: "
            f"{count:5d} samples "
            f"({ratio * 100:6.2f}%)"
        )

    print(f"  {'TOTAL':12s}: {total:5d} samples")

    if len(counts) >= 2:
        nonzero_counts = [
            v for v in counts.values()
            if v > 0
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

            if imbalance_ratio > 2.0:
                print(
                    "  [Warning] Difficulty groups are noticeably "
                    "imbalanced. Consider difficulty-aware sampling "
                    "if the minority group is scientifically important."
                )

    print("-" * 80)


# ============================================================
# Main
# ============================================================

def main():

    rank, local_rank, world_size, is_dist = (
        get_ddp_info()
    )

    is_main = rank == 0

    global_batch_size = (
        BATCH_SIZE_PER_DEVICE
        * world_size
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    if is_main:
        print("=" * 80)
        print(
            "[v3] Single-stage joint training "
            "— ALL combined data"
        )

        print(f"PROJECT_ROOT = {PROJECT_ROOT}")

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

        print("=" * 80)

    # ========================================================
    # Data
    # ========================================================

    COMBINED_TRAIN = (
        PROJECT_ROOT
        / "cached_datasets/coarseflow_v6/"
          "combined_train_manifest.json"
    )

    COMBINED_VAL = (
        PROJECT_ROOT
        / "cached_datasets/coarseflow_v6/"
          "combined_val_manifest.json"
    )

    if is_main:

        print(
            "\n[Data] Combined train manifest:"
        )

        summarize_manifest(
            str(COMBINED_TRAIN)
        )

        summarize_difficulty_distribution(
            COMBINED_TRAIN
        )

        print(
            "\n[Data] Combined val manifest:"
        )

        summarize_manifest(
            str(COMBINED_VAL)
        )

        summarize_difficulty_distribution(
            COMBINED_VAL
        )

    # ========================================================
    # Model configuration
    # ========================================================

    model_config = dict(

        # ----------------------------------------------------
        # Core matching configuration
        # ----------------------------------------------------

        dim=96,

        radius=(4, 3, 3),

        # Dot-product score temperature
        temperature=0.05,

        # Final matching-distribution temperature
        coord_temperature=0.5,

        use_learned_matching=True,
        matcher_mode="hybrid",

        # Raw control points every 16 XY pixels
        control_stride=16,

        # Swin encoder output at XY /8
        encoder_stride=8,

        num_refine_iters=1,

        query_chunk_size=512,

        # IMPORTANT:
        # learned /8 -> /16 instead of adaptive average pooling
        query_downsample_mode="learned",

        # ----------------------------------------------------
        # Legacy encoder arguments
        #
        # Unused when encoder_type="swin3d".
        # Kept for checkpoint/config compatibility.
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
        # Swin3D encoder
        # ----------------------------------------------------

        encoder_type="swin3d",

        # Preserve Z resolution;
        # XY starts with /2 patch embedding.
        swin_patch_size=(
            1,
            2,
            2,
        ),

        swin_embed_dim=24,

        # Enlarged stage-3 capacity
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

        # Save activation memory
        swin_use_checkpoint=True,

        # Residual multi-scale feature fusion
        swin_use_fusion=True,
        swin_fuse_dim=32,

        # ----------------------------------------------------
        # Coordinate residual head
        #
        # OFF in unified coarse training.
        # ----------------------------------------------------

        use_coord_residual=False,

        residual_type="spatial",

        residual_hidden_dim=256,
        residual_num_blocks=5,

        residual_max_delta=(
            1.5,
            3.0,
            3.0,
        ),

        residual_use_disp=True,
        residual_use_3d=True,

        residual_detach_coarse=True,
        residual_detach_features=True,
    )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    if is_main:

        print("\n" + "=" * 80)
        print("[Config] Model configuration")

        for key, value in model_config.items():
            print(
                f"  {key}: {value}"
            )

        print("=" * 80)

    # ========================================================
    # Data loaders
    # ========================================================

    train_loader, _, _ = (
        build_sameShape_loader(
            str(COMBINED_TRAIN),

            batch_size=(
                BATCH_SIZE_PER_DEVICE
            ),

            shuffle=True,

            num_workers=NUM_WORKERS,

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

    # Validation only runs on rank 0.
    if is_main:

        val_loader, _, _ = (
            build_sameShape_loader(
                str(COMBINED_VAL),

                batch_size=1,

                shuffle=False,

                num_workers=NUM_WORKERS,

                pin_memory=True,

                drop_last=False,

                verbose=True,

                distributed=False,
            )
        )

    else:
        val_loader = None

    # ========================================================
    # Train
    # ========================================================

    model = train_coarse_matching_model(

        train_dataset=None,
        val_dataset=None,

        train_loader=train_loader,
        val_loader=val_loader,

        save_dir=(
            "checkpoints/"
            "coarseflow_swin3d_v3_all"
        ),

        log_filename="train.log",
        log_mode="w",

        # ----------------------------------------------------
        # Optimization
        # ----------------------------------------------------

        num_epochs=NUM_EPOCHS,

        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,

        batch_size=(
            BATCH_SIZE_PER_DEVICE
        ),

        num_workers=NUM_WORKERS,

        use_amp=False,

        # ----------------------------------------------------
        # LR schedule
        #
        # Epoch   1-300: 1e-4
        # Epoch 301-600: cosine 1e-4 -> 1e-6
        # ----------------------------------------------------

        use_cosine_lr=True,

        cosine_start_epoch=(
            COSINE_START_EPOCH
        ),

        cosine_eta_min=(
            COSINE_ETA_MIN
        ),

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        **model_config,

        # ----------------------------------------------------
        # Loss
        #
        # Fixed objective for the entire run.
        #
        # Slightly sharper than the previous
        # 0.3 KL / 0.7 CE / 0.3 coord setting.
        # ----------------------------------------------------

        loss_mode="match",

        lambda_match=1.0,

        # Match distribution
        lambda_match_kl=0.25,
        lambda_match_ce=0.75,

        # Continuous coordinate accuracy
        lambda_coord=0.40,

        # Do not duplicate coordinate supervision
        lambda_disp=0.0,

        # Weak spatial regularization only
        lambda_smooth=0.005,
        lambda_z_spacing=0.005,

        # Prevent systematic displacement
        # magnitude under-estimation
        lambda_disp_mag=0.10,

        # ----------------------------------------------------
        # Chunked local match supervision
        # ----------------------------------------------------

        compute_chunk_match_loss=True,

        match_sigma=(
            0.4,
            0.6,
            0.6,
        ),

        match_inside_threshold=4.0,

        # ----------------------------------------------------
        # Fresh training
        # ----------------------------------------------------

        resume_path=None,

        resume_optimizer=False,
        resume_best_val_loss=False,

        strict_load=False,
    )

    if is_main:
        print(
            "[Done] v3 training finished."
        )


if __name__ == "__main__":
    main()