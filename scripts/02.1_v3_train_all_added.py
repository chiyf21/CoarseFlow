# v3 Stage 2:
# Iterative local matching + sharper correspondence fine-tuning
#
# Initialization:
#   Stage 1 best checkpoint
#
# Main changes from Stage 1:
#   1. num_refine_iters: 1 -> 2
#   2. KL / CE: 0.25 / 0.75 -> 0.10 / 0.90
#   3. lambda_coord: 0.40 -> 0.50
#   4. LR: 1e-4 -> 1e-5
#   5. 120 additional epochs
#   6. reset optimizer
#
# Data:
#   Same combined train / val dataset
#
# Residual coordinate refinement remains OFF.

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
# Stage 2 training configuration
# ============================================================

# Additional epochs after loading Stage 1 best checkpoint
NUM_EPOCHS = 300

# Smaller LR for fine-tuning
BASE_LR = 1e-5
WEIGHT_DECAY = 1e-4

# Hold LR at 1e-5 for the first 20 Stage-2 epochs,
# then cosine decay toward 1e-6.
COSINE_START_EPOCH = 20
COSINE_ETA_MIN = 1e-6

BATCH_SIZE_PER_DEVICE = 4
NUM_WORKERS = 4

SEED = 1234


# ============================================================
# Checkpoint
# ============================================================

STAGE1_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "coarseflow_swin3d_v3_all"
    / "best.pth"
)

STAGE2_SAVE_DIR = (
    "checkpoints/"
    "coarseflow_swin3d_v3_refine2_sharpen"
)


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
    Print sample count for each difficulty group.

    This is only a diagnostic.
    It does not rebalance the training dataset.
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

            if imbalance_ratio > 2.0:
                print(
                    "  [Warning] Difficulty groups are noticeably "
                    "imbalanced."
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
            "[v3 Stage 2] "
            "Iterative local matching + sharper fine-tuning"
        )

        print(
            f"PROJECT_ROOT = {PROJECT_ROOT}"
        )

        print(
            f"Stage 1 checkpoint = "
            f"{STAGE1_CHECKPOINT}"
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

        print("=" * 80)

    # --------------------------------------------------------
    # Check Stage 1 checkpoint
    # --------------------------------------------------------

    if not STAGE1_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found:\n"
            f"{STAGE1_CHECKPOINT}"
        )

    # ========================================================
    # Data
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

        # Dot-product similarity temperature
        temperature=0.05,

        # Coordinate expectation / final score distribution
        coord_temperature=0.5,

        use_learned_matching=True,

        matcher_mode="hybrid",

        # Raw XY control-point spacing
        control_stride=16,

        # Swin encoder XY output stride
        encoder_stride=8,

        # ----------------------------------------------------
        # STAGE 2 CHANGE:
        #
        # One local match:
        #     initial -> prediction
        #
        # becomes:
        #
        #     initial
        #        -> local match
        #        -> refined center
        #        -> second local match
        #        -> final prediction
        # ----------------------------------------------------

        num_refine_iters=2,

        query_chunk_size=512,

        # Learned /8 -> /16 query downsampling
        query_downsample_mode="learned",

        # ----------------------------------------------------
        # Legacy encoder arguments
        #
        # Unused with encoder_type="swin3d".
        # Kept for config / checkpoint compatibility.
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
        #
        # IMPORTANT:
        # Stage 2 does NOT increase backbone capacity.
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

        # Multi-scale /2 + /4 + /8 fusion
        swin_use_fusion=True,
        swin_fuse_dim=32,

        # ----------------------------------------------------
        # Residual coordinate refinement
        #
        # Still OFF in Stage 2.
        #
        # We first test whether iterative local matching alone
        # can improve Top-1 and coordinate accuracy.
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
    # Print model configuration
    # --------------------------------------------------------

    if is_main:

        print("\n" + "=" * 80)
        print("[Config] Stage 2 model configuration")

        for key, value in model_config.items():
            print(
                f"  {key}: {value}"
            )

        print("=" * 80)

    # ========================================================
    # Data loaders
    #
    # IMPORTANT:
    # Same dataset as Stage 1.
    # No easy -> hard curriculum.
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

    # Validation only on rank 0
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
    # Train Stage 2
    # ========================================================

    model = train_coarse_matching_model(

        train_dataset=None,
        val_dataset=None,

        train_loader=train_loader,
        val_loader=val_loader,

        # ----------------------------------------------------
        # Separate Stage-2 directory
        # ----------------------------------------------------

        save_dir=STAGE2_SAVE_DIR,

        log_filename="train.log",
        log_mode="w",

        # ====================================================
        # Optimization
        # ====================================================

        # IMPORTANT:
        # With the current train.py resume semantics,
        # this means 120 ADDITIONAL epochs.
        num_epochs=NUM_EPOCHS,

        # 10x smaller than Stage 1
        lr=BASE_LR,

        weight_decay=WEIGHT_DECAY,

        batch_size=(
            BATCH_SIZE_PER_DEVICE
        ),

        num_workers=NUM_WORKERS,

        use_amp=False,

        # ====================================================
        # LR schedule
        #
        # Stage-2 local epoch 1-20:
        #     1e-5
        #
        # Remaining:
        #     cosine 1e-5 -> 1e-6
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
        # Stage 2 loss
        #
        # Stage 1:
        #   KL    = 0.25
        #   CE    = 0.75
        #   coord = 0.40
        #
        # Stage 2:
        #   KL    = 0.10
        #   CE    = 0.90
        #   coord = 0.50
        #
        # Goal:
        #   emphasize exact candidate discrimination
        #   and final coordinate localization.
        # ====================================================

        loss_mode="match",

        lambda_match=1.0,

        lambda_match_kl=0.10,
        lambda_match_ce=0.90,

        lambda_coord=0.50,

        # Do not duplicate coordinate L1 using displacement
        lambda_disp=0.0,

        # Keep weak regularization unchanged
        lambda_smooth=0.005,
        lambda_z_spacing=0.005,

        # Keep displacement magnitude supervision
        lambda_disp_mag=0.10,

        # ====================================================
        # Local matching supervision
        #
        # Keep these unchanged to isolate the effect of:
        #   - 2 matching iterations
        #   - sharper loss
        # ====================================================

        compute_chunk_match_loss=True,

        match_sigma=(
            0.4,
            0.6,
            0.6,
        ),

        match_inside_threshold=4.0,

        # ====================================================
        # Resume
        # ====================================================

        # Load Stage 1 best MODEL WEIGHTS
        resume_path=str(
            STAGE1_CHECKPOINT
        ),

        # IMPORTANT:
        #
        # Do NOT restore Stage-1 AdamW state.
        #
        # Stage 2 has:
        #   - new LR
        #   - different loss weighting
        #   - two matching passes
        resume_optimizer=False,

        # Stage 1 and Stage 2 total losses are not directly
        # comparable because the loss weights changed.
        resume_best_val_loss=False,

        # num_refine_iters does not create new parameters,
        # so model state should match exactly.
        strict_load=True,
    )

    if is_main:

        print("=" * 80)
        print(
            "[Done] v3 Stage 2 training finished."
        )
        print(
            f"Outputs saved to: "
            f"{STAGE2_SAVE_DIR}"
        )
        print("=" * 80)


if __name__ == "__main__":
    main()