# training/staged_train.py

import os
import time
from dataclasses import dataclass
from collections import deque

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from models.CoarseNet_v2 import CoarseMatchingNetV2
from datasets.synthetic_dataset import SparseZStackSyntheticDataset,CachedDataset
from training.losses import total_coarse_loss



@dataclass
class TrainStage:
    name: str

    # dataset difficulty
    num_samples_per_volume: int
    num_sparse_slices: int
    amp_xy: float
    art_R_xy: float
    noise_std: float
    use_cached: bool
    num_cached: int = 100

    # training
    epochs: int = 20
    batch_size: int = 1
    lr: float = 1e-4

    # loss weights
    lambda_smooth: float = 0.0
    lambda_z_spacing: float = 0.0

def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            v = v.to(device, non_blocking=True)
            if k in ["mov", "ref", "spacing", "gt_disp", "gt_coords"]:
                v = v.float()
            out[k] = v
        else:
            out[k] = v
    return out


def save_checkpoint(save_path, model, optimizer, stage_name, epoch, global_step):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stage_name": stage_name,
            "epoch": epoch,
            "global_step": global_step,
        },
        save_path,
    )


# =========================================================
# 4. 构建某一阶段 Dataset
# =========================================================
def build_stage_dataset(
    stage,
    volumes,
    motion_fn,
    warp_fn,
    ref_spacing,
    mov_spacing,
    control_stride,
    zRatio,
    gt_direction="mov_to_ref",
    normalize=True,
):
    base_dataset = SparseZStackSyntheticDataset(
        volumes=volumes,
        motion_fn=motion_fn,
        warp_fn=warp_fn,
        num_samples_per_volume=stage.num_samples_per_volume,
        num_sparse_slices=stage.num_sparse_slices,
        control_stride=control_stride,
        ref_spacing=ref_spacing,
        mov_spacing=mov_spacing,
        motion_kwargs=dict(
            art_R_xy=stage.art_R_xy,
            amp_xy=stage.amp_xy,
            zRatio=zRatio,
            use_incompressibility=True,
        ),
        noise_std=stage.noise_std,
        normalize=normalize,
        gt_direction=gt_direction,
    )

    if stage.use_cached:
        dataset = CachedDataset(
            base_dataset=base_dataset,
            num_cached=stage.num_cached,
        )
    else:
        dataset = base_dataset

    return dataset


# =========================================================
# 5. 训练一个阶段
# =========================================================
def train_one_stage(
    model,
    optimizer,
    scaler,
    dataset,
    stage,
    device,
    save_dir,
    global_step=0,
    use_amp=True,
    num_workers=0,
    log_interval=20,
    grad_clip=1.0,
):
    loader = DataLoader(
        dataset,
        batch_size=stage.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    loss_ma = deque(maxlen=100)
    disp_ma = deque(maxlen=100)

    print("\n" + "=" * 80)
    print(f"Start stage: {stage.name}")
    print(f"dataset length: {len(dataset)}")
    print(f"batch size: {stage.batch_size}")
    print(f"steps per epoch: {len(loader)}")
    print(
        f"amp_xy={stage.amp_xy}, art_R_xy={stage.art_R_xy}, "
        f"noise={stage.noise_std}, K={stage.num_sparse_slices}, "
        f"cached={stage.use_cached}"
    )
    print(
        f"lambda_smooth={stage.lambda_smooth}, "
        f"lambda_z_spacing={stage.lambda_z_spacing}"
    )
    print("=" * 80)

    model.train()

    for epoch in range(1, stage.epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)

            mov = batch["mov"]
            ref = batch["ref"]
            spacing = batch["spacing"]
            gt_disp = batch["gt_disp"]
            sparse_z_idx = batch["sparse_z_idx"]

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                pred_disp = model(mov, ref, spacing)

                loss, loss_dict = total_coarse_loss(
                    pred_disp=pred_disp,
                    gt_disp=gt_disp,
                    sparse_z_idx=sparse_z_idx,
                    spacing=spacing,
                    lambda_smooth=stage.lambda_smooth,
                    lambda_z_spacing=stage.lambda_z_spacing,
                )

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            epoch_loss += loss.item()

            loss_ma.append(loss.item())
            disp_ma.append(loss_dict["loss_disp"].item())

            if global_step % log_interval == 0:
                smooth_raw = loss_dict["loss_smooth"].item()
                z_raw = loss_dict["loss_z_spacing"].item()

                smooth_w = stage.lambda_smooth * smooth_raw
                z_w = stage.lambda_z_spacing * z_raw

                print(
                    f"[{stage.name}] "
                    f"epoch {epoch:03d}/{stage.epochs} "
                    f"step {step:04d}/{len(loader)} "
                    f"global {global_step:06d} | "
                    f"loss={loss.item():.4f} "
                    f"disp={loss_dict['loss_disp'].item():.4f} "
                    f"smooth={smooth_raw:.4f}(w={smooth_w:.4f}) "
                    f"z={z_raw:.4f}(w={z_w:.4f}) | "
                    f"ma100_loss={sum(loss_ma)/len(loss_ma):.4f} "
                    f"ma100_disp={sum(disp_ma)/len(disp_ma):.4f}"
                )

        avg_epoch_loss = epoch_loss / max(len(loader), 1)

        print(
            f"[{stage.name}] epoch {epoch:03d} done | "
            f"avg_loss={avg_epoch_loss:.4f} | "
            f"time={time.time() - t0:.1f}s"
        )

        save_checkpoint(
            save_path=os.path.join(save_dir, f"{stage.name}_latest.pth"),
            model=model,
            optimizer=optimizer,
            stage_name=stage.name,
            epoch=epoch,
            global_step=global_step,
        )

    return global_step


# =========================================================
# 6. 分阶段训练主函数
# =========================================================
def staged_train(
    volumes,
    motion_fn,
    warp_fn,
    save_dir="./checkpoints/coarse_matching_staged",
    ref_spacing=(1.0, 1.0, 1.0),
    mov_spacing=(1.0, 1.0, 5.0),
    control_stride=16,
    stage1_epoch = 100,
    stage2_epoch = 100,
    stage3_epoch = 100,
    stage4_epoch = 100,
    zRatio=5,
    use_vit=False,
    use_amp=True,
    device=None,
    num_workers=0,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)

    model = CoarseMatchingNetV2(use_vit=use_vit).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    scaler = GradScaler(enabled=use_amp)

    # -------------------------
    # 训练阶段设计
    # -------------------------
    stages = [
        # Stage 1: 固定小数据集，简单 motion，无正则
        TrainStage(
            name="stage1_cached_easy",
            num_samples_per_volume=200,
            num_sparse_slices=16,
            amp_xy=2.0,
            art_R_xy=160,
            noise_std=0.0,
            use_cached=True,
            num_cached=100,
            epochs=stage1_epoch,
            batch_size=1,
            lr=1e-4,
            lambda_smooth=0.0,
            lambda_z_spacing=0.0,
        ),

        # Stage 2: 固定小数据集，中等 motion，加轻微正则
        TrainStage(
            name="stage2_cached_medium",
            num_samples_per_volume=300,
            num_sparse_slices=12,
            amp_xy=4.0,
            art_R_xy=120,
            noise_std=0.01,
            use_cached=True,
            num_cached=150,
            epochs=40,
            batch_size=1,
            lr=1e-4,
            lambda_smooth=0.001,
            lambda_z_spacing=0.001,
        ),

        # Stage 3: online 简单随机训练
        TrainStage(
            name="stage3_online_medium",
            num_samples_per_volume=100,
            num_sparse_slices=12,
            amp_xy=4.0,
            art_R_xy=120,
            noise_std=0.01,
            use_cached=False,
            epochs=80,
            batch_size=1,
            lr=1e-4,
            lambda_smooth=0.001,
            lambda_z_spacing=0.001,
        ),

        # Stage 4: online 较难训练
        TrainStage(
            name="stage4_online_hard",
            num_samples_per_volume=100,
            num_sparse_slices=8,
            amp_xy=8.0,
            art_R_xy=80,
            noise_std=0.05,
            use_cached=False,
            epochs=100,
            batch_size=1,
            lr=5e-5,
            lambda_smooth=0.003,
            lambda_z_spacing=0.001,
        ),
    ]

    global_step = 0

    for stage in stages:
        # 每个阶段可以换学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = stage.lr

        dataset = build_stage_dataset(
            stage=stage,
            volumes=volumes,
            motion_fn=motion_fn,
            warp_fn=warp_fn,
            ref_spacing=ref_spacing,
            mov_spacing=mov_spacing,
            control_stride=control_stride,
            zRatio=zRatio,
        )

        global_step = train_one_stage(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            dataset=dataset,
            stage=stage,
            device=device,
            save_dir=save_dir,
            global_step=global_step,
            use_amp=use_amp,
            num_workers=num_workers,
        )

        save_checkpoint(
            save_path=os.path.join(save_dir, f"{stage.name}_final.pth"),
            model=model,
            optimizer=optimizer,
            stage_name=stage.name,
            epoch=stage.epochs,
            global_step=global_step,
        )

    save_checkpoint(
        save_path=os.path.join(save_dir, "final.pth"),
        model=model,
        optimizer=optimizer,
        stage_name="final",
        epoch=-1,
        global_step=global_step,
    )

    return model