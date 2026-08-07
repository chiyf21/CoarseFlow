"""Smoke test: CoarseMatchingNet with Swin3D encoders (small-scale, end-to-end).

Tests:
- CoarseMatchingNet forward with encoder_type="swin3d"
- pred_coords, pred_disp, coords0 shapes are correct
- All outputs are finite
- Loss backward works
- Moving and reference encoders receive gradients
- Legacy mode still instantiates and forward-passes

This test uses random data and checks functional correctness only.
It does NOT require the model to predict identity motion.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

# -------------------------------------------------------------------
# 1. Swin3D mode — forward + backward
# -------------------------------------------------------------------
print("=" * 60)
print("[1/4] Swin3D mode: forward + backward")
print("=" * 60)

from models.SparseGMFlow3D import CoarseMatchingNet

model = CoarseMatchingNet(
    dim=96,
    radius=(1, 1, 1),  # small radius for small test shapes
    encoder_stride=8,

    encoder_type="swin3d",
    swin_patch_size=(1, 2, 2),
    swin_embed_dim=24,
    swin_depths=(2, 2, 6),
    swin_num_heads=(3, 6, 12),
    moving_swin_window_sizes=((2, 4, 4), (2, 4, 4), (2, 4, 4)),
    ref_swin_window_sizes=((2, 4, 4), (2, 4, 4), (4, 4, 4)),
    swin_mlp_ratio=4.0,
    swin_qkv_bias=True,
    swin_drop_rate=0.0,
    swin_attn_drop_rate=0.0,
    swin_drop_path_rate=0.1,
    swin_patch_norm=True,
    swin_use_checkpoint=False,

    # Keep the matcher simple
    use_learned_matching=True,
    matcher_mode="hybrid",
    use_coord_embed=True,
    use_spacing_embed=False,
    use_offset_encoding=False,
    use_offset_bias=False,
    use_local_cross_attn=False,
    num_refine_iters=1,
    query_chunk_size=512,
    use_coord_residual=False,
)

print(f"encoder_type = {model.encoder_type}")
print(f"moving_encoder.xy_stride = {model.moving_encoder.xy_stride}")
print(f"reference_encoder.xy_stride = {model.reference_encoder.xy_stride}")
print(f"moving_encoder.z_stride = {model.moving_encoder.z_stride}")
print(f"reference_encoder.z_stride = {model.reference_encoder.z_stride}")

total = sum(p.numel() for p in model.parameters())
moving_params = sum(p.numel() for p in model.moving_encoder.parameters())
ref_params = sum(p.numel() for p in model.reference_encoder.parameters())
print(f"Total parameters: {total / 1e6:.3f} M")
print(f"Moving encoder params: {moving_params / 1e6:.3f} M")
print(f"Reference encoder params: {ref_params / 1e6:.3f} M")

# Small test shapes: mov=(1,1,3,33,35), ref=(1,1,7,33,35)
mov = torch.randn(1, 1, 3, 33, 35)
ref = torch.randn(1, 1, 7, 33, 35)
z_init = torch.tensor([1.0, 3.0, 5.0])  # (K,)

model.train()
outputs = model(mov, ref, z_init=z_init)

print(f"\npred_coords shape: {outputs['pred_coords'].shape}")  # (1,3,Hc,Wc,3)
print(f"pred_disp shape:   {outputs['pred_disp'].shape}")
print(f"coords0 shape:     {outputs['coords0'].shape}")

assert outputs['pred_coords'].shape[0] == 1
assert outputs['pred_coords'].shape[1] == 3  # K
assert outputs['pred_disp'].shape[1] == 3
assert outputs['coords0'].shape[1] == 3
print("✓ Shape assertions passed")

assert torch.isfinite(outputs['pred_coords']).all(), "pred_coords has NaN/Inf"
assert torch.isfinite(outputs['pred_disp']).all(), "pred_disp has NaN/Inf"
assert torch.isfinite(outputs['coords0']).all(), "coords0 has NaN/Inf"
print("✓ Finiteness assertions passed")

# Backward test
loss = outputs['pred_disp'].abs().sum()
loss.backward()

# Check encoder gradients
mov_grad = any(
    p.grad is not None and p.grad.abs().sum() > 0
    for p in model.moving_encoder.parameters()
)
ref_grad = any(
    p.grad is not None and p.grad.abs().sum() > 0
    for p in model.reference_encoder.parameters()
)
print(f"Moving encoder has gradients: {mov_grad}")
print(f"Reference encoder has gradients: {ref_grad}")
assert mov_grad, "Moving encoder has no gradients!"
assert ref_grad, "Reference encoder has no gradients!"
print("✓ Gradient flow assertions passed")

# -------------------------------------------------------------------
# 2. Chunked match loss
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("[2/4] Swin3D mode: chunked match loss")
print("=" * 60)

model.train()
model.zero_grad()
outputs = model(
    mov, ref, z_init=z_init,
    compute_chunk_match_loss=True,
    gt_coords=outputs['coords0'].clone() + 0.3,  # small offset
    match_sigma=(0.5, 0.75, 0.75),
    match_inside_threshold=4.0,
)
loss_keys = [k for k in outputs.keys() if k.startswith("loss_")]
print(f"Loss keys: {loss_keys}")
for k in loss_keys:
    print(f"  {k}: {outputs[k].item():.4f}")

# -------------------------------------------------------------------
# 3. Eval mode consistency
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("[3/4] Swin3D mode: eval consistency")
print("=" * 60)

model.eval()
with torch.no_grad():
    out1 = model(mov, ref, z_init=z_init)['pred_coords'].clone()
    out2 = model(mov, ref, z_init=z_init)['pred_coords'].clone()

assert torch.allclose(out1, out2, atol=1e-5), "Eval outputs not consistent!"
print("✓ Eval consistency passed")

# -------------------------------------------------------------------
# 4. Legacy mode still works
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("[4/4] Legacy mode: instantiate + forward")
print("=" * 60)

legacy_model = CoarseMatchingNet(
    dim=96,
    encoder_type="legacy",
    radius=(1, 1, 1),
    encoder_stride=8,
    num_refine_iters=1,
    query_chunk_size=512,
    moving_window_attn_layers=0,  # no window attn for speed
    ref_attn_layers=0,
    use_learned_matching=True,
    matcher_mode="hybrid",
    use_coord_embed=True,
    use_spacing_embed=False,
    use_coord_residual=False,
)
legacy_model.eval()
with torch.no_grad():
    legacy_out = legacy_model(mov, ref, z_init=z_init)
print(f"Legacy pred_coords shape: {legacy_out['pred_coords'].shape}")
assert torch.isfinite(legacy_out['pred_coords']).all()
print("✓ Legacy mode passed")

# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
