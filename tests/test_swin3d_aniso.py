"""Unit tests for models/swin3d_aniso.py.

These tests use pure PyTorch (cpu), no torch_npu required.
"""
import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.swin3d_aniso import (
    DropPath,
    Mlp,
    window_partition_3d,
    window_reverse_3d,
    WindowAttention3D,
    SwinTransformerBlock3D,
    PatchEmbed3DAniso,
    PatchMerging3DXY,
    BasicLayer3D,
    AnisotropicSwinEncoder3D,
    MovingQuerySwinEncoder,
    ReferenceMemorySwinEncoder,
)


class TestWindowPartition(unittest.TestCase):
    """window_partition_3d / window_reverse_3d round-trip."""

    def test_round_trip(self):
        B, Z, H, W, C = 1, 6, 32, 32, 24
        window_size = (2, 4, 4)
        x = torch.randn(B, Z, H, W, C)

        windows = window_partition_3d(x, window_size)
        self.assertEqual(windows.shape, (1 * 3 * 8 * 8, 2, 4, 4, C))

        recovered = window_reverse_3d(windows, window_size, B, Z, H, W)
        self.assertEqual(recovered.shape, x.shape)
        self.assertTrue(torch.allclose(x, recovered, atol=1e-5))


class TestRelativePositionIndex(unittest.TestCase):
    """Verify relative_position_index shape and max index."""

    def test_shape_and_bounds(self):
        Wz, Wy, Wx = 2, 4, 4
        attn = WindowAttention3D(dim=24, window_size=(Wz, Wy, Wx), num_heads=3)

        idx = attn.relative_position_index
        N = Wz * Wy * Wx
        self.assertEqual(idx.shape, (N, N))

        max_idx = (2 * Wz - 1) * (2 * Wy - 1) * (2 * Wx - 1) - 1
        self.assertLessEqual(idx.max().item(), max_idx)
        self.assertGreaterEqual(idx.min().item(), 0)

        table_shape = attn.relative_position_bias_table.shape
        self.assertEqual(table_shape, ((2 * Wz - 1) * (2 * Wy - 1) * (2 * Wx - 1), 3))


class TestSwinBlockNormal(unittest.TestCase):
    """Normal (non-shifted) window block preserves shape."""

    def test_normal_window_shape(self):
        B, Z, H, W, C = 1, 4, 16, 16, 24
        x = torch.randn(B, Z, H, W, C)
        block = SwinTransformerBlock3D(
            dim=C, num_heads=3, window_size=(2, 4, 4),
            shift_size=(0, 0, 0),
        )
        out = block(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())


class TestSwinBlockShifted(unittest.TestCase):
    """Shifted window block preserves shape."""

    def test_shifted_window_shape(self):
        B, Z, H, W, C = 1, 4, 16, 16, 24
        x = torch.randn(B, Z, H, W, C)
        block = SwinTransformerBlock3D(
            dim=C, num_heads=3, window_size=(2, 4, 4),
            shift_size=(1, 2, 2),
        )
        out = block(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())


class TestNonDivisibleInput(unittest.TestCase):
    """Input sizes not divisible by window should not error or NaN."""

    def test_non_divisible_no_nan(self):
        B, Z, H, W, C = 1, 3, 15, 17, 24
        x = torch.randn(B, Z, H, W, C)
        # Both normal and shifted blocks
        for shift in [(0, 0, 0), (1, 2, 2)]:
            block = SwinTransformerBlock3D(
                dim=C, num_heads=3, window_size=(2, 4, 4),
                shift_size=shift,
            )
            out = block(x)
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(torch.isfinite(out).all(),
                            msg=f"NaN/Inf with shift={shift}, input={(Z,H,W)}")


class TestZSmallerThanWindow(unittest.TestCase):
    """Z < window_z should not error."""

    def test_small_z_no_error(self):
        B, Z, H, W, C = 1, 1, 16, 16, 24
        x = torch.randn(B, Z, H, W, C)
        # Window z=2 but input z=1 -> z shift clamped to 0 dynamically in forward
        block = SwinTransformerBlock3D(
            dim=C, num_heads=3, window_size=(2, 4, 4),
            shift_size=(1, 2, 2),
        )
        out = block(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())
        # Shift_size stored in module reflects window-based clamping,
        # but forward() also clamps per dim when input dim <= window dim.
        # Input Z=1 <= window_z=2, so z shift is functionally 0.
        self.assertEqual(block.shift_size, (1, 2, 2))  # stored as-is


class TestPatchEmbedAniso(unittest.TestCase):
    """PatchEmbed3DAniso tests."""

    def test_basic(self):
        B, C, Z, H, W = 1, 1, 5, 32, 32
        x = torch.randn(B, C, Z, H, W)
        embed = PatchEmbed3DAniso(patch_size=(1, 2, 2), in_chans=1, embed_dim=24)
        out = embed(x)
        # Without norm: channel-first output
        self.assertEqual(out.shape[:2], (B, 24))
        self.assertEqual(out.shape[2], Z)  # Z unchanged
        self.assertEqual(out.shape[3], H // 2)
        self.assertEqual(out.shape[4], W // 2)

    def test_with_norm(self):
        B, C, Z, H, W = 1, 1, 5, 32, 32
        x = torch.randn(B, C, Z, H, W)
        embed = PatchEmbed3DAniso(patch_size=(1, 2, 2), in_chans=1, embed_dim=24,
                                  norm_layer=nn.LayerNorm)
        out = embed(x)
        # With norm: channel-last output
        self.assertEqual(out.shape, (B, Z, H // 2, W // 2, 24))


class TestPatchMergingXY(unittest.TestCase):
    """PatchMerging3DXY tests."""

    def test_basic(self):
        B, Z, H, W, C = 1, 5, 16, 16, 24
        x = torch.randn(B, Z, H, W, C)
        merge = PatchMerging3DXY(dim=C)
        out = merge(x)
        self.assertEqual(out.shape, (B, Z, H // 2, W // 2, C * 2))

    def test_odd_hw(self):
        B, Z, H, W, C = 1, 5, 15, 17, 24
        x = torch.randn(B, Z, H, W, C)
        merge = PatchMerging3DXY(dim=C)
        out = merge(x)
        self.assertEqual(out.shape[0], B)
        self.assertEqual(out.shape[1], Z)
        self.assertEqual(out.shape[2], (H + 1) // 2)  # ceil(15/2) = 8
        self.assertEqual(out.shape[3], (W + 1) // 2)  # ceil(17/2) = 9
        self.assertEqual(out.shape[4], C * 2)


class TestEncoderOutputShape(unittest.TestCase):
    """Encoder output shape tests."""

    def test_moving_output_shape(self):
        """Moving: (B,1,K=5,H=191,W=193) -> (B,5,96,24,25)"""
        B, C, K, H, W = 1, 1, 5, 191, 193
        x = torch.randn(B, C, K, H, W)
        enc = MovingQuerySwinEncoder(
            patch_size=(1, 2, 2),
            embed_dim=24,
            depths=(2, 2, 6),
            num_heads=(3, 6, 12),
            window_sizes=((2, 4, 4), (2, 4, 4), (2, 4, 4)),
            out_dim=96,
        )
        out = enc(x)
        expected_h = (H + 1) // 2 // 2 // 2  # ceil each /2 three times
        expected_w = (W + 1) // 2 // 2 // 2
        # But patch merging pads odd dims before halving, so sequential ceil:
        # W=193: patch_embed -> 194//2=97, merg1 pads 97->98, 98//2=49, merg2 pads 49->50, 50//2=25
        # H=191: patch_embed -> 192//2=96, merg1 96//2=48, merg2 48//2=24
        self.assertEqual(out.shape[:3], (B, K, 96))
        self.assertEqual(out.shape[3], 24)  # H
        self.assertEqual(out.shape[4], 25)  # W
        self.assertEqual(out.shape, (B, 5, 96, 24, 25))
        self.assertTrue(torch.isfinite(out).all())

    def test_reference_output_shape(self):
        """Reference: (B,1,D=17,H=191,W=193) -> (B,96,17,24,25)"""
        B, C, D, H, W = 1, 1, 17, 191, 193
        x = torch.randn(B, C, D, H, W)
        enc = ReferenceMemorySwinEncoder(
            patch_size=(1, 2, 2),
            embed_dim=24,
            depths=(2, 2, 6),
            num_heads=(3, 6, 12),
            window_sizes=((2, 4, 4), (2, 4, 4), (4, 4, 4)),
            out_dim=96,
        )
        out = enc(x)
        self.assertEqual(out.shape, (B, 96, D, 24, 25))
        self.assertTrue(torch.isfinite(out).all())

    def test_strides(self):
        enc = AnisotropicSwinEncoder3D(
            patch_size=(1, 2, 2),
            embed_dim=24,
            depths=(2, 2, 6),
            num_heads=(3, 6, 12),
        )
        self.assertEqual(enc.xy_stride, 8)
        self.assertEqual(enc.z_stride, 1)


class TestGradientFlow(unittest.TestCase):
    """Verify gradients flow through all key parameters."""

    def test_gradient_flow(self):
        enc = AnisotropicSwinEncoder3D(
            patch_size=(1, 2, 2),
            in_chans=1,
            embed_dim=24,
            depths=(2, 2, 2),
            num_heads=(3, 6, 6),
            window_sizes=((2, 4, 4), (2, 4, 4), (2, 4, 4)),
            out_dim=48,
            drop_path_rate=0.0,  # no stochastic depth for gradient test
        )
        x = torch.randn(1, 1, 4, 32, 32)
        out = enc(x)
        loss = out.sum()
        loss.backward()

        # Check key parameter types received finite non-zero gradients
        param_names_with_grad = set()
        for name, p in enc.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                param_names_with_grad.add(name.rsplit(".", 1)[0] if "." in name else name)

        # Verify relative_position_bias_table, qkv, proj, mlp all got gradients
        for layer in enc.layers:
            for blk in layer.blocks:
                self.assertIsNotNone(blk.attn.qkv.weight.grad)
                self.assertTrue((blk.attn.qkv.weight.grad.abs().sum() > 0).item())
                self.assertIsNotNone(blk.attn.proj.weight.grad)
                self.assertTrue((blk.attn.proj.weight.grad.abs().sum() > 0).item())
                self.assertIsNotNone(blk.mlp.fc1.weight.grad)
                self.assertTrue((blk.mlp.fc1.weight.grad.abs().sum() > 0).item())
                self.assertIsNotNone(blk.attn.relative_position_bias_table.grad)
                self.assertTrue((blk.attn.relative_position_bias_table.grad.abs().sum() > 0).item())


class TestStateDictRoundTrip(unittest.TestCase):
    """Save/load state_dict round-trip."""

    def test_save_load_strict(self):
        enc = AnisotropicSwinEncoder3D(
            patch_size=(1, 2, 2),
            in_chans=1,
            embed_dim=24,
            depths=(2, 2, 2),
            num_heads=(3, 6, 6),
            out_dim=48,
        )
        x = torch.randn(1, 1, 4, 32, 32)
        enc.eval()
        with torch.no_grad():
            out_before = enc(x).clone()

        state = enc.state_dict()

        enc2 = AnisotropicSwinEncoder3D(
            patch_size=(1, 2, 2),
            in_chans=1,
            embed_dim=24,
            depths=(2, 2, 2),
            num_heads=(3, 6, 6),
            out_dim=48,
        )
        load_msg = enc2.load_state_dict(state, strict=True)
        # Should be empty (no missing or unexpected keys)
        self.assertEqual(len(load_msg.missing_keys), 0)
        self.assertEqual(len(load_msg.unexpected_keys), 0)

        enc2.eval()
        with torch.no_grad():
            out_after = enc2(x).clone()

        self.assertTrue(torch.allclose(out_before, out_after, atol=1e-5))


class TestEvalModeConsistency(unittest.TestCase):
    """Outputs should be identical in eval mode across reloads."""

    def test_eval_allclose(self):
        enc = AnisotropicSwinEncoder3D(
            patch_size=(1, 2, 2),
            in_chans=1,
            embed_dim=24,
            depths=(2, 2, 2),
            num_heads=(3, 6, 6),
            out_dim=48,
        )
        enc.eval()
        x = torch.randn(1, 1, 4, 32, 32)

        with torch.no_grad():
            out1 = enc(x).clone()
            out2 = enc(x).clone()

        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))


class TestDropPath(unittest.TestCase):
    """DropPath basic behavior."""

    def test_train_identity_when_zero(self):
        dp = DropPath(0.0)
        dp.train()
        x = torch.randn(4, 3, 16, 16)
        out = dp(x)
        self.assertTrue(torch.equal(x, out))

    def test_eval_identity(self):
        dp = DropPath(0.5)
        dp.eval()
        x = torch.randn(4, 3, 16, 16)
        out = dp(x)
        self.assertTrue(torch.equal(x, out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
