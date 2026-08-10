# ------------------------------------------------------------------------
# 3D Anisotropic Swin Transformer Encoder
# Adapted from Microsoft Swin Transformer, MIT License.
#
# Original: https://github.com/microsoft/Swin-Transformer
# Reference commit: f82860bfb5225915aca09c3227159ee9e1df874d
# Copyright (c) Microsoft Corporation.
#
# This is a ground-up rewrite for dynamic-size 3D anisotropic volumes.
# It does NOT import or depend on the Microsoft Swin-Transformer repo.
# ------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


# ============================================================
# DropPath
# ============================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# ============================================================
# MLP
# ============================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================
# 3D window partition / reverse
# ============================================================

def window_partition_3d(x, window_size):
    """
    Args:
        x: (B, Z, H, W, C)  channel-last
        window_size: (Wz, Wy, Wx)

    Returns:
        windows: (B * nWz * nWy * nWx, Wz, Wy, Wx, C)
    """
    B, Z, H, W, C = x.shape
    Wz, Wy, Wx = window_size
    x = x.view(B,
               Z // Wz, Wz,
               H // Wy, Wy,
               W // Wx, Wx,
               C)
    # (B, nWz, Wz, nWy, Wy, nWx, Wx, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    # (B, nWz, nWy, nWx, Wz, Wy, Wx, C)
    windows = windows.view(-1, Wz, Wy, Wx, C)
    return windows


def window_reverse_3d(windows, window_size, B, Z, H, W):
    """
    Args:
        windows: (B * nWz * nWy * nWx, Wz, Wy, Wx, C)
        window_size: (Wz, Wy, Wx)
        B, Z, H, W: original spatial dims

    Returns:
        x: (B, Z, H, W, C)
    """
    Wz, Wy, Wx = window_size
    C = windows.shape[-1]
    nWz = Z // Wz
    nWy = H // Wy
    nWx = W // Wx
    x = windows.view(B, nWz, nWy, nWx, Wz, Wy, Wx, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, Z, H, W, C)
    return x


# ============================================================
# WindowAttention3D
# ============================================================

class WindowAttention3D(nn.Module):
    """Window-based multi-head self-attention with 3D relative position bias.

    Args:
        dim: Number of input channels.
        window_size: (Wz, Wy, Wx) — local window size.
        num_heads: Number of attention heads.
        qkv_bias: If True, add learnable bias to qkv projection.
        attn_drop: Attention dropout rate.
        proj_drop: Output projection dropout rate.
    """

    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = tuple(window_size)  # (Wz, Wy, Wx)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        Wz, Wy, Wx = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wz - 1) * (2 * Wy - 1) * (2 * Wx - 1), num_heads)
        )

        # Build relative position index
        coords_z = torch.arange(Wz)
        coords_y = torch.arange(Wy)
        coords_x = torch.arange(Wx)
        coords = torch.stack(torch.meshgrid(
            coords_z, coords_y, coords_x, indexing="ij"
        ))  # 3, Wz, Wy, Wx
        coords_flatten = torch.flatten(coords, 1)  # 3, Wz*Wy*Wx
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 3, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 3

        relative_coords[:, :, 0] += Wz - 1
        relative_coords[:, :, 1] += Wy - 1
        relative_coords[:, :, 2] += Wx - 1

        relative_coords[:, :, 0] *= (2 * Wy - 1) * (2 * Wx - 1)
        relative_coords[:, :, 1] *= (2 * Wx - 1)

        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B_ , N, C)  where B_ = B * nW, N = Wz*Wy*Wx
            mask: (B_ , N, N) or None.  0 = keep, -100 = mask out.
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B_, num_heads, N, head_dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, num_heads, N, N)

        # Relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)  # N, N, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, N, N
        attn = attn + relative_position_bias.unsqueeze(0)

        # Mask: additive, 0 = keep, -100 = mask
        if mask is not None:
            # mask shape: (B_, N, N)
            attn = attn + mask.unsqueeze(1)  # (B_, num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, window_size={self.window_size}, "
                f"num_heads={self.num_heads}")


# ============================================================
# SwinTransformerBlock3D
# ============================================================

class SwinTransformerBlock3D(nn.Module):
    """3D Swin Transformer block with shifted windows.

    Operates entirely in channel-last (B, Z, H, W, C).
    """

    def __init__(self, dim, num_heads, window_size=(2, 4, 4),
                 shift_size=(0, 0, 0), mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = tuple(window_size)  # (Wz, Wy, Wx)
        self.shift_size = tuple(shift_size)  # (sz, sy, sx)
        self.mlp_ratio = mlp_ratio

        # Store intended shift; actual shift may be clamped further in forward()
        # when the input dims are smaller than the window.
        Wz, Wy, Wx = self.window_size
        sz, sy, sx = self.shift_size
        self.shift_size = (0 if Wz <= 1 else min(sz, Wz // 2),
                           0 if Wy <= 1 else min(sy, Wy // 2),
                           0 if Wx <= 1 else min(sx, Wx // 2))

        assert all(0 <= s < w for s, w in zip(self.shift_size, self.window_size)), \
            f"shift_size {self.shift_size} must be in [0, window_size {self.window_size})"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def _build_cyclic_shift_mask_3d(self, Z, H, W, sz, sy, sx, device, dtype):
        """Build attention mask for cyclic-shifted windows.

        Returns:
            mask: (nW, N, N) where N = Wz*Wy*Wx, nW = nWz*nWy*nWx.
                  0 = keep, -100 = mask out.
            None if no shift.
        """
        if sz == 0 and sy == 0 and sx == 0:
            return None

        Wz, Wy, Wx = self.window_size
        N = Wz * Wy * Wx

        # Build img_mask similar to official Swin but in 3D
        img_mask = torch.zeros((1, Z, H, W, 1), device=device, dtype=dtype)

        cnt = 0
        # z slices
        if sz > 0 and Wz > 1:
            z_slices = (slice(0, -Wz), slice(-Wz, -sz), slice(-sz, None))
        else:
            z_slices = (slice(0, Z),)

        if sy > 0 and Wy > 1:
            y_slices = (slice(0, -Wy), slice(-Wy, -sy), slice(-sy, None))
        else:
            y_slices = (slice(0, H),)

        if sx > 0 and Wx > 1:
            x_slices = (slice(0, -Wx), slice(-Wx, -sx), slice(-sx, None))
        else:
            x_slices = (slice(0, W),)

        for zs in z_slices:
            for ys in y_slices:
                for xs in x_slices:
                    img_mask[:, zs, ys, xs, :] = cnt
                    cnt += 1

        mask_windows = window_partition_3d(img_mask, self.window_size)  # nW, Wz, Wy, Wx, 1
        mask_windows = mask_windows.view(-1, N)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask  # (nW, N, N)

    @staticmethod
    def _build_padding_mask_3d(padding_mask_spatial, window_size):
        """Build per-window padding attention mask.

        Args:
            padding_mask_spatial: (B, Z_pad, H_pad, W_pad), 1=real, 0=pad.
            window_size: (Wz, Wy, Wx)

        Returns:
            mask: (B * nW, N, N), 0=keep, -100=mask out.
        """
        Wz, Wy, Wx = window_size
        N = Wz * Wy * Wx

        # Partition padding mask into windows
        pad_win = window_partition_3d(
            padding_mask_spatial.unsqueeze(-1), window_size
        )  # (B*nW, Wz, Wy, Wx, 1)
        pad_win = pad_win.view(-1, N)  # (B*nW, N)

        # Create pair-wise mask: 1 if BOTH are real tokens
        attn_mask = pad_win.unsqueeze(1) * pad_win.unsqueeze(2)  # (B*nW, N, N)
        # Convert: 0 -> -100 (at least one is padding), 1 -> 0 (both real)
        attn_mask = (1.0 - attn_mask) * (-100.0)
        return attn_mask

    def forward(self, x):
        """
        Args:
            x: (B, Z, H, W, C)  channel-last

        Returns:
            x: (B, Z, H, W, C)
        """
        B, Z, H, W, C = x.shape
        Wz, Wy, Wx = self.window_size
        N = Wz * Wy * Wx

        # Clamp shift per-axis if actual input dim <= window dim
        sz, sy, sx = self.shift_size
        sz = 0 if Z <= Wz else sz
        sy = 0 if H <= Wy else sy
        sx = 0 if W <= Wx else sx

        shortcut = x
        x = self.norm1(x)

        # --------------------------------------------------------
        # 1. Pad FIRST to window multiples
        # --------------------------------------------------------
        pad_z = (Wz - Z % Wz) % Wz
        pad_h = (Wy - H % Wy) % Wy
        pad_w = (Wx - W % Wx) % Wx

        need_pad = (
            pad_z > 0
            or pad_h > 0
            or pad_w > 0
        )

        # Real/padding mask in the SAME coordinate system as x
        padding_mask = torch.ones(
            B,
            Z,
            H,
            W,
            device=x.device,
            dtype=torch.float32,
        )

        if need_pad:
            x_cf = x.permute(
                0, 4, 1, 2, 3
            ).contiguous()

            x_cf = F.pad(
                x_cf,
                (0, pad_w, 0, pad_h, 0, pad_z),
                mode="constant",
                value=0.0,
            )

            x_pad = x_cf.permute(
                0, 2, 3, 4, 1
            ).contiguous()

            padding_mask = F.pad(
                padding_mask,
                (0, pad_w, 0, pad_h, 0, pad_z),
                value=0.0,
            )

        else:
            x_pad = x

        Zp, Hp, Wp = x_pad.shape[1:4]

        # --------------------------------------------------------
        # 2. Cyclic shift AFTER padding
        # --------------------------------------------------------
        if any(s > 0 for s in (sz, sy, sx)):

            shifted_x = torch.roll(
                x_pad,
                shifts=(-sz, -sy, -sx),
                dims=(1, 2, 3),
            )

            # Padding locations must shift together with features.
            padding_mask = torch.roll(
                padding_mask,
                shifts=(-sz, -sy, -sx),
                dims=(1, 2, 3),
            )

        else:
            shifted_x = x_pad

        # --------------------------------------------------------
        # 3. Window partition
        # --------------------------------------------------------
        x_windows = window_partition_3d(
            shifted_x,
            self.window_size,
        )

        nW_total = x_windows.shape[0]

        x_windows = x_windows.view(
            nW_total,
            N,
            C,
        )
        # --------------------------------------------------------
        # 4. Build attention mask
        # --------------------------------------------------------
        mask_cyclic = self._build_cyclic_shift_mask_3d(
            Zp, Hp, Wp, sz, sy, sx, x.device, x.dtype
        )  # (nW_per_sample, N, N) or None

        mask_pad = self._build_padding_mask_3d(
            padding_mask, self.window_size
        )  # (B * nW, N, N)

        # Combine masks
        nW_per_sample = nW_total // B
        if mask_cyclic is not None:
            # mask_cyclic: (nW_per_sample, N, N), expand to (B * nW_per_sample, N, N)
            mask_cyclic_expanded = mask_cyclic.unsqueeze(0).expand(B, -1, -1, -1)
            mask_cyclic_expanded = mask_cyclic_expanded.reshape(nW_total, N, N)
            attn_mask = mask_cyclic_expanded + mask_pad
        else:
            attn_mask = mask_pad

        # --------------------------------------------------------
        # 5. Window attention
        # --------------------------------------------------------
        attn_windows = self.attn(x_windows, mask=attn_mask)
        # (B * nW, N, C)

        # --------------------------------------------------------
        # 6. Window reverse
        # --------------------------------------------------------
        attn_windows = attn_windows.view(nW_total, Wz, Wy, Wx, C)
        shifted_x = window_reverse_3d(attn_windows, self.window_size,
                                      B, Zp, Hp, Wp)


        # --------------------------------------------------------
        # 7. Reverse cyclic shift on the padded lattice
        # --------------------------------------------------------
        if any(s > 0 for s in (sz, sy, sx)):
            x = torch.roll(
                shifted_x,
                shifts=(sz, sy, sx),
                dims=(1, 2, 3),
            )
        else:
            x = shifted_x

        # --------------------------------------------------------
        # 8. Crop padding AFTER reversing the shift
        # --------------------------------------------------------
        if need_pad:
            x = x[:, :Z, :H, :W, :]

        # --------------------------------------------------------
        # 9. Residual + MLP
        # --------------------------------------------------------
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, num_heads={self.num_heads}, "
                f"window_size={self.window_size}, shift_size={self.shift_size}, "
                f"mlp_ratio={self.mlp_ratio}")


# ============================================================
# PatchEmbed3DAniso
# ============================================================

class PatchEmbed3DAniso(nn.Module):
    """3D anisotropic patch embedding via Conv3d.

    patch_size=(1, 2, 2): no downsampling in Z, 2x in H and W.
    """

    def __init__(self, patch_size=(1, 2, 2), in_chans=1, embed_dim=96,
                 norm_layer=None):
        super().__init__()
        self.patch_size = tuple(patch_size)  # (pz, py, px)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=self.patch_size,
                              stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        """
        Args:
            x: (B, in_chans, Z, H, W) channel-first

        Returns:
            x: (B, Z, embed_dim, ceil(H/2), ceil(W/2)) channel-first (if no norm)
               or (B, Z, ceil(H/2), ceil(W/2), embed_dim) channel-last (if norm)
        """
        B, C, Z, H, W = x.shape
        pz, py, px = self.patch_size

        # Pad H, W on the right to make divisible by patch size
        pad_h = (py - H % py) % py
        pad_w = (px - W % px) % px

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0))

        x = self.proj(x)  # (B, embed_dim, Z, H_out, W_out)

        if self.norm is not None:
            # Convert to channel-last for LayerNorm
            _, _, Z_out, H_out, W_out = x.shape
            x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, Z, H, W, embed_dim)
            x = self.norm(x)
            return x  # channel-last

        return x  # channel-first

    def flops(self, Z, H, W):
        pz, py, px = self.patch_size
        Ho = H // py
        Wo = W // px
        flops = Z * Ho * Wo * self.embed_dim * self.in_chans * pz * py * px
        if self.norm is not None:
            flops += Z * Ho * Wo * self.embed_dim
        return flops


# ============================================================
# PatchMerging3DXY
# ============================================================

class PatchMerging3DXY(nn.Module):
    """XY-only patch merging. Z dimension is untouched.

    Input:  (B, Z, H, W, C)  channel-last
    Output: (B, Z, ceil(H/2), ceil(W/2), 2*C)
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        Args:
            x: (B, Z, H, W, C) channel-last
        """
        B, Z, H, W, C = x.shape

        # Pad H, W if odd
        pad_h = H % 2
        pad_w = W % 2
        if pad_h > 0 or pad_w > 0:
            x_cf = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, Z, H, W)
            x_cf = F.pad(x_cf, (0, pad_w, 0, pad_h))
            x = x_cf.permute(0, 2, 3, 4, 1).contiguous()  # (B, Z, Hp, Wp, C)

        x0 = x[:, :, 0::2, 0::2, :]  # B, Z, H/2, W/2, C
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)  # B, Z, H/2, W/2, 4C
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


# ============================================================
# BasicLayer3D
# ============================================================

class BasicLayer3D(nn.Module):
    """A single Swin Transformer stage with optional downsampling.

    Blocks alternate between W-MSA (shift=0) and SW-MSA (shift=window_size//2).
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (
                    window_size[0] // 2,
                    window_size[1] // 2,
                    window_size[2] // 2,
                ),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            self.blocks.append(block)

        # Downsample
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward_blocks(self, x):
        """Run only the transformer blocks, return pre-downsample features."""
        for blk in self.blocks:
            if self.use_checkpoint:
                try:
                    x = checkpoint.checkpoint(blk, x, use_reentrant=False)
                except Exception:
                    x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def forward(self, x):
        x = self.forward_blocks(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


# ============================================================
# FeatureFusion3D
# ============================================================

class XYDownsampleBlock(nn.Module):
    """
    Learnable XY-only downsampling.

    Spatial:
        (Z, H, W) -> (Z, ceil(H/2), ceil(W/2))

    Z is untouched.
    """

    def __init__(self, channels):
        super().__init__()

        num_groups = 8 if channels % 8 == 0 else 1

        self.block = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class FeatureFusion3D(nn.Module):
    """
    Residual multi-scale feature fusion.

    Inputs:
        feat_s1:
            /2 XY feature
            (B, dims[0], Z, H/2, W/2)

        feat_s2:
            /4 XY feature
            (B, dims[1], Z, H/4, W/4)

        feat_final:
            /8 XY feature
            (B, dims[2], Z, H/8, W/8)

    Output:
        fused /8 feature
        (B, out_dim, Z, H/8, W/8)

    Design:
        1. /2 and /4 features are projected and learnably downsampled.
        2. The full /8 feature is preserved instead of compressed to fuse_dim.
        3. Multi-scale information predicts a residual correction.
        4. Final output = base /8 feature + gamma * residual.
    """

    def __init__(
        self,
        dims=(24, 48, 96),
        fuse_dim=32,
        out_dim=96,
    ):
        super().__init__()

        self.fuse_dim = int(fuse_dim)
        self.out_dim = int(out_dim)

        fuse_groups = 8 if self.fuse_dim % 8 == 0 else 1
        out_groups = 8 if self.out_dim % 8 == 0 else 1

        # --------------------------------------------------
        # /2 branch
        # --------------------------------------------------
        self.proj_s1 = nn.Sequential(
            nn.Conv3d(
                dims[0],
                self.fuse_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(fuse_groups, self.fuse_dim),
            nn.GELU(),
        )

        # /2 -> /4 -> /8
        self.down_s1 = nn.Sequential(
            XYDownsampleBlock(self.fuse_dim),
            XYDownsampleBlock(self.fuse_dim),
        )

        # --------------------------------------------------
        # /4 branch
        # --------------------------------------------------
        self.proj_s2 = nn.Sequential(
            nn.Conv3d(
                dims[1],
                self.fuse_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(fuse_groups, self.fuse_dim),
            nn.GELU(),
        )

        # /4 -> /8
        self.down_s2 = XYDownsampleBlock(self.fuse_dim)

        # --------------------------------------------------
        # Preserve full /8 branch
        # --------------------------------------------------
        if dims[2] == self.out_dim:
            self.base_proj = nn.Identity()
        else:
            self.base_proj = nn.Conv3d(
                dims[2],
                self.out_dim,
                kernel_size=1,
                bias=False,
            )

        # --------------------------------------------------
        # Multi-scale residual predictor
        #
        # Important:
        # keep full out_dim-dimensional /8 feature here.
        # Do NOT compress /8 from 96 -> 32.
        # --------------------------------------------------
        fusion_in_dim = (
            self.fuse_dim
            + self.fuse_dim
            + self.out_dim
        )

        self.mix = nn.Sequential(
            nn.Conv3d(
                fusion_in_dim,
                self.out_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(out_groups, self.out_dim),
            nn.GELU(),
            nn.Conv3d(
                self.out_dim,
                self.out_dim,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
            ),
        )

        # Small residual contribution initially.
        # 0.1 is preferable to 0 here because it still allows
        # gradients to flow into all fusion branches immediately.
        self.gamma = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float32)
        )

    def forward(
        self,
        feat_s1,
        feat_s2,
        feat_final,
    ):
        # Full /8 feature is preserved.
        base = self.base_proj(feat_final)

        # /2 -> /8
        f1 = self.proj_s1(feat_s1)
        f1 = self.down_s1(f1)

        # /4 -> /8
        f2 = self.proj_s2(feat_s2)
        f2 = self.down_s2(f2)

        target_shape = base.shape[-3:]

        if f1.shape[-3:] != target_shape:
            raise RuntimeError(
                "FeatureFusion3D /2 branch spatial mismatch: "
                f"f1={tuple(f1.shape)}, "
                f"base={tuple(base.shape)}"
            )

        if f2.shape[-3:] != target_shape:
            raise RuntimeError(
                "FeatureFusion3D /4 branch spatial mismatch: "
                f"f2={tuple(f2.shape)}, "
                f"base={tuple(base.shape)}"
            )

        fused = torch.cat(
            [f1, f2, base],
            dim=1,
        )

        delta = self.mix(fused)

        return base + self.gamma * delta


# ============================================================
# AnisotropicSwinEncoder3D
# ============================================================

class AnisotropicSwinEncoder3D(nn.Module):
    """Anisotropic 3D Swin Transformer encoder.

    Input:  (B, in_chans, Z, H, W)  channel-first
    Output: (B, out_dim, Z, ceil(H/xy_stride), ceil(W/xy_stride))  channel-first

    Three stages with XY patch merging after stages 1 and 2.
    Z stride is always 1.
    XY stride = patch_size_y * 2^(stages_with_merging) = 2 * 2 * 2 = 8.

    By default:
        patch_size = (1, 2, 2)
        embed_dim  = 24
        depths     = (2, 2, 6)
        num_heads  = (3, 6, 12)
        final dim after 3 stages = 24 * 2^2 = 96
    """

    def __init__(self,
                 patch_size=(1, 2, 2),
                 in_chans=1,
                 embed_dim=24,
                 depths=(2, 2, 6),
                 num_heads=(3, 6, 12),
                 window_sizes=((2, 4, 4), (2, 4, 4), (2, 4, 4)),
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 patch_norm=True,
                 use_checkpoint=False,
                 out_dim=None,
                 use_fusion=False,
                 fuse_dim=32):
        super().__init__()

        self.patch_size = tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.xy_stride = self.patch_size[1] * (2 ** (self.num_layers - 1))  # 2 * 2^2 = 8
        self.z_stride = self.patch_size[0]  # 1
        self.use_fusion = use_fusion

        final_dim = int(embed_dim * 2 ** (self.num_layers - 1))
        self.out_channels = out_dim if out_dim is not None else final_dim

        norm_layer = nn.LayerNorm

        # --------------------------------------------------------
        # Patch embedding
        # --------------------------------------------------------
        self.patch_embed = PatchEmbed3DAniso(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # --------------------------------------------------------
        # Stochastic depth schedule
        # --------------------------------------------------------
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # --------------------------------------------------------
        # Build layers
        # --------------------------------------------------------
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            layer = BasicLayer3D(
                dim=layer_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_sizes[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging3DXY if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # Final LayerNorm (channel-last)
        final_dim = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = norm_layer(final_dim)

        # Multi-scale feature fusion
        if use_fusion:
            dims_s1 = int(embed_dim * 2 ** 0)   # embed_dim
            dims_s2 = int(embed_dim * 2 ** 1)   # 2 * embed_dim
            dims_s3 = final_dim                  # 4 * embed_dim
            self.fusion = FeatureFusion3D(
                dims=(dims_s1, dims_s2, dims_s3),
                fuse_dim=fuse_dim,
                out_dim=self.out_channels,
            )
            # When fusion is used, the output projection is redundant
            # (fusion already projects to out_dim)
            self.proj_out = nn.Identity()
        else:
            self.fusion = None
            if self.out_channels != final_dim:
                self.proj_out = nn.Conv3d(final_dim, self.out_channels, kernel_size=1)
            else:
                self.proj_out = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        """
        Args:
            x: (B, in_chans, Z, H, W) channel-first

        Returns:
            (B, out_channels, Z, ceil(H/xy_stride), ceil(W/xy_stride)) channel-first
        """
        # Patch embedding: channel-first -> channel-last
        x = self.patch_embed(x)  # (B, Z, ceil(H/2), ceil(W/2), embed_dim)
        x = self.pos_drop(x)

        intermediates_cf = []
        for i, layer in enumerate(self.layers):
            # Run blocks, capture pre-downsample feature
            x = layer.forward_blocks(x)

            if self.use_fusion and i < self.num_layers - 1:
                # Convert to channel-first for fusion
                intermediates_cf.append(
                    x.permute(0, 4, 1, 2, 3).contiguous()
                )

            # Downsample if present
            if layer.downsample is not None:
                x = layer.downsample(x)

        # Final norm (channel-last)
        x = self.norm(x)  # (B, Z, Hf, Wf, C)

        # Convert to channel-first
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, Z, Hf, Wf)

        # Feature fusion or simple output projection
        if self.use_fusion and len(intermediates_cf) >= 2:
            x = self.fusion(intermediates_cf[0], intermediates_cf[1], x)
        x = self.proj_out(x)

        return x


# ============================================================
# MovingQuerySwinEncoder
# ============================================================

class MovingQuerySwinEncoder(nn.Module):
    """Moving (query) encoder using AnisotropicSwinEncoder3D.

    Input:  (B, 1, K, H, W)  — K moving slices as Z.
    Output: (B, K, out_dim, ceil(H/xy_stride), ceil(W/xy_stride))
    """

    def __init__(self, use_fusion=False, fuse_dim=32, **kwargs):
        super().__init__()
        self.encoder = AnisotropicSwinEncoder3D(
            in_chans=1, use_fusion=use_fusion, fuse_dim=fuse_dim, **kwargs
        )

    @property
    def xy_stride(self):
        return self.encoder.xy_stride

    @property
    def z_stride(self):
        return self.encoder.z_stride

    @property
    def out_channels(self):
        return self.encoder.out_channels

    def forward(self, mov):
        """
        Args:
            mov: (B, 1, K, H, W)

        Returns:
            (B, K, out_dim, ceil(H/xy_stride), ceil(W/xy_stride))
        """
        # mov: (B, 1, K, H, W) -> use K as Z
        x = self.encoder(mov)  # (B, out_dim, K, Hf, Wf)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, K, out_dim, Hf, Wf)
        return x


# ============================================================
# ReferenceMemorySwinEncoder
# ============================================================

class ReferenceMemorySwinEncoder(nn.Module):
    """Reference (memory) encoder using AnisotropicSwinEncoder3D.

    Input:  (B, 1, D, H, W)  — D reference slices as Z.
    Output: (B, out_dim, D, ceil(H/xy_stride), ceil(W/xy_stride))
    """

    def __init__(self, use_fusion=False, fuse_dim=32, **kwargs):
        super().__init__()
        self.encoder = AnisotropicSwinEncoder3D(
            in_chans=1, use_fusion=use_fusion, fuse_dim=fuse_dim, **kwargs
        )

    @property
    def xy_stride(self):
        return self.encoder.xy_stride

    @property
    def z_stride(self):
        return self.encoder.z_stride

    @property
    def out_channels(self):
        return self.encoder.out_channels

    def forward(self, ref):
        """
        Args:
            ref: (B, 1, D, H, W)

        Returns:
            (B, out_dim, D, ceil(H/xy_stride), ceil(W/xy_stride))
        """
        return self.encoder(ref)


# ============================================================
__all__ = [
    "DropPath",
    "Mlp",
    "window_partition_3d",
    "window_reverse_3d",
    "WindowAttention3D",
    "SwinTransformerBlock3D",
    "PatchEmbed3DAniso",
    "PatchMerging3DXY",
    "BasicLayer3D",
    "AnisotropicSwinEncoder3D",
    "MovingQuerySwinEncoder",
    "ReferenceMemorySwinEncoder",
]
