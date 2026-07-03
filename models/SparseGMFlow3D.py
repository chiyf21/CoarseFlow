# SparseGMFlow3D.py
# CoarseMatchingNet: sparse-to-dense 3D medical image registration via coarse-to-fine matching.
#   - moving encoder output:    XY /8, Z/K unchanged
#   - reference encoder output: XY /8, Z unchanged
#   - reference enhancement:    6-layer local 3D window attention by default
#   - matching/refinement/loss outputs are compatible with the existing training/losses.py path.

import math
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic blocks
# ============================================================
def count_parameters_by_module(model, max_depth=1):
    """
    max_depth=1: 只看一级模块，比如 moving_encoder/reference_encoder/matcher
    max_depth=2: 看更细一级
    """
    module_params = {}

    for name, p in model.named_parameters():
        parts = name.split(".")
        module_name = ".".join(parts[:max_depth])

        if module_name not in module_params:
            module_params[module_name] = {
                "total": 0,
                "trainable": 0,
            }

        n = p.numel()
        module_params[module_name]["total"] += n
        if p.requires_grad:
            module_params[module_name]["trainable"] += n

    print("=" * 80)
    print(f"{'Module':40s} {'Total(M)':>12s} {'Trainable(M)':>15s}")
    print("-" * 80)

    for name, stat in sorted(module_params.items()):
        total_m = stat["total"] / 1e6
        train_m = stat["trainable"] / 1e6
        print(f"{name:40s} {total_m:12.3f} {train_m:15.3f}")

    print("=" * 80)

    return module_params

def count_parameters(model, verbose=True):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    if verbose:
        print("=" * 60)
        print(f"Total parameters    : {total / 1e6:.3f} M")
        print(f"Trainable parameters: {trainable / 1e6:.3f} M")
        print(f"Frozen parameters   : {frozen / 1e6:.3f} M")
        print("=" * 60)

    return total, trainable, frozen




class WindowSelfAttention2D(nn.Module):
    """
    Local window self-attention for each moving slice.

    Input : (B*K, C, H, W)
    Output: (B*K, C, H, W)
    """

    def __init__(self, dim, num_heads=4, window_size=8, mlp_ratio=2.0):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = int(window_size)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x_pad.shape[-2:]

        # (B,C,Hp,Wp) -> (B*nW, ws*ws, C)
        B, C, H_pad, W_pad = x_pad.shape
        assert H_pad % ws == 0 and W_pad % ws == 0

        nH = H_pad // ws
        nW = W_pad // ws

        x_win = (
            x_pad.contiguous()
            .view(B, C, nH, ws, nW, ws)
            .permute(0, 1, 2, 4, 3, 5)
            .contiguous()
        )
        x_win = x_win.permute(0, 2, 3, 4, 5, 1).contiguous()
        x_win = x_win.view(-1, ws * ws, C)

        h = self.norm1(x_win)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x_win = x_win + attn_out
        x_win = x_win + self.ffn(self.norm2(x_win))

        # restore
        x_win = x_win.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x_win = x_win.permute(0, 5, 1, 3, 2, 4).contiguous()
        x_pad = x_win.view(B, C, Hp, Wp)

        return x_pad[:, :, :H, :W]


class WindowSelfAttention3D(nn.Module):
    """
    Local 3D window self-attention for reference memory.

    Input : (B, C, D, H, W)
    Output: (B, C, D, H, W)

    This is intentionally windowed and residual. The residual branches
    are layer-scaled so that inserting many layers starts close to identity.
    """

    def __init__(
        self,
        dim,
        num_heads=4,
        window_size=(4, 8, 8),
        mlp_ratio=2.0,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        init_gamma=1e-4,
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim={dim} must be divisible by num_heads={num_heads}"
        )

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = tuple(window_size)

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_drop),
        )

        self.gamma_attn = nn.Parameter(torch.ones(1) * init_gamma)
        self.gamma_mlp = nn.Parameter(torch.ones(1) * init_gamma)

    def _window_partition(self, x):
        """
        x: (B, Dp, Hp, Wp, C)
        return: (B*nW, Wd*Wh*Ww, C)
        """
        B, Dp, Hp, Wp, C = x.shape
        Wd, Wh, Ww = self.window_size

        x = x.view(
            B,
            Dp // Wd, Wd,
            Hp // Wh, Wh,
            Wp // Ww, Ww,
            C,
        )
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(-1, Wd * Wh * Ww, C)

    def _window_reverse(self, windows, B, Dp, Hp, Wp):
        Wd, Wh, Ww = self.window_size
        C = windows.shape[-1]

        x = windows.view(
            B,
            Dp // Wd,
            Hp // Wh,
            Wp // Ww,
            Wd,
            Wh,
            Ww,
            C,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(B, Dp, Hp, Wp, C)

    def _attention_forward(self, x):
        B, C, D, H, W = x.shape
        Wd, Wh, Ww = self.window_size

        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B,D,H,W,C)

        pad_d = (Wd - D % Wd) % Wd
        pad_h = (Wh - H % Wh) % Wh
        pad_w = (Ww - W % Ww) % Ww

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_ch = x.permute(0, 4, 1, 2, 3).contiguous()
            x_ch = F.pad(
                x_ch,
                pad=(0, pad_w, 0, pad_h, 0, pad_d),
                mode="constant",
                value=0.0,
            )
            x = x_ch.permute(0, 2, 3, 4, 1).contiguous()

        Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

        windows = self._window_partition(x)  # (B*nW,N,C)

        h = self.norm1(windows)
        qkv = self.qkv(h)
        qkv = qkv.view(
            qkv.shape[0],
            qkv.shape[1],
            3,
            self.num_heads,
            C // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(windows.shape[0], windows.shape[1], C)

        out = self.proj(out)
        out = self.proj_drop(out)

        x_out = self._window_reverse(out, B, Dp, Hp, Wp)
        x_out = x_out[:, :D, :H, :W, :].contiguous()
        x_out = x_out.permute(0, 4, 1, 2, 3).contiguous()

        return x_out

    def _mlp_forward(self, x):
        x_perm = x.permute(0, 2, 3, 4, 1).contiguous()
        x_norm = self.norm2(x_perm)
        x_mlp = self.mlp(x_norm)
        return x_mlp.permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x):
        x = x + self.gamma_attn * self._attention_forward(x)
        x = x + self.gamma_mlp * self._mlp_forward(x)
        return x


class ConvNeXtBlock2D(nn.Module):
    """ConvNeXt-style 2D block. Input/Output: (B,C,H,W)."""

    def __init__(self, dim, kernel_size=7, mlp_ratio=2.0, layer_scale_init=1e-6):
        super().__init__()
        padding = kernel_size // 2
        hidden_dim = int(dim * mlp_ratio)

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x):
        shortcut = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()

        return shortcut + x


class ConvNeXtBlock3DAniso(nn.Module):
    """
    Anisotropic ConvNeXt-style 3D block.
    Input/Output: (B,C,D,H,W).
    """

    def __init__(
        self,
        dim,
        kernel_size=(3, 7, 7),
        mlp_ratio=2.0,
        layer_scale_init=1e-6,
    ):
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)
        hidden_dim = int(dim * mlp_ratio)

        self.dwconv = nn.Conv3d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x):
        shortcut = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return shortcut + x


def _make_2d_stage(channels, num_blocks, kernel_size=7, mlp_ratio=2.0):
    return nn.Sequential(
        *[
            ConvNeXtBlock2D(
                channels,
                kernel_size=kernel_size,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(int(num_blocks))
        ]
    )


def _make_3d_stage(channels, num_blocks, kernel_size=(3, 7, 7), mlp_ratio=2.0):
    return nn.Sequential(
        *[
            ConvNeXtBlock3DAniso(
                channels,
                kernel_size=kernel_size,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(int(num_blocks))
        ]
    )


# ============================================================
# Moving / reference encoders
# ============================================================

class MovingQueryEncoder(nn.Module):
    """
    Moving sparse stack encoder with XY /8 output.

    Input:
        mov: (B,1,K,H,W)

    Output:
        F_mov: (B,K,C,ceil(H/8),ceil(W/8))
    """

    def __init__(
        self,
        in_ch=1,
        dim=96,
        base_channels=(24, 48, 96),
        num_blocks=(1, 2, 1),
        mlp_ratio=2.0,
        use_window_attn=True,
        window_attn_layers=1,
        window_size=8,
        num_heads=4,
        slice_fusion_blocks=1,
    ):
        super().__init__()

        c1, c2, c3 = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c1),
            nn.GELU(),
        )
        self.stage1 = _make_2d_stage(c1, num_blocks[0], kernel_size=7, mlp_ratio=mlp_ratio)

        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c2),
            nn.GELU(),
        )
        self.stage2 = _make_2d_stage(c2, num_blocks[1], kernel_size=7, mlp_ratio=mlp_ratio)

        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c3),
            nn.GELU(),
        )
        self.stage3 = _make_2d_stage(c3, num_blocks[2], kernel_size=7, mlp_ratio=mlp_ratio)

        self.proj = nn.Sequential(
            nn.Conv2d(c3, dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )

        if use_window_attn and window_attn_layers > 0:
            self.slice_attn = nn.Sequential(
                *[
                    WindowSelfAttention2D(
                        dim=dim,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(int(window_attn_layers))
                ]
            )
        else:
            self.slice_attn = nn.Identity()

        if slice_fusion_blocks > 0:
            self.slice_fusion = nn.Sequential(
                *[
                    ConvNeXtBlock3DAniso(
                        dim,
                        kernel_size=(3, 3, 3),
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(int(slice_fusion_blocks))
                ]
            )
        else:
            self.slice_fusion = nn.Identity()

    def forward(self, mov):
        B, C, K, H, W = mov.shape

        x = mov.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * K, C, H, W)

        x = self.stem(x)    # /2
        x = self.stage1(x)
        x = self.down1(x)   # /4
        x = self.stage2(x)
        x = self.down2(x)   # /8
        x = self.stage3(x)
        x = self.proj(x)
        x = self.slice_attn(x)

        _, dim, Hf, Wf = x.shape

        x = x.view(B, K, dim, Hf, Wf)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B,C,K,Hf,Wf)
        x = self.slice_fusion(x)

        return x.permute(0, 2, 1, 3, 4).contiguous()


class ReferenceMemoryEncoder(nn.Module):
    """
    Reference volume encoder with XY /8 output and no Z downsampling.

    Input:
        ref: (B,1,D,H,W)

    Output:
        F_ref: (B,C,D,ceil(H/8),ceil(W/8))
    """

    def __init__(
        self,
        in_ch=1,
        dim=96,
        base_channels=(24, 48, 96),
        num_blocks=(1, 2, 2),
        refine_blocks=1,
        mlp_ratio=2.0,
        attn_layers=6,
        attn_num_heads=4,
        attn_window_size=(4, 8, 8),
        attn_mlp_ratio=2.0,
        attn_init_gamma=1e-4,
    ):
        super().__init__()

        c1, c2, c3 = base_channels

        self.stem = nn.Sequential(
            nn.Conv3d(
                in_ch,
                c1,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(8, c1),
            nn.GELU(),
        )
        self.stage1 = _make_3d_stage(
            c1,
            num_blocks[0],
            kernel_size=(1, 7, 7),
            mlp_ratio=mlp_ratio,
        )

        self.down1 = nn.Sequential(
            nn.Conv3d(
                c1,
                c2,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(8, c2),
            nn.GELU(),
        )
        self.stage2 = _make_3d_stage(
            c2,
            num_blocks[1],
            kernel_size=(3, 7, 7),
            mlp_ratio=mlp_ratio,
        )

        self.down2 = nn.Sequential(
            nn.Conv3d(
                c2,
                c3,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(8, c3),
            nn.GELU(),
        )
        self.stage3 = _make_3d_stage(
            c3,
            num_blocks[2],
            kernel_size=(3, 7, 7),
            mlp_ratio=mlp_ratio,
        )

        self.proj = nn.Sequential(
            nn.Conv3d(c3, dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )

        self.refine = _make_3d_stage(
            dim,
            refine_blocks,
            kernel_size=(3, 7, 7),
            mlp_ratio=mlp_ratio,
        )

        if attn_layers > 0:
            self.ref_attn = nn.Sequential(
                *[
                    WindowSelfAttention3D(
                        dim=dim,
                        num_heads=attn_num_heads,
                        window_size=attn_window_size,
                        mlp_ratio=attn_mlp_ratio,
                        init_gamma=attn_init_gamma,
                    )
                    for _ in range(int(attn_layers))
                ]
            )
        else:
            self.ref_attn = nn.Identity()

    def forward(self, ref):
        x = self.stem(ref)    # D, H/2, W/2
        x = self.stage1(x)

        x = self.down1(x)     # D, H/4, W/4
        x = self.stage2(x)

        x = self.down2(x)     # D, H/8, W/8
        x = self.stage3(x)

        x = self.proj(x)
        x = self.refine(x)
        x = self.ref_attn(x)

        return x


# ============================================================
# Coordinate utilities
# ============================================================

def normalize_grid(coords, D, H, W):
    """
    Feature-space z,y,x -> grid_sample normalized x,y,z.
    coords: (...,3), order=(z,y,x)
    """
    z = coords[..., 0]
    y = coords[..., 1]
    x = coords[..., 2]

    x_norm = 2.0 * x / max(W - 1, 1) - 1.0
    y_norm = 2.0 * y / max(H - 1, 1) - 1.0
    z_norm = 2.0 * z / max(D - 1, 1) - 1.0

    return torch.stack([x_norm, y_norm, z_norm], dim=-1)


def sample_local_3d_window(F_ref, center_coords, radius=(4, 3, 3)):
    """
    Sample local 3D candidate windows from reference memory.

    Args:
        F_ref:
            (B,C,Dr,Hr,Wr)

        center_coords:
            (B,N,3), feature-space coords, order=(z,y,x)

        radius:
            (Rz,Ry,Rx) in feature-space.

    Returns:
        sampled_feat:
            (B,N,M,C)

        candidate_coords_clamped:
            (B,N,M,3)

        candidate_valid_mask:
            (B,N,M), True if original candidate was in bounds.
    """
    B, C, Dr, Hr, Wr = F_ref.shape
    _, N, _ = center_coords.shape
    Rz, Ry, Rx = radius

    dz = torch.arange(-Rz, Rz + 1, device=F_ref.device)
    dy = torch.arange(-Ry, Ry + 1, device=F_ref.device)
    dx = torch.arange(-Rx, Rx + 1, device=F_ref.device)

    oz, oy, ox = torch.meshgrid(dz, dy, dx, indexing="ij")
    offsets = torch.stack([oz, oy, ox], dim=-1).float()
    offsets = offsets.view(1, 1, -1, 3)

    candidate_coords = center_coords.unsqueeze(2) + offsets
    M = candidate_coords.shape[2]

    candidate_valid_mask = (
        (candidate_coords[..., 0] >= 0) & (candidate_coords[..., 0] <= Dr - 1) &
        (candidate_coords[..., 1] >= 0) & (candidate_coords[..., 1] <= Hr - 1) &
        (candidate_coords[..., 2] >= 0) & (candidate_coords[..., 2] <= Wr - 1)
    )

    candidate_coords_clamped = candidate_coords.clone()
    candidate_coords_clamped[..., 0].clamp_(0, Dr - 1)
    candidate_coords_clamped[..., 1].clamp_(0, Hr - 1)
    candidate_coords_clamped[..., 2].clamp_(0, Wr - 1)

    grid = normalize_grid(candidate_coords_clamped, Dr, Hr, Wr)
    grid = grid.view(B, N * M, 1, 1, 3)

    sampled = F.grid_sample(
        F_ref,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    sampled = sampled.squeeze(-1).squeeze(-1)        # (B,C,N*M)
    sampled = sampled.permute(0, 2, 1).contiguous()  # (B,N*M,C)
    sampled = sampled.view(B, N, M, C)

    return sampled, candidate_coords_clamped, candidate_valid_mask


def get_ref_feature_scale(ref, F_ref):
    """
    Return feature->raw coordinate scale for z,y,x.

    ref:   (B,1,D,H,W)
    F_ref: (B,C,Dr,Hr,Wr)
    """
    _, _, D, H, W = ref.shape
    _, _, Dr, Hr, Wr = F_ref.shape

    scale_z = (D - 1) / max(Dr - 1, 1)
    scale_y = (H - 1) / max(Hr - 1, 1)
    scale_x = (W - 1) / max(Wr - 1, 1)

    return scale_z, scale_y, scale_x


def feature_to_raw_coords(coords_feat, scale):
    """coords_feat: (...,3), order=(z,y,x)."""
    scale_z, scale_y, scale_x = scale

    coords_raw = coords_feat.clone()
    coords_raw[..., 0] = coords_feat[..., 0] * scale_z
    coords_raw[..., 1] = coords_feat[..., 1] * scale_y
    coords_raw[..., 2] = coords_feat[..., 2] * scale_x

    return coords_raw


def raw_to_feature_coords(coords_raw, scale):
    """coords_raw: (...,3), order=(z,y,x)."""
    scale_z, scale_y, scale_x = scale

    coords_feat = coords_raw.clone()
    coords_feat[..., 0] = coords_raw[..., 0] / scale_z
    coords_feat[..., 1] = coords_raw[..., 1] / scale_y
    coords_feat[..., 2] = coords_raw[..., 2] / scale_x

    return coords_feat


def build_raw_control_coords_stride_grid(
    B,
    K,
    Hc,
    Wc,
    z_init,
    control_stride,
    device,
):
    """
    Build raw-space control grid:
        z = z_init[k]
        y = 0, stride, 2*stride, ...
        x = 0, stride, 2*stride, ...
    """
    if z_init is None:
        raise ValueError("z_init or sparse_z_idx must be provided.")

    if not torch.is_tensor(z_init):
        z_init = torch.as_tensor(z_init, device=device)

    z_init = z_init.to(device=device, dtype=torch.float32)

    if z_init.dim() == 1:
        z_init = z_init.unsqueeze(0).repeat(B, 1)

    ys = torch.arange(Hc, device=device).float() * float(control_stride)
    xs = torch.arange(Wc, device=device).float() * float(control_stride)

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    zz = z_init[:, :, None, None].expand(B, K, Hc, Wc)
    yy = yy[None, None, :, :].expand(B, K, Hc, Wc)
    xx = xx[None, None, :, :].expand(B, K, Hc, Wc)

    return torch.stack([zz, yy, xx], dim=-1)


def sample_ref_features_at_feature_coords(
    F_ref,
    coords_feat_zyx,
    padding_mode="border",
    align_corners=True,
):
    """
    Sample reference features at feature-space coordinates.

    F_ref:           (B,C,D,Hf,Wf)
    coords_feat_zyx: (B,N,3), order=(z,y,x)

    Return:
        sampled: (B,N,C)
    """
    B, C, D, Hf, Wf = F_ref.shape

    coords = coords_feat_zyx.to(dtype=F_ref.dtype)
    zf = coords[..., 0]
    yf = coords[..., 1]
    xf = coords[..., 2]

    x_norm = 2.0 * xf / (Wf - 1) - 1.0 if Wf > 1 else torch.zeros_like(xf)
    y_norm = 2.0 * yf / (Hf - 1) - 1.0 if Hf > 1 else torch.zeros_like(yf)
    z_norm = 2.0 * zf / (D - 1) - 1.0 if D > 1 else torch.zeros_like(zf)

    grid = torch.stack([x_norm, y_norm, z_norm], dim=-1)
    grid = grid.view(B, -1, 1, 1, 3)

    sampled = F.grid_sample(
        F_ref,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    sampled = sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1).contiguous()
    return sampled


# ============================================================
# Matcher
# ============================================================

class LocalCrossAttentionBlock(nn.Module):
    """
    Multi-head local cross-attention block.

    Query:
        queries:     (B, N, C)

    Key/Value:
        sampled_ref: (B, N, M, C)

    Optional mask:
        candidate_valid_mask: (B, N, M), bool

    This is local query-to-reference-window cross-attention:
        each query only attends to its own local candidate window.
    """

    def __init__(
        self,
        dim,
        num_heads=4,
        ffn_ratio=2.0,
        attn_temperature=0.20,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        init_gamma=1e-3,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}."
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_temperature = float(attn_temperature)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        hidden_dim = int(dim * ffn_ratio)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_drop),
        )

        # Small residual scale: stable when inserted into existing model.
        self.gamma_attn = nn.Parameter(torch.ones(1) * init_gamma)
        self.gamma_ffn = nn.Parameter(torch.ones(1) * init_gamma)

    def forward(
        self,
        queries,
        sampled_ref,
        candidate_valid_mask=None,
    ):
        """
        Args:
            queries:
                (B, N, C)

            sampled_ref:
                (B, N, M, C)

            candidate_valid_mask:
                (B, N, M), bool. True means valid candidate.

        Returns:
            updated queries:
                (B, N, C)
        """
        B, N, C = queries.shape
        _, _, M, _ = sampled_ref.shape
        H = self.num_heads
        Dh = self.head_dim

        q_in = self.norm_q(queries)
        r_in = self.norm_kv(sampled_ref)

        q = self.q_proj(q_in).view(B, N, H, Dh)
        k = self.k_proj(r_in).view(B, N, M, H, Dh)
        v = self.v_proj(r_in).view(B, N, M, H, Dh)

        # Use normalized Q/K for stable local attention.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # logits: (B, N, H, M)
        attn_logits = torch.einsum("bnhd,bnmhd->bnhm", q, k)
        attn_logits = attn_logits / max(float(self.attn_temperature), 1e-6)

        if candidate_valid_mask is not None:
            valid = candidate_valid_mask.to(
                device=attn_logits.device,
                dtype=torch.bool,
            )
            attn_logits = attn_logits.masked_fill(
                ~valid.unsqueeze(2),
                -1e4,
            )

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)

        # context: (B, N, H, Dh)
        context = torch.einsum("bnhm,bnmhd->bnhd", attn, v)
        context = context.reshape(B, N, C)

        attn_out = self.out_proj(context)
        attn_out = self.proj_drop(attn_out)

        queries = queries + self.gamma_attn * attn_out

        ffn_out = self.ffn(self.norm_ffn(queries))
        queries = queries + self.gamma_ffn * ffn_out

        return queries

class LocalQueryToVolumeMatcher(nn.Module):
    """
    Query-to-local-volume matcher.

    Design:
        - normalized dot score
        - optional learned residual pairwise matcher
        - optional relative offset feature embedding
        - optional relative offset scalar bias
        - optional local cross-attention query update
    """

    def __init__(
        self,
        dim=96,
        radius=(4, 3, 3),
        temperature=0.05,
        use_learned_matching=True,
        matcher_mode="hybrid",
        zero_init_residual=True,
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

        # new
        coord_temperature=0.5,
    ):
        super().__init__()

        self.dim = dim
        self.radius = tuple(radius)
        self.temperature = float(temperature)
        self.use_learned_matching = use_learned_matching
        self.matcher_mode = matcher_mode

        self.use_offset_encoding = use_offset_encoding
        self.use_offset_bias = use_offset_bias
        self.use_local_cross_attn = use_local_cross_attn
        self.local_attn_temperature = float(local_attn_temperature)
        self.coord_temperature = float(coord_temperature)

        self.matcher_cross_attn_layers = int(matcher_cross_attn_layers)
        self.matcher_cross_attn_heads = int(matcher_cross_attn_heads)
        self.matcher_ffn_ratio = float(matcher_ffn_ratio)

        if matcher_mode not in ["dot", "mlp", "hybrid"]:
            raise ValueError("matcher_mode must be one of: 'dot', 'mlp', 'hybrid'.")

        if use_learned_matching or matcher_mode in ["mlp", "hybrid"]:
            self.match_mlp = nn.Sequential(
                nn.LayerNorm(dim * 4),
                nn.Linear(dim * 4, dim),
                nn.GELU(),
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1),
            )
            if zero_init_residual:
                nn.init.zeros_(self.match_mlp[-1].weight)
                nn.init.zeros_(self.match_mlp[-1].bias)
        else:
            self.match_mlp = None

        if use_offset_encoding:
            self.offset_feat = nn.Sequential(
                nn.Linear(3, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            nn.init.zeros_(self.offset_feat[-1].weight)
            nn.init.zeros_(self.offset_feat[-1].bias)
        else:
            self.offset_feat = None

        if use_offset_bias:
            self.offset_bias = nn.Sequential(
                nn.Linear(3, dim),
                nn.GELU(),
                nn.Linear(dim, 1),
            )
            nn.init.zeros_(self.offset_bias[-1].weight)
            nn.init.zeros_(self.offset_bias[-1].bias)
        else:
            self.offset_bias = None

        if use_local_cross_attn and self.matcher_cross_attn_layers > 0:
            self.cross_attn_blocks = nn.ModuleList(
                [
                    LocalCrossAttentionBlock(
                        dim=dim,
                        num_heads=matcher_cross_attn_heads,
                        ffn_ratio=matcher_ffn_ratio,
                        attn_temperature=local_attn_temperature,
                        attn_drop=matcher_attn_drop,
                        proj_drop=matcher_proj_drop,
                        init_gamma=matcher_init_gamma,
                    )
                    for _ in range(self.matcher_cross_attn_layers)
                ]
            )
        else:
            self.cross_attn_blocks = None

    def normalize_relative_offset(self, candidate_coords, center_coords):
        rz, ry, rx = self.radius

        rel_offset = candidate_coords - center_coords.unsqueeze(2)
        rel_offset_norm = rel_offset.clone()

        rel_offset_norm[..., 0] = rel_offset_norm[..., 0] / max(float(rz), 1.0)
        rel_offset_norm[..., 1] = rel_offset_norm[..., 1] / max(float(ry), 1.0)
        rel_offset_norm[..., 2] = rel_offset_norm[..., 2] / max(float(rx), 1.0)

        rel_offset_norm = rel_offset_norm.clamp(-2.0, 2.0)
        return rel_offset, rel_offset_norm

    def apply_local_cross_attention(
        self,
        queries,
        sampled_ref,
        candidate_valid_mask=None,
    ):
        """
        Multi-layer local cross-attention.

        queries:
            (B, N, C)

        sampled_ref:
            (B, N, M, C)

        candidate_valid_mask:
            (B, N, M)
        """
        if (
            not self.use_local_cross_attn
            or self.cross_attn_blocks is None
            or len(self.cross_attn_blocks) == 0
        ):
            return queries

        for block in self.cross_attn_blocks:
            queries = block(
                queries=queries,
                sampled_ref=sampled_ref,
                candidate_valid_mask=candidate_valid_mask,
            )

        return queries

    def compute_local_scores(self, queries, sampled_ref, rel_offset_norm=None):
        q = F.normalize(queries, dim=-1)
        r = F.normalize(sampled_ref, dim=-1)

        dot_scores = (q.unsqueeze(2) * r).sum(dim=-1)
        dot_scores = dot_scores / max(self.temperature, 1e-6)

        if self.matcher_mode == "dot" or not self.use_learned_matching:
            scores = dot_scores
        else:
            q_expand = q.unsqueeze(2).expand_as(r)
            pair_feat = torch.cat(
                [
                    q_expand,
                    r,
                    q_expand * r,
                    torch.abs(q_expand - r),
                ],
                dim=-1,
            )
            learned_delta = self.match_mlp(pair_feat).squeeze(-1)

            if self.matcher_mode == "mlp":
                scores = learned_delta
            elif self.matcher_mode == "hybrid":
                scores = dot_scores + learned_delta
            else:
                raise RuntimeError(f"Unexpected matcher_mode: {self.matcher_mode}")

        if self.use_offset_bias and rel_offset_norm is not None:
            scores = scores + self.offset_bias(rel_offset_norm).squeeze(-1)

        return scores

    @staticmethod
    def coordinate_expectation(prob, candidate_coords):
        return torch.sum(prob.unsqueeze(-1) * candidate_coords, dim=2)

    def forward(self, queries, F_ref, center_coords, return_aux=False):
        sampled_ref, candidate_coords, candidate_valid_mask = sample_local_3d_window(
            F_ref=F_ref,
            center_coords=center_coords,
            radius=self.radius,
        )

        _, rel_offset_norm = self.normalize_relative_offset(
            candidate_coords=candidate_coords,
            center_coords=center_coords,
        )

        if self.use_offset_encoding:
            sampled_ref = sampled_ref + self.offset_feat(rel_offset_norm)

        queries = self.apply_local_cross_attention(
            queries=queries,
            sampled_ref=sampled_ref,
            candidate_valid_mask=candidate_valid_mask,
        )

        scores = self.compute_local_scores(
            queries=queries,
            sampled_ref=sampled_ref,
            rel_offset_norm=rel_offset_norm,
        )

        scores = scores.masked_fill(~candidate_valid_mask, -1e4)

        # Match probability used for CE/KL diagnostics and returned aux.
        prob = torch.softmax(scores, dim=-1)

        # Sharper probability used only for coordinate expectation.
        # This keeps the output differentiable, but reduces the conservative
        # averaging effect of a soft distribution.
        prob_coord = torch.softmax(
            scores / max(float(self.coord_temperature), 1e-6),
            dim=-1,
        )

        pred_coords = self.coordinate_expectation(
            prob=prob_coord,
            candidate_coords=candidate_coords,
        )

        if return_aux:
            return pred_coords, prob, scores, candidate_coords, candidate_valid_mask

        return pred_coords, None, None, None, None


# ============================================================
# Chunked local match loss
# ============================================================

def compute_local_match_loss_chunk(
    scores,
    candidate_coords,
    candidate_valid_mask,
    gt_coords_feat,
    valid_mask_chunk=None,
    sigma=(0.5, 1.0, 1.0),
    inside_threshold=4.0,
    eps=1e-8,
):
    """
    Chunk-level CE/KL matching loss.

    scores:               (B,Nc,M)
    candidate_coords:     (B,Nc,M,3), feature-space z,y,x
    candidate_valid_mask: (B,Nc,M)
    gt_coords_feat:       (B,Nc,3), feature-space z,y,x
    valid_mask_chunk:     (B,Nc)
    """
    B, Nc, M = scores.shape
    device = scores.device
    dtype = scores.dtype

    if valid_mask_chunk is None:
        valid_mask_chunk = torch.ones(B, Nc, device=device, dtype=torch.bool)
    else:
        valid_mask_chunk = valid_mask_chunk.to(device=device).bool()

    gt_coords_feat = gt_coords_feat.to(device=device, dtype=dtype)

    sigma_t = torch.tensor(sigma, device=device, dtype=dtype).view(1, 1, 1, 3)

    diff = (candidate_coords - gt_coords_feat.unsqueeze(2)) / sigma_t
    dist2 = torch.sum(diff * diff, dim=-1)

    dist2_valid = dist2.masked_fill(~candidate_valid_mask, 1e8)

    min_dist2, target_idx = dist2_valid.min(dim=-1)
    inside = torch.sqrt(min_dist2.clamp_min(0.0)) <= inside_threshold

    train_valid = valid_mask_chunk & inside
    train_valid_f = train_valid.float()
    valid_count = train_valid_f.sum().clamp_min(1.0)

    scores_valid = scores.masked_fill(~candidate_valid_mask, -1e4)

    ce_per = F.cross_entropy(
        scores_valid.reshape(B * Nc, M),
        target_idx.reshape(B * Nc),
        reduction="none",
    ).reshape(B, Nc)

    loss_ce_sum = (ce_per * train_valid_f).sum()

    target_logits = -0.5 * dist2_valid
    target_logits = target_logits.masked_fill(~candidate_valid_mask, -1e4)

    target_prob = torch.softmax(target_logits, dim=-1)
    log_prob = torch.log_softmax(scores_valid, dim=-1)

    kl_per = torch.sum(
        target_prob * (torch.log(target_prob.clamp_min(eps)) - log_prob),
        dim=-1,
    )

    loss_kl_sum = (kl_per * train_valid_f).sum()

    with torch.no_grad():
        pred_idx = scores_valid.argmax(dim=-1)
        top1 = (pred_idx == target_idx) & train_valid

        k = min(5, M)
        topk_idx = scores_valid.topk(k=k, dim=-1).indices
        top5 = (topk_idx == target_idx.unsqueeze(-1)).any(dim=-1) & train_valid

        prob = torch.softmax(scores_valid, dim=-1)
        pmax = prob.max(dim=-1).values
        entropy = -torch.sum(prob * torch.log(prob.clamp_min(eps)), dim=-1)

        valid_inside_all = valid_mask_chunk & inside
        all_count = torch.tensor(float(B * Nc), device=device, dtype=dtype)

    return {
        "loss_ce_sum": loss_ce_sum,
        "loss_kl_sum": loss_kl_sum,
        "train_valid_count": valid_count.detach(),

        "match_top1_sum": top1.float().sum(),
        "match_top5_sum": top5.float().sum(),
        "pmax_sum": (pmax * train_valid_f).sum(),
        "entropy_sum": (entropy * train_valid_f).sum(),

        "inside_valid_sum": (inside.float() * valid_mask_chunk.float()).sum(),
        "candidate_valid_sum": candidate_valid_mask.float().sum(),
        "candidate_total": torch.tensor(
            float(candidate_valid_mask.numel()),
            device=device,
            dtype=dtype,
        ),

        "inside_all_sum": inside.float().sum(),
        "valid_inside_all_sum": valid_inside_all.float().sum(),
        "all_count": all_count,
    }


# ============================================================
# Optional coordinate residual refinement
# ============================================================

class CoordResidualHead(nn.Module):
    """
    Per-token continuous coordinate residual head.

    Input:
        q_feat:    (B,N,C)
        r_feat:    (B,N,C)
        disp_feat: (B,N,3), feature-space coarse displacement

    Output:
        delta_raw: (B,N,3), raw coordinate residual, order=(z,y,x)
    """

    def __init__(
        self,
        dim,
        hidden_dim=128,
        max_delta=(0.5, 1.0, 1.0),
        use_disp=True,
    ):
        super().__init__()
        self.use_disp = use_disp

        in_dim = dim * 4 + (3 if use_disp else 0)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

        self.register_buffer(
            "max_delta",
            torch.tensor(max_delta, dtype=torch.float32).view(1, 1, 3),
            persistent=False,
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, q_feat, r_feat, disp_feat=None):
        feats = [
            q_feat,
            r_feat,
            q_feat - r_feat,
            q_feat * r_feat,
        ]

        if self.use_disp:
            if disp_feat is None:
                raise ValueError("disp_feat is required when use_disp=True.")
            feats.append(disp_feat)

        x = torch.cat(feats, dim=-1)
        return torch.tanh(self.mlp(x)) * self.max_delta.to(
            device=x.device,
            dtype=x.dtype,
        )


class SpatialCoordResidualHead(nn.Module):
    """
    Spatial residual refiner on the control grid.

    Input:
        feat: (B,N,Cin), N=K*Hc*Wc

    Output:
        delta_raw: (B,K,Hc,Wc,3)
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        num_blocks=3,
        max_delta=(1.0, 2.0, 2.0),
        use_3d=True,
    ):
        super().__init__()
        self.use_3d = use_3d

        self.register_buffer(
            "max_delta",
            torch.tensor(max_delta, dtype=torch.float32).view(1, 1, 1, 1, 3),
            persistent=False,
        )

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        blocks = []
        for _ in range(int(num_blocks)):
            kernel_size = (3, 3, 3) if use_3d else (1, 3, 3)
            padding = (1, 1, 1) if use_3d else (0, 1, 1)

            blocks.append(
                nn.Sequential(
                    nn.Conv3d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(8, hidden_dim),
                    nn.GELU(),
                    nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(8, hidden_dim),
                    nn.GELU(),
                )
            )

        self.blocks = nn.ModuleList(blocks)

        self.out_proj = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, 3, kernel_size=1),
        )

        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(self, feat, K, Hc, Wc):
        B, N, C = feat.shape
        if N != K * Hc * Wc:
            raise ValueError(f"N={N} does not match K*Hc*Wc={K * Hc * Wc}.")

        x = self.in_proj(feat)
        x = x.view(B, K, Hc, Wc, -1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        for block in self.blocks:
            x = x + block(x)

        delta = self.out_proj(x)
        delta = delta.permute(0, 2, 3, 4, 1).contiguous()

        return torch.tanh(delta) * self.max_delta.to(
            device=delta.device,
            dtype=delta.dtype,
        )


# ============================================================
# Full model: CoarseMatchingNet
# ============================================================

class CoarseMatchingNet(nn.Module):
    """
    Sparse moving control-point queries -> dense reference volume memory.

    Design:
        1. moving feature:    XY /8
        2. reference feature: XY /8, no Z downsampling
        3. reference enhancement: 6 local 3D window attention layers by default
        4. matching path remains local query-to-volume matching, compatible
           with the existing chunked CE/KL training path.

    Inputs:
        mov:       (B,1,K,H,W)
        ref:       (B,1,D,H,W)
        z_init:    (K,) or (B,K), raw reference z coordinate for each moving slice
        spacing:   optional (B,6) or (6,)
                   [sx_ref,sy_ref,sz_ref,sx_mov,sy_mov,sz_mov]

    Outputs:
        pred_coords:      (B,K,Hc,Wc,3), raw z,y,x
        pred_disp:        (B,K,Hc,Wc,3), raw z,y,x
        coords0:          (B,K,Hc,Wc,3), raw z,y,x
        pred_coords_feat: (B,K,Hc,Wc,3), reference feature-space z,y,x
        coords0_feat:     (B,K,Hc,Wc,3), reference feature-space z,y,x
    """

    def __init__(
        self,
        dim=96,
        radius=(4, 3, 3),
        temperature=0.05,
        use_learned_matching=True,
        num_refine_iters=1,
        control_stride=8,
        encoder_stride=8,
        matcher_mode="hybrid",
        query_chunk_size=512,

        # Moving encoder
        moving_base_channels=(24, 48, 96),
        moving_num_blocks=(1, 2, 1),
        moving_mlp_ratio=2.0,
        moving_window_attn_layers=6,
        moving_window_size=8,
        moving_attn_num_heads=4,
        moving_slice_fusion_blocks=1,

        # Reference encoder
        ref_base_channels=(24, 48, 96),
        ref_num_blocks=(1, 2, 2),
        ref_refine_blocks=1,
        ref_mlp_ratio=2.0,
        ref_attn_layers=6,
        ref_attn_num_heads=4,
        ref_attn_window_size=(4, 8, 8),
        ref_attn_mlp_ratio=2.0,

        # Embeddings
        use_coord_embed=True,
        use_spacing_embed=True,

        # Matcher options
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
        coord_temperature=0.5,

        # Optional residual coordinate refinement
        use_coord_residual=False,
        residual_type="mlp",          # "mlp" or "spatial"
        residual_hidden_dim=128,
        residual_max_delta=(0.5, 1.0, 1.0),
        residual_use_disp=True,
        residual_detach_coarse=True,
        residual_detach_features=True,
        residual_num_blocks=3,
        residual_use_3d=True,
    ):
        super().__init__()

        self.dim = dim
        self.num_refine_iters = int(num_refine_iters)
        self.control_stride = int(control_stride)
        self.encoder_stride = int(encoder_stride)
        self.query_chunk_size = int(query_chunk_size)

        self.use_coord_embed = bool(use_coord_embed)
        self.use_spacing_embed = bool(use_spacing_embed)

        self.use_coord_residual = bool(use_coord_residual)
        self.residual_type = residual_type
        self.residual_use_disp = bool(residual_use_disp)
        self.residual_detach_coarse = bool(residual_detach_coarse)
        self.residual_detach_features = bool(residual_detach_features)

        if self.control_stride % self.encoder_stride != 0:
            raise ValueError(
                f"control_stride={control_stride} must be divisible by "
                f"encoder_stride={encoder_stride}."
            )

        self.query_downsample = self.control_stride // self.encoder_stride

        # ----------------------------------------------------
        # 1. Moving and reference encoders
        # ----------------------------------------------------
        self.moving_encoder = MovingQueryEncoder(
            in_ch=1,
            dim=dim,
            base_channels=moving_base_channels,
            num_blocks=moving_num_blocks,
            mlp_ratio=moving_mlp_ratio,
            use_window_attn=moving_window_attn_layers > 0,
            window_attn_layers=moving_window_attn_layers,
            window_size=moving_window_size,
            num_heads=moving_attn_num_heads,
            slice_fusion_blocks=moving_slice_fusion_blocks,
        )

        self.reference_encoder = ReferenceMemoryEncoder(
            in_ch=1,
            dim=dim,
            base_channels=ref_base_channels,
            num_blocks=ref_num_blocks,
            refine_blocks=ref_refine_blocks,
            mlp_ratio=ref_mlp_ratio,
            attn_layers=ref_attn_layers,
            attn_num_heads=ref_attn_num_heads,
            attn_window_size=ref_attn_window_size,
            attn_mlp_ratio=ref_attn_mlp_ratio,
        )

        # ----------------------------------------------------
        # 2. Local matcher
        # ----------------------------------------------------
        self.matcher = LocalQueryToVolumeMatcher(
            dim=dim,
            radius=radius,
            temperature=temperature,
            use_learned_matching=use_learned_matching,
            matcher_mode=matcher_mode,
            zero_init_residual=True,
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

            # new
            coord_temperature=coord_temperature,
        )

        # ----------------------------------------------------
        # 3. Token projections and embeddings
        # ----------------------------------------------------
        if self.use_coord_embed:
            self.coord_mlp = nn.Sequential(
                nn.Linear(3, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            nn.init.zeros_(self.coord_mlp[-1].weight)
            nn.init.zeros_(self.coord_mlp[-1].bias)
        else:
            self.coord_mlp = None

        if self.use_spacing_embed:
            self.spacing_mlp = nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(4, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            nn.init.zeros_(self.spacing_mlp[-1].weight)
            nn.init.zeros_(self.spacing_mlp[-1].bias)
        else:
            self.spacing_mlp = None

        self.query_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.ref_proj = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
        )

        # ----------------------------------------------------
        # 4. Optional coordinate residual refinement
        # ----------------------------------------------------
        if self.use_coord_residual:
            if residual_type == "mlp":
                self.coord_residual_refiner = CoordResidualHead(
                    dim=dim,
                    hidden_dim=residual_hidden_dim,
                    max_delta=residual_max_delta,
                    use_disp=residual_use_disp,
                )
            elif residual_type == "spatial":
                residual_in_dim = dim * 4 + (3 if residual_use_disp else 0)
                self.coord_residual_refiner = SpatialCoordResidualHead(
                    in_dim=residual_in_dim,
                    hidden_dim=residual_hidden_dim,
                    num_blocks=residual_num_blocks,
                    max_delta=residual_max_delta,
                    use_3d=residual_use_3d,
                )
            else:
                raise ValueError(
                    f"Unknown residual_type={residual_type}. "
                    "Expected 'mlp' or 'spatial'."
                )
        else:
            self.coord_residual_refiner = None

    # ========================================================
    # Utility methods
    # ========================================================

    @staticmethod
    def check_z_init(z_init, B, K, device):
        if z_init is None:
            raise ValueError(
                "z_init or sparse_z_idx must be provided. "
                "Default arange(K) is unsafe for sparse stack-to-volume matching."
            )

        if not torch.is_tensor(z_init):
            z_init = torch.as_tensor(z_init, device=device)

        z_init = z_init.to(device=device, dtype=torch.float32)

        if z_init.dim() == 1:
            if z_init.numel() != K:
                raise ValueError(
                    f"z_init has shape {tuple(z_init.shape)}, but K={K}. "
                    f"Expected z_init shape (K,) or (B,K)."
                )
            z_init = z_init.unsqueeze(0).repeat(B, 1)
        elif z_init.dim() == 2:
            if z_init.shape != (B, K):
                raise ValueError(
                    f"z_init has shape {tuple(z_init.shape)}, but expected {(B, K)}."
                )
        else:
            raise ValueError(
                f"z_init must have shape (K,) or (B,K), got {tuple(z_init.shape)}."
            )

        return z_init

    @staticmethod
    def normalize_raw_coords_for_embedding(coords_raw, ref):
        """
        Normalize raw z,y,x coords to [-1,1] for coordinate embedding.
        """
        _, _, D, H, W = ref.shape

        z = coords_raw[..., 0] / max(D - 1, 1)
        y = coords_raw[..., 1] / max(H - 1, 1)
        x = coords_raw[..., 2] / max(W - 1, 1)

        coords_norm = torch.stack([z, y, x], dim=-1)
        return coords_norm * 2.0 - 1.0

    @staticmethod
    def build_spacing_features(spacing, K, device, dtype):
        """
        spacing:
            (B,6) or (6,)
            [sx_ref,sy_ref,sz_ref,sx_mov,sy_mov,sz_mov]

        Return:
            spacing_feat: (B,4)
        """
        if spacing is None:
            return None

        if not torch.is_tensor(spacing):
            spacing = torch.as_tensor(spacing, device=device)

        spacing = spacing.to(device=device, dtype=dtype)

        if spacing.dim() == 1:
            spacing = spacing.unsqueeze(0)

        sx_ref = spacing[:, 0]
        sy_ref = spacing[:, 1]
        sz_ref = spacing[:, 2]
        sx_mov = spacing[:, 3]
        sy_mov = spacing[:, 4]
        sz_mov = spacing[:, 5]

        xy_ref = 0.5 * (sx_ref + sy_ref)
        xy_mov = 0.5 * (sx_mov + sy_mov)

        K_tensor = torch.full_like(sz_ref, float(K))

        return torch.stack(
            [
                torch.log(sz_ref / (xy_ref + 1e-8) + 1e-8),
                torch.log(sz_mov / (xy_mov + 1e-8) + 1e-8),
                torch.log(sz_mov / (sz_ref + 1e-8) + 1e-8),
                torch.log(K_tensor + 1e-8),
            ],
            dim=-1,
        )

    def downsample_query_features(self, F_mov, input_hw):
        """
        F_mov: (B,K,C,Hf,Wf)
        input_hw: raw moving H,W

        Return: (B,K,C,ceil(H/control_stride),ceil(W/control_stride))

        For the default setting, moving encoder output stride and
        control_stride are both 8, so this is usually shape-preserving.
        """
        H, W = input_hw
        target_h = math.ceil(H / self.control_stride)
        target_w = math.ceil(W / self.control_stride)

        B, K, C, Hf, Wf = F_mov.shape

        x = F_mov.permute(0, 2, 1, 3, 4).contiguous()  # (B,C,K,Hf,Wf)
        x = F.adaptive_avg_pool3d(x, output_size=(K, target_h, target_w))

        return x.permute(0, 2, 1, 3, 4).contiguous()

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        mov,
        ref,
        z_init=None,
        sparse_z_idx=None,
        spacing=None,
        return_match_aux=False,

        compute_chunk_match_loss=False,
        gt_coords=None,
        valid_mask=None,
        match_sigma=(0.5, 1.0, 1.0),
        match_inside_threshold=4.0,
    ):
        if z_init is None:
            z_init = sparse_z_idx

        B, _, K_in, H_in, W_in = mov.shape

        z_init = self.check_z_init(
            z_init=z_init,
            B=B,
            K=K_in,
            device=mov.device,
        )

        # ----------------------------------------------------
        # 1. Encode moving queries to XY /8
        # ----------------------------------------------------
        F_mov = self.moving_encoder(mov)
        F_mov = self.downsample_query_features(
            F_mov,
            input_hw=(H_in, W_in),
        )

        B, K, C, Hc, Wc = F_mov.shape

        # ----------------------------------------------------
        # 2. Initial raw control coordinates
        # ----------------------------------------------------
        coords0_raw = build_raw_control_coords_stride_grid(
            B=B,
            K=K,
            Hc=Hc,
            Wc=Wc,
            z_init=z_init,
            control_stride=self.control_stride,
            device=mov.device,
        )

        # ----------------------------------------------------
        # 3. Encode reference memory to XY /8
        # ----------------------------------------------------
        F_ref = self.reference_encoder(ref)
        F_ref = self.ref_proj(F_ref)

        # ----------------------------------------------------
        # 4. Build query tokens
        # ----------------------------------------------------
        queries = F_mov.permute(0, 1, 3, 4, 2).contiguous()
        queries = queries.reshape(B, K * Hc * Wc, C)

        if self.use_coord_embed:
            coords0_norm = self.normalize_raw_coords_for_embedding(
                coords_raw=coords0_raw,
                ref=ref,
            )
            coord_embed = self.coord_mlp(
                coords0_norm.reshape(B, K * Hc * Wc, 3)
            )
            queries = queries + coord_embed

        if self.use_spacing_embed and spacing is not None:
            spacing_feat = self.build_spacing_features(
                spacing=spacing,
                K=K,
                device=mov.device,
                dtype=queries.dtype,
            )
            if spacing_feat.shape[0] == 1 and B > 1:
                spacing_feat = spacing_feat.repeat(B, 1)

            spacing_embed = self.spacing_mlp(spacing_feat)
            queries = queries + spacing_embed[:, None, :]

        queries = self.query_proj(queries)

        # ----------------------------------------------------
        # 5. Raw coords -> reference feature coords
        # ----------------------------------------------------
        scale = get_ref_feature_scale(ref, F_ref)

        coords0_feat = raw_to_feature_coords(coords0_raw, scale)
        coords_feat = coords0_feat.reshape(B, K * Hc * Wc, 3)

        gt_coords_feat_flat = None
        valid_mask_flat = None

        if compute_chunk_match_loss:
            if gt_coords is None:
                raise ValueError("compute_chunk_match_loss=True requires gt_coords.")

            gt_coords = gt_coords.to(device=mov.device, dtype=coords0_feat.dtype)
            gt_coords_feat = raw_to_feature_coords(gt_coords, scale)
            gt_coords_feat_flat = gt_coords_feat.reshape(B, K * Hc * Wc, 3)

            if valid_mask is None:
                valid_mask_flat = torch.ones(
                    B,
                    K * Hc * Wc,
                    device=mov.device,
                    dtype=torch.bool,
                )
            else:
                valid_mask_flat = valid_mask.to(device=mov.device).bool()
                valid_mask_flat = valid_mask_flat.reshape(B, K * Hc * Wc)

        # ----------------------------------------------------
        # 6. Local query-to-reference matching
        # ----------------------------------------------------
        last_prob = None
        last_scores = None
        last_candidate_coords = None
        last_candidate_valid_mask = None
        chunk_match_stats = None

        for it in range(self.num_refine_iters):
            compute_loss_this_iter = (
                compute_chunk_match_loss
                and it == self.num_refine_iters - 1
            )

            (
                coords_feat,
                last_prob,
                last_scores,
                last_candidate_coords,
                last_candidate_valid_mask,
                chunk_match_stats,
            ) = self.match_queries_in_chunks(
                queries=queries,
                F_ref=F_ref,
                coords=coords_feat,
                return_match_aux=return_match_aux,
                compute_chunk_match_loss=compute_loss_this_iter,
                gt_coords_feat=gt_coords_feat_flat,
                valid_mask_flat=valid_mask_flat,
                match_sigma=match_sigma,
                match_inside_threshold=match_inside_threshold,
            )

        # ----------------------------------------------------
        # 7. Feature coords -> raw coords
        # ----------------------------------------------------
        pred_coords_feat = coords_feat.reshape(B, K, Hc, Wc, 3)
        coords0_feat = coords0_feat.reshape(B, K, Hc, Wc, 3)

        pred_coords_raw_coarse = feature_to_raw_coords(pred_coords_feat, scale)
        pred_disp_raw_coarse = pred_coords_raw_coarse - coords0_raw

        coord_scale = torch.tensor(
            scale,
            device=mov.device,
            dtype=pred_coords_raw_coarse.dtype,
        )

        pred_coords_raw = pred_coords_raw_coarse
        pred_disp_raw = pred_disp_raw_coarse
        residual_delta_raw = None

        # ----------------------------------------------------
        # 8. Optional coordinate residual refinement
        # ----------------------------------------------------
        if self.use_coord_residual:
            N = K * Hc * Wc

            q_feat = queries
            pred_coords_feat_for_sample = pred_coords_feat
            coords0_feat_for_res = coords0_feat

            if self.residual_detach_coarse:
                pred_coords_feat_for_sample = pred_coords_feat_for_sample.detach()
                coords0_feat_for_res = coords0_feat_for_res.detach()

            r_feat = sample_ref_features_at_feature_coords(
                F_ref=F_ref,
                coords_feat_zyx=pred_coords_feat_for_sample.reshape(B, N, 3),
                padding_mode="border",
                align_corners=True,
            )

            if self.residual_detach_features:
                q_feat = q_feat.detach()
                r_feat = r_feat.detach()

            disp_feat = (
                pred_coords_feat_for_sample - coords0_feat_for_res
            ).reshape(B, N, 3)

            if self.residual_detach_coarse:
                disp_feat = disp_feat.detach()

            if self.residual_type == "mlp":
                residual_delta_raw = self.coord_residual_refiner(
                    q_feat=q_feat,
                    r_feat=r_feat,
                    disp_feat=disp_feat if self.residual_use_disp else None,
                )
                residual_delta_raw = residual_delta_raw.reshape(B, K, Hc, Wc, 3)

            elif self.residual_type == "spatial":
                if self.residual_use_disp:
                    residual_feat = torch.cat(
                        [
                            q_feat,
                            r_feat,
                            q_feat - r_feat,
                            q_feat * r_feat,
                            disp_feat,
                        ],
                        dim=-1,
                    )
                else:
                    residual_feat = torch.cat(
                        [
                            q_feat,
                            r_feat,
                            q_feat - r_feat,
                            q_feat * r_feat,
                        ],
                        dim=-1,
                    )

                residual_delta_raw = self.coord_residual_refiner(
                    residual_feat,
                    K=K,
                    Hc=Hc,
                    Wc=Wc,
                )
            else:
                raise ValueError(f"Unknown residual_type={self.residual_type}")

            pred_coords_raw = pred_coords_raw_coarse + residual_delta_raw
            pred_disp_raw = pred_coords_raw - coords0_raw
            pred_coords_feat = raw_to_feature_coords(pred_coords_raw, scale)

        # ----------------------------------------------------
        # 9. Outputs
        # ----------------------------------------------------
        out = {
            "pred_coords": pred_coords_raw,
            "pred_disp": pred_disp_raw,
            "coords0": coords0_raw,

            "pred_coords_feat": pred_coords_feat,
            "coords0_feat": coords0_feat,

            "coord_scale": coord_scale,
        }

        if self.use_coord_residual:
            out["pred_coords_coarse"] = pred_coords_raw_coarse
            out["pred_disp_coarse"] = pred_disp_raw_coarse
            out["residual_delta"] = residual_delta_raw
            out["coord_residual_delta"] = residual_delta_raw

        if chunk_match_stats is not None:
            count = chunk_match_stats["train_valid_count"].clamp_min(1.0)

            out["loss_match_kl_chunked"] = (
                chunk_match_stats["loss_kl_sum"] / count
            )
            out["loss_match_ce_chunked"] = (
                chunk_match_stats["loss_ce_sum"] / count
            )
            out["match_top1_chunked"] = (
                chunk_match_stats["match_top1_sum"] / count
            )
            out["match_top5_chunked"] = (
                chunk_match_stats["match_top5_sum"] / count
            )
            out["match_prob_max_chunked"] = (
                chunk_match_stats["pmax_sum"] / count
            )
            out["match_entropy_chunked"] = (
                chunk_match_stats["entropy_sum"] / count
            )
            out["match_inside_valid_chunked"] = (
                chunk_match_stats["inside_valid_sum"] / count
            )
            out["candidate_valid_ratio_chunked"] = (
                chunk_match_stats["candidate_valid_sum"]
                / chunk_match_stats["candidate_total"].clamp_min(1.0)
            )
            out["match_inside_all_chunked"] = (
                chunk_match_stats["inside_all_sum"]
                / chunk_match_stats["all_count"].clamp_min(1.0)
            )
            out["match_valid_and_inside_all_chunked"] = (
                chunk_match_stats["valid_inside_all_sum"]
                / chunk_match_stats["all_count"].clamp_min(1.0)
            )

        if return_match_aux:
            out.update(
                {
                    "prob": last_prob,
                    "scores": last_scores,
                    "candidate_coords_feat": last_candidate_coords,
                    "candidate_valid_mask": last_candidate_valid_mask,
                    "coord_scale": coord_scale,
                }
            )

            if last_prob is not None:
                out["confidence"] = last_prob.max(dim=-1).values
                out["entropy"] = -torch.sum(
                    last_prob * torch.log(last_prob.clamp_min(1e-8)),
                    dim=-1,
                )

        return out

    def match_queries_in_chunks(
        self,
        queries,
        F_ref,
        coords,
        return_match_aux=False,
        compute_chunk_match_loss=False,
        gt_coords_feat=None,
        valid_mask_flat=None,
        match_sigma=(0.5, 1.0, 1.0),
        match_inside_threshold=4.0,
    ):
        pred_chunks = []

        if return_match_aux:
            prob_chunks = []
            score_chunks = []
            cand_chunks = []
            valid_chunks = []
        else:
            prob_chunks = None
            score_chunks = None
            cand_chunks = None
            valid_chunks = None

        stats = None
        if compute_chunk_match_loss:
            zero = queries.sum() * 0.0
            stats = {
                "loss_kl_sum": zero,
                "loss_ce_sum": zero,
                "train_valid_count": zero,

                "match_top1_sum": zero,
                "match_top5_sum": zero,
                "pmax_sum": zero,
                "entropy_sum": zero,

                "inside_valid_sum": zero,
                "candidate_valid_sum": zero,
                "candidate_total": zero,

                "inside_all_sum": zero,
                "valid_inside_all_sum": zero,
                "all_count": zero,
            }

        B, N, C = queries.shape

        for start in range(0, N, self.query_chunk_size):
            end = min(start + self.query_chunk_size, N)

            q_chunk = queries[:, start:end, :]
            c_chunk = coords[:, start:end, :]

            need_aux_this_chunk = return_match_aux or compute_chunk_match_loss

            pred_chunk, prob_chunk, score_chunk, cand_chunk, valid_chunk = self.matcher(
                queries=q_chunk,
                F_ref=F_ref,
                center_coords=c_chunk,
                return_aux=need_aux_this_chunk,
            )

            pred_chunks.append(pred_chunk)

            if compute_chunk_match_loss:
                if gt_coords_feat is None:
                    raise ValueError(
                        "compute_chunk_match_loss=True requires gt_coords_feat."
                    )

                gt_chunk = gt_coords_feat[:, start:end, :]

                valid_chunk_mask = (
                    None
                    if valid_mask_flat is None
                    else valid_mask_flat[:, start:end]
                )

                chunk_stats = compute_local_match_loss_chunk(
                    scores=score_chunk,
                    candidate_coords=cand_chunk,
                    candidate_valid_mask=valid_chunk,
                    gt_coords_feat=gt_chunk,
                    valid_mask_chunk=valid_chunk_mask,
                    sigma=match_sigma,
                    inside_threshold=match_inside_threshold,
                )

                for k, v in chunk_stats.items():
                    stats[k] = stats[k] + v

            if return_match_aux:
                prob_chunks.append(prob_chunk)
                score_chunks.append(score_chunk)
                cand_chunks.append(cand_chunk)
                valid_chunks.append(valid_chunk)

        pred_coords = torch.cat(pred_chunks, dim=1)

        if return_match_aux:
            prob = torch.cat(prob_chunks, dim=1)
            scores = torch.cat(score_chunks, dim=1)
            candidate_coords = torch.cat(cand_chunks, dim=1)
            candidate_valid_mask = torch.cat(valid_chunks, dim=1)
        else:
            prob = None
            scores = None
            candidate_coords = None
            candidate_valid_mask = None

        return pred_coords, prob, scores, candidate_coords, candidate_valid_mask, stats


__all__ = [
    "CoarseMatchingNet",
    "MovingQueryEncoder",
    "ReferenceMemoryEncoder",
    "LocalQueryToVolumeMatcher",
    "compute_local_match_loss_chunk",
    "build_raw_control_coords_stride_grid",
    "get_ref_feature_scale",
    "raw_to_feature_coords",
    "feature_to_raw_coords",
]
