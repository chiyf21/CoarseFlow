import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class ConvBlock2D(nn.Module):
    """
    Input : (B*K, Cin, H, W)
    Output: (B*K, Cout, H/stride, W/stride)
    """
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)

class ConvBlock3D(nn.Module):
    """
    Input : (B, Cin, D, H, W)
    Output: (B, Cout, D/stride_z, H/stride_y, W/stride_x)
    """
    def __init__(self, cin, cout, stride=(1, 1, 1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)

class WindowSelfAttention2D(nn.Module):
    """
    Lightweight local window attention for each moving slice.

    Input : (B*K, C, Hc, Wc)
    Output: (B*K, C, Hc, Wc)
    """
    def __init__(self, dim, num_heads=4, window_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x_pad.shape[-2:]

        # (B, C, Hp, Wp) -> windows: (num_windows*B, ws*ws, C)
        x_win = x_pad.unfold(2, ws, ws).unfold(3, ws, ws)
        x_win = x_win.permute(0, 2, 3, 4, 5, 1).contiguous()
        x_win = x_win.view(-1, ws * ws, C)

        h = self.norm(x_win)
        attn_out, _ = self.attn(h, h, h)
        x_win = x_win + attn_out
        x_win = x_win + self.ffn(self.norm(x_win))

        # restore
        x_win = x_win.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x_win = x_win.permute(0, 5, 1, 3, 2, 4).contiguous()
        x_pad = x_win.view(B, C, Hp, Wp)

        return x_pad[:, :, :H, :W]

class WindowSelfAttention3D(nn.Module):
    """
    3D window self-attention for reference memory feature.

    Input:
        x: (B, C, D, H, W)

    Output:
        x: (B, C, D, H, W)

    Notes:
        - Attention is computed within local 3D windows.
        - This block is residual and initialized close to identity.
        - It is safe to insert into pretrained ConvNeXt encoder.
    """

    def __init__(
        self,
        dim,
        num_heads=4,
        window_size=(3, 8, 8),
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
        self.window_size = window_size

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

        # Very important:
        # initialize attention branch close to identity.
        self.gamma_attn = nn.Parameter(torch.ones(1) * init_gamma)
        self.gamma_mlp = nn.Parameter(torch.ones(1) * init_gamma)

    def _window_partition(self, x):
        """
        x: (B, Dp, Hp, Wp, C)
        return windows: (B*num_windows, Wd*Wh*Ww, C)
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

        windows = x.permute(
            0, 1, 3, 5, 2, 4, 6, 7
        ).contiguous()

        windows = windows.view(
            -1,
            Wd * Wh * Ww,
            C,
        )

        return windows

    def _window_reverse(self, windows, B, Dp, Hp, Wp):
        """
        windows: (B*num_windows, Wd*Wh*Ww, C)
        return x: (B, Dp, Hp, Wp, C)
        """
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

        x = x.permute(
            0, 1, 4, 2, 5, 3, 6, 7
        ).contiguous()

        x = x.view(B, Dp, Hp, Wp, C)

        return x

    def _attention_forward(self, x):
        """
        x: (B, C, D, H, W)
        return: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        Wd, Wh, Ww = self.window_size

        # (B, C, D, H, W) -> (B, D, H, W, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        # pad D/H/W to multiples of window size
        pad_d = (Wd - D % Wd) % Wd
        pad_h = (Wh - H % Wh) % Wh
        pad_w = (Ww - W % Ww) % Ww

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # F.pad works on last dimensions.
            # Current x is (B, D, H, W, C), so move C before padding.
            x_ch = x.permute(0, 4, 1, 2, 3).contiguous()
            x_ch = F.pad(
                x_ch,
                pad=(0, pad_w, 0, pad_h, 0, pad_d),
                mode="constant",
                value=0.0,
            )
            x = x_ch.permute(0, 2, 3, 4, 1).contiguous()

        Dp = D + pad_d
        Hp = H + pad_h
        Wp = W + pad_w

        # window partition
        windows = self._window_partition(x)  # (B*nW, N, C)

        # window attention
        windows_norm = self.norm1(windows)

        qkv = self.qkv(windows_norm)
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

        # reverse windows
        x_out = self._window_reverse(out, B, Dp, Hp, Wp)

        # remove padding
        x_out = x_out[:, :D, :H, :W, :].contiguous()

        # (B, D, H, W, C) -> (B, C, D, H, W)
        x_out = x_out.permute(0, 4, 1, 2, 3).contiguous()

        return x_out

    def _mlp_forward(self, x):
        """
        x: (B, C, D, H, W)
        return: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape

        x_perm = x.permute(0, 2, 3, 4, 1).contiguous()
        x_norm = self.norm2(x_perm)
        x_mlp = self.mlp(x_norm)

        x_mlp = x_mlp.permute(0, 4, 1, 2, 3).contiguous()

        return x_mlp

    def forward(self, x):
        # attention residual
        x = x + self.gamma_attn * self._attention_forward(x)

        # MLP residual
        x = x + self.gamma_mlp * self._mlp_forward(x)

        return x
    
class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        # x: (B,C,H,W)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class LayerNorm3d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        # x: (B,C,D,H,W)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

class ConvNeXtBlock2D(nn.Module):
    """
    Input/Output: (B,C,H,W)
    """
    def __init__(self, dim, mlp_ratio=4.0, layer_scale_init=1e-6):
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)

        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim),
            requires_grad=True,
        )

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

class ConvNeXtBlock3D(nn.Module):
    """
    Input/Output: (B,C,D,H,W)
    """
    def __init__(self, dim, mlp_ratio=4.0, layer_scale_init=1e-6):
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)

        self.dwconv = nn.Conv3d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)

        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim),
            requires_grad=True,
        )

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

class ConvNeXtBlock3DAniso(nn.Module):
    def __init__(self, dim, kernel_size=(3, 7, 7), mlp_ratio=4.0, layer_scale_init=1e-6):
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
# ============================================================
# Moving Query Encoder
# ============================================================

class MovingQueryEncoder(nn.Module):
    """
    Input:
        I_mov: (B, 1, K, H, W)

    Output:
        F_mov: (B, K, C, Hc, Wc)
    """
    def __init__(self, in_ch=1, dim=128):
        super().__init__()

        self.slice_encoder = nn.Sequential(
            ConvBlock2D(in_ch, 32, stride=2),    # H/2, W/2
            ConvBlock2D(32, 64, stride=2),       # H/4, W/4
            ConvBlock2D(64, dim, stride=1),      # H/4, W/4
        )

        self.slice_attn = WindowSelfAttention2D(
            dim=dim,
            num_heads=4,
            window_size=8,
        )

        self.slice_fusion = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )

    def forward(self, mov):
        B, C, K, H, W = mov.shape

        # (B, 1, K, H, W) -> (B*K, 1, H, W)
        x = mov.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * K, C, H, W)

        # (B*K, dim, Hc, Wc)
        x = self.slice_encoder(x)
        x = self.slice_attn(x)

        _, dim, Hc, Wc = x.shape

        # (B*K, dim, Hc, Wc) -> (B, K, dim, Hc, Wc)
        x = x.view(B, K, dim, Hc, Wc)

        # 3D fusion over sparse K dimension
        # (B, K, C, Hc, Wc) -> (B, C, K, Hc, Wc)
        x3d = x.permute(0, 2, 1, 3, 4).contiguous()
        x3d = self.slice_fusion(x3d)

        # (B, C, K, Hc, Wc) -> (B, K, C, Hc, Wc)
        F_mov = x3d.permute(0, 2, 1, 3, 4).contiguous()
        return F_mov

class MovingQueryEncoderV2(nn.Module):
    """
    Stronger moving encoder.

    Input:
        mov: (B,1,K,H,W)

    Output:
        F_mov: (B,K,C,H/4,W/4)
    """
    def __init__(
        self,
        in_ch=1,
        dim=96,
        num_convnext_blocks=3,
        use_window_attn=True,
        window_size=8,
        num_heads=4,
        window_attn_layers=1,
    ):
        super().__init__()

        self.use_window_attn = use_window_attn

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )

        self.stage1 = nn.Sequential(
            ConvNeXtBlock2D(32),
            ConvNeXtBlock2D(32),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        self.stage2 = nn.Sequential(
            *[ConvNeXtBlock2D(64) for _ in range(num_convnext_blocks)]
        )

        self.proj = nn.Sequential(
            nn.Conv2d(64, dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )

        if use_window_attn:
            self.slice_attn = WindowSelfAttention2D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
            )
        else:
            self.slice_attn = nn.Identity()

        extra_attn_layers = max(int(window_attn_layers) - 1, 0)

        if use_window_attn and extra_attn_layers > 0:
            self.extra_slice_attn = nn.Sequential(
                *[
                    WindowSelfAttention2D(
                        dim=dim,
                        num_heads=num_heads,
                        window_size=window_size,
                    )
                    for _ in range(extra_attn_layers)
                ]
            )
        else:
            self.extra_slice_attn = nn.Identity()

        self.slice_fusion = nn.Sequential(
            ConvNeXtBlock3D(dim),
            ConvNeXtBlock3D(dim),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )

    def forward(self, mov):
        B, C, K, H, W = mov.shape

        x = mov.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * K, C, H, W)

        x = self.stem(x)       # (B*K,32,H/2,W/2)
        x = self.stage1(x)
        x = self.down1(x)      # (B*K,64,H/4,W/4)
        x = self.stage2(x)
        x = self.proj(x)       # (B*K,dim,H/4,W/4)

        x = self.slice_attn(x)
        x = self.extra_slice_attn(x)
        _, dim, Hf, Wf = x.shape

        x = x.view(B, K, dim, Hf, Wf)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B,dim,K,Hf,Wf)

        x = self.slice_fusion(x)

        F_mov = x.permute(0, 2, 1, 3, 4).contiguous()
        return F_mov
# ============================================================
# Reference Dense Memory Encoder
# ============================================================

class ReferenceMemoryEncoder(nn.Module):
    """
    Input:
        I_ref: (B, 1, D, H, W)

    Output:
        F_ref: (B, C, Dr, Hr, Wr)
    """
    def __init__(self, in_ch=1, dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock3D(in_ch, 32, stride=(1, 2, 2)),   # D, H/2, W/2
            ConvBlock3D(32, 64, stride=(1, 2, 2)),      # D, H/4, W/4
            ConvBlock3D(64, dim, stride=(1, 1, 1)),     # D, H/4, W/4
        )

        self.refine = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )

    def forward(self, ref):
        F_ref = self.encoder(ref)
        F_ref = self.refine(F_ref)
        return F_ref

class ReferenceMemoryEncoderV2(nn.Module):
    """
    Stronger reference memory encoder.

    Input:
        ref: (B,1,D,H,W)

    Output:
        F_ref: (B,C,D,H/4,W/4)
    """
    def __init__(
        self,
        in_ch=1,
        dim=96,
        num_blocks=(2, 3, 3),
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(
                in_ch,
                32,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )

        self.stage1 = nn.Sequential(
            *[ConvNeXtBlock3D(32) for _ in range(num_blocks[0])]
        )

        self.down1 = nn.Sequential(
            nn.Conv3d(
                32,
                64,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        self.stage2 = nn.Sequential(
            *[ConvNeXtBlock3D(64) for _ in range(num_blocks[1])]
        )

        self.proj = nn.Sequential(
            nn.Conv3d(64, dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )

        self.refine = nn.Sequential(
            *[ConvNeXtBlock3D(dim) for _ in range(num_blocks[2])]
        )

    def forward(self, ref):
        x = self.stem(ref)     # (B,32,D,H/2,W/2)
        x = self.stage1(x)

        x = self.down1(x)      # (B,64,D,H/4,W/4)
        x = self.stage2(x)

        x = self.proj(x)       # (B,dim,D,H/4,W/4)
        x = self.refine(x)

        return x

class ReferenceMemoryEncoderV3(nn.Module):
    """
    Stronger anisotropic reference memory encoder.

    Input:
        ref: (B, 1, D, H, W)

    Output:
        F_ref: (B, C, D, H/4, W/4)

    Notes:
        - z direction is never downsampled.
        - early stage mainly extracts xy features: kernel=(1,7,7)
        - deeper stages mix local z context: kernel=(3,7,7)
    """
    def __init__(
        self,
        in_ch=1,
        dim=96,
        base_channels=(32, 64, 96),
        num_blocks=(2, 4, 4),
        refine_blocks=2,
        use_attention=False,
        attn_layers=0,
        attn_num_heads=4,
        attn_window_size=(4, 8, 8),
        attn_mlp_ratio=2.0,
    ):
        super().__init__()

        c1, c2, c3 = base_channels

        # ----------------------------------------------------
        # Stage 0: shallow xy downsampling, no z mixing
        # ----------------------------------------------------
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

        self.stage1 = nn.Sequential(
            *[
                ConvNeXtBlock3DAniso(
                    c1,
                    kernel_size=(1, 7, 7),
                )
                for _ in range(num_blocks[0])
            ]
        )

        # ----------------------------------------------------
        # Stage 1: further xy downsampling, start local z mixing
        # ----------------------------------------------------
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

        self.stage2 = nn.Sequential(
            *[
                ConvNeXtBlock3DAniso(
                    c2,
                    kernel_size=(3, 7, 7),
                )
                for _ in range(num_blocks[1])
            ]
        )

        # ----------------------------------------------------
        # Stage 2: increase channel capacity without further downsampling
        # ----------------------------------------------------
        self.expand = nn.Sequential(
            nn.Conv3d(c2, c3, kernel_size=1, bias=False),
            nn.GroupNorm(8, c3),
            nn.GELU(),
        )

        self.stage3 = nn.Sequential(
            *[
                ConvNeXtBlock3DAniso(
                    c3,
                    kernel_size=(3, 7, 7),
                )
                for _ in range(num_blocks[2])
            ]
        )

        # ----------------------------------------------------
        # Project to matching dimension
        # ----------------------------------------------------
        self.proj = nn.Sequential(
            nn.Conv3d(c3, dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )

        self.refine = nn.Sequential(
            *[
                ConvNeXtBlock3DAniso(
                    dim,
                    kernel_size=(3, 7, 7),
                )
                for _ in range(refine_blocks)
            ]
        )
        self.use_attention = use_attention

        if use_attention and attn_layers > 0:
            self.ref_attn = nn.Sequential(
                *[
                    WindowSelfAttention3D(
                        dim=dim,
                        num_heads=attn_num_heads,
                        window_size=attn_window_size,
                        mlp_ratio=attn_mlp_ratio,
                        init_gamma=1e-4,
                    )
                    for _ in range(attn_layers)
                ]
            )
        else:
            self.ref_attn = nn.Identity()
    def forward(self, ref):
        x = self.stem(ref)       # (B, c1, D, H/2, W/2)
        x = self.stage1(x)

        x = self.down1(x)        # (B, c2, D, H/4, W/4)
        x = self.stage2(x)

        x = self.expand(x)       # (B, c3, D, H/4, W/4)
        x = self.stage3(x)

        x = self.proj(x)         # (B, dim, D, H/4, W/4)
        x = self.refine(x)

        # optional window attention on final reference memory
        x = self.ref_attn(x)

        return x
# ============================================================
# Coordinate utilities
# ============================================================

def normalize_grid(coords, D, H, W):
    """
    Convert feature-space z,y,x coordinates to grid_sample coordinates.

    Input:
        coords: (..., 3), order z,y,x

    Output:
        grid: (..., 3), order x,y,z, normalized to [-1, 1]
    """
    z = coords[..., 0]
    y = coords[..., 1]
    x = coords[..., 2]

    x_norm = 2.0 * x / max(W - 1, 1) - 1.0
    y_norm = 2.0 * y / max(H - 1, 1) - 1.0
    z_norm = 2.0 * z / max(D - 1, 1) - 1.0

    return torch.stack([x_norm, y_norm, z_norm], dim=-1)


def sample_local_3d_window(F_ref, center_coords, radius=(2, 4, 4)):
    """
    Sample local 3D windows from reference memory.

    Args:
        F_ref:
            (B, C, Dr, Hr, Wr)

        center_coords:
            (B, N, 3), feature-space coordinates, order z,y,x

        radius:
            (Rz, Ry, Rx)

    Returns:
        sampled_feat:
            (B, N, M, C), where M = (2Rz+1)*(2Ry+1)*(2Rx+1)

        candidate_coords:
            (B, N, M, 3), order z,y,x
    """
    B, C, Dr, Hr, Wr = F_ref.shape
    _, N, _ = center_coords.shape
    Rz, Ry, Rx = radius

    dz = torch.arange(-Rz, Rz + 1, device=F_ref.device)
    dy = torch.arange(-Ry, Ry + 1, device=F_ref.device)
    dx = torch.arange(-Rx, Rx + 1, device=F_ref.device)

    oz, oy, ox = torch.meshgrid(dz, dy, dx, indexing="ij")
    offsets = torch.stack([oz, oy, ox], dim=-1).float()
    offsets = offsets.view(1, 1, -1, 3)  # (1, 1, M, 3)

    candidate_coords = center_coords.unsqueeze(2) + offsets
    M = candidate_coords.shape[2]

    # Clamp only for numerical stability.
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

    # grid_sample input: (B, C, D, H, W)
    # grid: (B, Dout, Hout, Wout, 3)
    sampled = F.grid_sample(
        F_ref,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    # sampled: (B, C, N*M, 1, 1)
    sampled = sampled.squeeze(-1).squeeze(-1)
    sampled = sampled.permute(0, 2, 1).contiguous()
    sampled = sampled.view(B, N, M, C)

    return sampled, candidate_coords_clamped, candidate_valid_mask

def get_ref_feature_scale(ref, F_ref):
    """
    ref:   (B,1,D,H,W)
    F_ref: (B,C,Dr,Hr,Wr)

    Return scale from feature coordinates to raw coordinates.
    Coordinate order: (z,y,x)
    """
    _, _, D, H, W = ref.shape
    _, _, Dr, Hr, Wr = F_ref.shape

    scale_z = (D - 1) / max(Dr - 1, 1)
    scale_y = (H - 1) / max(Hr - 1, 1)
    scale_x = (W - 1) / max(Wr - 1, 1)

    return scale_z, scale_y, scale_x

def feature_to_raw_coords(coords_feat, scale):
    """
    coords_feat: (...,3), order=(z,y,x)
    """
    scale_z, scale_y, scale_x = scale

    coords_raw = coords_feat.clone()
    coords_raw[..., 0] = coords_feat[..., 0] * scale_z
    coords_raw[..., 1] = coords_feat[..., 1] * scale_y
    coords_raw[..., 2] = coords_feat[..., 2] * scale_x

    return coords_raw

def raw_to_feature_coords(coords_raw, scale):
    """
    coords_raw: (...,3), order=(z,y,x)
    """
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
    if z_init is None:
        raise ValueError("z_init or sparse_z_idx must be provided.")

    if not torch.is_tensor(z_init):
        z_init = torch.as_tensor(z_init, device=device)

    z_init = z_init.to(device=device, dtype=torch.float32)

    if z_init.dim() == 1:
        z_init = z_init.unsqueeze(0).repeat(B, 1)

    ys = torch.arange(Hc, device=device).float() * control_stride
    xs = torch.arange(Wc, device=device).float() * control_stride

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    zz = z_init[:, :, None, None].expand(B, K, Hc, Wc)
    yy = yy[None, None, :, :].expand(B, K, Hc, Wc)
    xx = xx[None, None, :, :].expand(B, K, Hc, Wc)

    coords0_raw = torch.stack([zz, yy, xx], dim=-1)
    return coords0_raw
# ============================================================
# Matching module
# ============================================================

class LocalQueryToVolumeMatcher(nn.Module):
    """
    Query-to-local-volume matching.

    mode:
        "dot"    : pure normalized dot product
        "mlp"    : pure learned pairwise matcher
        "hybrid" : dot product + learned residual correction
    """
    def __init__(
        self,
        dim=128,
        radius=(2, 4, 4),
        temperature=0.07,
        use_learned_matching=True,
        matcher_mode="hybrid",   # "dot", "mlp", "hybrid"
        zero_init_residual=True,
    ):
        super().__init__()

        self.radius = radius
        self.temperature = temperature
        self.use_learned_matching = use_learned_matching
        self.matcher_mode = matcher_mode

        if matcher_mode not in ["dot", "mlp", "hybrid"]:
            raise ValueError("matcher_mode must be 'dot', 'mlp', or 'hybrid'.")

        if use_learned_matching or matcher_mode in ["mlp", "hybrid"]:
            self.match_mlp = nn.Sequential(
                nn.LayerNorm(dim * 4),
                nn.Linear(dim * 4, dim),
                nn.GELU(),
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1),
            )

            # Important:
            # For hybrid mode, initialize residual branch as zero.
            # Then the model initially behaves exactly like pure dot product.
            if zero_init_residual:
                last = self.match_mlp[-1]
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def compute_local_correlation(self, queries, sampled_ref):
        """
        queries:     (B, N, C)
        sampled_ref: (B, N, M, C)

        return:
            scores: (B, N, M)
        """
        q = F.normalize(queries, dim=-1)
        r = F.normalize(sampled_ref, dim=-1)

        # baseline dot score
        dot_scores = (q.unsqueeze(2) * r).sum(dim=-1)
        dot_scores = dot_scores / self.temperature

        if self.matcher_mode == "dot" or not self.use_learned_matching:
            return dot_scores

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
            return learned_delta

        # Recommended mode:
        # residual correction over old dot product.
        if self.matcher_mode == "hybrid":
            return dot_scores + learned_delta

    @staticmethod
    def coordinate_expectation(prob, candidate_coords):
        return torch.sum(prob.unsqueeze(-1) * candidate_coords, dim=2)

    def forward(self, queries, F_ref, center_coords):
        sampled_ref, candidate_coords,_ = sample_local_3d_window(
            F_ref=F_ref,
            center_coords=center_coords,
            radius=self.radius,
        )

        scores = self.compute_local_correlation(queries, sampled_ref)
        prob = torch.softmax(scores, dim=-1)
        pred_coords = self.coordinate_expectation(prob, candidate_coords)

        return pred_coords, prob, scores, candidate_coords

class LocalQueryToVolumeMatcherV2(nn.Module):
    """
    Query-to-local-volume matcher V2.

    Compared with LocalQueryToVolumeMatcher, this version adds:

    1. Relative offset feature encoding:
        candidate feature = candidate image feature + offset embedding

    2. Relative offset bias:
        score = appearance score + learned offset bias

    3. Local cross-attention query update:
        query attends to all local candidate features first, then uses
        the updated query to compute final matching scores.

    Coordinate convention:
        center_coords and candidate_coords are in reference feature-space,
        order = (z, y, x).

    Args:
        dim:
            feature dimension.

        radius:
            local search radius in feature-space, order=(Rz,Ry,Rx).

        temperature:
            final matching softmax temperature.

        matcher_mode:
            "dot"    : normalized dot product only
            "mlp"    : learned pairwise MLP only
            "hybrid" : dot product + learned residual

        use_offset_encoding:
            whether to add offset embedding to candidate features.

        use_offset_bias:
            whether to add learned scalar offset bias to scores.

        use_local_cross_attn:
            whether to update query by attending to the local candidate window.
    """

    def __init__(
        self,
        dim=96,
        radius=(4, 5, 5),
        temperature=0.05,
        use_learned_matching=True,
        matcher_mode="hybrid",
        zero_init_residual=True,
        use_offset_encoding=True,
        use_offset_bias=True,
        use_local_cross_attn=True,
        local_attn_temperature=0.20,
    ):
        super().__init__()

        self.dim = dim
        self.radius = tuple(radius)
        self.temperature = temperature
        self.use_learned_matching = use_learned_matching
        self.matcher_mode = matcher_mode

        self.use_offset_encoding = use_offset_encoding
        self.use_offset_bias = use_offset_bias
        self.use_local_cross_attn = use_local_cross_attn
        self.local_attn_temperature = local_attn_temperature

        if matcher_mode not in ["dot", "mlp", "hybrid"]:
            raise ValueError(
                "matcher_mode must be one of: 'dot', 'mlp', 'hybrid'."
            )

        # ----------------------------------------------------
        # 1. Pairwise learned matcher branch
        # ----------------------------------------------------
        if use_learned_matching or matcher_mode in ["mlp", "hybrid"]:
            self.match_mlp = nn.Sequential(
                nn.LayerNorm(dim * 4),
                nn.Linear(dim * 4, dim),
                nn.GELU(),
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1),
            )

            # For hybrid mode, start from pure dot product.
            if zero_init_residual:
                last = self.match_mlp[-1]
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        else:
            self.match_mlp = None

        # ----------------------------------------------------
        # 2. Relative offset feature embedding
        # ----------------------------------------------------
        if use_offset_encoding:
            self.offset_feat = nn.Sequential(
                nn.Linear(3, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

            # Start as no-op.
            nn.init.zeros_(self.offset_feat[-1].weight)
            nn.init.zeros_(self.offset_feat[-1].bias)
        else:
            self.offset_feat = None

        # ----------------------------------------------------
        # 3. Relative offset scalar bias
        # ----------------------------------------------------
        if use_offset_bias:
            self.offset_bias = nn.Sequential(
                nn.Linear(3, dim),
                nn.GELU(),
                nn.Linear(dim, 1),
            )

            # Start as no-op.
            nn.init.zeros_(self.offset_bias[-1].weight)
            nn.init.zeros_(self.offset_bias[-1].bias)
        else:
            self.offset_bias = None

        # ----------------------------------------------------
        # 4. Local cross-attention query update
        # ----------------------------------------------------
        if use_local_cross_attn:
            self.query_update = nn.Sequential(
                nn.LayerNorm(dim * 4),
                nn.Linear(dim * 4, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
            )

            # Residual branch starts as zero, so V2 initially behaves
            # like the original matcher.
            nn.init.zeros_(self.query_update[-1].weight)
            nn.init.zeros_(self.query_update[-1].bias)
        else:
            self.query_update = None

    def normalize_relative_offset(self, candidate_coords, center_coords):
        """
        Args:
            candidate_coords:
                (B,N,M,3), feature-space coordinates, order=(z,y,x)

            center_coords:
                (B,N,3), feature-space coordinates, order=(z,y,x)

        Returns:
            rel_offset:
                (B,N,M,3), feature-space offset

            rel_offset_norm:
                (B,N,M,3), normalized offset, roughly in [-1,1]
        """
        rz, ry, rx = self.radius

        rel_offset = candidate_coords - center_coords.unsqueeze(2)

        rel_offset_norm = rel_offset.clone()
        rel_offset_norm[..., 0] = rel_offset_norm[..., 0] / max(float(rz), 1.0)
        rel_offset_norm[..., 1] = rel_offset_norm[..., 1] / max(float(ry), 1.0)
        rel_offset_norm[..., 2] = rel_offset_norm[..., 2] / max(float(rx), 1.0)

        # Near image boundary, clamping can make offsets slightly unusual.
        # This clamp prevents very large values from destabilizing the MLP.
        rel_offset_norm = rel_offset_norm.clamp(-2.0, 2.0)

        return rel_offset, rel_offset_norm

    def apply_local_cross_attention(self, queries, sampled_ref):
        """
        Let each query attend to its local candidate window, producing
        a context-aware query.

        Args:
            queries:
                (B,N,C)

            sampled_ref:
                (B,N,M,C)

        Returns:
            updated queries:
                (B,N,C)
        """
        if not self.use_local_cross_attn:
            return queries

        q = F.normalize(queries, dim=-1)
        r = F.normalize(sampled_ref, dim=-1)

        attn_logits = torch.einsum("bnc,bnmc->bnm", q, r)
        attn_logits = attn_logits / max(float(self.local_attn_temperature), 1e-6)

        attn = torch.softmax(attn_logits, dim=-1)

        context = torch.sum(attn.unsqueeze(-1) * sampled_ref, dim=2)
        # (B,N,C)

        update_input = torch.cat(
            [
                queries,
                context,
                queries * context,
                torch.abs(queries - context),
            ],
            dim=-1,
        )

        updated_queries = queries + self.query_update(update_input)

        return updated_queries

    def compute_local_scores(self, queries, sampled_ref, rel_offset_norm=None):
        """
        Args:
            queries:
                (B,N,C)

            sampled_ref:
                (B,N,M,C)

            rel_offset_norm:
                (B,N,M,3)

        Returns:
            scores:
                (B,N,M)
        """
        q = F.normalize(queries, dim=-1)
        r = F.normalize(sampled_ref, dim=-1)

        # ----------------------------------------------------
        # Dot-product score
        # ----------------------------------------------------
        dot_scores = (q.unsqueeze(2) * r).sum(dim=-1)
        dot_scores = dot_scores / max(float(self.temperature), 1e-6)

        # ----------------------------------------------------
        # Matcher mode
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # Add relative offset bias
        # ----------------------------------------------------
        if self.use_offset_bias and rel_offset_norm is not None:
            scores = scores + self.offset_bias(rel_offset_norm).squeeze(-1)

        return scores

    @staticmethod
    def coordinate_expectation(prob, candidate_coords):
        """
        Args:
            prob:
                (B,N,M)

            candidate_coords:
                (B,N,M,3)

        Returns:
            pred_coords:
                (B,N,3)
        """
        return torch.sum(prob.unsqueeze(-1) * candidate_coords, dim=2)

    def forward(self, queries, F_ref, center_coords,return_aux=False):
        """
        Args:
            queries:
                (B,N,C)

            F_ref:
                (B,C,Dr,Hr,Wr)

            center_coords:
                (B,N,3), feature-space coordinates, order=(z,y,x)

        Returns:
            pred_coords:
                (B,N,3), feature-space coordinates

            prob:
                (B,N,M)

            scores:
                (B,N,M)

            candidate_coords:
                (B,N,M,3), feature-space candidate coordinates
        """
        sampled_ref, candidate_coords, candidate_valid_mask = sample_local_3d_window(
            F_ref=F_ref,
            center_coords=center_coords,
            radius=self.radius,
        )
        # sampled_ref:      (B,N,M,C)
        # candidate_coords: (B,N,M,3)

        _, rel_offset_norm = self.normalize_relative_offset(
            candidate_coords=candidate_coords,
            center_coords=center_coords,
        )

        # ----------------------------------------------------
        # Add relative offset embedding to candidate feature
        # ----------------------------------------------------
        if self.use_offset_encoding:
            sampled_ref = sampled_ref + self.offset_feat(rel_offset_norm)

        # ----------------------------------------------------
        # Query update by local cross-attention
        # ----------------------------------------------------
        queries = self.apply_local_cross_attention(
            queries=queries,
            sampled_ref=sampled_ref,
        )

        # ----------------------------------------------------
        # Final local matching scores
        # ----------------------------------------------------
        scores = self.compute_local_scores(
            queries=queries,
            sampled_ref=sampled_ref,
            rel_offset_norm=rel_offset_norm,
        )
        scores = scores.masked_fill(~candidate_valid_mask, -1e4)
        prob = torch.softmax(scores, dim=-1)

        pred_coords = self.coordinate_expectation(
            prob=prob,
            candidate_coords=candidate_coords,
        )
        if return_aux:
            return pred_coords, prob, scores, candidate_coords, candidate_valid_mask
        else:
            return pred_coords, None, None, None, None
    

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

    scores:               (B, Nc, M)
    candidate_coords:     (B, Nc, M, 3), feature-space z,y,x
    candidate_valid_mask: (B, Nc, M)
    gt_coords_feat:       (B, Nc, 3), feature-space z,y,x
    valid_mask_chunk:     (B, Nc)
    """
    B, Nc, M = scores.shape
    device = scores.device
    dtype = scores.dtype

    if valid_mask_chunk is None:
        valid_mask_chunk = torch.ones(
            B, Nc,
            device=device,
            dtype=torch.bool,
        )
    else:
        valid_mask_chunk = valid_mask_chunk.to(device=device).bool()

    gt_coords_feat = gt_coords_feat.to(device=device, dtype=dtype)

    sigma_t = torch.tensor(
        sigma,
        device=device,
        dtype=dtype,
    ).view(1, 1, 1, 3)

    diff = (candidate_coords - gt_coords_feat.unsqueeze(2)) / sigma_t
    dist2 = torch.sum(diff * diff, dim=-1)  # (B,Nc,M)

    dist2_valid = dist2.masked_fill(~candidate_valid_mask, 1e8)

    min_dist2, target_idx = dist2_valid.min(dim=-1)

    inside = torch.sqrt(min_dist2.clamp_min(0.0)) <= inside_threshold

    train_valid = valid_mask_chunk & inside

    train_valid_f = train_valid.float()
    valid_count = train_valid_f.sum().clamp_min(1.0)

    scores_valid = scores.masked_fill(~candidate_valid_mask, -1e4)

    # hard CE target: nearest candidate to GT
    ce_per = F.cross_entropy(
        scores_valid.reshape(B * Nc, M),
        target_idx.reshape(B * Nc),
        reduction="none",
    ).reshape(B, Nc)

    loss_ce_sum = (ce_per * train_valid_f).sum()

    # soft KL target: Gaussian around GT
    target_logits = -0.5 * dist2_valid
    target_logits = target_logits.masked_fill(~candidate_valid_mask, -1e4)

    target_prob = torch.softmax(target_logits, dim=-1)
    log_prob = torch.log_softmax(scores_valid, dim=-1)

    kl_per = torch.sum(
        target_prob * (
            torch.log(target_prob.clamp_min(eps)) - log_prob
        ),
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
        entropy = -torch.sum(
            prob * torch.log(prob.clamp_min(eps)),
            dim=-1,
        )

        candidate_valid_ratio = candidate_valid_mask.float().mean()

        all_count = torch.tensor(
            float(B * Nc),
            device=device,
            dtype=dtype,
        )

        valid_inside_all = valid_mask_chunk & inside

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

class CoordResidualHead(nn.Module):
    """
    Predict continuous residual offset after coarse matching.

    Input:
        q_feat:       (B,N,C), moving/query feature
        r_feat:       (B,N,C), sampled reference feature at coarse coord
        disp_feat:    (B,N,3), coarse displacement in feature-normalized units

    Output:
        delta_raw:    (B,N,3), residual in raw coordinate, order=(z,y,x)
    """
    def __init__(
        self,
        dim,
        hidden_dim=128,
        max_delta=(0.5, 1.0, 1.0),  # raw z,y,x
        use_disp=True,
    ):
        super().__init__()

        self.use_disp = use_disp

        in_dim = dim * 4
        if use_disp:
            in_dim += 3

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

        # 让 residual 初始接近 0，避免刚加载 checkpoint 后输出突变
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
        delta = torch.tanh(self.mlp(x)) * self.max_delta.to(x.device)
        return delta

class SpatialCoordResidualHead(nn.Module):
    """
    Spatial residual refiner on control grid.

    Input:
        feat: (B, N, C), where N = K * Hc * Wc

    Output:
        delta_raw: (B, K, Hc, Wc, 3), order=(z,y,x)
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

        for _ in range(num_blocks):
            if use_3d:
                kernel_size = (3, 3, 3)
                padding = (1, 1, 1)
            else:
                kernel_size = (1, 3, 3)
                padding = (0, 1, 1)

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
                    nn.Conv3d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=1,
                        bias=False,
                    ),
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

        # very important: initial residual = 0
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(self, feat, K, Hc, Wc):
        """
        Args:
            feat:
                (B, N, C), N = K * Hc * Wc

        Returns:
            delta_raw:
                (B, K, Hc, Wc, 3)
        """
        B, N, C = feat.shape

        if N != K * Hc * Wc:
            raise ValueError(
                f"N={N} does not match K*Hc*Wc={K * Hc * Wc}."
            )

        x = self.in_proj(feat)  # (B,N,hidden)

        x = x.view(B, K, Hc, Wc, -1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        # (B,hidden,K,Hc,Wc)

        for block in self.blocks:
            x = x + block(x)

        delta = self.out_proj(x)
        # (B,3,K,Hc,Wc)

        delta = delta.permute(0, 2, 3, 4, 1).contiguous()
        # (B,K,Hc,Wc,3)

        delta = torch.tanh(delta) * self.max_delta.to(
            device=delta.device,
            dtype=delta.dtype,
        )

        return delta

def sample_ref_features_at_feature_coords(
    F_ref,
    coords_feat_zyx,
    padding_mode="border",
    align_corners=True,
):
    """
    Sample reference feature at feature-space coordinates.

    Args:
        F_ref:
            (B,C,D,Hf,Wf)

        coords_feat_zyx:
            (B,N,3), feature-space coordinates, order=(z,y,x)

    Returns:
        sampled:
            (B,N,C)
    """
    B, C, D, Hf, Wf = F_ref.shape

    coords = coords_feat_zyx.to(dtype=F_ref.dtype)

    zf = coords[..., 0]
    yf = coords[..., 1]
    xf = coords[..., 2]

    if Wf > 1:
        x_norm = 2.0 * xf / (Wf - 1) - 1.0
    else:
        x_norm = torch.zeros_like(xf)

    if Hf > 1:
        y_norm = 2.0 * yf / (Hf - 1) - 1.0
    else:
        y_norm = torch.zeros_like(yf)

    if D > 1:
        z_norm = 2.0 * zf / (D - 1) - 1.0
    else:
        z_norm = torch.zeros_like(zf)

    # grid_sample expects order=(x,y,z)
    grid = torch.stack([x_norm, y_norm, z_norm], dim=-1)
    grid = grid.view(B, -1, 1, 1, 3)

    sampled = F.grid_sample(
        F_ref,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    # (B,C,N,1,1) -> (B,N,C)
    sampled = sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1).contiguous()
    return sampled


#  ============================================================
# Full model
# ============================================================

class CoarseMatchingNetV3(nn.Module):
    """
    Sparse moving control-point queries -> dense reference volume memory.

    Input:
        mov: (B, 1, K, H, W)
        ref: (B, 1, D, H, W)

    Output:
        pred_coords: (B, K, Hc, Wc, 3), feature-space z,y,x
        pred_disp:   (B, K, Hc, Wc, 3)
        coords0:     (B, K, Hc, Wc, 3)
        prob:        (B, N, M)
        scores:      (B, N, M)
    """
    def __init__(
        self,
        dim=128,
        radius=(2, 4, 4),
        temperature=0.07,
        use_learned_matching=True,
        num_refine_iters=1,
        control_stride=32,
        encoder_stride=4,
        matcher_mode = "hybrid",
        query_chunk_size=1024,
    ):
        super().__init__()

        self.moving_encoder = MovingQueryEncoderV2(
            in_ch=1,
            dim=dim,
            num_convnext_blocks=3,
            use_window_attn=True,
            window_size=8,
            num_heads=4,
        )

        self.reference_encoder = ReferenceMemoryEncoderV2(
            in_ch=1,
            dim=dim,
            num_blocks=(2, 3, 3),
        )

        self.matcher = LocalQueryToVolumeMatcher(
            dim=dim,
            radius=radius,
            temperature=temperature,
            use_learned_matching=use_learned_matching,
            matcher_mode=matcher_mode,
            zero_init_residual=True,
        )

        self.num_refine_iters = num_refine_iters

        # Moving query embedding projection
        self.query_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Reference memory embedding projection
        self.ref_proj = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
        )

        # Optional learnable temperature scale
        self.control_stride = control_stride
        self.encoder_stride = encoder_stride
        self.query_chunk_size = query_chunk_size

        if control_stride % encoder_stride != 0:
            raise ValueError(
                f"control_stride={control_stride} must be divisible by "
                f"encoder_stride={encoder_stride}."
            )

        self.query_downsample = control_stride // encoder_stride
    def downsample_query_features(self, F_mov, input_hw):
        """
        F_mov:
            (B,K,C,Hf,Wf)

        input_hw:
            (H,W), where mov input is (B,1,K,H,W)

        Return:
            (B,K,C,Hc,Wc)

        Hc = ceil(H / control_stride)
        Wc = ceil(W / control_stride)
        """
        H, W = input_hw

        target_h = math.ceil(H / self.control_stride)
        target_w = math.ceil(W / self.control_stride)

        B, K, C, Hf, Wf = F_mov.shape

        x = F_mov.permute(0, 2, 1, 3, 4).contiguous()
        # (B,C,K,Hf,Wf)

        x = F.adaptive_avg_pool3d(
            x,
            output_size=(K, target_h, target_w),
        )

        F_mov = x.permute(0, 2, 1, 3, 4).contiguous()
        # (B,K,C,target_h,target_w)

        return F_mov
    def forward(self, mov, ref, z_init=None, sparse_z_idx=None, spacing=None):
        if z_init is None:
            z_init = sparse_z_idx

        B, _, K_in, H_in, W_in = mov.shape

        # -------------------------
        # 1. Encode moving queries
        # -------------------------
        F_mov = self.moving_encoder(mov)  # (B,K,C,Hf,Wf)

        F_mov = self.downsample_query_features(
            F_mov,
            input_hw=(H_in, W_in),
        )  # (B,K,C,Hc,Wc)

        # -------------------------
        # 2. Encode reference memory
        # -------------------------
        F_ref = self.reference_encoder(ref)  # (B,C,Dr,Hr,Wr)

        # NEW: project reference memory into matching embedding space
        F_ref = self.ref_proj(F_ref)

        B, K, C, Hc, Wc = F_mov.shape
        _, _, Dr, Hr, Wr = F_ref.shape

        # -------------------------
        # 3. Build moving queries
        # -------------------------
        queries = F_mov.permute(0, 1, 3, 4, 2).contiguous()
        queries = queries.view(B, K * Hc * Wc, C)

        # NEW / existing: project moving query into matching embedding space
        coords0_norm = coords0_raw.clone()
        coords0_norm[..., 0] = coords0_norm[..., 0] / max(ref.shape[2] - 1, 1)
        coords0_norm[..., 1] = coords0_norm[..., 1] / max(ref.shape[3] - 1, 1)
        coords0_norm[..., 2] = coords0_norm[..., 2] / max(ref.shape[4] - 1, 1)

        coord_embed = self.coord_mlp(
            coords0_norm.view(B, K * Hc * Wc, 3)
        )

        queries = queries + coord_embed
        queries = self.query_proj(queries)

        # -------------------------
        # 4. Build initial raw coordinates
        # -------------------------
        coords0_raw = build_raw_control_coords_stride_grid(
            B=B,
            K=K,
            Hc=Hc,
            Wc=Wc,
            z_init=z_init,
            control_stride=self.control_stride,
            device=mov.device,
        )
        # -------------------------
        # 5. Convert raw coords to feature coords for matching
        # -------------------------
        scale = get_ref_feature_scale(ref, F_ref)

        coords0_feat = raw_to_feature_coords(coords0_raw, scale)
        coords_feat = coords0_feat.view(B, K * Hc * Wc, 3)

        # -------------------------
        # 6. Query-to-reference matching in feature space
        # -------------------------
        last_prob = None
        last_scores = None
        last_candidate_coords = None

        for _ in range(self.num_refine_iters):
            coords_feat, last_prob, last_scores, last_candidate_coords = self.match_queries_in_chunks(
                queries=queries,
                F_ref=F_ref,
                coords=coords_feat,
            )
        pred_coords_feat = coords_feat.view(B, K, Hc, Wc, 3)
        coords0_feat = coords0_feat.view(B, K, Hc, Wc, 3)

        pred_coords_raw = feature_to_raw_coords(pred_coords_feat, scale)
        pred_disp_raw = pred_coords_raw - coords0_raw

        coord_scale = torch.tensor(
            scale,
            device=mov.device,
            dtype=pred_coords_raw.dtype,
        )

        return {
            "pred_coords": pred_coords_raw,
            "pred_disp": pred_disp_raw,
            "coords0": coords0_raw,
            "pred_coords_feat": pred_coords_feat,
            "coords0_feat": coords0_feat,
            "F_mov": F_mov,
            "F_ref": F_ref,
            "queries": queries,
            "prob": last_prob,
            "scores": last_scores,
            "candidate_coords_feat": last_candidate_coords,
            "coord_scale": coord_scale,
        }
    def match_queries_in_chunks(self, queries, F_ref, coords,return_match_aux=False):
        """
        queries: (B,N,C)
        F_ref:   (B,C,Dr,Hr,Wr)
        coords:  (B,N,3), feature-space z,y,x

        Returns:
            pred_coords:       (B,N,3)
            scores:            (B,N,M)
            candidate_coords:  (B,N,M,3)
        """
        pred_chunks = []
        if return_match_aux:
            score_chunks = []
            cand_chunks = []
            prob_chunks = []
            valid_chunks = []
        else:
            score_chunks = None
            cand_chunks = None
            prob_chunks = None
            valid_chunks = None
        B, N, C = queries.shape

        for start in range(0, N, self.query_chunk_size):
            end = min(start + self.query_chunk_size, N)

            q_chunk = queries[:, start:end, :]
            c_chunk = coords[:, start:end, :]

            pred_chunk, prob_chunk, score_chunk, cand_chunk, valid_chunk = self.matcher(
                queries=q_chunk,
                F_ref=F_ref,
                center_coords=c_chunk,
            )

            pred_chunks.append(pred_chunk)
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
            prob=None
            scores=None
            candidate_coords=None
            candidate_valid_mask=None
        return pred_coords, prob, scores, candidate_coords, candidate_valid_mask
    
class CoarseMatchingNetV4(nn.Module):
    """
    Sparse moving control-point queries -> dense reference volume memory.

    Input:
        mov: (B, 1, K, H, W)
        ref: (B, 1, D, H, W)

    Output:
        pred_coords: raw reference coordinates, order=(z,y,x)
        pred_disp:   raw displacement, order=(z,y,x)
        coords0:     initial raw control-point coordinates
    """
    def __init__(
        self,
        dim=96,
        radius=(2, 4, 4),
        temperature=0.07,
        use_learned_matching=True,
        num_refine_iters=1,
        control_stride=32,
        encoder_stride=4,
        matcher_mode="hybrid",
        query_chunk_size=1024,
        moving_num_convnext_blocks=3,
        ref_base_channels=(32, 64, 96),
        ref_num_blocks=(2, 4, 4),
        ref_refine_blocks=2,
        use_coord_embed=True,
    ):
        super().__init__()

        self.dim = dim
        self.num_refine_iters = num_refine_iters
        self.control_stride = control_stride
        self.encoder_stride = encoder_stride
        self.query_chunk_size = query_chunk_size
        self.use_coord_embed = use_coord_embed

        if control_stride % encoder_stride != 0:
            raise ValueError(
                f"control_stride={control_stride} must be divisible by "
                f"encoder_stride={encoder_stride}."
            )

        self.query_downsample = control_stride // encoder_stride

        # ----------------------------------------------------
        # 1. Moving query encoder
        # ----------------------------------------------------
        self.moving_encoder = MovingQueryEncoderV2(
            in_ch=1,
            dim=dim,
            num_convnext_blocks=moving_num_convnext_blocks,
            use_window_attn=True,
            window_size=8,
            num_heads=4,
        )

        # ----------------------------------------------------
        # 2. Stronger anisotropic reference memory encoder
        # ----------------------------------------------------
        self.reference_encoder = ReferenceMemoryEncoderV3(
            in_ch=1,
            dim=dim,
            base_channels=ref_base_channels,
            num_blocks=ref_num_blocks,
            refine_blocks=ref_refine_blocks,
        )

        # ----------------------------------------------------
        # 3. Local query-to-volume matcher
        # ----------------------------------------------------
        self.matcher = LocalQueryToVolumeMatcher(
            dim=dim,
            radius=radius,
            temperature=temperature,
            use_learned_matching=use_learned_matching,
            matcher_mode=matcher_mode,
            zero_init_residual=True,
        )

        # ----------------------------------------------------
        # 4. Coordinate embedding for query tokens
        # ----------------------------------------------------
        if use_coord_embed:
            self.coord_mlp = nn.Sequential(
                nn.Linear(3, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

            # Optional but recommended:
            # initialize coordinate embedding as zero residual.
            # This makes the model initially behave like the old version.
            nn.init.zeros_(self.coord_mlp[-1].weight)
            nn.init.zeros_(self.coord_mlp[-1].bias)
        else:
            self.coord_mlp = None

        # ----------------------------------------------------
        # 5. Moving query projection
        # ----------------------------------------------------
        self.query_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # ----------------------------------------------------
        # 6. Reference memory projection
        # ----------------------------------------------------
        self.ref_proj = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
        )

    # ========================================================
    # Utility functions
    # ========================================================

    def downsample_query_features(self, F_mov, input_hw):
        """
        F_mov:
            (B, K, C, Hf, Wf)

        input_hw:
            raw moving image size: (H, W)

        Return:
            (B, K, C, Hc, Wc)

        where:
            Hc = ceil(H / control_stride)
            Wc = ceil(W / control_stride)
        """
        H, W = input_hw

        target_h = math.ceil(H / self.control_stride)
        target_w = math.ceil(W / self.control_stride)

        B, K, C, Hf, Wf = F_mov.shape

        x = F_mov.permute(0, 2, 1, 3, 4).contiguous()
        # (B, C, K, Hf, Wf)

        x = F.adaptive_avg_pool3d(
            x,
            output_size=(K, target_h, target_w),
        )

        F_mov = x.permute(0, 2, 1, 3, 4).contiguous()
        # (B, K, C, target_h, target_w)

        return F_mov

    @staticmethod
    def normalize_raw_coords_for_embedding(coords_raw, ref):
        """
        Normalize raw coordinates to [-1, 1] for coordinate embedding.

        coords_raw:
            (B, K, Hc, Wc, 3), order=(z,y,x)

        ref:
            (B, 1, D, H, W)
        """
        _, _, D, H, W = ref.shape

        z = coords_raw[..., 0] / max(D - 1, 1)
        y = coords_raw[..., 1] / max(H - 1, 1)
        x = coords_raw[..., 2] / max(W - 1, 1)

        coords_norm = torch.stack([z, y, x], dim=-1)
        coords_norm = coords_norm * 2.0 - 1.0

        return coords_norm

    @staticmethod
    def check_z_init(z_init, B, K, device):
        """
        Standardize z_init to shape (B, K).
        """
        if z_init is None:
            raise ValueError(
                "z_init or sparse_z_idx must be provided. "
                "For sparse stack-to-volume matching, default arange(K) is unsafe."
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
                f"z_init must have shape (K,) or (B,K), but got {tuple(z_init.shape)}."
            )

        return z_init

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, mov, ref, z_init=None, sparse_z_idx=None):
        """
        Args:
            mov:
                (B, 1, K, H, W)

            ref:
                (B, 1, D, H, W)

            z_init / sparse_z_idx:
                initial raw reference z coordinate for each moving slice.
                Shape: (K,) or (B,K)

        Returns:
            dict
        """
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
        # 1. Encode moving queries
        # ----------------------------------------------------
        F_mov = self.moving_encoder(mov)
        # (B, K, C, H/4, W/4)

        F_mov = self.downsample_query_features(
            F_mov,
            input_hw=(H_in, W_in),
        )
        # (B, K, C, Hc, Wc)

        B, K, C, Hc, Wc = F_mov.shape

        # ----------------------------------------------------
        # 2. Build initial raw control coordinates
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
        # (B, K, Hc, Wc, 3), raw coordinates, order=(z,y,x)

        # ----------------------------------------------------
        # 3. Encode reference memory
        # ----------------------------------------------------
        F_ref = self.reference_encoder(ref)
        # (B, C, Dr, Hr, Wr)

        F_ref = self.ref_proj(F_ref)
        # (B, C, Dr, Hr, Wr)

        _, _, Dr, Hr, Wr = F_ref.shape

        # ----------------------------------------------------
        # 4. Build moving query tokens
        # ----------------------------------------------------
        queries = F_mov.permute(0, 1, 3, 4, 2).contiguous()
        queries = queries.reshape(B, K * Hc * Wc, C)
        # (B, N, C)

        # ----------------------------------------------------
        # 5. Add coordinate embedding
        # ----------------------------------------------------
        if self.use_coord_embed:
            coords0_norm = self.normalize_raw_coords_for_embedding(
                coords_raw=coords0_raw,
                ref=ref,
            )
            # (B, K, Hc, Wc, 3)

            coord_embed = self.coord_mlp(
                coords0_norm.reshape(B, K * Hc * Wc, 3)
            )
            # (B, N, C)

            queries = queries + coord_embed

        queries = self.query_proj(queries)
        # (B, N, C)

        # ----------------------------------------------------
        # 6. Convert raw coordinates to reference feature coordinates
        # ----------------------------------------------------
        scale = get_ref_feature_scale(ref, F_ref)

        coords0_feat = raw_to_feature_coords(coords0_raw, scale)
        # (B, K, Hc, Wc, 3)

        coords_feat = coords0_feat.reshape(B, K * Hc * Wc, 3)
        # (B, N, 3)

        # ----------------------------------------------------
        # 7. Query-to-reference local matching
        # ----------------------------------------------------
        last_prob = None
        last_scores = None
        last_candidate_coords = None

        for _ in range(self.num_refine_iters):
            coords_feat, last_prob, last_scores, last_candidate_coords = (
                self.match_queries_in_chunks(
                    queries=queries,
                    F_ref=F_ref,
                    coords=coords_feat,
                )
            )

        # ----------------------------------------------------
        # 8. Convert predicted feature coordinates back to raw coordinates
        # ----------------------------------------------------
        pred_coords_feat = coords_feat.reshape(B, K, Hc, Wc, 3)
        coords0_feat = coords0_feat.reshape(B, K, Hc, Wc, 3)

        pred_coords_raw = feature_to_raw_coords(pred_coords_feat, scale)
        pred_disp_raw = pred_coords_raw - coords0_raw

        coord_scale = torch.tensor(
            scale,
            device=mov.device,
            dtype=pred_coords_raw.dtype,
        )

        # ----------------------------------------------------
        # 9. Optional confidence diagnostics
        # ----------------------------------------------------
        if last_prob is not None:
            confidence = last_prob.max(dim=-1).values
            entropy = -torch.sum(
                last_prob * torch.log(last_prob.clamp_min(1e-8)),
                dim=-1,
            )
        else:
            confidence = None
            entropy = None

        return {
            "pred_coords": pred_coords_raw,
            "pred_disp": pred_disp_raw,
            "coords0": coords0_raw,

            "pred_coords_feat": pred_coords_feat,
            "coords0_feat": coords0_feat,

            "F_mov": F_mov,
            "F_ref": F_ref,
            "queries": queries,

            "prob": last_prob,
            "scores": last_scores,
            "candidate_coords_feat": last_candidate_coords,

            "confidence": confidence,
            "entropy": entropy,

            "coord_scale": coord_scale,
        }

    def match_queries_in_chunks(self, queries, F_ref, coords):
        """
        queries:
            (B, N, C)

        F_ref:
            (B, C, Dr, Hr, Wr)

        coords:
            (B, N, 3), feature-space coordinates, order=(z,y,x)

        Returns:
            pred_coords:
                (B, N, 3)

            prob:
                (B, N, M)

            scores:
                (B, N, M)

            candidate_coords:
                (B, N, M, 3)
        """
        pred_chunks = []
        prob_chunks = []
        score_chunks = []
        cand_chunks = []

        B, N, C = queries.shape

        for start in range(0, N, self.query_chunk_size):
            end = min(start + self.query_chunk_size, N)

            q_chunk = queries[:, start:end, :]
            c_chunk = coords[:, start:end, :]

            pred_chunk, prob_chunk, score_chunk, cand_chunk = self.matcher(
                queries=q_chunk,
                F_ref=F_ref,
                center_coords=c_chunk,
            )

            pred_chunks.append(pred_chunk)
            prob_chunks.append(prob_chunk)
            score_chunks.append(score_chunk)
            cand_chunks.append(cand_chunk)

        pred_coords = torch.cat(pred_chunks, dim=1)
        prob = torch.cat(prob_chunks, dim=1)
        scores = torch.cat(score_chunks, dim=1)
        candidate_coords = torch.cat(cand_chunks, dim=1)

        return pred_coords, prob, scores, candidate_coords

class CoarseMatchingNetV5(nn.Module):
    """
    Sparse moving control-point queries -> dense reference volume memory.

    V5 = V4 backbone + LocalQueryToVolumeMatcherV2.

    Input:
        mov:
            (B, 1, K, H, W)

        ref:
            (B, 1, D, H, W)

    Output:
        pred_coords:
            raw reference coordinates, order=(z,y,x)

        pred_disp:
            raw displacement, order=(z,y,x)

        coords0:
            initial raw control-point coordinates, order=(z,y,x)
    """

    def __init__(
        self,
        dim=96,
        radius=(4, 5, 5),
        temperature=0.05,
        use_learned_matching=True,
        num_refine_iters=1,
        control_stride=16,
        encoder_stride=4,
        matcher_mode="hybrid",
        query_chunk_size=1024,

        # Moving encode
        moving_num_convnext_blocks=3,
        moving_window_attn_layers=1,

        # Reference encoder
        ref_base_channels=(32, 64, 96),
        ref_num_blocks=(2, 4, 4),
        ref_refine_blocks=2,
        ref_use_attention=False,
        ref_attn_layers=2 ,
        ref_attn_num_heads = 4,
        ref_attn_window_size=(4, 8, 8),
        ref_attn_mlp_ratio=2.0,

        # Coordinate embedding
        use_coord_embed=True,

        # Matcher V2 options
        use_offset_encoding=True,
        use_offset_bias=True,
        use_local_cross_attn=True,
        local_attn_temperature=0.20,
        use_spacing_embed=True,

        return_match_aux = False,

        use_coord_residual=False,
        residual_hidden_dim=128,
        residual_max_delta=(0.5, 1.0, 1.0),
        residual_use_disp=True,
        residual_detach_coarse=True,
        residual_detach_features=True,
        residual_type="mlp",          # "mlp" or "spatial"
        residual_num_blocks=3,
        residual_use_3d=True,

    ):
        super().__init__()

        self.dim = dim
        self.num_refine_iters = num_refine_iters
        self.control_stride = control_stride
        self.encoder_stride = encoder_stride
        self.query_chunk_size = query_chunk_size
        self.use_coord_embed = use_coord_embed
        self.use_spacing_embed = use_spacing_embed
        self.use_coord_residual = use_coord_residual
        self.residual_detach_coarse = residual_detach_coarse
        self.residual_detach_features = residual_detach_features
        self.use_coord_residual = use_coord_residual
        self.residual_type = residual_type
        self.residual_use_disp = residual_use_disp
        if use_spacing_embed:
            self.spacing_mlp = nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(4, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

            # 初始为 0，保证一开始不破坏原来的 V5 行为
            nn.init.zeros_(self.spacing_mlp[-1].weight)
            nn.init.zeros_(self.spacing_mlp[-1].bias)
        else:
            self.spacing_mlp = None
        if control_stride % encoder_stride != 0:
            raise ValueError(
                f"control_stride={control_stride} must be divisible by "
                f"encoder_stride={encoder_stride}."
            )

        self.query_downsample = control_stride // encoder_stride

        if use_coord_residual:
            if residual_type == "mlp":
                self.coord_residual_refiner = CoordResidualHead(
                    dim=dim,
                    hidden_dim=residual_hidden_dim,
                    max_delta=residual_max_delta,
                    use_disp=residual_use_disp,
                )

            elif residual_type == "spatial":
                residual_in_dim = dim * 4
                if residual_use_disp:
                    residual_in_dim += 3

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
        # ----------------------------------------------------
        # 1. Moving query encoder
        # ----------------------------------------------------
        self.moving_encoder = MovingQueryEncoderV2(
            in_ch=1,
            dim=dim,
            num_convnext_blocks=moving_num_convnext_blocks,
            use_window_attn=True,
            window_size=8,
            num_heads=4,
            window_attn_layers=moving_window_attn_layers,
        )

        # ----------------------------------------------------
        # 2. Stronger anisotropic reference memory encoder
        # ----------------------------------------------------
        self.reference_encoder = ReferenceMemoryEncoderV3(
            in_ch=1,
            dim=dim,
            base_channels=ref_base_channels,
            num_blocks=ref_num_blocks,
            refine_blocks=ref_refine_blocks,

            use_attention=ref_use_attention,
            attn_layers=ref_attn_layers,
            attn_num_heads=ref_attn_num_heads,
            attn_window_size=ref_attn_window_size,
            attn_mlp_ratio=ref_attn_mlp_ratio,
        )

        # ----------------------------------------------------
        # 3. Local query-to-volume matcher V2
        # ----------------------------------------------------
        self.matcher = LocalQueryToVolumeMatcherV2(
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
        )

        # ----------------------------------------------------
        # 4. Coordinate embedding for query tokens
        # ----------------------------------------------------
        if use_coord_embed:
            self.coord_mlp = nn.Sequential(
                nn.Linear(3, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

            # Initialize as zero residual.
            nn.init.zeros_(self.coord_mlp[-1].weight)
            nn.init.zeros_(self.coord_mlp[-1].bias)
        else:
            self.coord_mlp = None

        # ----------------------------------------------------
        # 5. Moving query projection
        # ----------------------------------------------------
        self.query_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # ----------------------------------------------------
        # 6. Reference memory projection
        # ----------------------------------------------------
        self.ref_proj = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
        )

    # ========================================================
    # Utility functions
    # ========================================================
    @staticmethod
    def build_spacing_features(spacing, K, device, dtype):
        """
        Args:
            spacing:
                (B,6) or (6,)
                order = [sx_ref, sy_ref, sz_ref, sx_mov, sy_mov, sz_mov]

            K:
                number of moving sparse slices

        Returns:
            spacing_feat:
                (B,4)
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

        spacing_feat = torch.stack(
            [
                torch.log(sz_ref / (xy_ref + 1e-8) + 1e-8),
                torch.log(sz_mov / (xy_mov + 1e-8) + 1e-8),
                torch.log(sz_mov / (sz_ref + 1e-8) + 1e-8),
                torch.log(K_tensor + 1e-8),
            ],
            dim=-1,
        )

        return spacing_feat
    def downsample_query_features(self, F_mov, input_hw):
        """
        Args:
            F_mov:
                (B,K,C,Hf,Wf)

            input_hw:
                raw moving image size: (H,W)

        Returns:
            F_mov:
                (B,K,C,Hc,Wc)

        Hc = ceil(H / control_stride)
        Wc = ceil(W / control_stride)
        """
        H, W = input_hw

        target_h = math.ceil(H / self.control_stride)
        target_w = math.ceil(W / self.control_stride)

        B, K, C, Hf, Wf = F_mov.shape

        x = F_mov.permute(0, 2, 1, 3, 4).contiguous()
        # (B,C,K,Hf,Wf)

        x = F.adaptive_avg_pool3d(
            x,
            output_size=(K, target_h, target_w),
        )

        F_mov = x.permute(0, 2, 1, 3, 4).contiguous()
        # (B,K,C,Hc,Wc)

        return F_mov

    @staticmethod
    def normalize_raw_coords_for_embedding(coords_raw, ref):
        """
        Normalize raw coordinates to [-1, 1] for coordinate embedding.

        Args:
            coords_raw:
                (B,K,Hc,Wc,3), order=(z,y,x)

            ref:
                (B,1,D,H,W)

        Returns:
            coords_norm:
                (B,K,Hc,Wc,3)
        """
        _, _, D, H, W = ref.shape

        z = coords_raw[..., 0] / max(D - 1, 1)
        y = coords_raw[..., 1] / max(H - 1, 1)
        x = coords_raw[..., 2] / max(W - 1, 1)

        coords_norm = torch.stack([z, y, x], dim=-1)
        coords_norm = coords_norm * 2.0 - 1.0

        return coords_norm

    @staticmethod
    def check_z_init(z_init, B, K, device):
        """
        Standardize z_init to shape (B,K).
        """
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
                f"z_init must have shape (K,) or (B,K), but got {tuple(z_init.shape)}."
            )

        return z_init

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

        # new: chunk-level CE/KL
        compute_chunk_match_loss=False,
        gt_coords=None,
        valid_mask=None,
        match_sigma=(0.5, 1.0, 1.0),
        match_inside_threshold=4.0,
    ):
        """
        Args:
            mov:
                (B,1,K,H,W)

            ref:
                (B,1,D,H,W)

            z_init / sparse_z_idx:
                initial raw reference z coordinate for each moving slice.
                Shape: (K,) or (B,K)

        Returns:
            dict
        """
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
        # 1. Encode moving queries
        # ----------------------------------------------------
        F_mov = self.moving_encoder(mov)
        # (B,K,C,H/4,W/4)

        F_mov = self.downsample_query_features(
            F_mov,
            input_hw=(H_in, W_in),
        )
        # (B,K,C,Hc,Wc)

        B, K, C, Hc, Wc = F_mov.shape

        # ----------------------------------------------------
        # 2. Build initial raw control coordinates
        # Must be consistent with dataset control grid:
        # 0, control_stride, 2*control_stride, ...
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
        # (B,K,Hc,Wc,3), raw coordinates, order=(z,y,x)

        # ----------------------------------------------------
        # 3. Encode reference memory
        # ----------------------------------------------------
        F_ref = self.reference_encoder(ref)
        # (B,C,Dr,Hr,Wr)

        F_ref = self.ref_proj(F_ref)
        # (B,C,Dr,Hr,Wr)

        # ----------------------------------------------------
        # 4. Build moving query tokens
        # ----------------------------------------------------
        queries = F_mov.permute(0, 1, 3, 4, 2).contiguous()
        queries = queries.reshape(B, K * Hc * Wc, C)
        # (B,N,C)

        # ----------------------------------------------------
        # 5. Add coordinate embedding
        # ----------------------------------------------------
        if self.use_coord_embed:
            coords0_norm = self.normalize_raw_coords_for_embedding(
                coords_raw=coords0_raw,
                ref=ref,
            )
            # (B,K,Hc,Wc,3)

            coord_embed = self.coord_mlp(
                coords0_norm.reshape(B, K * Hc * Wc, 3)
            )
            # (B,N,C)

            queries = queries + coord_embed
        if self.use_spacing_embed and spacing is not None:
            spacing_feat = self.build_spacing_features(
                spacing=spacing,
                K=K,
                device=mov.device,
                dtype=queries.dtype,
            )  # (B,4)

            if spacing_feat.shape[0] == 1 and B > 1:
                spacing_feat = spacing_feat.repeat(B, 1)

            spacing_embed = self.spacing_mlp(spacing_feat)  # (B,C)

            queries = queries + spacing_embed[:, None, :]
        queries = self.query_proj(queries)
        # (B,N,C)

        # ----------------------------------------------------
        # 6. Convert raw coordinates to reference feature coordinates
        # ----------------------------------------------------
        scale = get_ref_feature_scale(ref, F_ref)

        coords0_feat = raw_to_feature_coords(coords0_raw, scale)
        # (B,K,Hc,Wc,3)

        coords_feat = coords0_feat.reshape(B, K * Hc * Wc, 3)
        # (B,N,3)
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
        # 7. Query-to-reference local matching
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

            coords_feat, last_prob, last_scores, last_candidate_coords, last_candidate_valid_mask, chunk_match_stats = (
                self.match_queries_in_chunks(
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
            )
        # ----------------------------------------------------
        # 8. Feature coords -> raw coords
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

        # 默认最终输出就是 coarse 输出
        pred_coords_raw = pred_coords_raw_coarse
        pred_disp_raw = pred_disp_raw_coarse
        residual_delta_raw = None

        # ----------------------------------------------------
        # 8.5 Optional coordinate residual refinement
        # ----------------------------------------------------
        if self.use_coord_residual:
            N = K * Hc * Wc

            # q_feat: moving/query-side feature
            # 推荐用 queries，因为它已经包含 moving feature + coord/spacing encoding + query projection
            q_feat = queries  # (B,N,C)

            # coarse feature-space coords
            pred_coords_feat_for_sample = pred_coords_feat
            coords0_feat_for_res = coords0_feat

            if self.residual_detach_coarse:
                pred_coords_feat_for_sample = pred_coords_feat_for_sample.detach()
                coords0_feat_for_res = coords0_feat_for_res.detach()

            # sample reference feature at coarse predicted coordinates
            pred_coords_feat_flat_for_sample = pred_coords_feat_for_sample.reshape(B, N, 3)

            r_feat = sample_ref_features_at_feature_coords(
                F_ref=F_ref,
                coords_feat_zyx=pred_coords_feat_flat_for_sample,
                padding_mode="border",
                align_corners=True,
            )
            # r_feat: (B,N,C)

            if self.residual_detach_features:
                q_feat = q_feat.detach()
                r_feat = r_feat.detach()

            # feature-space coarse displacement, order=(z,y,x)
            disp_feat = (
                pred_coords_feat_for_sample - coords0_feat_for_res
            ).reshape(B, N, 3)

            if self.residual_detach_coarse:
                disp_feat = disp_feat.detach()

            # ------------------------------------------------
            # Build residual input
            # ------------------------------------------------
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

            # ------------------------------------------------
            # MLP residual or spatial residual
            # ------------------------------------------------
            if self.residual_type == "mlp":
                residual_delta_raw = self.coord_residual_refiner(
                    q_feat=q_feat,
                    r_feat=r_feat,
                    disp_feat=disp_feat if self.residual_use_disp else None,
                )
                residual_delta_raw = residual_delta_raw.reshape(B, K, Hc, Wc, 3)

            elif self.residual_type == "spatial":
                residual_delta_raw = self.coord_residual_refiner(
                    residual_feat,
                    K=K,
                    Hc=Hc,
                    Wc=Wc,
                )

            else:
                raise ValueError(f"Unknown residual_type={self.residual_type}")

            # final refined raw coordinates
            pred_coords_raw = pred_coords_raw_coarse + residual_delta_raw
            pred_disp_raw = pred_coords_raw - coords0_raw

            # keep pred_coords_feat consistent with final raw coords
            pred_coords_feat = raw_to_feature_coords(pred_coords_raw, scale)
        # ----------------------------------------------------
        # 9. Confidence diagnostics
        # ----------------------------------------------------
        if last_prob is not None:
            confidence = last_prob.max(dim=-1).values
            entropy = -torch.sum(
                last_prob * torch.log(last_prob.clamp_min(1e-8)),
                dim=-1,
            )
        else:
            confidence = None
            entropy = None
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

            # 建议加这个别名，方便 train.py / diagnostics 统一读取
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

                if valid_mask_flat is None:
                    valid_chunk_mask = None
                else:
                    valid_chunk_mask = valid_mask_flat[:, start:end]

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