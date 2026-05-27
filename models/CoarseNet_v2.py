# models/coarse_matching_net_v2.py
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    """
    x: (B, C, H, W)
    return:
        windows: (num_windows*B, window_size, window_size, C)
        pad_hw:  (Hp, Wp)
    """
    B, C, H, W = x.shape
    Wh, Ww = window_size

    pad_h = (Wh - H % Wh) % Wh
    pad_w = (Ww - W % Ww) % Ww

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    _, _, Hp, Wp = x.shape

    x = x.view(B, C, Hp // Wh, Wh, Wp // Ww, Ww)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()

    windows = x.view(-1, Wh, Ww, C)
    return windows, (Hp, Wp)

def window_reverse(windows, window_size, pad_hw, original_hw, B):
    """
    windows: (num_windows*B, Wh, Ww, C)
    return:
        x: (B, C, H, W)
    """
    Wh, Ww = window_size
    Hp, Wp = pad_hw
    H, W = original_hw

    C = windows.shape[-1]

    x = windows.view(B, Hp // Wh, Wp // Ww, Wh, Ww, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, Hp, Wp)

    x = x[:, :, :H, :W]
    return x

class WindowAttentionBlock2D(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        window_size=(8, 8),
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        windows, pad_hw = window_partition(x, self.window_size)
        Wh, Ww = self.window_size

        # (num_windows*B, Wh, Ww, C)
        tokens = windows.view(windows.shape[0], Wh * Ww, C)

        shortcut = tokens

        tokens_norm = self.norm1(tokens)
        attn_out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm)

        tokens = shortcut + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))

        windows = tokens.view(-1, Wh, Ww, C)

        x = window_reverse(
            windows=windows,
            window_size=self.window_size,
            pad_hw=pad_hw,
            original_hw=(H, W),
            B=B,
        )

        return x

class WindowAttention2D(nn.Module):
    def __init__(
        self,
        dim,
        depth=2,
        num_heads=4,
        window_size=(8, 8),
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            WindowAttentionBlock2D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        for blk in self.blocks:
            x = blk(x)
        return x

# =========================================================
# Basic residual blocks
# =========================================================
class ResidualBlock2D(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class ResidualBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


# =========================================================
# Spacing FiLM encoder
# =========================================================
class SpacingEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim * 2),
        )

    def forward(self, spacing):
        out = self.mlp(spacing)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


# =========================================================
# Moving encoder: 2D ResNet encoder
# =========================================================
class MovingEncoderRes2D(nn.Module):
    def __init__(
        self,
        in_ch=1,
        feat_ch=64,
        use_vit=True,
        vit_depth=2,
        vit_heads=4,
        window_size=(8, 8),
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),  # /2
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            ResidualBlock2D(32),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),     # /4
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            ResidualBlock2D(64),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),     # /8
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            ResidualBlock2D(64),

            nn.Conv2d(64, feat_ch, 3, stride=2, padding=1),  # /16
            nn.GroupNorm(8, feat_ch),
            nn.ReLU(inplace=True),
            ResidualBlock2D(feat_ch),
        )

        self.use_vit = use_vit
        if use_vit:
            self.vit = WindowAttention2D(
                dim=feat_ch,
                depth=vit_depth,
                num_heads=vit_heads,
                window_size=window_size,
            )

    def forward(self, x):
        # x: (B,1,X,Y,K)
        B, C, X, Y, K = x.shape

        x = x.permute(0, 4, 1, 2, 3).reshape(B * K, C, X, Y)
        feat = self.stem(x)  # (B*K,C,Hc,Wc)

        if self.use_vit:
            feat = self.vit(feat)

        _, C2, H, W = feat.shape
        feat = feat.reshape(B, K, C2, H, W)

        return feat


# =========================================================
# Moving slice fusion
# =========================================================
class SliceFusion(nn.Module):
    """
    Fuse neighboring sparse slices.

    Input:
        (B,K,C,H,W)
    Internally:
        (B,C,K,H,W)
    """

    def __init__(self, ch=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, ch),
            nn.ReLU(inplace=True),
            ResidualBlock3D(ch),
        )

    def forward(self, x):
        # x: (B,K,C,H,W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B,C,K,H,W)

        y = self.net(x)

        x = x + y

        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B,K,C,H,W)
        return x


# =========================================================
# Reference encoder: 3D ResNet encoder
# =========================================================
class ReferenceEncoderRes3D(nn.Module):
    def __init__(self, in_ch=1, feat_ch=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, stride=(2, 2, 2), padding=1),  # xyz /2
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            ResidualBlock3D(32),

            nn.Conv3d(32, 64, 3, stride=(2, 2, 2), padding=1),     # xy /4, z /4
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64),

            nn.Conv3d(64, 64, 3, stride=(2, 2, 1), padding=1),     # xy /8
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64),

            nn.Conv3d(64, feat_ch, 3, stride=(2, 2, 1), padding=1),  # xy /16
            nn.GroupNorm(8, feat_ch),
            nn.ReLU(inplace=True),
            ResidualBlock3D(feat_ch),
        )

    def forward(self, x):
        """
        x: (B,1,X,Y,Z)
        return:
            feat: (B,C,Hc,Wc,Dc)
        """
        return self.net(x)


# =========================================================
# Learned matching module
# =========================================================
class LearnedMatching(nn.Module):
    """
    Replace dot product with learned matching.

    For each moving slice k and each reference z d:

        concat = [Fm, Fr, Fm*Fr, |Fm-Fr|]
        score = CNN(concat)

    Output:
        cost: (B,K,H,W,D)
    """

    def __init__(self, feat_ch=64, hidden_ch=64):
        super().__init__()

        in_ch = feat_ch * 4

        self.matcher = nn.Sequential(
            nn.Conv3d(in_ch, hidden_ch, kernel_size=1),
            nn.GroupNorm(8, hidden_ch),
            nn.ReLU(inplace=True),

            nn.Conv3d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_ch),
            nn.ReLU(inplace=True),

            nn.Conv3d(hidden_ch, 1, kernel_size=1),
        )

    def forward(self, F_mov, F_ref):
        """
        F_mov: (B,K,C,H,W)
        F_ref: (B,C,H,W,D)

        return:
            cost: (B,K,H,W,D)
        """
        B, K, C, H, W = F_mov.shape
        _, _, _, _, D = F_ref.shape

        Fm = F_mov.unsqueeze(-1).expand(B, K, C, H, W, D)
        Fr = F_ref.unsqueeze(1).expand(B, K, C, H, W, D)

        feat = torch.cat(
            [
                Fm,
                Fr,
                Fm * Fr,
                torch.abs(Fm - Fr),
            ],
            dim=2,
        )  # (B,K,4C,H,W,D)

        feat = feat.reshape(B * K, 4 * C, H, W, D)

        cost = self.matcher(feat)  # (B*K,1,H,W,D)

        cost = cost.reshape(B, K, H, W, D)

        return cost


# =========================================================
# Cost aggregator
# =========================================================
class CostAggregator(nn.Module):
    def __init__(self, hidden_ch=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(1, hidden_ch, 3, padding=1),
            nn.GroupNorm(8, hidden_ch),
            nn.ReLU(inplace=True),

            ResidualBlock3D(hidden_ch),
            ResidualBlock3D(hidden_ch),
        )

    def forward(self, cost):
        """
        cost: (B,K,H,W,D)
        return:
            feat: (B,K,2*hidden_ch,H,W)
        """
        B, K, H, W, D = cost.shape

        x = cost.reshape(B * K, 1, H, W, D)

        x = self.net(x)  # (B*K,C,H,W,D)

        x_mean = x.mean(dim=-1)
        x_max = x.max(dim=-1).values

        x = torch.cat([x_mean, x_max], dim=1)  # (B*K,2C,H,W)

        C2 = x.shape[1]
        x = x.reshape(B, K, C2, H, W)

        return x


# =========================================================
# Prediction head
# =========================================================
class PredictionHead(nn.Module):
    def __init__(self, in_ch=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.GroupNorm(8, in_ch),
            nn.ReLU(inplace=True),

            ResidualBlock2D(in_ch),

            nn.Conv2d(in_ch, 3, 1),
        )

    def forward(self, x):
        """
        x: (B,K,C,H,W)
        return:
            disp: (B,K,H,W,3)
        """
        B, K, C, H, W = x.shape

        x = x.reshape(B * K, C, H, W)
        out = self.net(x)

        out = out.reshape(B, K, 3, H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()

        return out


# =========================================================
# Full model v2
# =========================================================
class CoarseMatchingNetV2(nn.Module):
    def __init__(
        self,
        feat_ch=64,
        use_slice_fusion=True,
        use_vit=True,
        vit_depth=2,
        vit_heads=4,
        window_size=(8, 8),
    ):
        super().__init__()

        self.mov_enc = MovingEncoderRes2D(
            in_ch=1,
            feat_ch=feat_ch,
            use_vit=use_vit,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            window_size=window_size,
        )

        self.ref_enc = ReferenceEncoderRes3D(
            in_ch=1,
            feat_ch=feat_ch,
        )

        self.use_slice_fusion = use_slice_fusion
        if use_slice_fusion:
            self.slice_fusion = SliceFusion(ch=feat_ch)

        self.spacing_enc = SpacingEncoder(out_dim=feat_ch)

        self.matcher = LearnedMatching(
            feat_ch=feat_ch,
            hidden_ch=feat_ch,
        )

        self.agg = CostAggregator(hidden_ch=32)
        self.head = PredictionHead(in_ch=64)
    def apply_film(self, feat, gamma, beta, feat_type):
        """
        gamma/beta: (B,C)

        feat_type:
            "mov": feat is (B,K,C,H,W)
            "ref": feat is (B,C,H,W,D)
        """

        if feat_type == "mov":
            gamma = gamma[:, None, :, None, None]
            beta = beta[:, None, :, None, None]

        elif feat_type == "ref":
            gamma = gamma[:, :, None, None, None]
            beta = beta[:, :, None, None, None]

        else:
            raise ValueError(f"Unknown feat_type: {feat_type}")

        return feat * (1.0 + gamma) + beta

    def forward(self, mov, ref, spacing):
        """
        mov:     (B,1,X,Y,K)
        ref:     (B,1,X,Y,Z)
        spacing: (B,6)

        return:
            pred_disp: (B,K,Hc,Wc,3)
        """

        F_mov = self.mov_enc(mov)  # (B,K,C,H,W)

        if self.use_slice_fusion:
            F_mov = self.slice_fusion(F_mov)

        F_ref = self.ref_enc(ref)  # (B,C,H,W,D)

        gamma, beta = self.spacing_enc(spacing)

        F_mov = self.apply_film(F_mov, gamma, beta, feat_type="mov")
        F_ref = self.apply_film(F_ref, gamma, beta, feat_type="ref")

        cost = self.matcher(F_mov, F_ref)  # (B,K,H,W,D)

        feat = self.agg(cost)  # (B,K,64,H,W)

        pred_disp = self.head(feat)  # (B,K,H,W,3)

        return pred_disp