import torch
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
# 1. Spacing Encoder (FiLM style)
# =========================================================
class SpacingEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim * 2)  # gamma + beta
        )

    def forward(self, spacing):
        """
        spacing: (B,6)
        [sx_ref, sy_ref, sz_ref, sx_mov, sy_mov, sz_mov]
        """
        out = self.mlp(spacing)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


# =========================================================
# 2. Lightweight ViT block
# =========================================================
class SimpleViT(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # x: (B,N,C)
        x2 = self.norm(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out
        x = x + self.mlp(self.norm(x))
        return x


# =========================================================
# 3. Moving Encoder (2D + optional ViT)
# =========================================================
class MovingEncoder(nn.Module):
    def __init__(
        self,
        use_vit=True,
        vit_depth=2,
        vit_heads=4,
        window_size=(8, 8),
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # /2
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # /4
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # /8
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # /16
            nn.ReLU(inplace=True),
        )

        self.use_vit = use_vit

        if use_vit:
            self.vit = WindowAttention2D(
                dim=64,
                depth=vit_depth,
                num_heads=vit_heads,
                window_size=window_size,
            )

    def forward(self, x):
        """
        x: (B,1,X,Y,K)
        """
        B, C, X, Y, K = x.shape

        x = x.permute(0, 4, 1, 2, 3).reshape(B * K, 1, X, Y)

        feat = self.cnn(x)  # (B*K,64,Hc,Wc)

        if self.use_vit:
            feat = self.vit(feat)

        _, C2, H, W = feat.shape
        feat = feat.view(B, K, C2, H, W)

        return feat


# =========================================================
# 4. Reference Encoder (3D CNN)
# =========================================================
class ReferenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=(2, 2, 2), padding=1),   # xy /2, z /2
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, 3, stride=(2, 2, 2), padding=1),  # xy /4, z /4
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, 3, stride=(2, 2, 1), padding=1),  # xy /8
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, 3, stride=(2, 2, 1), padding=1),  # xy /16
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B,1,X,Y,Z)
        return self.net(x)

# =========================================================
# 5. Cost Volume
# =========================================================
def build_cost_volume(F_mov, F_ref):
    """
    F_mov: (B,K,C,H,W)
    F_ref: (B,C,H,W,D)
    """
    B, K, C, H, W = F_mov.shape
    _, _, _, _, D = F_ref.shape

    F_mov = F_mov.unsqueeze(-1)  # (B,K,C,H,W,1)
    F_ref = F_ref.unsqueeze(1)   # (B,1,C,H,W,D)

    cost = (F_mov * F_ref).sum(dim=2)
    return cost  # (B,K,H,W,D)


# =========================================================
# 6. Aggregation
# =========================================================
class CostAggregator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, cost):
        # cost: (B,K,H,W,D)
        B, K, H, W, D = cost.shape

        x = cost.reshape(B * K, 1, H, W, D)
        x = self.net(x)  # (B*K,32,H,W,D)

        # better z pooling
        x_mean = x.mean(dim=-1)          # (B*K,32,H,W)
        x_max = x.max(dim=-1).values     # (B*K,32,H,W)

        x = torch.cat([x_mean, x_max], dim=1)  # (B*K,64,H,W)

        x = x.reshape(B, K, 64, H, W)
        return x


# =========================================================
# 7. Prediction Head
# =========================================================
class Head(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1)
        )

    def forward(self, x):
        # x: (B,K,C,H,W)
        B, K, C, H, W = x.shape

        x = x.view(B*K, C, H, W)
        out = self.net(x)

        out = out.view(B, K, 3, H, W)
        out = out.permute(0,1,3,4,2)

        return out  # (B,K,H,W,3)


# =========================================================
# 8. Full Model
# =========================================================
class CoarseMatchingNetV1(nn.Module):
    def __init__(
        self,
        use_vit=True,
        vit_depth=2,
        vit_heads=4,
        window_size=(8, 8),
    ):
        super().__init__()

        self.mov_enc = MovingEncoder(
            use_vit=use_vit,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            window_size=window_size,
        )

        self.ref_enc = ReferenceEncoder()

        self.spacing_enc = SpacingEncoder(out_dim=64)

        self.agg = CostAggregator()
        self.head = Head()

    def apply_film(self, feat, gamma, beta, feat_type):
        """
        feat_type:
            "mov": feat is (B,K,C,H,W)
            "ref": feat is (B,C,H,W,D)
        gamma/beta: (B,C)
        """

        if feat_type == "mov":
            # (B,C) -> (B,1,C,1,1)
            gamma = gamma[:, None, :, None, None]
            beta = beta[:, None, :, None, None]

        elif feat_type == "ref":
            # (B,C) -> (B,C,1,1,1)
            gamma = gamma[:, :, None, None, None]
            beta = beta[:, :, None, None, None]

        else:
            raise ValueError(f"Unknown feat_type: {feat_type}")

        return feat * (1.0 + gamma) + beta

    def forward(self, mov, ref, spacing):
        """
        mov: (B,1,X,Y,K)
        ref: (B,1,X,Y,Z)
        spacing: (B,6)
        """

        # feature extraction
        F_mov = self.mov_enc(mov)   # (B,K,64,H,W)
        F_ref = self.ref_enc(ref)   # (B,64,H,W,D)

        # spacing conditioning
        gamma, beta = self.spacing_enc(spacing)
        F_mov = self.apply_film(F_mov, gamma, beta, feat_type="mov")
        F_ref = self.apply_film(F_ref, gamma, beta, feat_type="ref")

        # cost volume
        cost = build_cost_volume(F_mov, F_ref)

        # aggregation
        feat = self.agg(cost)

        # prediction
        disp = self.head(feat)

        return disp