import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Positional Embedding
def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float)
    omega /= (embed_dim / 2.)
    omega = 1. / (10000 ** omega)
    pos = torch.arange(grid_size, dtype=torch.float).reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    if cls_token:
        emb = torch.cat([torch.zeros(1, embed_dim), emb], dim=0)
    return emb

# 2. Patch Embedding
class PatchEmbed1D(nn.Module):
    def __init__(self, in_chans=12, patch_size=300, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x

# 3. Transformer Block
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

# 4. Encoder
class Encoder1D(nn.Module):
    def __init__(self, seq_len=5000, in_chans=12, patch_size=300, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbed1D(in_chans, patch_size, embed_dim)
        self.num_patches = seq_len // patch_size
        pos_embed = get_1d_sincos_pos_embed(embed_dim, self.num_patches)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x, mask_indices=None):
        tokens = self.patch_embed(x)
        B, N, D = tokens.shape
        if mask_indices is not None:
            expanded_indices = mask_indices.unsqueeze(-1).expand(-1, -1, D)
            x = tokens.gather(1, expanded_indices)
            pos = self.pos_embed.expand(B, -1, -1).gather(1, expanded_indices)
            x = x + pos
        else:
            x = tokens + self.pos_embed.expand(B, -1, -1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

# 5. Predictor
class Predictor1D(nn.Module):
    def __init__(self, embed_dim=128, pred_dim=64, depth=2, num_heads=4, num_patches=16):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, pred_dim)
        self.out_proj = nn.Linear(pred_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        pos_embed = get_1d_sincos_pos_embed(pred_dim, num_patches)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))
        self.blocks = nn.ModuleList([Block(pred_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim, eps=1e-6)

    def forward(self, ctx_tokens, ctx_indices, tgt_indices):
        B, N_ctx, D = ctx_tokens.shape
        N_tgt = tgt_indices.shape[1]
        ctx_x = self.in_proj(ctx_tokens)
        pred_dim = self.in_proj.out_features
        ctx_pos = self.pos_embed.expand(B, -1, -1).gather(1, ctx_indices.unsqueeze(-1).expand(-1, -1, pred_dim))
        ctx_x = ctx_x + ctx_pos
        mask_tokens = self.mask_token.expand(B, N_tgt, -1)
        tgt_pos = self.pos_embed.expand(B, -1, -1).gather(1, tgt_indices.unsqueeze(-1).expand(-1, -1, pred_dim))
        mask_tokens = mask_tokens + tgt_pos
        x = torch.cat([ctx_x, mask_tokens], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pred_x = x[:, -N_tgt:]
        return self.out_proj(pred_x)

# 6. SIGReg Loss
class SIGRegLoss(nn.Module):
    def __init__(self, embedding_dim, num_slices=128, num_t=17, t_max=5.0, min_batch_size=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_slices = num_slices
        self.num_t = num_t
        self.t_max = t_max
        self.min_batch_size = min_batch_size
        
    def forward(self, z):
        if z.ndim != 2:
            return torch.zeros((), device=z.device)
        if z.size(0) < self.min_batch_size:
            return z.sum() * 0.0
            
        z_float = z.float()
        raw = torch.randn(self.embedding_dim, self.num_slices, device=z.device, dtype=torch.float32)
        directions = raw / raw.norm(dim=0, keepdim=True).clamp(min=1e-12)
        
        projected = z_float @ directions
        t_grid = torch.linspace(-self.t_max, self.t_max, self.num_t, device=z.device, dtype=torch.float32)
        
        phase = projected.unsqueeze(-1) * t_grid.view(1, 1, -1)
        real_phi = torch.cos(phase).mean(dim=0)
        imag_phi = torch.sin(phase).mean(dim=0)
        
        normal_phi = torch.exp(-0.5 * t_grid.square()).view(1, -1)
        loss = (real_phi - normal_phi).square() + imag_phi.square()
        return loss.mean().to(dtype=z.dtype)

# 7. JEPA 1D
class JEPA_1D(nn.Module):
    def __init__(self, seq_len=5000, in_chans=12, patch_size=300, embed_dim=128):
        super().__init__()
        self.context_encoder = Encoder1D(seq_len, in_chans, patch_size, embed_dim)
        import copy
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = Predictor1D(embed_dim, pred_dim=embed_dim // 2, num_patches=self.context_encoder.num_patches)
        self.sigreg_loss = SIGRegLoss(embedding_dim=embed_dim)

    @torch.no_grad()
    def update_target_encoder(self, m=0.996):
        for pt, po in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.mul_(m).add_(po.detach(), alpha=1 - m)

    def forward(self, x, ctx_indices, tgt_indices):
        with torch.no_grad():
            full_targets = self.target_encoder(x, mask_indices=None)
            B, N_all, D = full_targets.shape
            expanded_tgt = tgt_indices.unsqueeze(-1).expand(-1, -1, D)
            target_reps = full_targets.gather(1, expanded_tgt)
            target_reps = F.layer_norm(target_reps, (D,))
        ctx_reps = self.context_encoder(x, mask_indices=ctx_indices)
        preds = self.predictor(ctx_reps, ctx_indices, tgt_indices)
        
        loss_pred = F.smooth_l1_loss(preds, target_reps)
        
        # SIGReg Loss to regularize target reps and predictions
        loss_sig_tgt = self.sigreg_loss(target_reps.reshape(-1, D))
        loss_sig_pred = self.sigreg_loss(preds.reshape(-1, D))
        loss_sig = 0.5 * (loss_sig_tgt + loss_sig_pred)
        
        total_loss = loss_pred + 0.1 * loss_sig
        return total_loss, preds, target_reps
