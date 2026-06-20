import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        return (x / rms) * self.gamma


class LMBlock(nn.Module):
    def __init__(self, dim: int, mem_dim: int, up_dim: int, dropout: float = 0.0):
        super().__init__()
        self.up_proj   = nn.Linear(dim,          up_dim,  bias=False)
        self.gate_proj = nn.Linear(dim + mem_dim, up_dim,  bias=False)
        self.mem_proj  = nn.Linear(up_dim,        mem_dim, bias=False)
        self.down_proj = nn.Linear(up_dim,        dim,     bias=False)
        self.norm      = RMSNorm(dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        residual = x
        x = self.norm(x)

        up     = self.up_proj(x)
        gate   = self.gate_proj(torch.cat([x, mem], dim=-1))
        latent = self.dropout(F.silu(gate) * up)

        out   = self.down_proj(latent)
        n_mem = mem + self.mem_proj(latent)

        return residual + out, n_mem


class LM(nn.Module):
    def __init__(self, dim: int, n_layers: int, vocab: int,
                 mem_dim: int, up_dim: int, dropout: float = 0.0):
        super().__init__()

        self.mem_dim  = mem_dim
        self.n_layers = n_layers

        self.emb        = nn.Embedding(vocab, dim)
        self.emb_drop   = nn.Dropout(dropout)
        self.token_proj = nn.Linear(dim + mem_dim, dim, bias=False)

        self.layers = nn.ModuleList([
            LMBlock(dim, mem_dim, up_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.mem_norm  = RMSNorm(mem_dim)
        self.mem_gate  = nn.Linear(2 * mem_dim, mem_dim)
        # linspace spans sigmoid≈0.27 (fast-forget) → sigmoid≈0.99 (long-term)
        self.mem_decay = nn.Parameter(torch.linspace(-1.0, 4.6, mem_dim))

        self.out_norm  = RMSNorm(dim)
        self.pred_head        = nn.Linear(dim, vocab, bias=False)
        self.pred_head.weight = self.emb.weight  # tied weights

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        for layer in self.layers:
            nn.init.normal_(layer.down_proj.weight, std=0.02 / (2 * self.n_layers) ** 0.5)
            nn.init.normal_(layer.mem_proj.weight,  std=0.02 / (2 * self.n_layers) ** 0.5)

    def forward(self, x: torch.Tensor, mem: torch.Tensor | None = None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, T = x.shape

        if mem is None:
            mem = x.new_zeros(B, self.mem_dim)

        h = self.emb_drop(self.emb(x))
        all_logits = []

        for t in range(T):
            h_t = h[:, t, :]
            h_t = self.token_proj(torch.cat([h_t, mem], dim=-1))

            n_mem = mem
            for layer in self.layers:
                h_t, n_mem = layer(h_t, n_mem)

            n_mem    = self.mem_norm(n_mem)
            decay    = torch.sigmoid(self.mem_decay)
            gate_out = F.silu(self.mem_gate(torch.cat([mem, n_mem], dim=-1)))
            mem      = decay * mem + (1.0 - decay) * gate_out * n_mem

            all_logits.append(self.pred_head(self.out_norm(h_t)))

        return torch.stack(all_logits, dim=1), mem

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
