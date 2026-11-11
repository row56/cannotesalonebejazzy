from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .repr import VOCAB_SIZE


@dataclass
class VAEConfig:
    vocab_size: int = VOCAB_SIZE
    embed_dim: int = 64
    enc_hidden: int = 512
    dec_hidden: int = 512
    latent_dim: int = 128
    dec_layers: int = 1
    dropout: float = 0.1
    word_dropout: float = 0.2


class MelodyVAE(nn.Module):

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.unk = cfg.vocab_size
        self.embed = nn.Embedding(cfg.vocab_size + 1, cfg.embed_dim)
        self.encoder = nn.GRU(cfg.embed_dim, cfg.enc_hidden, batch_first=True, bidirectional=True)
        self.to_mu = nn.Linear(2 * cfg.enc_hidden, cfg.latent_dim)
        self.to_logvar = nn.Linear(2 * cfg.enc_hidden, cfg.latent_dim)
        self.z_to_hidden = nn.Linear(cfg.latent_dim, cfg.dec_hidden * cfg.dec_layers)
        self.decoder = nn.GRU(
            cfg.embed_dim + cfg.latent_dim,
            cfg.dec_hidden,
            num_layers=cfg.dec_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.dec_layers > 1 else 0.0,
        )
        self.out = nn.Linear(cfg.dec_hidden, cfg.vocab_size)
        self.drop = nn.Dropout(cfg.dropout)

    def encode(self, x: torch.Tensor):
    # print(x.shape)
        _, h = self.encoder(self.drop(self.embed(x)))
        h = torch.cat([h[0], h[1]], dim=-1)
        return (self.to_mu(h), self.to_logvar(h).clamp(-8.0, 8.0))

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def _init_hidden(self, z: torch.Tensor):
        h = torch.tanh(self.z_to_hidden(z))
        return h.view(-1, self.cfg.dec_layers, self.cfg.dec_hidden).transpose(0, 1).contiguous()

    def decode(self, z: torch.Tensor, x: torch.Tensor):
        b, t = x.shape
        inp = torch.full_like(x[:, :1], 0)
        inp = torch.cat([inp, x[:, :-1]], dim=1)
        if self.training and self.cfg.word_dropout > 0:
            mask = torch.rand_like(inp, dtype=torch.float) < self.cfg.word_dropout
            inp = torch.where(mask, torch.full_like(inp, self.unk), inp)
        emb = self.drop(self.embed(inp))
        zt = z.unsqueeze(1).expand(b, t, z.size(-1))
        h, _ = self.decoder(torch.cat([emb, zt], dim=-1), self._init_hidden(z))
        return self.out(self.drop(h))

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        steps: int,
        temperature: float = 1.0,
        greedy: bool = False,
        top_p: float = 1.0,
        logit_bias: torch.Tensor | None = None,
    ):
        self.eval()
        b = z.size(0)
        tok = torch.zeros(b, 1, dtype=torch.long, device=z.device)
        hidden = self._init_hidden(z)
        zt = z.unsqueeze(1)
        out = []
        for _ in range(steps):
            h, hidden = self.decoder(torch.cat([self.embed(tok), zt], dim=-1), hidden)
            logits = self.out(h[:, -1])
            if logit_bias is not None:
                logits = logits + logit_bias
            if greedy:
                tok = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-05)
                if top_p < 1.0:
                    ordered, idx = logits.sort(dim=-1, descending=True)
                    cum = F.softmax(ordered, dim=-1).cumsum(-1)
                    drop = cum - F.softmax(ordered, dim=-1) >= top_p
                    ordered = ordered.masked_fill(drop, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(1, idx, ordered)
                tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            out.append(tok)
        return torch.cat(out, dim=1)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar) if self.training else mu
        return (self.decode(z, x), mu, logvar, z)


class _GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return (-ctx.lambd * grad, None)


class CorpusAdversary(nn.Module):

    def __init__(self, latent_dim: int, n_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, z: torch.Tensor, lambd: float = 1.0):
        return self.net(_GradReverse.apply(z, lambd))


def vae_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    free_bits: float = 0.05,
):
    b, t, v = logits.shape
    recon = F.cross_entropy(logits.reshape(b * t, v), target.reshape(b * t), reduction="sum") / b
    kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_raw = kl_dim.sum(-1).mean()
    kl_clamped = kl_dim.mean(0).clamp(min=free_bits).sum()
    return {
        "loss": recon + beta * kl_clamped,
        "recon": recon.detach(),
        "kl": kl_raw.detach(),
        "active_dims": (kl_dim.mean(0) > 0.02).sum().detach(),
    }
