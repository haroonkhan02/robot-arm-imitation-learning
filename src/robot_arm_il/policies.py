"""Policies: a behaviour-cloning MLP and a state-based ACT (action-chunking transformer)."""

from __future__ import annotations

import torch
from torch import nn

from .config import ACTION_DIM, STATE_DIM


class MLPPolicy(nn.Module):
    """Behaviour-cloning MLP: the last `history` observations -> the next `chunk` actions.

    `history=1, chunk=1` is plain behaviour cloning. The other settings exist as controls, to
    separate what ACT gains from memory, from predicting several steps, and from its architecture.
    """

    def __init__(
        self,
        hidden: int = 256,
        layers: int = 3,
        dropout: float = 0.0,
        history: int = 1,
        chunk: int = 1,
    ):
        super().__init__()
        self.history, self.chunk = history, chunk
        dims = [STATE_DIM * history] + [hidden] * layers
        blocks: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            blocks += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*blocks, nn.Linear(hidden, ACTION_DIM * chunk))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # (B, S * H) -> (B, K, A)
        return self.net(obs).view(obs.shape[0], self.chunk, ACTION_DIM)

    def loss(self, obs: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor) -> dict:
        l1 = ((self(obs) - actions).abs().mean(-1) * mask).sum() / mask.sum()
        return {"loss": l1, "l1": l1.detach()}


class ACTPolicy(nn.Module):
    """Action chunking with a transformer (Zhao et al. 2023) on state observations.

    Training: a CVAE encoder reads the observation and the whole target chunk and produces a
    latent z; the decoder predicts the chunk from (obs, z) with one query token per future
    step. Inference: z = 0, so the output is the mean behaviour. Loss: L1 + beta * KL.
    """

    def __init__(
        self,
        chunk: int = 20,
        d_model: int = 128,
        heads: int = 4,
        enc_layers: int = 2,
        dec_layers: int = 2,
        latent: int = 16,
        beta: float = 10.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk, self.latent, self.beta = chunk, latent, beta

        def stack(n: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True
            )
            return nn.TransformerEncoder(layer, n, enable_nested_tensor=False)

        # CVAE encoder over [CLS, obs, a_1..a_K]
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.enc_obs = nn.Linear(STATE_DIM, d_model)
        self.enc_act = nn.Linear(ACTION_DIM, d_model)
        self.enc_pos = nn.Parameter(torch.randn(1, chunk + 2, d_model) * 0.02)
        self.style_encoder = stack(enc_layers)
        self.to_latent = nn.Linear(d_model, 2 * latent)
        # policy: encoder over [z, obs], decoder with learned per-step queries
        self.z_proj = nn.Linear(latent, d_model)
        self.obs_proj = nn.Linear(STATE_DIM, d_model)
        self.ctx_pos = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        self.encoder = stack(enc_layers)
        self.queries = nn.Parameter(torch.randn(1, chunk, d_model) * 0.02)
        dec = nn.TransformerDecoderLayer(
            d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.head = nn.Linear(d_model, ACTION_DIM)

    def _decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        ctx = torch.stack([self.z_proj(z), self.obs_proj(obs)], dim=1) + self.ctx_pos
        memory = self.encoder(ctx)
        q = self.queries.expand(obs.shape[0], -1, -1)
        return self.head(self.decoder(q, memory))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # (B, S) -> (B, K, A)
        return self._decode(obs, torch.zeros(obs.shape[0], self.latent, device=obs.device))

    def loss(self, obs: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor) -> dict:
        b = obs.shape[0]
        tokens = torch.cat(
            [self.cls.expand(b, -1, -1), self.enc_obs(obs).unsqueeze(1), self.enc_act(actions)], 1
        )
        pad = torch.cat([torch.zeros(b, 2, dtype=torch.bool, device=obs.device), ~mask], 1)
        h = self.style_encoder(tokens + self.enc_pos, src_key_padding_mask=pad)[:, 0]
        mu, logvar = self.to_latent(h).chunk(2, dim=-1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        pred = self._decode(obs, z)
        l1 = ((pred - actions).abs().mean(-1) * mask).sum() / mask.sum()
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        return {"loss": l1 + self.beta * kl, "l1": l1.detach(), "kl": kl.detach()}


def build_policy(kind: str, **kwargs) -> nn.Module:
    if kind == "bc":
        return MLPPolicy(**kwargs)
    if kind == "act":
        return ACTPolicy(**kwargs)
    raise ValueError(f"unknown policy kind {kind!r}")


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
