"""
Physics-guided residual SAC for RIS phase configuration (contextual-bandit form).

Idea
----
A model-based solver (alternating optimisation, AO) is optimal ONLY for the hardware model it
assumes. Real RIS unit cells deviate from the nominal PDA curve (manufacturing spread, ageing,
temperature). Instead of learning the whole configuration from scratch, the agent outputs a
bounded per-element phase RESIDUAL  delta  on top of the AO solution:

        theta = AO_nominal(CSI) + delta(features),     G = ZF(theta)

* the policy starts at delta ~ 0, i.e. it starts exactly at the model-based solution
  (no "learning from random phases" phase, no safety gap at initialisation);
* it can only help by learning the hardware deviation from rate feedback;
* inference is one forward pass (zero probing measurements per slot).

Each slot's channel is redrawn, so the problem is a contextual bandit: SAC with gamma = 0
(critic regresses the immediate reward, no bootstrapping).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 1.0


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    def __init__(self, in_dim, act_dim, hidden, delta_max):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        # start at the physics prior: mean residual ~ 0, moderate exploration noise
        nn.init.uniform_(self.mu.weight, -1e-3, 1e-3); nn.init.zeros_(self.mu.bias)
        nn.init.uniform_(self.log_std.weight, -1e-3, 1e-3); nn.init.constant_(self.log_std.bias, -1.0)
        self.delta_max = delta_max

    def forward(self, x, deterministic=False):
        h = self.body(x)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        if deterministic:
            return torch.tanh(mu) * self.delta_max, None
        u = mu + log_std.exp() * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / log_std.exp()) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(-1) - mu.shape[-1] * np.log(self.delta_max)
        return a * self.delta_max, logp


class ResidualSAC:
    def __init__(self, in_dim, act_dim, device='cpu', hidden=256, delta_max=1.0, lr=3e-4,
                 alpha_init=0.1, target_entropy=None):
        self.device = torch.device(device)
        self.in_dim, self.act_dim, self.delta_max, self.hidden = in_dim, act_dim, delta_max, hidden
        self.actor = Actor(in_dim, act_dim, hidden, delta_max).to(self.device)
        self.q1 = _mlp(in_dim + act_dim, hidden, 1).to(self.device)
        self.q2 = _mlp(in_dim + act_dim, hidden, 1).to(self.device)
        self.log_alpha = torch.tensor(float(np.log(alpha_init)), device=self.device, requires_grad=True)
        self.target_entropy = -float(act_dim) if target_entropy is None else target_entropy
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)

    @torch.no_grad()
    def act(self, x, deterministic=True):
        t = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(t, deterministic=deterministic)
        return a[0].cpu().numpy().astype(np.float64)

    def update(self, x, a, r):
        """x:(B,in) a:(B,act) r:(B,) tensors on self.device. gamma = 0 (contextual bandit)."""
        xa = torch.cat([x, a], -1)
        q_loss = F.mse_loss(self.q1(xa).squeeze(-1), r) + F.mse_loss(self.q2(xa).squeeze(-1), r)
        self.opt_q.zero_grad(); q_loss.backward(); self.opt_q.step()

        a_new, logp = self.actor(x)
        xa_new = torch.cat([x, a_new], -1)
        q_min = torch.min(self.q1(xa_new), self.q2(xa_new)).squeeze(-1)
        alpha = self.log_alpha.exp().detach()
        a_loss = (alpha * logp - q_min).mean()
        self.opt_a.zero_grad(); a_loss.backward(); self.opt_a.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()
        return dict(q_loss=q_loss.item(), a_loss=a_loss.item(), alpha=self.log_alpha.exp().item(),
                    logp=logp.mean().item())

    def save(self, path):
        torch.save(dict(actor=self.actor.state_dict(), q1=self.q1.state_dict(), q2=self.q2.state_dict(),
                        log_alpha=self.log_alpha.detach().cpu(),
                        meta=dict(in_dim=self.in_dim, act_dim=self.act_dim, delta_max=self.delta_max,
                                  hidden=self.hidden)), path)

    @classmethod
    def load(cls, path, device='cpu'):
        ck = torch.load(path, map_location=device)
        m = ck['meta']
        agent = cls(m['in_dim'], m['act_dim'], device=device, hidden=m['hidden'], delta_max=m['delta_max'])
        agent.actor.load_state_dict(ck['actor']); agent.q1.load_state_dict(ck['q1']); agent.q2.load_state_dict(ck['q2'])
        agent.log_alpha.data.copy_(ck['log_alpha'].to(agent.device))
        return agent
