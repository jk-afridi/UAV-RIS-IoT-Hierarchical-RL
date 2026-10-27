"""
Hybrid-action PPO for the macro level: a CATEGORICAL head picks the cluster to serve and a
Gaussian head picks the 2-D UAV velocity. (The original code produced the cluster index by
thresholding a continuous Gaussian sample, which is a poorly-conditioned way to learn a discrete
choice and has no meaningful probability for it.)

State: env.get_high_level_state()  (normalised UAV position, energy, cluster centres relative to
the UAV, per-device AoI), so the policy can generalise across city layouts.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class HybridPPO:
    def __init__(self, state_dim, n_clusters, device='cpu', hidden=256, lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=8, minibatch=256, ent_coef=0.01, vf_coef=0.5, init_log_std=-0.5):
        self.device = torch.device(device)
        self.n_clusters, self.state_dim, self.hidden = n_clusters, state_dim, hidden
        self.gamma, self.lam, self.clip, self.epochs = gamma, lam, clip, epochs
        self.minibatch, self.ent_coef, self.vf_coef = minibatch, ent_coef, vf_coef
        self.body = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head_c = nn.Linear(hidden, n_clusters)
        self.head_v = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.full((2,), float(init_log_std)))
        self.vf = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, 1))
        for m in (self.body, self.head_c, self.head_v, self.vf):
            m.to(self.device)
        self.log_std.data = self.log_std.data.to(self.device)
        self.params = (list(self.body.parameters()) + list(self.head_c.parameters())
                       + list(self.head_v.parameters()) + [self.log_std] + list(self.vf.parameters()))
        self.opt = torch.optim.Adam(self.params, lr=lr)
        self.buf = dict(s=[], c=[], v=[], logp=[], r=[], d=[], val=[])

    # ------------------------------------------------------------------ policy
    def _dists(self, s):
        h = self.body(s)
        return Categorical(logits=self.head_c(h)), Normal(torch.tanh(self.head_v(h)), self.log_std.exp())

    @torch.no_grad()
    def value(self, state):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.vf(s).item())

    @torch.no_grad()
    def act(self, state, deterministic=False):
        """-> (cluster:int, vel in [-1,1]^2 (np), raw_v (np), logp:float, value:float)"""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dc, dv = self._dists(s)
        if deterministic:
            c, raw = dc.probs.argmax(-1), dv.mean
        else:
            c, raw = dc.sample(), dv.sample()
        logp = dc.log_prob(c) + dv.log_prob(raw).sum(-1)
        return (int(c.item()), raw[0].clamp(-1, 1).cpu().numpy().astype(np.float64), raw[0].cpu().numpy(),
                float(logp.item()), float(self.vf(s).item()))

    def env_action(self, state, v_max, deterministic=True):
        """Convenience for evaluation: -> (velocity in m/s, cluster id)."""
        c, vel, _, _, _ = self.act(state, deterministic=deterministic)
        return vel * v_max, c

    # ------------------------------------------------------------------ learning
    def store(self, s, c, raw_v, logp, r, done, val):
        b = self.buf
        b['s'].append(np.asarray(s, np.float32)); b['c'].append(c); b['v'].append(np.asarray(raw_v, np.float32))
        b['logp'].append(logp); b['r'].append(r); b['d'].append(float(done)); b['val'].append(val)

    def update(self):
        b = self.buf
        T = len(b['r'])
        if T == 0:
            return {}
        r, d, val = np.array(b['r'], np.float32), np.array(b['d'], np.float32), np.array(b['val'], np.float32)
        adv = np.zeros(T, np.float32); last = 0.0
        for t in reversed(range(T)):
            nonterm = 1.0 - d[t]
            nxt = val[t + 1] if t + 1 < T else 0.0
            delta = r[t] + self.gamma * nxt * nonterm - val[t]
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        S = torch.as_tensor(np.array(b['s']), device=self.device)
        C = torch.as_tensor(np.array(b['c']), device=self.device)
        V = torch.as_tensor(np.array(b['v']), device=self.device)
        LP = torch.as_tensor(np.array(b['logp'], np.float32), device=self.device)
        ADV, RET = torch.as_tensor(adv, device=self.device), torch.as_tensor(ret, device=self.device)

        stats = {}
        for _ in range(self.epochs):
            perm = torch.randperm(T, device=self.device)
            for i in range(0, T, self.minibatch):
                idx = perm[i:i + self.minibatch]
                dc, dv = self._dists(S[idx])
                logp = dc.log_prob(C[idx]) + dv.log_prob(V[idx]).sum(-1)
                ratio = (logp - LP[idx]).exp()
                pg = -torch.min(ratio * ADV[idx], ratio.clamp(1 - self.clip, 1 + self.clip) * ADV[idx]).mean()
                vloss = ((self.vf(S[idx]).squeeze(-1) - RET[idx]) ** 2).mean()
                ent = (dc.entropy() + dv.entropy().sum(-1)).mean()
                loss = pg + self.vf_coef * vloss - self.ent_coef * ent
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.params, 0.5); self.opt.step()
                stats = dict(pg=pg.item(), vf=vloss.item(), ent=ent.item())
        for k in b:
            b[k].clear()
        return stats

    # ------------------------------------------------------------------ io
    def save(self, path):
        torch.save(dict(body=self.body.state_dict(), head_c=self.head_c.state_dict(), head_v=self.head_v.state_dict(),
                        log_std=self.log_std.data.cpu(), vf=self.vf.state_dict(),
                        meta=dict(state_dim=self.state_dim, n_clusters=self.n_clusters, hidden=self.hidden)), path)

    @classmethod
    def load(cls, path, device='cpu'):
        ck = torch.load(path, map_location=device)
        m = ck['meta']
        agent = cls(m['state_dim'], m['n_clusters'], device=device, hidden=m['hidden'])
        agent.body.load_state_dict(ck['body']); agent.head_c.load_state_dict(ck['head_c'])
        agent.head_v.load_state_dict(ck['head_v']); agent.vf.load_state_dict(ck['vf'])
        agent.log_std.data.copy_(ck['log_std'].to(agent.device))
        return agent
