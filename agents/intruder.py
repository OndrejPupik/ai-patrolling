"""The intruder's real decision process: a causal, risk-seeking optimal-
stopping planner.

What it knows (per Kerckhoffs's principle -- assume the mechanism, not its
secrecy, is the only real defense):
  - The graph (weights, edge lengths) exactly.
  - The patroller's policy network exactly (but frozen -- this is always a
    best response to one snapshot of the patroller, not a moving target).
  - The current state (position, idleness vector) exactly -- this is all
    "history" reduces to, since idleness is a deterministic function of it.

What it does NOT know:
  - The realized future draws of the patroller's stochastic policy. It only
    knows the probability distribution over them.

At every decision point it chooses between "attack the best available node
now" and "wait", using bounded Monte-Carlo lookahead to estimate, for each
candidate node, the CVaR (upper-tail risk measure) of weight(v) * time-to-
next-visit -- i.e. it is actively hunting for the tail, not the average case.
This is the practical, bounded-depth approximation to the full infinite-
horizon risk-sensitive Bellman recursion we derived by hand.
"""
import numpy as np
import torch

from env.patrol_env import PatrolEnv


class CVaRLookaheadIntruder:
    def __init__(self, model, device, horizon: int = 25, rollouts: int = 16, alpha: float = 0.3):
        self.model = model
        self.device = device
        self.horizon = horizon
        self.rollouts = rollouts
        self.alpha = alpha

    def _sample_hitting_times(self, env: PatrolEnv) -> np.ndarray:
        """hitting[k, v] = ticks until node v is first (re-)visited in
        simulated continuation k, capped at self.horizon if never visited."""
        n = env.n
        hitting = np.full((self.rollouts, n), self.horizon, dtype=np.float32)
        for k in range(self.rollouts):
            sim = env.clone()
            visited = np.zeros(n, dtype=bool)
            visited[sim.position] = True
            hitting[k, sim.position] = 0.0
            ticks_used = 0
            while ticks_used < self.horizon:
                obs = torch.tensor(sim._obs(), dtype=torch.float32, device=self.device).unsqueeze(0)
                mask = torch.tensor(sim.action_mask(), dtype=torch.bool, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = self.model.act(obs, mask)
                a = int(action.item())
                neighbors = dict(sim.graph.neighbors(sim.position))
                step_ticks = max(1, int(np.ceil(neighbors[a])))
                ticks_used += step_ticks
                sim.step(a)
                if not visited[a]:
                    visited[a] = True
                    hitting[k, a] = min(float(ticks_used), float(self.horizon))
        return hitting

    def _cvar(self, samples: np.ndarray) -> float:
        """Upper-tail CVaR_alpha: mean of the worst (largest-loss) alpha
        fraction of samples -- the risk-seeking value the intruder cares
        about, not the mean."""
        k = max(1, int(np.ceil(self.alpha * len(samples))))
        worst = np.sort(samples)[-k:]
        return float(np.mean(worst))

    def best_attack(self, env: PatrolEnv):
        """Best current target under the CVaR_alpha risk measure of
        weight(v) * hitting_time(v). Returns (node, value)."""
        hitting = self._sample_hitting_times(env)
        weights = env.weights.astype(np.float32)
        best_v, best_val = -1, -np.inf
        for v in range(env.n):
            val = self._cvar(weights[v] * hitting[:, v])
            if val > best_val:
                best_val, best_v = val, v
        return best_v, best_val

    def decide(self, env: PatrolEnv):
        """Optimal-stopping decision: attack now if that is at least as good
        as the (CVaR-weighted) value of waiting one more step and
        re-deciding. Returns ("attack", node, value) or ("wait", None, value).
        """
        attack_v, attack_val = self.best_attack(env)

        continuation_samples = []
        n_lookahead = max(4, self.rollouts // 4)
        for _ in range(n_lookahead):
            sim = env.clone()
            obs = torch.tensor(sim._obs(), dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = torch.tensor(sim.action_mask(), dtype=torch.bool, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = self.model.act(obs, mask)
            sim.step(int(action.item()))
            _, next_val = self.best_attack(sim)
            continuation_samples.append(next_val)
        continuation_val = self._cvar(np.array(continuation_samples, dtype=np.float32))

        if attack_val >= continuation_val:
            return "attack", attack_v, attack_val
        return "wait", None, continuation_val
