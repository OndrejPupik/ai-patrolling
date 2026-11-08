"""Discrete-time single-agent patrol environment, specialized to one fixed graph.

Decisions happen only at nodes (a semi-MDP): choosing to move along an edge of
length L advances the simulation by ceil(L) idleness ticks internally, and the
step reward is the sum of per-tick weighted idleness over that whole transit
(a dense proxy for an intruder able to always strike the currently-worst
node). See agents/intruder.py for the real, causal, risk-seeking attacker
used at evaluation time.
"""
import numpy as np

from graphs.graph import Graph


class PatrolEnv:
    def __init__(
        self,
        graph: Graph,
        idleness_norm: float = 20.0,
        reward_scale: float = 0.02,
        idleness_cap: float = 200.0,
    ):
        self.graph = graph
        self.n = graph.n
        self.weights = np.array(graph.weights, dtype=np.float64)
        self.idleness_norm = idleness_norm
        # Only the reward handed to the RL optimizer is scaled down (a
        # monotonic rescaling that doesn't change the optimal policy, but
        # keeps return magnitudes in a range where PPO's value-loss and
        # policy-loss gradients stay balanced). worst_instant (used for
        # CVaR monitoring and by the intruder planner) stays in true,
        # unscaled weight*idleness units.
        self.reward_scale = reward_scale
        # Hard cap on any single node's idleness. Without this, a policy
        # that gets unlucky early and settles into a cycle neglecting one
        # node can let that node's idleness (and the reward signal) blow up
        # without bound, poisoning the gradient and making the trap
        # self-reinforcing (observed empirically on the grid graph). The
        # cap keeps the penalty for severe neglect very large but finite.
        self.idleness_cap = idleness_cap
        self.position = 0
        self.idleness = np.zeros(self.n, dtype=np.float64)

    def reset(self, position: int = 0) -> np.ndarray:
        self.position = position
        self.idleness = np.zeros(self.n, dtype=np.float64)
        return self._obs()

    def clone(self) -> "PatrolEnv":
        """A deep-enough copy for simulated lookahead rollouts, without
        touching the real environment's state."""
        new_env = PatrolEnv(self.graph, self.idleness_norm, self.reward_scale, self.idleness_cap)
        new_env.position = self.position
        new_env.idleness = self.idleness.copy()
        return new_env

    def _obs(self) -> np.ndarray:
        onehot = np.zeros(self.n, dtype=np.float32)
        onehot[self.position] = 1.0
        norm_idle = np.clip(self.idleness / self.idleness_norm, 0.0, 5.0).astype(np.float32)
        return np.concatenate([onehot, norm_idle])

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n, dtype=bool)
        for neighbor, _ in self.graph.neighbors(self.position):
            mask[neighbor] = True
        return mask

    def step(self, action: int):
        neighbors = dict(self.graph.neighbors(self.position))
        if action not in neighbors:
            raise ValueError(f"Invalid move: node {self.position} has no edge to {action}")
        length = neighbors[action]
        ticks = max(1, int(np.ceil(length)))

        total_reward = 0.0
        worst_instant = 0.0
        for _ in range(ticks):
            self.idleness = np.minimum(self.idleness + 1.0, self.idleness_cap)
            step_weighted_idle = self.weights * self.idleness
            total_reward += -float(step_weighted_idle.sum())
            worst_instant = max(worst_instant, float(step_weighted_idle.max()))

        self.position = action
        self.idleness[action] = 0.0

        info = {"worst_instant": worst_instant, "ticks": ticks}
        return self._obs(), total_reward * self.reward_scale, False, info
