"""Actor-critic network for the patroller, with action masking so the policy
only ever places probability mass on valid moves (neighbors of the current
node). Keeping a non-trivial entropy bonus on this policy at training time is
what keeps the deployed policy genuinely randomized rather than collapsing to
a deterministic, and therefore fully exploitable, route.
"""
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor):
        h = self.body(obs)
        logits = self.policy_head(h)
        logits = logits.masked_fill(~action_mask, float("-inf"))
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor, action_mask: torch.Tensor):
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs: torch.Tensor, action_mask: torch.Tensor, action: torch.Tensor):
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value
