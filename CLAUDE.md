# CLAUDE.md

Context for working on this repo. See README.md for the user-facing
explanation of what this project is and how to run it.

## What this is

RL-trained graph patrolling agent (PPO) graded/benchmarked against a
risk-seeking optimal-stopping intruder planner. Full design rationale (why
a deterministic policy can be exploitable, why the intruder is modeled as a
causal real-time planner rather than an omniscient one, why training needs
to be risk-sensitive/CVaR rather than average-reward) was worked out in
conversation before building and is summarized in README.md's "Design
notes" — read that first if a change touches training objective, reward
shape, or the intruder.

Plan file (original build plan + rationale) at
`~/.claude/plans/scalable-whistling-pebble.md`.

## Architecture

- `graphs/graph.py` — plain `Graph` dataclass + builders (`make_cycle`,
  `make_grid`, `make_star`, `make_random`). Each `train.py --graph` run is
  specialized to one fixed graph (no GNN / cross-graph generalization yet).
- `env/patrol_env.py` — `PatrolEnv`, a semi-MDP: one `step(action)` = move to
  a neighbor, taking `ceil(edge_length)` internal idleness ticks, returning
  the summed per-tick weighted-idleness reward for the whole transit.
  Decisions only happen at nodes.
- `agents/policy.py` — small MLP actor-critic with **action masking**
  (invalid moves get `-inf` logits before the softmax) so the policy only
  ever proposes real edges.
- `agents/ppo.py` — from-scratch clipped PPO. Treats the env as a
  *continuing* task across rollout segments within one `train.py` run (GAE
  bootstraps across segment boundaries rather than resetting) — this is
  deliberate, so training doesn't create an artificial "safe window" near a
  fake episode end. Also does CVaR-style upweighting of the worst-`alpha%`
  segments per batch (the risk-sensitive training signal).
- `agents/intruder.py` — `CVaRLookaheadIntruder`. A causal, real-time
  planner (not a trained network): at each tick it either attacks the
  currently-best node or waits, using bounded Monte-Carlo rollouts of the
  *frozen* patroller policy to estimate CVaR of `weight(v) * hitting_time(v)`
  per candidate, plus a one-step lookahead continuation-value comparison.
  Used in `evaluate.py` to grade a trained policy; not yet wired into
  `train.py`'s reward (see "Not yet done" below).
- `training/train.py` — Phase 1 trainer against the dense proxy reward
  (`-sum(weight * idleness)` per tick). Periodically probes the current
  policy with the intruder planner for monitoring only.
- `training/evaluate.py` — Phase 2: runs the trained policy against the real
  intruder planner over one continuous (no-reset) window, reports attack
  loss stats, optionally renders a replay GIF.
- `viz/plots.py`, `viz/replay.py` — matplotlib training curves and an
  animated patrol replay (networkx layout).

## Things that will bite you if changed carelessly

- **Reward scaling** (`PatrolEnv.reward_scale`, default 0.02): raw weighted-
  idleness reward is large (tens to hundreds per tick); without scaling it
  down, the value-loss term dominates PPO's gradient and the policy doesn't
  learn at all (confirmed empirically — see git history / plan file). Only
  the *training* reward is scaled; `info["worst_instant"]` stays in true
  units for CVaR/monitoring/intruder-planner use. If you change the reward
  function, re-check this balance (watch whether reward visibly improves
  over a measured random-walk baseline within the first ~100 iterations).
- **Idleness cap** (`PatrolEnv.idleness_cap`, default from
  `train.py --idleness-cap`, currently 80 for the saved grid checkpoint):
  without a cap, a policy that gets unlucky early and settles into a cycle
  permanently neglecting one node lets that node's idleness (and the
  resulting reward magnitude) blow up without bound — observed empirically
  on the grid graph, with per-iteration reward reaching -1.6M before the cap
  was added. The cap is saved into the checkpoint (`idleness_cap` key) and
  `evaluate.py` reconstructs the env with the same cap — don't drop that
  plumbing.
- **Periodic resets during training** (`train.py --reset-every`, default
  20): a pure continuing task can get trapped in the same way even with the
  cap in place (recovers more slowly without resets). Resets are
  training-only — `evaluate.py` never resets mid-run, since the point of
  evaluation is to grade one continuous, uninterrupted patrol the way a real
  deployment would run.
- **Advantage clipping** in `ppo.py` (`np.clip(advantages, -5.0, 5.0)`): a
  safety net against a single outlier tick (e.g. briefly hitting the
  idleness cap) dominating an otherwise-converged policy's gradient update.
  Don't remove without re-testing the grid graph for the instability it was
  added to fix (see below).
- **Entropy coefficient**: this is the knob controlling how much
  randomization survives training. Too low → policy can collapse to
  near-zero entropy. Empirically this was *fine* on `cycle` and `star`
  (deterministic patrol there isn't actually exploitable, since the
  intruder already observes state in real time rather than needing to infer
  a hidden strategy — see README design notes) but caused outright training
  instability on `grid` combined with low batch size. Current default
  (`0.02`) + `n-segments=16` + `lr=1e-3` is the result of empirical tuning
  across all three graphs, not an arbitrary choice — change with the same
  three-graph check.

## Known unresolved issue

`grid` training (4-way branching, the hardest of the three shipped graphs)
is noticeably less stable than `cycle`/`star`: reward dips and recovers
mid-training rather than monotonically improving. The final checkpoint still
clearly beats a random-walk baseline, but isn't as cleanly converged. Likely
next steps if revisited: learning-rate annealing, more segments/iterations,
or value-loss clipping (PPO2-style) in addition to the existing advantage
clipping.

## Not yet done (deliberately deferred, see plan file)

- The intruder planner is only used for post-hoc grading, not folded back
  into `train.py`'s reward signal (the "alternating best-response" loop from
  the design discussion is implemented in spirit via the dense proxy reward
  + periodic monitoring, but not as literal alternating training rounds).
- No graph-generalizing (GNN) policy — one checkpoint per graph.
- Single patroller only, no multi-agent cooperation.

## Running things

```bash
.venv/bin/python tests/test_env_sanity.py                              # unit sanity
.venv/bin/python training/train.py --graph <cycle|star|grid|random>     # train
.venv/bin/python training/evaluate.py --checkpoint <path> --replay       # grade + GIF
```

Each full training run (600 iterations, default batch size) takes roughly
2-4 minutes on CPU for these small (6-9 node) graphs.
