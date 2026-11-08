# AI Patrolling

An RL-trained patrol agent that moves around a graph (think: a security guard
walking between rooms) trying to minimize damage from an intruder who can
attack any room and does damage proportional to **how long that room goes
unvisited**. The twist: the intruder isn't random — it watches the guard's
position and history in real time and picks the best moment to strike.

## The idea, in plain terms

- The graph has **nodes** (places worth protecting, each with a **weight** =
  how much it matters) and **edges** (paths between them, each with a
  **length** = how long it takes to walk).
- The **loss** if the intruder attacks node `v` and the guard doesn't show up
  for `t` time units is `weight(v) * t`. So leaving an important node
  unvisited for a long time is expensive.
- The **patroller** is trained with reinforcement learning (PPO) to decide,
  at every node, which neighbor to walk to next — trying to keep every node
  visited often enough, with priority on high-weight ones.
- The **intruder** is not a second trained AI — it's a planner
  (`agents/intruder.py`) that is handed the patroller's own strategy and the
  map, and calculates the statistically best moment to strike, the way a
  patient, well-informed attacker actually would. This is used to grade the
  patroller after training (and to sanity-check it isn't learning a route
  that's easy to predict and exploit).
- Because the intruder is assumed to know the patroller's strategy, a
  **completely predictable** (always-the-same) route can be dangerous. The
  training deliberately keeps some randomness in the guard's behavior and
  specifically penalizes the *worst* moments it's caught out, not just the
  average — see "Design notes" below for why.

## Project layout

```
graphs/graph.py         the map: nodes, weights, edges, lengths
                         (a few ready-made maps: cycle, grid, star, random)
env/patrol_env.py        the simulation the guard trains in
agents/policy.py          the guard's neural network (what to do next)
agents/ppo.py              the training algorithm
agents/intruder.py          the "smart attacker" used to grade the guard
training/train.py            run this to train a guard on a chosen map
training/evaluate.py          run this to grade a trained guard + make a replay
viz/plots.py, viz/replay.py    charts and an animated GIF of a patrol
tests/test_env_sanity.py         quick check that the simulation math is right
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

(Already set up in this repo's `.venv/` if you're reading this right after
the initial build.)

## Running it

**1. Train a guard on a map** (a few minutes on a laptop CPU):

```bash
.venv/bin/python training/train.py --graph cycle --iterations 400 --out checkpoint_cycle.pt
```

`--graph` can be `cycle`, `star`, `grid`, or `random`. This saves:
- `checkpoint_cycle.pt` — the trained guard
- `checkpoint_cycle.pt.curves.png` — training progress chart (reward, how
  bad the worst moment got, and how "random" the guard's behavior still is)

**2. Grade it against the real smart attacker, and make a replay GIF:**

```bash
.venv/bin/python training/evaluate.py --checkpoint checkpoint_cycle.pt --replay
```

This prints how much damage the attacker managed to inflict over a test run,
and (with `--replay`) saves `patrol_replay.gif` — an animation of the guard
walking the map, nodes glowing red the longer they've gone unvisited, with a
dashed ring marking a node under attack.

**3. Sanity-check the simulation itself:**

```bash
.venv/bin/python tests/test_env_sanity.py
```

## What's actually in this repo right now

Three maps have been trained and graded already:

| Map    | Trained guard's avg. damage/attack | Worst single attack | vs. random-walk baseline |
|--------|-------------------------------------|----------------------|---------------------------|
| cycle  | 19.5                                | 20                    | up to 216 (10x+ worse)    |
| star   | 9.3                                 | 11                    | up to 96 (8x+ worse)      |
| grid   | 12.4                                | 61                    | up to 126                 |

`cycle` and `star` converged cleanly to efficient, low-risk patrol routes.
`grid` (more complex — 4-way branches instead of 2 or 1) still beats random
movement clearly but is less consistent and would benefit from more training
time/tuning — see Known limitations below.

## Design notes (the "why" behind some non-obvious choices)

- **Why isn't the intruder just an unpredictable random attacker?** Because
  a real intruder watching a guard wouldn't attack randomly — it would wait
  for the worst moment. Training against a weak/random attacker teaches the
  guard nothing about actually being safe; it needs to be graded (and
  ideally also trained) against an attacker that's doing the smartest thing
  it can.
- **Why does the guard need to stay somewhat random, not always take the
  "best" route?** If the guard's route is 100% predictable and the intruder
  knows it, the intruder can time an attack with certainty. In practice, we
  found this matters less on simple maps (a plain loop) where there's no
  real decision to hide, and matters more on maps with real branching
  choices (multiple different paths from one spot) — see the training
  curves' "entropy" chart for how much randomness survived in each case.
- **Why cap how bad a single node's idleness can get, and reset
  occasionally during training?** Early experiments without these hit a
  nasty failure mode: an unlucky training run can get the guard stuck
  permanently ignoring one node, whose "badness" score then grows without
  limit, which makes training actively worse rather than better. Capping it
  and giving periodic fresh starts during training fixes this. (Doesn't
  apply to how the trained guard is graded — grading always runs one long,
  uninterrupted patrol.)

## Known limitations / good next steps

- `grid` training is noticeably less stable than `cycle`/`star` — occasional
  dips mid-training where it re-learns after a bad patch. More training
  iterations, a learning-rate schedule, or more segments per batch would
  likely smooth this out.
- The intruder planner (`agents/intruder.py`) is currently used only to
  *grade* the trained guard, not as part of training itself. A natural next
  step is folding it back into training (have the guard actually train
  against its attacks, not just a proxy reward) to close the loop fully.
- Each map needs its own separately-trained guard right now (no
  generalization to graphs the guard hasn't seen). Teaching one guard to
  handle any map would need a graph neural network instead of the current
  simple network — a bigger follow-up project.
