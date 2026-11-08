"""Phase 1: train the patroller with PPO against the dense worst-case-proxy
reward. Periodically probes the current policy with the real causal intruder
planner, purely for monitoring (not as a training signal yet) -- so we can
see whether the proxy reward and what a real adversary could extract start to
diverge as training progresses.

Usage:
    python training/train.py --graph cycle --iterations 150
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from agents.intruder import CVaRLookaheadIntruder
from agents.policy import ActorCritic
from agents.ppo import collect_rollouts, ppo_update
from env.patrol_env import PatrolEnv
from graphs.graph import make_cycle, make_grid, make_random, make_star
from viz.plots import plot_training_curves

GRAPHS = {
    "cycle": lambda: make_cycle(6, weights=[1, 1, 4, 1, 1, 1]),
    "grid": lambda: make_grid(3, 3),
    "star": lambda: make_star(6),
    "random": lambda: make_random(8, p=0.3, seed=1),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", choices=GRAPHS.keys(), default="cycle")
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--segment-len", type=int, default=64)
    parser.add_argument("--n-segments", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.02,
                         help="Entropy bonus weight. This is the knob controlling how much "
                              "randomization survives at convergence -- too low and the policy "
                              "drifts toward a predictable, exploitable route (or, on graphs with "
                              "real branching, can collapse into a self-reinforcing trap that "
                              "permanently neglects a node); too high and it never commits to an "
                              "efficient patrol. Watch the entropy subplot.")
    parser.add_argument("--cvar-alpha", type=float, default=0.2,
                         help="Fraction of worst segments per batch to upweight (risk-sensitive "
                              "training signal).")
    parser.add_argument("--cvar-weight", type=float, default=1.5,
                         help="Upweighting factor applied to the worst cvar-alpha segments.")
    parser.add_argument("--reset-every", type=int, default=20,
                         help="Reset the environment to a random start position every N "
                              "iterations. This is a training-stability measure (bounds how long "
                              "a bad early policy can keep neglecting a node before getting a "
                              "fresh start) -- it does not affect how the final policy is "
                              "evaluated, since evaluate.py never resets mid-run.")
    parser.add_argument("--idleness-cap", type=float, default=80.0,
                         help="Hard cap on any single node's idleness, bounding how extreme a "
                              "single neglected-node event can be before the env forces it back "
                              "down -- keeps rare outlier ticks from destabilizing an otherwise "
                              "converged policy.")
    parser.add_argument("--monitor-every", type=int, default=25)
    parser.add_argument("--out", default="checkpoint.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = GRAPHS[args.graph]()
    env = PatrolEnv(graph, idleness_cap=args.idleness_cap)
    env.reset(position=0)

    obs_dim = 2 * graph.n
    model = ActorCritic(obs_dim, graph.n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    intruder = CVaRLookaheadIntruder(model, device, horizon=25, rollouts=12, alpha=0.3)

    rng = np.random.default_rng(0)
    history = []
    for it in range(args.iterations):
        if args.reset_every and it > 0 and it % args.reset_every == 0:
            env.reset(position=int(rng.integers(graph.n)))

        batch, seg_rewards, seg_worst = collect_rollouts(env, model, device, args.segment_len, args.n_segments)
        stats = ppo_update(
            model, optimizer, batch, seg_rewards, seg_worst, device,
            entropy_coef=args.entropy_coef, cvar_alpha=args.cvar_alpha, cvar_weight=args.cvar_weight,
        )
        history.append(stats)
        print(
            f"iter {it:4d} | reward {stats['mean_reward']:9.2f} | "
            f"mean_worst {stats['mean_worst']:7.2f} | max_worst {stats['max_worst']:7.2f} | "
            f"entropy {stats['entropy']:.3f}"
        )

        if args.monitor_every and (it + 1) % args.monitor_every == 0:
            probe_env = env.clone()
            v, val = intruder.best_attack(probe_env)
            print(f"   [intruder probe] best current target: node {v}, CVaR value {val:.2f}")

    torch.save(
        {
            "model": model.state_dict(),
            "graph_key": args.graph,
            "obs_dim": obs_dim,
            "n_actions": graph.n,
            "idleness_cap": args.idleness_cap,
        },
        args.out,
    )
    with open(args.out + ".history.json", "w") as f:
        json.dump(history, f)
    curve_path = plot_training_curves(history, save_path=args.out + ".curves.png")
    print(f"Saved checkpoint to {args.out}, curves to {curve_path}")


if __name__ == "__main__":
    main()
