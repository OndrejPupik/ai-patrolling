"""Phase 2: benchmark a trained patroller against the real causal, risk-
seeking intruder (agents/intruder.py) -- not the training-time proxy reward.

The intruder repeatedly runs its optimal-stopping decision process: whenever
it is not currently holding an attack, it decides attack-now vs. wait; once
it commits to a node, loss accrues until the patroller actually visits it,
then the intruder resumes deciding for the remainder of the run (the
renewal/continuing-process view we settled on conceptually).

Usage:
    python training/evaluate.py --checkpoint checkpoint.pt --replay
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from agents.intruder import CVaRLookaheadIntruder
from agents.policy import ActorCritic
from env.patrol_env import PatrolEnv
from graphs.graph import make_cycle, make_grid, make_random, make_star
from viz.replay import animate_patrol

GRAPHS = {
    "cycle": lambda: make_cycle(6, weights=[1, 1, 4, 1, 1, 1]),
    "grid": lambda: make_grid(3, 3),
    "star": lambda: make_star(6),
    "random": lambda: make_random(8, p=0.3, seed=1),
}


def load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ActorCritic(ckpt["obs_dim"], ckpt["n_actions"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["graph_key"], ckpt.get("idleness_cap", 200.0)


def cvar(samples, alpha: float = 0.2) -> float:
    if not samples:
        return 0.0
    k = max(1, int(np.ceil(alpha * len(samples))))
    worst = np.sort(samples)[-k:]
    return float(np.mean(worst))


def run_evaluation(env: PatrolEnv, model, device, intruder: CVaRLookaheadIntruder, total_ticks: int = 300):
    attack_losses, frames, attacked_at = [], [], {}
    ticks_elapsed, decisions = 0, 0
    current_attack = None  # [node, start_tick]

    while ticks_elapsed < total_ticks:
        if current_attack is None:
            decision, v, _ = intruder.decide(env)
            if decision == "attack":
                current_attack = [v, ticks_elapsed]
                attacked_at[decisions] = v

        obs = torch.tensor(env._obs(), dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.tensor(env.action_mask(), dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = model.act(obs, mask)
        a = int(action.item())
        neighbors = dict(env.graph.neighbors(env.position))
        step_ticks = max(1, int(np.ceil(neighbors[a])))

        env.step(a)
        ticks_elapsed += step_ticks
        frames.append((env.position, env.idleness.copy()))
        decisions += 1

        if current_attack is not None and a == current_attack[0]:
            duration = ticks_elapsed - current_attack[1]
            loss = env.graph.weights[current_attack[0]] * duration
            attack_losses.append(loss)
            current_attack = None

    return attack_losses, attacked_at, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--total-ticks", type=int, default=300)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--replay-out", default="patrol_replay.gif")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, graph_key, idleness_cap = load_model(args.checkpoint, device)
    graph = GRAPHS[graph_key]()
    env = PatrolEnv(graph, idleness_cap=idleness_cap)
    env.reset(position=0)

    intruder = CVaRLookaheadIntruder(model, device, horizon=25, rollouts=16, alpha=0.3)
    losses, attacked_at, frames = run_evaluation(env, model, device, intruder, args.total_ticks)

    print(f"Attacks landed: {len(losses)}")
    if losses:
        print(f"  mean loss per attack : {np.mean(losses):.2f}")
        print(f"  worst single attack  : {np.max(losses):.2f}")
        print(f"  CVaR_0.2 of attacks   : {cvar(losses, 0.2):.2f}")
    else:
        print("  (no attacks landed within the evaluation window)")

    if args.replay:
        path = animate_patrol(graph, frames, save_path=args.replay_out, attacked_nodes=attacked_at)
        print(f"Replay saved to {path}")


if __name__ == "__main__":
    main()
