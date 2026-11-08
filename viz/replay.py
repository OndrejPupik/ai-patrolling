"""Animated replay of a patrol episode: node color intensity = current
weighted idleness, blue ring = patroller's current position, dashed black
ring = a node the intruder is actively attacking at that frame."""
import networkx as nx
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch

from graphs.graph import Graph
from env.patrol_env import PatrolEnv


def build_networkx(graph: Graph) -> nx.Graph:
    G = nx.Graph()
    for u in range(graph.n):
        G.add_node(u, weight=graph.weights[u])
    for u, nbrs in graph.adjacency.items():
        for v, length in nbrs:
            G.add_edge(u, v, length=length)
    return G


def record_episode(env: PatrolEnv, model, device, n_decisions: int = 60, seed: int = 0):
    """Simple policy-only rollout (no intruder) -- a quick visual sanity
    check that the trained patroller behaves sensibly and non-repetitively."""
    rng = np.random.default_rng(seed)
    env.reset(position=int(rng.integers(env.n)))
    frames = []
    for _ in range(n_decisions):
        obs = torch.tensor(env._obs(), dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.tensor(env.action_mask(), dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = model.act(obs, mask)
        env.step(int(action.item()))
        frames.append((env.position, env.idleness.copy()))
    return frames


def animate_patrol(graph: Graph, frames, save_path: str = "patrol_replay.gif", attacked_nodes=None):
    G = build_networkx(graph)
    pos = nx.spring_layout(G, seed=0)
    weights = np.array(graph.weights)
    attacked_nodes = attacked_nodes or {}

    fig, ax = plt.subplots(figsize=(6, 6))

    def draw(frame_idx):
        ax.clear()
        position, idleness = frames[frame_idx]
        node_colors = idleness * weights
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors, cmap="Reds",
            vmin=0, vmax=max(1.0, float(node_colors.max())), node_size=500,
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        ax.scatter(*pos[position], s=900, facecolors="none", edgecolors="tab:blue", linewidths=3)
        if frame_idx in attacked_nodes:
            v = attacked_nodes[frame_idx]
            ax.scatter(*pos[v], s=1100, facecolors="none", edgecolors="black",
                       linewidths=3, linestyle="--")
        ax.set_title(f"step {frame_idx} -- blue ring = patroller, dashed = under attack")
        ax.axis("off")

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=400)
    anim.save(save_path, writer="pillow")
    plt.close(fig)
    return save_path
