"""Training-curve plots: reward, worst-case weighted idleness, and policy
entropy (to visibly confirm randomization survives convergence rather than
collapsing to a deterministic, exploitable policy)."""
import matplotlib.pyplot as plt


def plot_training_curves(history, save_path: str = "training_curves.png") -> str:
    iterations = list(range(len(history)))
    rewards = [h["mean_reward"] for h in history]
    mean_worst = [h["mean_worst"] for h in history]
    max_worst = [h["max_worst"] for h in history]
    entropy = [h["entropy"] for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(iterations, rewards)
    axes[0].set_ylabel("Mean segment reward")
    axes[0].set_title("Patroller training progress")

    axes[1].plot(iterations, mean_worst, label="mean worst_instant")
    axes[1].plot(iterations, max_worst, label="max worst_instant")
    axes[1].set_ylabel("Weighted idleness")
    axes[1].legend()

    axes[2].plot(iterations, entropy)
    axes[2].set_ylabel("Policy entropy")
    axes[2].set_xlabel("PPO iteration")
    axes[2].axhline(0, color="gray", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
