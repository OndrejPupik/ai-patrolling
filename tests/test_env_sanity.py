"""Hand-checked sanity test for PatrolEnv idleness/reward bookkeeping.

Run with: python tests/test_env_sanity.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphs.graph import make_cycle
from env.patrol_env import PatrolEnv


def test_three_cycle():
    graph = make_cycle(3, weights=[1.0, 1.0, 1.0], length=1.0)
    env = PatrolEnv(graph, idleness_norm=10.0, reward_scale=1.0)
    env.reset(position=0)

    # Step 0 -> 1: every node's idleness becomes 1 for this one tick, then
    # node 1 resets to 0. reward = -(1+1+1) = -3, worst_instant = 1.
    obs, reward, done, info = env.step(1)
    assert abs(reward - (-3.0)) < 1e-9, reward
    assert abs(info["worst_instant"] - 1.0) < 1e-9, info
    assert list(env.idleness) == [1.0, 0.0, 1.0], env.idleness

    # Step 1 -> 2: idleness becomes [2, 1, 2] for this tick, then node 2
    # resets. reward = -(2+1+2) = -5, worst_instant = 2.
    obs, reward, done, info = env.step(2)
    assert abs(reward - (-5.0)) < 1e-9, reward
    assert abs(info["worst_instant"] - 2.0) < 1e-9, info
    assert list(env.idleness) == [2.0, 1.0, 0.0], env.idleness

    print("test_three_cycle passed")


def test_action_mask_and_clone():
    graph = make_cycle(4, weights=[1.0, 1.0, 1.0, 1.0], length=1.0)
    env = PatrolEnv(graph)
    env.reset(position=0)
    mask = env.action_mask()
    assert list(mask) == [False, True, False, True], mask

    env.step(1)
    clone = env.clone()
    clone.step(2)
    # Original env must be unaffected by the clone's step.
    assert env.position == 1, env.position
    assert clone.position == 2, clone.position
    print("test_action_mask_and_clone passed")


def test_multi_tick_edge():
    graph = make_cycle(3, weights=[1.0, 1.0, 1.0], length=3.0)
    env = PatrolEnv(graph, idleness_norm=10.0, reward_scale=1.0)
    env.reset(position=0)
    obs, reward, done, info = env.step(1)
    assert info["ticks"] == 3, info
    # idleness ramps 1,2,3 on nodes 0 and 2 across the 3 ticks; node 1 ramps
    # 1,2,3 then resets to 0. reward = -sum over ticks of weighted idleness.
    # tick1: [1,1,1]=3 tick2: [2,2,2]=6 tick3: [3,3,3]=9 => total -18
    assert abs(reward - (-18.0)) < 1e-9, reward
    assert abs(info["worst_instant"] - 3.0) < 1e-9, info
    print("test_multi_tick_edge passed")


if __name__ == "__main__":
    test_three_cycle()
    test_action_mask_and_clone()
    test_multi_tick_edge()
    print("All env sanity tests passed.")
