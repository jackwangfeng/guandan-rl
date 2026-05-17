"""Quick sanity: rule_agent vs random head-to-head (no learning)."""
import random
import numpy as np

from env import GuandanEnv
from rule_agent import rule_choose


def play(seed, who_is_rule_team=0):
    env = GuandanEnv(seed=seed)
    obs = env.obs()
    safety = 0
    while not env.done:
        safety += 1
        if safety > 10_000:
            break
        legal = env.legal()
        if not legal:
            break
        if env.cur % 2 == who_is_rule_team:
            m = rule_choose(env.hands[env.cur], env.last_play, env.last_player,
                            obs['hand_sizes'], env.cur)
            if m is None:
                m = random.choice(legal)
        else:
            m = random.choice(legal)
        obs, _, _, _ = env.step(m)
    return env.winner_team


def main(n=500):
    rule_wins = 0
    for i in range(n):
        w = play(seed=i, who_is_rule_team=0)
        if w == 0:
            rule_wins += 1
    print(f"rule(team0) vs random(team1): rule wins {rule_wins}/{n} = {rule_wins/n:.3f}")
    # Symmetry check
    rule_wins2 = 0
    for i in range(n):
        w = play(seed=i + 10**6, who_is_rule_team=1)
        if w == 1:
            rule_wins2 += 1
    print(f"rule(team1) vs random(team0): rule wins {rule_wins2}/{n} = {rule_wins2/n:.3f}")


if __name__ == '__main__':
    main()
