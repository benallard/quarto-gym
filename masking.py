"""Masking of the invalid actions

Requires PR #25 of sb3_contrib
"""
import sys
sys.path.insert(0, '../stable-baselines3-contrib')

import gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskedActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskedPPO

from gym_quarto import RandomPlayer, OnePlayerWrapper

def mask_fn(env: gym.Env) -> np.ndarray:
    positions = set(a[0] for a in env.legal_actions)
    pieces = set(a[1] for a in env.legal_actions)
    action_mask = [[], []]
    for i in range(16):
        action_mask[0].append(i in positions)
        action_mask[1].append(i in pieces)
    return action_mask

env = gym.make('quarto-v1')
player = RandomPlayer(env)
env = OnePlayerWrapper(env, player)
env = ActionMasker(env, mask_fn)

model = MaskedPPO(MaskedActorCriticPolicy, env, verbose=1)
model.learn(1e6)


