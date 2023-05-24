#!/usr/bin/env python

"""Helper code for Q-Learning"""


import argparse
from dataclasses import dataclass
import sys
from typing import Any, Iterable, Sequence, Tuple, Union

import gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from PIL import Image

@dataclass
class EpResult:
    """The result of running an episode, includes the total reward and (if available) the sequence of the rendering of each state (see Gym documentation)"""
    reward: float
    frames: Union[Sequence[Any], None] = None


class QTable(object):
    """Abstract class for Q-tables."""

    def update(self, obs, action, value: float) -> None:
        """Updates the content of the table for [obs, action] to the new value"""
        raise NotImplementedError

    def expected(self, obs, action = None) -> float:
        """Returns the expected value for the given action in the observation. If the action is None, it returns the maximum outcome across all the actions."""
        raise NotImplementedError

    def policy(self, obs) -> Any:
        """Returns the action that maximises the expected outcome from the current observation"""
        raise NotImplementedError


class DiscreteQTable(QTable):
    """QTable for discrete observation and action spaces. The table is implemented as a Numpy array."""
    def __init__(self, env: gym.Env, value: Union[float, Tuple[float, float]] = 0.0):
        """Creates a table for the given Gym environment; both observation and action spaces must be discrete. If `value` argument is provided, the table is initialised with its value (default to 0); if a tuple of two floats is provided, the table is initialised with random values within the given range."""
        assert isinstance(env.observation_space, gym.spaces.Discrete), f"only for discrete observation spaces, not {type(env.observation_space)}"
        obs_size = env.observation_space.n
        assert isinstance(env.observation_space, gym.spaces.Discrete), f"only for discrete action spaces, not {type(env.action_space)}"
        act_size = env.action_space.n
        if isinstance(value, (int, float)):
            self._table = np.full((obs_size, act_size), value)
        else:
            self._table = np.random.uniform(low=value[0], high=value[1], size=(obs_size, act_size))

    def __repr__(self):
        return self._table

    def update(self, obs: int, action: int, value: float) -> None:
        self._table[obs, action] = value

    def expected(self, obs: int, action: int = None) -> float:
        if action is None:
            return np.max(self._table[obs])
        else:
            return self._table[obs, action]

    def policy(self, obs):
        return np.argmax(self._table[obs])


def run_episode(env: gym.Env, qtable: QTable = None, frames: str = None) -> EpResult:
    """Runs a single episode over the given Gym environment, returning the accumulated reward and (if specified) the sequence of frames rendered using the specified frame (see the Gym `render` method)."""
    reward = 0
    cframes = [] if frames in env.metadata.get('render.modes', []) else None

    obs = env.reset()

    while True:
        if cframes is not None:
            cframes.append(env.render(frames))

        action = env.action_space.sample() if qtable is None else qtable.policy(obs)
        obs, r, done, info = env.step(action)
        reward += r
        if done:
            if cframes is not None:
                cframes.append(env.render(frames))
            break

    env.close()

    return EpResult(reward=reward, frames=cframes)


def plot_rewards(rewards: Iterable[float], ax: plt.Axes = None) -> plt.Axes:
    """
    Creates a Matplot graph showing the training history of the given list of rewards. If the axes is provided, it adds the graph to the given axes, otherwise it creates a new axes in the current axes.

    Returns the axes in which the plot has been created.
    """
    rews = np.array(rewards).T
    # calculate the running average <https://stackoverflow.com/a/30141358>
    smoothed_rews = pd.Series(rews).rolling(max(1, int(len(rews) * .01))).mean()

    ax = ax if ax is not None else plt.axes()

    ax.plot(smoothed_rews)
    ax.plot([np.mean(rews)] * len(rews), label='Mean', linestyle='--')
    ax.plot(rews, color='grey', alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')

    return ax


def frames_to_animation(frames: Iterable[np.ndarray], fig: plt.Figure = None) -> FuncAnimation:
    """Creates a new animation with the list of bitmats in `frames` within the given figure, returning
    the animation.
    If no figure is specified, a new one will be created.
    """
    fig = fig if fig is not None else plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')

    images = [Image.fromarray(f) for f in frames]

    return FuncAnimation(
        fig, (lambda i: [ax.imshow(i)]),
        frames=images, interval=50, blit=True, repeat=False)



def basic_training(env: gym.Env, qtable: QTable, episodes: int,
    alpha=0.1, gamma=0.6,
    epsilon=0.1, epsilon_decay=0, epsilon_min=0.01,
    verbose: bool=False) -> Iterable[float]:
    rewards = []
    epsilon_c = epsilon

    for episode in range(episodes):
        obs = env.reset()
        ep_reward = 0
        random_count = 0
        while True:
            if epsilon_c > np.random.random():
                action = env.action_space.sample()
                random_count += 1
            else:
                action = qtable.policy(obs)

            new_obs, r, done, info = env.step(action)
            ep_reward += r

            exp_reward = (1 - alpha) * qtable.expected(obs, action) + alpha * (r + gamma * qtable.expected(new_obs))
            qtable.update(obs, action, exp_reward)

            obs = new_obs
            if done:
                if verbose:
                    print(f'Episode {episode}/{episodes}: reward {ep_reward}, random actions {random_count}, epsilon {epsilon_c}')
                epsilon_c = epsilon_min + (epsilon - epsilon_min) * np.exp(-epsilon_decay * episode)
                rewards.append(ep_reward)
                break

    env.close()
    return rewards

def main(*cargs: str) -> int:
    """Run an episode of a Gym environment using a random agent"""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('env', nargs='?', default='Taxi-v3', help='Gym environment to use, see <https://www.gymlibrary.dev/>')
    args = parser.parse_args(cargs)

    env = gym.make(args.env)
    print(env.metadata)

    result = run_episode(env=env, frames='ansi')
    for f in result.frames or []:
        print(f)
    print(f'Total reward: {result.reward}')
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))