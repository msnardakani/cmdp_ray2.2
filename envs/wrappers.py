from typing import Optional

import gym.spaces
import numpy as np
# from gym import spaces
from gymnasium import ObservationWrapper
from gymnasium.spaces import Dict
from gymnasium.wrappers import TimeLimit
# from gym.wrappers import FlattenObservation
# return spaces.flatten(self.env.observation_space, observation)
class CtxAugmentedObs(ObservationWrapper):
    r"""Augment the observation with current time step in the trajectory.

    .. note::
        Currently it only works with one-dimensional observation space. It doesn't
        support pixel observation space yet.

    """
    def __init__(self, env):
        super().__init__(env)
        # assert isinstance(env.observation_space, Box)
        # assert env.observation_space.dtype == np.float32
        # low = np.append(self.observation_space.low, 0.0)
        # high = np.append(self.observation_space.high, np.inf)
        # self.observation_space = Box(low, high, dtype=np.float32)
        self.observation_space = Dict({'obs': env.observation_space,
                                        'ctx': env.context_space})
    def observation(self, observation):
        return {'obs': np.array(observation),
                'ctx': self.env.get_task()}



    def step(self, action):
        # self.t += 1
        return super().step(action)

    def reset(self, *, seed=None, options=None):
        # self.t = 0
        return super().reset(seed=seed, options=options)


class BrxEnvObs(ObservationWrapper):
    r"""Augment the observation with current time step in the trajectory.

    .. note::
        Currently it only works with one-dimensional observation space. It doesn't
        support pixel observation space yet.

    """
    def __init__(self, env):
        super().__init__(env)
        # assert isinstance(env.observation_space, Box)
        # assert env.observation_space.dtype == np.float32
        # low = np.append(self.observation_space.low, 0.0)
        # high = np.append(self.observation_space.high, np.inf)
        self.observation_space = env.observation_space
        # self.observation_space = Dict({'observation': env.observation_space,
        #                                 'context': env.context_space})
    def observation(self, observation):
        return  np.array(observation)



    def step(self, action):
        # self.t += 1
        return super().step(action)

    def reset(self,*, seed=None, options=None):
        # self.t = 0
        return super().reset( seed=seed, options=options)



class TimeLimitRewardWrapper(TimeLimit):
    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: int,
            key):
        super().__init__(env, max_episode_steps= max_episode_steps)
        self.key = key

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action=action)

        if truncated or terminated:
            reward += -info[self.key]
        return observation, reward, terminated, truncated, info