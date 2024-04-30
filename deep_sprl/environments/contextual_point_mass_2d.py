import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from .contextual_point_mass import ContextualPointMass


class ContextualPointMass2D(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, context=np.array([0., 2.])):
        self.env = ContextualPointMass(np.concatenate((context, [0.])))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.context_space = Box(low = np.array([-4., .5], dtype=np.float64),
                                 high = np.array([4., 4.], dtype=np.float64), dtype=np.float64)

    def set_context(self, context):
        self.env.context = np.concatenate((context, [0.]))

    def get_context(self):
        return self.env.context.copy()

    context = property(get_context, set_context)

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed = seed, options = options)

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='rgb_array'):
        return self.env.render(mode)
