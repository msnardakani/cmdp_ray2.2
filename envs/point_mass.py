import numpy as np
from gym import Env
from gym.spaces import Box
from deep_sprl.environments.contextual_point_mass import ContextualPointMass
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env import EnvContext


class TaskSettablePointMass(ContextualPointMass, TaskSettableEnv):
    def __init__(self, config:EnvContext):

        self.friction = config.get('friction', 0)
        self.gate = config.get('gate', [2, 0.5])
        super().__init__(context= np.array(self.gate + [self.friction, ]))
        self.context_space = dict(gate = Box(low=np.array([-3.5, .5]),
                                             high=np.array([3.5, 6])),
                                  friction= Box(low= np.array([0,]), high=np.array([1,])))

    def set_task(self, task):
        if 'gate' in task:
            self.gate = task['gate']
            self.context[0:2] = np.array(self.gate)

        if 'friction' in task:
            self.friction = task['friction']
            self.context[2] = self.friction

        return

    def get_task(self):
        return dict(friction = self.friction, gate = self.gate)


    def sample_task(self, n_task):
        ctxs = list(np.array([8, 5.5, 1]) * np.random.random(n_task*3).reshape(n_task, -1) - np.array([4, -.5, 0]))
        return [{'gate': [ctx[0], ctx[1]], 'friction':[ctx[2], ]} for ctx in ctxs]