
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union

import importlib
import inspect
import json
import os
from types import ModuleType

import gym
import numpy as np
from gym import Wrapper, spaces



brax_spec = importlib.util.find_spec("brax")
if brax_spec is not None:
    import jax.numpy as jnp
    import jaxlib


class CtxBraxEnv(Wrapper):


    def __init__(
        self,
        env,
        context_keys = None
    ):
        super().__init__(env=env)
        self.context_space = spaces.Box(self.ctx_lb, self.ctx_ub)

        self.init_context(context_keys)


    def init_context(self, keys):
        if keys:
            self.context = {k:self.DEFAULT_CONTEXT[k] for k in keys}
        else:
            self.context = self.DEFAULT_CONTEXT
        return

    def __getattr__(self, name: str) -> Any:
        if name in ["sys"]:
            return getattr(self.env._env, name)
        else:
            return getattr(self, name)

    def get_task(self):
        return np.array(list(self.context.values()))


    def get_context(self):

        return self.context


    def set_task(self, task):
        for i, k in enumerate(self.context.keys()):
            self.context[k] = task[i]
        self._update_context()


        return
    # def