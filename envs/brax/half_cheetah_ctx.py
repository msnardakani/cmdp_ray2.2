import brax
import gym
import numpy as np
from brax.envs.half_cheetah import Halfcheetah,  _SYSTEM_CONFIG_SPRING as _SYSTEM_CONFIG
from brax.envs.wrappers import GymWrapper
from google.protobuf import json_format, text_format
from google.protobuf.json_format import MessageToDict
from numpyencoder import NumpyEncoder
from typing import Any, Dict, List, Optional, Union

import copy
import json

from .contextual_brax_env import CtxBraxEnv

DEFAULT_CONTEXT = {
    "joint_stiffness": 25000.0,
    "gravity": -9.8,
    "friction": 0.77459666924,
    "angular_damping": -0.009999999776482582,
    "joint_angular_damping": 20,
    "torso_mass": 9.457333,
}

CONTEXT_BOUNDS = {
    "joint_stiffness": (1, np.inf, float),
    "gravity": (-np.inf, -0.1, float),
    "friction": (-np.inf, np.inf, float),
    "angular_damping": (-np.inf, np.inf, float),
    "joint_angular_damping": (0, np.inf, float),
    "torso_mass": (0.1, np.inf, float),
}

class HalfcheetahCTX(CtxBraxEnv):
    DEFAULT_CONTEXT = DEFAULT_CONTEXT
    CONTEXT_BOUNDS = CONTEXT_BOUNDS
    def __init__(
            self,
            context_keys=DEFAULT_CONTEXT.keys()):
        env = GymWrapper(Halfcheetah())
        self.ctx_lb = np.array([self.CONTEXT_BOUNDS[k][0] for k in context_keys])
        self.ctx_ub = np.array([self.CONTEXT_BOUNDS[k][1] for k in context_keys])
        super().__init__(
            env=env,
            context_keys=context_keys
        )
        self.base_config = MessageToDict(
            text_format.Parse(_SYSTEM_CONFIG, brax.Config())
        )
        self._context = self.DEFAULT_CONTEXT.copy()

    def _update_context(self) -> None:
        config = copy.deepcopy(self.base_config)
        self._context.update(self.context)

        config["gravity"] = {"z": self._context["gravity"]}
        config["friction"] = self._context["friction"]
        config["angularDamping"] = self._context["angular_damping"]
        for j in range(len(config["joints"])):
            config["joints"][j]["angularDamping"] = self._context[
                "joint_angular_damping"
            ]
            config["joints"][j]["stiffness"] = self._context["joint_stiffness"]
        config["bodies"][0]["mass"] = self._context["torso_mass"]
        # This converts the dict to a JSON String, then parses it into an empty brax config
        self.env.sys = brax.System(
            json_format.Parse(json.dumps(config, cls=NumpyEncoder), brax.Config())
        )
