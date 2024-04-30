import numpy as np

import copy
import json

import brax
import numpy as np
from brax.envs.half_cheetah import _SYSTEM_CONFIG, Halfcheetah
from Box2D import Box2D
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
# from brax.envs.half_cheetah import Halfcheetah
from gym import Wrapper
from gym.utils import EzPickle
from gym.vector.utils import spaces
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from brax.envs.half_cheetah import Halfcheetah

DEFAULT_CONTEXT = {
    "joint_stiffness": 15000.0,
    "gravity": -9.8,
    "friction": 0.6,
    "angular_damping": -0.05,
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


class HalfCheetahCtx(Halfcheetah, TaskSettableEnv, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array"],}
    def __init__(self, continuous=True, body_mass= -10 ):


        super().__init__()




    def set_task(self, task: TaskType) -> None:
        self.world.__setattr__("gravity", (0, task))

    def get_task(self) -> TaskType:

        return self.world.gravity[1]



if __name__ == "__main__":
    env = HalfCheetahEnv()
    # print(env['torso_mass'])
    env.reset()
    # env.model.
    for _ in range(150):
        env.step(env.action_space.sample())
        env.render()


    bm = np.array(env.model.body_mass)
    bm[1] = 10
    env.model.body_mass = bm
    env.reset()
    print(env.model.body_mass)
    for _ in range(150):

        env.step(env.action_space.sample())
        env.render()

#
#
# class HalfCheetahCtx(CARLEnv):
#     def __init__(
#         self,
#         env: Halfcheetah = Halfcheetah(),
#         n_envs: int = 1,
#         contexts: Contexts = {},
#         hide_context: bool = False,
#         add_gaussian_noise_to_context: bool = False,
#         gaussian_noise_std_percentage: float = 0.01,
#         logger: Optional[TrialLogger] = None,
#         scale_context_features: str = "no",
#         default_context: Optional[Context] = DEFAULT_CONTEXT,
#         state_context_features: Optional[List[str]] = None,
#         context_mask: Optional[List[str]] = None,
#         dict_observation_space: bool = False,
#         context_selector: Optional[
#             Union[AbstractSelector, type[AbstractSelector]]
#         ] = None,
#         context_selector_kwargs: Optional[Dict] = None,
#     ):
#         if n_envs == 1:
#             env = GymWrapper(env)
#         else:
#             env = VectorGymWrapper(VectorWrapper(env, n_envs))
#
#         self.base_config = MessageToDict(
#             text_format.Parse(_SYSTEM_CONFIG, brax.Config())
#         )
#         if not contexts:
#             contexts = {0: DEFAULT_CONTEXT}
#         super().__init__(
#             env=env,
#             n_envs=n_envs,
#             contexts=contexts,
#             hide_context=hide_context,
#             add_gaussian_noise_to_context=add_gaussian_noise_to_context,
#             gaussian_noise_std_percentage=gaussian_noise_std_percentage,
#             logger=logger,
#             scale_context_features=scale_context_features,
#             default_context=default_context,
#             state_context_features=state_context_features,
#             dict_observation_space=dict_observation_space,
#             context_selector=context_selector,
#             context_selector_kwargs=context_selector_kwargs,
#             context_mask=context_mask,
#         )
#         self.whitelist_gaussian_noise = list(
#             DEFAULT_CONTEXT.keys()
#         )  # allow to augment all values
#
#
#
#     def __getattr__(self, name: str) -> Any:
#         if name in ["sys"]:
#             return getattr(self.env._environment, name)
#         else:
#             return getattr(self, name)