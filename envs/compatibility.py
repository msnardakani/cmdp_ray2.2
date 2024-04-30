import gymnasium as gym
import logging
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type

from ray.rllib import MultiAgentEnv
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import (
    ExperimentalAPI,
    override,
    PublicAPI,
    DeveloperAPI,
)
from ray.rllib.utils.typing import (
    AgentID,
    EnvCreator,
    EnvID,
    EnvType,
    MultiAgentDict,
    MultiEnvDict,
)
from ray.util import log_once
from ray.workflow.common import TaskType


def make_multi_agent_divide_and_conquer(
    env_name_or_creator: Union[str, EnvCreator],
) -> Type["MultiAgentEnv"]:
    """Convenience wrapper for any single-agent env to be converted into MA.

    Allows you to convert a simple (single-agent) `gym.Env` class
    into a `MultiAgentEnv` class. This function simply stacks n instances
    of the given ```gym.Env``` class into one unified ``MultiAgentEnv`` class
    and returns this class, thus pretending the agents act together in the
    same environment, whereas - under the hood - they live separately from
    each other in n parallel single-agent envs.

    Agent IDs in the resulting and are int numbers starting from 0
    (first agent).

    Args:
        env_name_or_creator: String specifier or env_maker function taking
            an EnvContext object as only arg and returning a gym.Env.

    Returns:
        New MultiAgentEnv class to be used as env.
        The constructor takes a config dict with `num_agents` key
        (default=1). The rest of the config dict will be passed on to the
        underlying single-agent env's constructor.

    Examples:
         >>> from ray.rllib.env.multi_agent_env import make_multi_agent
         >>> # By gym string:
         >>> ma_cartpole_cls = make_multi_agent("CartPole-v1") # doctest: +SKIP
         >>> # Create a 2 agent multi-agent cartpole.
         >>> ma_cartpole = ma_cartpole_cls({"num_agents": 2}) # doctest: +SKIP
         >>> obs = ma_cartpole.reset() # doctest: +SKIP
         >>> print(obs) # doctest: +SKIP
         {0: [...], 1: [...]}
         >>> # By env-maker callable:
         >>> from ray.rllib.examples.env.stateless_cartpole # doctest: +SKIP
         ...    import StatelessCartPole
         >>> ma_stateless_cartpole_cls = make_multi_agent( # doctest: +SKIP
         ...    lambda config: StatelessCartPole(config)) # doctest: +SKIP
         >>> # Create a 3 agent multi-agent stateless cartpole.
         >>> ma_stateless_cartpole = ma_stateless_cartpole_cls( # doctest: +SKIP
         ...    {"num_agents": 3}) # doctest: +SKIP
         >>> print(obs) # doctest: +SKIP
         {0: [...], 1: [...], 2: [...]}
    """

    class MultiEnvDnC(MultiAgentEnv):
        def __init__(self, config: EnvContext = None):
            MultiAgentEnv.__init__(self)
            # Note(jungong) : explicitly check for None here, because config
            # can have an empty dict but meaningful data fields (worker_index,
            # vector_index) etc.
            # TODO(jungong) : clean this up, so we are not mixing up dict fields
            # with data fields.
            if config is None:
                config = {}
            num = config.pop("num_agents", 1)
            if isinstance(env_name_or_creator, str):
                self.envs = [gym.make(env_name_or_creator) for _ in range(num)]
            else:
                self.envs = [env_name_or_creator(config) for _ in range(num)]
            self.terminateds = set()
            self.truncateds = set()
            self.observation_space = self.envs[0].observation_space
            self.action_space = self.envs[0].action_space
            self._agent_ids = set(range(num))

        @override(MultiAgentEnv)
        def observation_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
            if agent_ids is None:
                agent_ids = list(range(len(self.envs)))
            obs = {agent_id: self.observation_space.sample() for agent_id in agent_ids}

            return obs

        @override(MultiAgentEnv)
        def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
            if agent_ids is None:
                agent_ids = list(range(len(self.envs)))
            actions = {agent_id: self.action_space.sample() for agent_id in agent_ids}

            return actions

        @override(MultiAgentEnv)
        def action_space_contains(self, x: MultiAgentDict) -> bool:
            if not isinstance(x, dict):
                return False
            return all(self.action_space.contains(val) for val in x.values())

        @override(MultiAgentEnv)
        def observation_space_contains(self, x: MultiAgentDict) -> bool:
            if not isinstance(x, dict):
                return False
            return all(self.observation_space.contains(val) for val in x.values())

        @override(MultiAgentEnv)
        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            self.terminateds = set()
            self.truncateds = set()
            obs, infos = {}, {}
            for i, env in enumerate(self.envs):
                obs[i], infos[i] = env.reset(seed=seed, options=options)
            return obs, infos

        @override(MultiAgentEnv)
        def step(self, action_dict):
            obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}

            # the environment is expecting action for at least one agent
            if len(action_dict) == 0:
                raise ValueError(
                    "The environment is expecting action for at least one agent."
                )

            for i, action in action_dict.items():
                obs[i], rew[i], terminated[i], truncated[i], info[i] = self.envs[
                    i
                ].step(action)
                if terminated[i]:
                    self.terminateds.add(i)
                if truncated[i]:
                    self.truncateds.add(i)
            # TODO: Flaw in our MultiAgentEnv API wrt. new gymnasium: Need to return
            #  an additional episode_done bool that covers cases where all agents are
            #  either terminated or truncated, but not all are truncated and not all are
            #  terminated. We can then get rid of the aweful `__all__` special keys!
            terminated["__all__"] = len(self.terminateds) + len(self.truncateds) == len(
                self.envs
            )
            truncated["__all__"] = len(self.truncateds) == len(self.envs)
            return obs, rew, terminated, truncated, info

        @override(MultiAgentEnv)
        def render(self):
            return self.envs[0].render(self.render_mode)



        def set_task(self, task_dict) -> None:
            """
            """
            for i, task in task_dict.items():
               self.envs[i].set_task(task)

        def copy_task(self, task):

            for a in self.envs:
                a.set_task(task)


            # [a.set_task(task) for i, a in enumerate(self.agents)]

        def get_task(self) -> TaskType:
            """Gets the task that the agent is performing in the current environment

            Returns:
                task: task of the meta-learning environment
            """

            return {i: a.get_task() for i, a in enumerate(self.envs)}

        def report_task(self):

            return {i: a.report_task() for i, a in enumerate(self.envs)}

        def get_statistics(self):
            return {i: a.get_statistics() for i, a in enumerate(self.envs)}

        def get_context_buffer(self):
            return {i: a.get_context_buffer() for i, a in enumerate(self.envs)}

        def update_distribution(self):
            for env in self.envs:
                # if isinstance(env, BaseWrapper):
                #     continue
                mean_rew, mean_disc_rew, mean_length = env.get_statistics()
                vf_inputs, contexts, rewards = env.get_context_buffer()
                env.teacher.update_distribution(mean_disc_rew, contexts,
                                                                    rewards )
            # new_task = {i: env.get_task() for i,env in enumerate(task_settable_env.agents)}

        def get_episodes_statistics(self):
            return {i: a.get_episodes_statistics() for i, a in enumerate(self.envs)}

        def get_env_episodes_statistics(self, i):
            return self.envs[i].get_episodes_statistics()

        def get_env_context_buffer(self, i):
            return self.envs[i].get_context_buffer()

        def get_env_teacher(self, i):

            return self.envs[i].get_teacher()

        def update_env_teacher(self, weights, idx):
            self.envs[idx].update_teacher(weights = weights)



    return MultiEnvDnC

