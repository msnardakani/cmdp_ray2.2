from ray.rllib import MultiAgentEnv

from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext

import gym
import logging
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type

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
import numpy as np

from deep_sprl.teachers.abstract_teacher import BaseWrapper


def Self_Paced_CL(
    train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
    """Function returning a possibly new task to set `task_settable_env` to.

    Args:
        train_results: The train results returned by Algorithm.train().
        task_settable_env: A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx: The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.

    Returns:
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """
    # Our env supports tasks 1 (default) to 5.
    # With each task, rewards get scaled up by a factor of 10, such that:
    # Level 1: Expect rewards between 0.0 and 1.0.
    # Level 2: Expect rewards between 1.0 and 10.0, etc..
    # We will thus raise the level/task each time we hit a new power of 10.0
    # new_task = {'mean':0.95*task_settable_env.get_task()['mean']+ 0.05*np.array([2,0.5]),
    #             'var': 0.95 * task_settable_env.get_task()['var'] + 0.05 * np.array([0.00016, 0.00016])}
    mean_rew, mean_disc_rew, mean_length = task_settable_env.get_statistics()
    vf_inputs, contexts, rewards = task_settable_env.get_context_buffer()
    task_settable_env.teacher.update_distribution(mean_disc_rew, contexts,
                                                        rewards )
    new_task = task_settable_env.get_task()
    # Clamp between valid values, just in case:
    # new_task = max(min(new_task, 5), 1)
    # print(
    #     f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
    #     f"\nR={train_results['episode_reward_mean']}"
    #     f"\nSetting env to task={new_task}"
    # )
    return new_task


def Self_Paced_MACL(
    train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
    """Function returning a possibly new task to set `task_settable_env` to.

    Args:
        train_results: The train results returned by Algorithm.train().
        task_settable_env: A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx: The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.

    Returns:
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """
    task_settable_env.iteration +=1
    if True:#task_settable_env.iteration %5 &task_settable_env.iteration > 9:
    # Our env supports tasks 1 (default) to 5.
    # With each task, rewards get scaled up by a factor of 10, such that:
    # Level 1: Expect rewards between 0.0 and 1.0.
    # Level 2: Expect rewards between 1.0 and 10.0, etc..
    # We will thus raise the level/task each time we hit a new power of 10.0
    # new_task = {'mean':0.95*task_settable_env.get_task()['mean']+ 0.05*np.array([2,0.5]),
    #             'var': 0.95 * task_settable_env.get_task()['var'] + 0.05 * np.array([0.00016, 0.00016])}
        for env in task_settable_env.agents:
            # if isinstance(env, BaseWrapper):
            #     continue
            if env.get_buffer_size()>16:
                # mean_rew, mean_disc_rew, mean_length = env.get_statistics()
                vf_inputs, contexts, rewards = env.get_context_buffer()
                env.teacher.update_distribution(0, np.array(contexts),
                                                                    np.array(rewards) )
    new_task = {i: env.get_task() for i,env in enumerate(task_settable_env.agents)}
        # Clamp between valid values, just in case:
        # new_task = max(min(new_task, 5), 1)
        # print(
        #     f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
        #     f"\nR={train_results['episode_reward_mean']}"
        #     f"\nSetting env to task={new_task}"
        # )
    return new_task
#

@PublicAPI
def make_multi_agent_divide_and_conquer(
    env_name_or_creator: Union[str, EnvCreator],
) -> Type["MultiAgentEnv"]:

    class MADnCEnv(MultiAgentEnv, TaskSettableEnv):
        iteration = 0
        TRAINING = True
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
            assert len(config["agent_config"]) == num

            self.env_mapping = config.get('env_mapping', None)
            self.non_training_envs = config.get('non_training_envs', 1)

            # self.TRAINING = config.get('training', True)

            if isinstance(env_name_or_creator, str):
                self.agents = [gym.make(env_name_or_creator) for _ in range(num)]
            elif self.env_mapping:
                self.agents = [env_name_or_creator(config["agent_config"][i]) for i in self.env_mapping ]
            else:
                self.agents = [env_name_or_creator(conf) for conf in config["agent_config"]]

            self.dones = set()
            self.dones_training = set()
            self.observation_space = self.agents[0].observation_space
            self.action_space = self.agents[0].action_space
            self._agent_ids = set(range(num))

        @override(MultiAgentEnv)
        def observation_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
            if agent_ids is None:
                agent_ids = list(range(len(self.agents)))
            obs = {agent_id: self.observation_space.sample() for agent_id in agent_ids}

            return obs

        @override(MultiAgentEnv)
        def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
            if agent_ids is None:
                agent_ids = list(range(len(self.agents)))
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
        def reset(self):
            self.dones = set()
            self.dones_training = set()
            return {i: a.reset() for i, a in enumerate(self.agents)}

        @override(MultiAgentEnv)
        def step(self, action_dict):
            obs, rew, done, info = {}, {}, {}, {}
            for i, action in action_dict.items():
                obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
                if done[i]:
                    self.dones.add(i)
                    if i >=self.non_training_envs:
                        self.dones_training.add(i)
            if self.TRAINING:
                if len(self.dones_training) == len(self.agents) - self.non_training_envs:
                    for i in range(self.non_training_envs):
                        self.dones.add(i)

            done["__all__"] = len(self.dones) == len(self.agents)

            return obs, rew, done, info

        def set_task(self, task_dict: TaskType) -> None:
            """
            """
            for i, task in task_dict.items():
               self.agents[i].set_task(task)

        def copy_task(self, task: TaskType):

            for a in self.agents:
                a.set_task(task)


            # [a.set_task(task) for i, a in enumerate(self.agents)]

        def get_task(self) -> TaskType:
            """Gets the task that the agent is performing in the current environment

            Returns:
                task: task of the meta-learning environment
            """

            return {i: a.get_task() for i, a in enumerate(self.agents)}

        def report_task(self):

            return {i: a.report_task() for i, a in enumerate(self.agents)}

        @override(MultiAgentEnv)
        def render(self, mode=None):
            return self.agents[0].render(mode)


        def get_statistics(self):
            return {i: a.get_statistics() for i, a in enumerate(self.agents)}

        def get_context_buffer(self):
            return {i: a.get_context_buffer() for i, a in enumerate(self.agents)}

        def update_distribution(self):
            for env in self.agents:
                # if isinstance(env, BaseWrapper):
                #     continue
                mean_rew, mean_disc_rew, mean_length = env.get_statistics()
                vf_inputs, contexts, rewards = env.get_context_buffer()
                env.teacher.update_distribution(mean_disc_rew, contexts,
                                                                    rewards )
            # new_task = {i: env.get_task() for i,env in enumerate(task_settable_env.agents)}

        def get_episodes_statistics(self):
            return {i: a.get_episodes_statistics() for i, a in enumerate(self.agents)}

        def get_env_episodes_statistics(self, i):
            return self.agents[i].get_episodes_statistics()

        def get_env_context_buffer(self, i):
            return self.agents[i].get_context_buffer()

        def get_env_buffer_size(self, i):
            return self.agents[i].get_buffer_size()

        def get_env_teacher(self, i):

            return self.agents[i].get_teacher()

        def update_env_teacher(self, weights, idx):
            self.agents[idx].update_teacher(weights = weights)
            return

        def reconfig_all(self, config):
            for a in self.agents:
                a.reconfig(config)
            return
        def get_ctx_hist(self):
            return {i: a.get_ctx_hist() for i, a in enumerate(self.agents)}
        def training_mode(self, mode):
            self.TRAINING = mode
    return MADnCEnv



@PublicAPI
def make_multi_agent_divide_and_conquer2(
    env_name_or_creator: Union[str, EnvCreator],
) -> Type["MultiAgentEnv"]:

    class MADnCEnv(MultiAgentEnv, TaskSettableEnv):
        TRAINING = True
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
            assert len(config["agent_config"]) == num
            if isinstance(env_name_or_creator, str):
                self.agents = [gym.make(env_name_or_creator) for _ in range(num)]
            else:
                self.agents = [env_name_or_creator(conf) for conf in config["agent_config"]]
            self.dones = set()
            self.observation_space = self.agents[0].observation_space
            self.action_space = self.agents[0].action_space
            self._agent_ids = set(range(num))

        @override(MultiAgentEnv)
        def observation_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
            if agent_ids is None:
                agent_ids = list(range(len(self.agents)))
            obs = {agent_id: self.observation_space.sample() for agent_id in agent_ids}

            return obs

        @override(MultiAgentEnv)
        def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
            if agent_ids is None:
                agent_ids = list(range(len(self.agents)))
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
        def reset(self):
            self.dones = set()
            return {i: a.reset() for i, a in enumerate(self.agents)}

        @override(MultiAgentEnv)
        def step(self, action_dict):
            obs, rew, done, info = {}, {}, {}, {}
            for i, action in action_dict.items():
                obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
                if done[i]:
                    self.dones.add(i)
            if self.TRAINING:
                if 0 not in self.dones and len(self.dones) == len(self.agents) - 1:
                    self.dones.add(0)

            done["__all__"] = len(self.dones) == len(self.agents)

            return obs, rew, done, info

        def set_task(self, task_dict: TaskType) -> None:
            """
            """
            for i, task in task_dict.items():
               self.agents[i].set_task(task)

        def copy_task(self, task: TaskType):

            for a in self.agents:
                a.set_task(task)


            # [a.set_task(task) for i, a in enumerate(self.agents)]

        def get_task(self) -> TaskType:
            """Gets the task that the agent is performing in the current environment

            Returns:
                task: task of the meta-learning environment
            """

            return {i: a.get_task() for i, a in enumerate(self.agents)}

        def report_task(self):

            return {i: a.report_task() for i, a in enumerate(self.agents)}

        @override(MultiAgentEnv)
        def render(self, mode=None):
            return self.agents[0].render(mode)


        def get_statistics(self):
            return {i: a.get_statistics() for i, a in enumerate(self.agents)}

        def get_context_buffer(self):
            return {i: a.get_context_buffer() for i, a in enumerate(self.agents)}

        def update_distribution(self):
            for env in self.agents:
                # if isinstance(env, BaseWrapper):
                #     continue
                mean_rew, mean_disc_rew, mean_length = env.get_statistics()
                vf_inputs, contexts, rewards = env.get_context_buffer()
                env.teacher.update_distribution(mean_disc_rew, contexts,
                                                                    rewards )
            # new_task = {i: env.get_task() for i,env in enumerate(task_settable_env.agents)}

        def get_episodes_statistics(self):
            return {i: a.get_episodes_statistics() for i, a in enumerate(self.agents)}

        def get_env_episodes_statistics(self, i):
            return self.agents[i].get_episodes_statistics()

        def get_env_context_buffer(self, i):
            return self.agents[i].get_context_buffer()

        def get_env_teacher(self, i):

            return self.agents[i].get_teacher()

        def update_env_teacher(self, weights, idx):
            self.agents[idx].update_teacher(weights = weights)

        def reconfig_all(self, config):
            for a in self.agents:
                a.reconfig(config)
            return

        def training_mode(self, mode):
            self.TRAINING = mode

    return MADnCEnv
#
#
#
#
# # from ray.rllib.env.multi_agent_env import MultiAgentEnv
#
#
# class TeacherStudentEnv(MultiAgentEnv):
#     def __init__(self, config_env={}, env="CartPole-v0"):
#         self.env = gym.make(env)
#         self.observation_space = self.env.observation_space
#         self.action_space = self.env.action_space
#
#     def reset(self):
#         obs = self.env.reset()
#         return {"teacher": obs, "student": obs}
#
#     def step(self, action_dict):
#         obs, rew, done, info = self.env.step(action_dict["teacher"])
#         student_reward = self._get_student_reward(action_dict["teacher"], action_dict["student"])
#         return {"teacher": obs, "student": obs}, \
#                {"teacher": rew, "student": student_reward}, \
#                {"teacher": done, "student": done, "__all__": done}, \
#                {"teacher": info, "student": info}
#
#     # would only work with discrete actions, would need extending for continuous action spaces
#     def _get_student_reward(self, teacher_action, student_action):
#         return 1 if student_action == teacher_action else -1
