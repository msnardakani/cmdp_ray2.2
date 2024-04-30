import itertools

import gymnasium as gym
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type

from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
import numpy as np
from ray.rllib.utils.annotations import (
    ExperimentalAPI,
    override,
    PublicAPI,
    DeveloperAPI,
)


from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext

from ray.rllib.utils.typing import (
    AgentID,
    EnvCreator,
    EnvID,
    EnvType,
    MultiAgentDict,
    MultiEnvDict,
)

from ray.rllib.utils.typing import EnvCreator, MultiAgentDict
from deep_sprl.teachers.dummy_teachers import GMMSampler
from flatten_dict import flatten, unflatten

import logging

from utils.self_paced_callback import Buffer

logger = logging.getLogger(__name__)

ctx_visibility = {0: 'ctx_hid',
                  1: 'ctx_vis',
                  2: 'ctx_vis'}

exp_group = {0: 'distral',
                  1: 'task_aug',
                  2: 'distill_aug'}


def find_category_idx(contexts_list, target_ctx):
    return next((idx for idx, val in enumerate(contexts_list) if np.array_equal(target_ctx, val)), 0)

class GMMCtxEnvWrapper(gym.ObservationWrapper, TaskSettableEnv):

    def identity_obs(self, obs):
        return obs

    def concat_obs(self, obs):
        # print(obs, self.cur_context)
        return np.concatenate((obs, self.cur_context))

    def dict_obs(self, obs):
        return ( obs, self.cur_context)

    def __init__(self,
                 env,
                 ctx_lb=None,
                 ctx_ub=None,
                 target_mean=None,
                 target_var=None,
                 target_priors=None,
                 ctx_mode=1, **kwargs):
        assert (target_mean is not None) or ((ctx_lb is not None) and (ctx_ub is not None))
        super().__init__(env)
        if ctx_lb is not None:
            self.ctx_lb = ctx_lb
        else:

            self.ctx_lb = np.ones_like(target_mean) * (-np.infty)

        if ctx_ub is not None:
            self.ctx_ub = ctx_ub
        else:
            self.ctx_ub = np.ones_like(target_mean) * (np.infty)

        self.observation = self.identity_obs
        if target_mean is None:
            target_mean = (ctx_lb + ctx_ub) / 2

        if target_var is None:

            target_var = np.ones_like(target_mean) / 100
            target_priors = np.array([1, ])

        elif target_priors is None:
            target_priors = np.array([1, ] * (target_mean.size // ctx_lb.size))


        if ctx_mode==1 :
            # print('context visible: task augmented')
            low_ext = np.concatenate((self.env.observation_space.low, ctx_lb))
            high_ext = np.concatenate((self.env.observation_space.high, ctx_ub))
            self._observation_space = gym.spaces.Box(low=low_ext, high=high_ext,dtype=np.float64)
            self.observation = self.concat_obs

        if ctx_mode == 2:
            # print('context visible: distillation augmented')
            self.observation = self.dict_obs
            self._observation_space = gym.spaces.Tuple((self.env.observation_space,
                                                       self.context_space))

        self.context_space = gym.spaces.Box(low=ctx_lb, high=ctx_ub, dtype=np.float64)

        self._ctx_sampler = GMMSampler(target_mean.copy(),
                                       target_var.copy(),
                                       target_priors.copy(),
                                       (self.ctx_lb.copy(),
                                        self.ctx_ub.copy()))
        self.cur_context = self._ctx_sampler.sample()
        self.env.set_task(self.cur_context.copy())
        # obs, info = self.env.reset(seed=seed, options=options)
        self.cur_context = self.env.get_task()

    def reset(self, *, seed= None, options = None):
        self.cur_context = self._ctx_sampler.sample()
        self.env.set_task(self.cur_context.copy())
        obs, info = self.env.reset(seed=seed, options=options)
        self.cur_context = self.env.get_task()
        return self.observation(obs=obs), info

    def sample_tasks(self, n_tasks):
        """Implement this to sample n random tasks."""
        return [self._ctx_sampler.sample() for _ in range(n_tasks)]

    def set_task(self, task: TaskType) -> None:
        self._ctx_sampler.reconfig(task)
        return

    def get_task(self):
        return self._ctx_sampler.export_config()

    def report_task(self):
        if hasattr(self.env, 'report_task'):
            self.env.set_task(self._ctx_sampler.mean())
            return self.env.report_task()

        return {'mean': self._ctx_sampler.mean(), 'var': np.diag(self._ctx_sampler.covariance_matrix())}

    def reconfig(self, config):
        self._ctx_sampler.reconfig(config)
        return

    def set_distribution(self, means, sigma2, priors):
        self._ctx_sampler.set_means(means)
        self._ctx_sampler.set_vars(sigma2)
        self._ctx_sampler.set_w(priors)
        return

    def get_context(self):
        return self.cur_context


class CtxDictWrapper(gym.ObservationWrapper, TaskSettableEnv):
    def identity_obs(self, obs):
        return obs

    def concat_obs(self, obs):
        return np.concatenate((obs, self.cur_context))

    def dict_obs(self,obs):
        return (obs,  self.cur_context)

    def __init__(self, env, key=None,
                 ctx_visible = 1, ):
        super().__init__(env)
        # self.current_ctx =
        self.flat_ctx = flatten(self.env.get_task())
        ctxkeys = list(self.flat_ctx.keys())
        if key is None:
            key = ctxkeys[0][0]

        self.selected_keys = [k for k in ctxkeys if k[0] in key]

        assert len(self.selected_keys) > 0
        self.ctx_map = dict()
        bias = 0

        for k in self.selected_keys:
            ctx_length = len(self.flat_ctx[k])
            self.ctx_map[k] = (bias, bias + ctx_length)
            bias += ctx_length

        self.current_flat_ctx = {k:self.flat_ctx[k] for k in self.selected_keys}
        self.cur_context = np.array(list(itertools.chain.from_iterable(self.current_flat_ctx.values())))

        if hasattr(env, 'context_space'):
            ctx_space = flatten(env.context_space)
            ctx_lb=np.zeros(0)
            ctx_ub=np.zeros(0)
            for k in self.selected_keys:
                ctx_lb = np.concatenate(( ctx_lb, ctx_space[k].low))

                ctx_ub = np.concatenate((ctx_ub, ctx_space[k].high))
        else:
            ctx_lb = np.ones(self.cur_context.size) * (-np.infty)
            ctx_ub = np.ones(self.cur_context.size) * (np.infty)

        self.context_space = gym.spaces.Box(low=ctx_lb, high=ctx_ub, dtype=np.float64)
        self.observation = self.identity_obs
        if ctx_visible==1 :
            low_ext = np.concatenate((self.env.observation_space.low, ctx_lb))
            high_ext = np.concatenate((self.env.observation_space.high, ctx_ub))
            self._observation_space = gym.spaces.Box(low=low_ext, high=high_ext, dtype=np.float64)
            self.observation = self.concat_obs
        elif ctx_visible == 2:
            self.observation = self.dict_obs
            self._observation_space = gym.spaces.Tuple((self.env.observation_space,
                                                       self.context_space))

        # self.reconfig(context)
    def set_task(self, task: TaskType) -> None:
        self.cur_context = task
        self.current_flat_ctx.update([(k, task[v[0]:v[1]]) for k, v in self.ctx_map.items()])
        self.env.set_task(unflatten(self.current_flat_ctx))
        return

    def report_task(self) -> Dict:
        ctx = self.env.report_task()
        ctx_flattened = flatten(ctx)
        # np.array(list(itertools.chain.from_iterable(self.current_flat_ctx.values())))
        ctx_array =  np.array(list(itertools.chain.from_iterable(ctx_flattened.values())))
        return {'mean': ctx_array, 'var': np.zeros(len(ctx_array)), 'unflattened': ctx}

    def get_task(self):
        return self.cur_context

    def reconfig(self, config):
        self.env.reconfig(config)
        ctx = flatten(self.env.get_task())
        self.current_flat_ctx.update([(k, ctx[k]) for k in self.selected_keys if k in ctx])
        self.cur_context = np.array(list(itertools.chain.from_iterable(self.current_flat_ctx.values())))
        # self.env.reconfig(config)




class DiscreteCtxDictWrapper(gym.Wrapper):


    def __init__(self, env, embedding_map,embeddings, key=None,
                 embedding_dim = 5 ,):
        super().__init__(env)
        # self.current_ctx =
        self.flat_ctx = flatten(self.env.get_task())
        ctxkeys = list(self.flat_ctx.keys())
        if key is None:
            key = ctxkeys[0][0]

        self.selected_keys = [k for k in ctxkeys if k[0] in key]

        assert len(self.selected_keys) > 0
        self.ctx_map = dict()
        bias = 0

        for k in self.selected_keys:
            ctx_length = len(self.flat_ctx[k])
            self.ctx_map[k] = (bias, bias + ctx_length)
            bias += ctx_length


        self.embedding_dim = embedding_dim

        self.embedding_map = embedding_map
        self.embeddings = embeddings


        self.current_flat_ctx = {k:self.flat_ctx[k] for k in self.selected_keys}
        self.cur_context_arr = np.array(list(itertools.chain.from_iterable(self.current_flat_ctx.values())))
        self.cur_context = self.embeddings[find_category_idx(self.embedding_map, self.cur_context_arr), :]

        # if hasattr(env, 'context_space'):
        #     ctx_space = flatten(env.context_space)
        #     ctx_lb=np.zeros(0)
        #     ctx_ub=np.zeros(0)
        #     for k in self.selected_keys:
        #         ctx_lb = np.concatenate(( ctx_lb, ctx_space[k].low))
        #
        #         ctx_ub = np.concatenate((ctx_ub, ctx_space[k].high))
        # else:
        ctx_lb = np.ones(embedding_dim) * -1
        ctx_ub = np.ones(embedding_dim) * 1

        self.context_space = gym.spaces.Box(low=ctx_lb, high=ctx_ub, dtype=np.float64)
        # self.observation = self.identity_obs
        # if ctx_visible==1 :
        #     low_ext = np.concatenate((self.env.observation_space.low, ctx_lb))
        #     high_ext = np.concatenate((self.env.observation_space.high, ctx_ub))
        #     self._observation_space = gym.spaces.Box(low=low_ext, high=high_ext, dtype=np.float64)
        #     self.observation = self.concat_obs
        # elif ctx_visible == 2:
        #     self.observation = self.dict_obs
        #     self._observation_space = gym.spaces.Tuple((self.env.observation_space,
        #                                                self.context_space))

        # self.reconfig(context)
    def set_task(self, task: TaskType) -> None:

        ctx_idx = np.argmin(np.sum((self.embeddings-task)**2, axis=1))
        self.cur_context = self.embeddings[ctx_idx, :]
        self.cur_context_arr = self.embedding_map[ctx_idx]
        self.current_flat_ctx.update([(k, self.cur_context_arr[v[0]: v[1]]) for k, v in self.ctx_map.items()])
        self.env.set_task(unflatten(self.current_flat_ctx))
        return

    def report_task(self) -> Dict:
        ctx = flatten(self.env.report_task())
        # np.array(list(itertools.chain.from_iterable(self.current_flat_ctx.values())))
        ctx_array =  np.array(list(itertools.chain.from_iterable(ctx.values())))
        return {'mean': ctx_array, 'var': np.zeros(len(ctx_array))}

    def get_task(self):
        return self.cur_context

    def reconfig(self, config):
        self.env.reconfig(config)
        ctx = flatten(self.env.get_task())
        self.current_flat_ctx.update([(k, ctx[k]) for k in self.selected_keys if k in ctx])
        self.cur_context_arr = np.array(list(itertools.chain.from_iterable(self.current_flat_ctx.values())))
        self.cur_context = self.embeddings[find_category_idx(self.embedding_map, self.cur_context_arr), :]
        # self.env.reconfig(config)




class DefaultCtxEnvWrapper(gym.ObservationWrapper, TaskSettableEnv):
    def identity_obs(self, obs):
        return obs

    def concat_obs(self, obs):
        return np.concatenate((obs, self.cur_context))

    def __init__(self,
                 env,
                 context=None,
                 ctx_visible=True, **kwargs):
        assert context is not None
        super().__init__(env)

        if isinstance(context, np.ndarray):
            ctx_lb = np.ones(context.size) * (-np.infty)
            ctx_ub = np.ones(context.size) * (np.infty)
        else:
            isinstance(context, list)
            ctx_lb = np.ones(len(context), dtype=np.float64) * (-np.infty)
            ctx_ub = np.ones(len(context), dtype=np.float64) * (np.infty)
        self.context_space = gym.spaces.Box(low=ctx_lb, high=ctx_ub, dtype=np.float64)
        self.observation = self.identity_obs



        if ctx_visible:
            low_ext = np.concatenate((self.env.observation_space.low, ctx_lb))
            high_ext = np.concatenate((self.env.observation_space.high, ctx_ub))
            self._observation_space = gym.spaces.Box(low=low_ext, high=high_ext, dtype=np.float64)
            self.observation = self.concat_obs



        self.cur_context = np.array(context)


    def sample_tasks(self, n_tasks):
        """Implement this to sample n random tasks."""
        return [self.cur_context,  ]*n_tasks

  #
    def report_task(self):
        return {'mean': self.cur_context, 'var': np.zeros_like(self.cur_context)}

    def reconfig(self, config):
        self.cur_context = np.array(config)
        self.env.set_task(self.cur_context)
        return

    def set_task(self, task: TaskType) -> None:
        self.reconfig(task)
        return

    def get_task(self):
        return self.cur_context
    def get_context(self):
        return self.cur_context


@PublicAPI
def make_multi_agent_divide_and_conquer(
        env_name_or_creator: Union[str, EnvCreator],
) -> Type["MultiAgentEnv"]:
    class MADnCEnv(MultiAgentEnv, TaskSettableEnv):
        # __name__= env_name_or_creator if isinstance(env_name_or_creator,str) else env_name_or_creator.__name__
        iteration = 0
        # TRAINING = True

        def class_name(cls) -> str:
            """Returns the class name of the wrapper."""
            return cls.__name__

        def __str__(self):
            """Returns the wrapper name and the :attr:`env` representation string."""
            return f"<{type(self).__name__}{self.envs[0]}>"

        def __repr__(self):
            """Returns the string representation of the wrapper."""
            return str(self)

        def __init__(self, config: EnvContext = None, **kwargs):
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

            # self.env_mapping = config.get('env_mapping', None)
            # self.non_training_envs = config.get('non_training_envs', 1)

            # self.TRAINING = config.get('training', True)

            if isinstance(env_name_or_creator, str):
                self.envs = [gym.make(env_name_or_creator) for _ in range(num)]
            # elif self.env_mapping:
            #     self.envs = [env_name_or_creator(config["agent_config"][i], **kwargs) for i in self.env_mapping]
            else:
                self.envs = [env_name_or_creator(conf) for conf in config["agent_config"]]
            self.__name__ = self.envs[0].unwrapped.__class__.__name__
            # self.truncateds_training = set()
            self.truncateds = set()
            # self.terminateds_training = set()
            self.terminateds = set()
            self.observation_space = self.envs[0].observation_space
            self.action_space = self.envs[0].action_space
            self._agent_ids = set(range(num))
            self.buffer = Buffer(n_elements=1, max_buffer_size=1000, reset_on_query=True)

        def update_buffer(self, data):
            self.buffer.update_buffer(data)

        def read_buffer(self, reset= True):
            return self.buffer.read_buffer( reset)


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
            self.truncateds = set()
            # self.truncateds_training = set()
            #
            # self.terminateds_training = set()
            self.terminateds = set()
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
                    # if i >self.non_training_envs:
                    #     self.terminateds_training.add(i)
                if truncated[i]:
                    self.truncateds.add(i)
                    # if i > self.non_training_envs:
                    #     self.truncateds_training.add(i)
            # if self.TRAINING:
            #     if len(self.terminateds_training) == len(self.envs) - self.non_training_envs:
            #         for i in range(self.non_training_envs):
            #             self.terminateds.add(i)
            #             terminated[i] = True

                # if len(self.truncateds_training) == len(self.envs) - self.non_training_envs:
                #     for i in range(self.non_training_envs):
                #         self.truncateds.add(i)
                #         truncated[i] = True




            # TODO: Flaw in our MultiAgentEnv API wrt. new gymnasium: Need to return
            #  an additional episode_done bool that covers cases where all agents are
            #  either terminated or truncated, but not all are truncated and not all are
            #  terminated. We can then get rid of the aweful `__all__` special keys!
            terminated["__all__"] = len(self.terminateds) + len(self.truncateds) >= len(
                self.envs
            )
            truncated["__all__"] = len(self.truncateds) == len(self.envs)
            return obs, rew, terminated, truncated, info

        def set_task(self, task_dict: TaskType) -> None:
            """
            """
            for i, task in task_dict.items():
                self.envs[i].set_task(task)

        def copy_task(self, task: TaskType):

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

        @override(MultiAgentEnv)
        def render(self, mode=None):
            return self.envs[0].render(mode)

        def reconfig_all(self, config):
            for a in self.envs:
                a.reconfig(config)
            return

        def get_context(self):
            return {i: a.get_context() for i, a in enumerate(self.envs)}

        def set_sampler_dist(self, agent_idx, means, sigma2, w):
            self.envs[agent_idx].set_distribution(means=means, sigma2=sigma2, priors=w)
            return

        def get_context_space(self):
            return self.envs[0].context_space

        def training_mode(self, mode):
            self.TRAINING = mode

    return MADnCEnv


@PublicAPI
def make_multi_agent(
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

    .. testcode::
        :skipif: True

        from ray.rllib.env.multi_agent_env import make_multi_agent
        # By gym string:
        ma_cartpole_cls = make_multi_agent("CartPole-v1")
        # Create a 2 agent multi-agent cartpole.
        ma_cartpole = ma_cartpole_cls({"num_agents": 2})
        obs = ma_cartpole.reset()
        print(obs)

        # By env-maker callable:
        from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
        ma_stateless_cartpole_cls = make_multi_agent(
           lambda config: StatelessCartPole(config))
        # Create a 3 agent multi-agent stateless cartpole.
        ma_stateless_cartpole = ma_stateless_cartpole_cls(
           {"num_agents": 3})
        print(obs)

    .. testoutput::

        {0: [...], 1: [...]}
        {0: [...], 1: [...], 2: [...]}
    """

    class MultiEnv(MultiAgentEnv, TaskSettableEnv):
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
                self.envs = [gym.make(env_name_or_creator) for _ in range(num)]
            elif self.env_mapping:
                self.envs = [env_name_or_creator(config["agent_config"][i]) for i in self.env_mapping]
            else:
                self.envs = [env_name_or_creator(conf) for conf in config["agent_config"]]
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
            terminated["__all__"] = len(self.terminateds) + len(self.truncateds) >=len(
                self.envs
            )
            truncated["__all__"] = len(self.truncateds) == len(self.envs)
            return obs, rew, terminated, truncated, info

        @override(MultiAgentEnv)
        def render(self):
            return self.envs[0].render(self.render_mode)

        def set_task(self, task_dict: TaskType) -> None:
            """
            """
            for i, task in task_dict.items():
                self.envs[i].set_task(task)

        def copy_task(self, task: TaskType):

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

        @override(MultiAgentEnv)
        def render(self, mode=None):
            return self.envs[0].render(mode)

        def reconfig_all(self, config):
            for a in self.envs:
                a.reconfig(config)
            return

        def get_context(self):
            return {i: a.get_context() for i, a in enumerate(self.envs)}

        def set_sampler_dist(self, agent_idx, means, sigma2, w):
            self.envs[agent_idx].set_distribution(means=means, sigma2=sigma2, priors=w)
            return

        def get_context_space(self):
            return self.envs[0].context_space

        def training_mode(self, mode):
            self.TRAINING = mode

    return MultiEnv

