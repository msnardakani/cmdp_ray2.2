import gym
# from gym.wrappers import TimeLimit
# from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
# from ray.rllib.env.env_context import EnvContext
# from ray.rllib.utils.annotations import override
from ray.rllib.env.env_context import EnvContext
import numpy as np
from gymnasium import spaces
from gym import spaces as gym_spaces
from gym.wrappers.normalize import NormalizeObservation
import gymnasium
from deep_sprl.teachers.dummy_teachers import GMMSampler
from deep_sprl.teachers.spl import SelfPacedTeacherV2#, SelfPacedWrapper
# from self_paced_wrappers import SelfPacedWrapper

def make_DnC_env(env_createor, ctx_lb, ctx_ub, ctx_visible=True,  ctx_norm = None, discount_factor= .99, std_lb = 0.1, dict_obs= False):
    class DnCEnv(gym.Env):
        DISCOUNT_FACTOR = discount_factor
        CTX_VISIBLE = ctx_visible
        STD_LOWER_BOUND = std_lb#np.array([0.2, 0.1875])
        MAX_KL = 0.05
        KL_THRESHOLD = 8000
        PERF_LB = 3.5
        STD_LB = std_lb
        CTX_NORM = ctx_norm
        DICT_OBS =dict_obs
        def __init__(self, config: EnvContext):
            # self.cur_level = config.get("init_mean", np.array([0, 4]))
            # self.max_timesteps = config.get("max_timesteps", 100)

            self.init_mean = config.get("init_mean", (ctx_lb + ctx_ub) / 2)
            self.init_var = config.get('init_var', ((-ctx_lb + ctx_ub)*2))
            self.init_priors = config.get('init_priors', np.array([1]))
            self.curriculum = config.get('curriculum', 'default')
            # if self.curriculum == 'self_paced':
            self.target_mean = config.get('target_mean', (ctx_lb + ctx_ub) / 2)
            self.target_var = config.get('target_var', (-ctx_lb + ctx_ub)/ 1000)
            self.target_priors = config.get('target_priors', np.array([1]))

            self.context_lb = ctx_lb
            self.context_ub = ctx_ub
            # self.frozen_lake = None
            self._make_env(obs_norm=self.CTX_NORM)  # create the gym env

            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
            self.context_space = self.env.context_space
            self.switch_env = False
            self._timesteps = 0

        def _make_env(self, obs_norm = None):
            base_env = env_createor()
            # if obs_norm:
            #     base_env = NormalizeObservation(base_env)
            if self.curriculum == 'self_paced':
                self.teacher = SelfPacedTeacherV2(self.target_mean.copy(), np.diag(self.target_var).copy(), self.init_mean.copy(),
                                                  np.diag(self.init_var).copy(), (self.context_lb.copy(), self.context_ub.copy()), self.PERF_LB,
                                                  max_kl=self.MAX_KL, std_lower_bound=self.STD_LB,
                                                  kl_threshold=self.KL_THRESHOLD, use_avg_performance=True)
                if self.DICT_OBS:
                    from .self_paced_wrappers import SelfPacedWrapper
                    self.env = SelfPacedWrapper(base_env, self.teacher, self.DISCOUNT_FACTOR, ctx_norm=obs_norm)
                else:
                    from deep_sprl.teachers.spl import SelfPacedWrapper
                    self.env = SelfPacedWrapper(base_env, self.teacher, self.DISCOUNT_FACTOR,
                                                context_visible=self.CTX_VISIBLE, ctx_norm=obs_norm)


                # self.env = SelfPacedWrapper(base_env, self.teacher, self.DISCOUNT_FACTOR, context_visible=self.CTX_VISIBLE,ctx_norm = obs_norm)


            else:
                self.teacher = GMMSampler(self.target_mean.copy(),
                                          self.target_var.copy(),
                                          self.target_priors.copy(),
                                          (self.context_lb.copy(),
                                           self.context_ub.copy()))
                if self.DICT_OBS:
                    from .self_paced_wrappers import BaseWrapper
                    self.env = BaseWrapper(base_env, self.teacher, self.DISCOUNT_FACTOR, ctx_norm=obs_norm)
                else:
                    from deep_sprl.teachers.abstract_teacher import BaseWrapper

                    self.env = BaseWrapper(base_env, self.teacher, self.DISCOUNT_FACTOR, context_visible=self.CTX_VISIBLE, ctx_norm = obs_norm)

        def step(self, action):
            return self.env.step(action)

        def reset(self):
            return self.env.reset()

        def sample_tasks(self, n_tasks):
            """Implement this to sample n random tasks."""
            return [self.teacher.sample() for _ in range(n_tasks)]

        def get_task(self):
            """Implement this to get the current task (curriculum level)."""
            if self.curriculum == 'default':
                return {'mean': self.teacher.mu, 'var': self.teacher.vars, 'prior': self.teacher.w0}
            else:
                return self.teacher.get_task()
        def get_ctx_hist(self):
            return self.env.get_ctx_hist()

        def report_task(self):
            if self.curriculum == 'default':
                return {'mean': self.teacher.mean(), 'var': np.diag(self.teacher.covariance_matrix())}
            else:
                return self.teacher.get_context()

        def set_task(self, task):
            """Implement this to set the task (curriculum level) for this env."""
            if self.curriculum == 'default':
                self.teacher.set_means( task['mean'])
                self.teacher.set_vars (task['var'])  # gets the var as 1d array and then make it 2d
                self.teacher.set_w(task['prior'])
            else:
                self.teacher.set_task(task)

            self.switch_env = True
        def get_context(self):
            return self.env.get_context()

        def get_statistics(self):
            return self.env.get_statistics()

        def get_episodes_statistics(self):
            return self.env.get_episodes_statistic()

        def get_context_buffer(self):
            return self.env.get_context_buffer()

        def get_teacher(self):
            return self.env.get_teacher()

        def reconfig(self, config):
            self.teacher.reconfig(config)
            return

        def update_teacher(self,weights):
            self.env.update_teacher(weights=weights)

        def get_buffer_size(self):
            return self.env.get_buffer_size()

    return DnCEnv






class gymnasiumEnvWrapper(gym.Wrapper):
    def __init__(self,  env: gymnasium.Env):
        super(gymnasiumEnvWrapper, self).__init__(env)
        if isinstance(env.action_space, spaces.Box):
            self._action_space = gym_spaces.Box(env.action_space.low, env.action_space.high,
                                            shape = env.action_space.shape, dtype= env.action_space.dtype)
        elif isinstance(env.action_space, spaces.Discrete):
            self._action_space = gym_spaces.Discrete(env.action_space.n)
        else:
            self._action_space = None


        if isinstance(env.observation_space, spaces.Box):
            self._observation_space = gym_spaces.Box(env.observation_space.low, env.observation_space.high,
                                                shape=env.observation_space.shape, dtype=env.observation_space.dtype)
        elif isinstance(env.observation_space, spaces.Discrete):
            self._observation_space  = gym_spaces.Discrete(env.observation_space.n)
        else:
            self._observation_space = None


    def step(self, action):

        observation, reward, terminated,truncated, info = self.env.step(action)

        return observation, reward, terminated or truncated, info

    def reset(self):
        return self.env.reset()[0]

# class LegacyEnvWrapper()

class gymEnvWrapper(gymnasium.Wrapper):
    def __init__(self,  env: gym.Env):
        super(gymEnvWrapper, self).__init__(env)
        if isinstance(env.action_space, gym_spaces.Box):
            self._action_space  =spaces.Box(env.action_space.low,
                                            env.action_space.high,
                                           shape = env.action_space.shape,
                                            dtype= env.action_space.dtype)
        elif isinstance(env.action_space, gym_spaces.Discrete):
            self._action_space = spaces.Discrete(env.action_space.n)
        else:
            self._action_space = None


        if isinstance(env.observation_space, gym_spaces.Box):
            self._observation_space =spaces.Box(env.observation_space.low,
                                                env.observation_space.high,
                                                shape=env.observation_space.shape,
                                                dtype=env.observation_space.dtype)
        elif isinstance(env.observation_space, spaces.Discrete):
            self._observation_space = spaces.Discrete(env.observation_space.n)
        else:
            self._observation_space = None


    def step(self, action):

        observation, reward, terminated, info = self.env.step(action)

        return observation, reward, terminated , False, info

    def reset(self, **kwargs):
        return self.env.reset(), {}