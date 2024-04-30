from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from deep_sprl.environments.contextual_point_mass_2d import ContextualPointMass2D
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import override
from deep_sprl.teachers.abstract_teacher import BaseWrapper
import numpy as np
from envs.utils import gymEnvWrapper

from deep_sprl.teachers.dummy_teachers import GaussianSampler, GMMSampler
from deep_sprl.teachers.spl import SelfPacedTeacherV2
from deep_sprl.teachers.spl.self_paced_wrapper import SelfPacedWrapper

class TaskSettablePointMass2D(TaskSettableEnv):
    DISCOUNT_FACTOR = 0.95
    STD_LOWER_BOUND =np.array([0.2, 0.1875])
    MAX_KL =0.05
    KL_THRESHOLD =8000
    PERF_LB = 3.5
    def __init__(self, config: EnvContext):
        # self.cur_level = config.get("init_mean", np.array([0, 4]))
        self.max_timesteps = config.get("max_timesteps", 100)

        self.init_mean = config.get("init_mean", np.array([0,4]))
        self.init_var = config.get('init_var', np.square([4, 4]))
        self.init_priors = config.get('init_priors', np.array([1]))
        self.curriculum = config.get('curriculum', 'default')
        # if self.curriculum == 'self_paced':
        self.target_mean = config.get('target_mean', np.array([2.5, 0.5]))
        self.target_var = config.get('target_var', np.square([4e-3, 3.75e-3]))
        self.target_priors = config.get('target_priors', np.array([1]))

        self.context_lb = np.array([-4,0.5])
        self.context_ub = np.array([4, 8])
        # self.frozen_lake = None
        self._make_env()  # create the gym env

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.switch_env = False
        self._timesteps = 0

    def step(self, action):
        self._timesteps += 1
        s, r, d, i = self.env.step(action)
        # Make rewards scale with the level exponentially:
        # Level 1: x1
        # Level 2: x10
        # Level 3: x100, etc..
        # r *= 10 ** (self.cur_level - 1)
        if self._timesteps >= self.max_timesteps:
            d = True
        return s, r, d, i

    def reset(self):
        if self.switch_env:
            self.switch_env = False
            # self._make_lake()
        self._timesteps = 0
        return self.env.reset()


    @override(TaskSettableEnv)
    def sample_tasks(self, n_tasks):
        """Implement this to sample n random tasks."""
        return [self.teacher.sample() for _ in range(n_tasks)]

    @override(TaskSettableEnv)
    def get_task(self):
        """Implement this to get the current task (curriculum level)."""
        if self.curriculum== 'default':
            return {'mean':self.teacher._mean, 'var':np.diag(self.teacher._covariance) }
        else:
            return self.teacher.get_task()

    @override(TaskSettableEnv)
    def set_task(self, task):
        """Implement this to set the task (curriculum level) for this env."""
        if self.curriculum == 'default':
            self.teacher._mean = task['mean']
            self.teacher._covariance = np.diag(task['var']) #gets the var as 1d array and then make it 2d
        else:
            self.teacher.set_task(task)

        self.switch_env = True

    def _make_env(self):
        base_env = ContextualPointMass2D()
        if self.curriculum =='self_paced':
            self.teacher =SelfPacedTeacherV2(self.target_mean, self.target_var, self.init_mean,
                                      self.init_var, (self.context_lb,self.context_ub), self.PERF_LB,
                                      max_kl=self.MAX_KL, std_lower_bound=self.STD_LOWER_BOUND,
                                      kl_threshold=self.KL_THRESHOLD, use_avg_performance=True)
            self.env = SelfPacedWrapper(base_env,self.teacher,self.DISCOUNT_FACTOR, context_visible=True)
        else:
            self.teacher = GMMSampler(self.target_mean,
                             self.target_var,
                                      self.target_priors,
                                  (self.context_lb,
                              self.context_ub))
            self.env = BaseWrapper(base_env, self.teacher, self.DISCOUNT_FACTOR, context_visible=True)

    def get_statistics(self):
        return self.env.get_statistics()

    def get_context_buffer(self):
        return self.env.get_context_buffer()


class PointMassEnv(ContextualPointMass2D, TaskSettableEnv):
    def __init__(self,context=np.array([0., 2.]) ):

        super().__init__(context=context)
        self.render_mode='rgb_array'


    def set_task(self, task) -> None:
        self.set_context(task)

    def get_task(self):
        return self.get_context()