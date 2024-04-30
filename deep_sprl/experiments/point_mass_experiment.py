import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import numpy as np
from torch.nn import functional as F
from deep_sprl.experiments.abstract_experiment import AbstractExperiment, Learner
# from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
# from deep_sprl.teachers.goal_gan import GoalGAN, GoalGANWrapper
from deep_sprl.teachers.spl import SelfPacedTeacher, SelfPacedTeacherV2, SelfPacedWrapper
from deep_sprl.teachers.spl.alpha_functions import PercentageAlphaFunction
from deep_sprl.teachers.dummy_teachers import GaussianSampler, UniformSampler
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from stable_baselines3.common.vec_env import DummyVecEnv


class PointMassExperiment(AbstractExperiment):
    LOWER_CONTEXT_BOUNDS = np.array([-4., 0.5, 0.])
    UPPER_CONTEXT_BOUNDS = np.array([4., 8., 4.])

    INITIAL_MEAN = np.array([0., 4.25, 2.])
    INITIAL_VARIANCE = np.diag(np.square([2, 1.875, 1]))

    TARGET_MEAN = np.array([2.5, 0.5, 0.])
    TARGET_VARIANCE = np.diag(np.square([4e-3, 3.75e-3, 2e-3]))

    DISCOUNT_FACTOR = 0.95
    STD_LOWER_BOUND = np.array([0.2, 0.1875, 0.1])
    KL_THRESHOLD = 8000.
    MAX_KL = 0.05

    ZETA = {Learner.TRPO: 1.6, Learner.PPO: 1.6, Learner.SAC: 1.1}
    ALPHA_OFFSET = {Learner.TRPO: 20, Learner.PPO: 10, Learner.SAC: 25}
    OFFSET = {Learner.TRPO: 5, Learner.PPO: 5, Learner.SAC: 5}
    PERF_LB = {Learner.TRPO: 3.5, Learner.PPO: 3.5, Learner.SAC: 3.5}

    STEPS_PER_ITER = 2048
    LAM = 0.99

    AG_P_RAND = {Learner.TRPO: 0.2, Learner.PPO: 0.3, Learner.SAC: 0.1}
    AG_FIT_RATE = {Learner.TRPO: 50, Learner.PPO: 100, Learner.SAC: 200}
    AG_MAX_SIZE = {Learner.TRPO: 500, Learner.PPO: 500, Learner.SAC: 500}

    GG_NOISE_LEVEL = {Learner.TRPO: 0.1, Learner.PPO: 0.025, Learner.SAC: 0.1}
    GG_FIT_RATE = {Learner.TRPO: 50, Learner.PPO: 100, Learner.SAC: 100}
    GG_P_OLD = {Learner.TRPO: 0.2, Learner.PPO: 0.1, Learner.SAC: 0.3}

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed, **kwargs):
        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed, **kwargs)
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)

    def create_environment(self, evaluation=False):
        env = gym.make("ContextualPointMass-v1")
        if evaluation or self.curriculum.default():
            teacher = GaussianSampler(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE,
                                      (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy()))
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        # elif self.curriculum.alp_gmm():
        #     teacher = ALPGMM(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(), seed=self.seed,
        #                      fit_rate=self.AG_FIT_RATE[self.learner], random_task_ratio=self.AG_P_RAND[self.learner],
        #                      max_size=self.AG_MAX_SIZE[self.learner])
        #     env = ALPGMMWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        # elif self.curriculum.goal_gan():
        #     samples = np.clip(np.random.multivariate_normal(self.INITIAL_MEAN, self.INITIAL_VARIANCE, size=1000),
        #                       self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
        #     teacher = GoalGAN(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(),
        #                       state_noise_level=self.GG_NOISE_LEVEL[self.learner], success_distance_threshold=0.01,
        #                       update_size=self.GG_FIT_RATE[self.learner], n_rollouts=2, goid_lb=0.25, goid_ub=0.75,
        #                       p_old=self.GG_P_OLD[self.learner], pretrain_samples=samples)
        #     env = GoalGANWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        # elif self.curriculum.self_paced() or self.curriculum.self_paced_v2():
        #     teacher = self.create_self_paced_teacher()
        #     env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.random():
            teacher = UniformSampler(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        else:
            raise RuntimeError("Invalid learning type")

        return env, DummyVecEnv([lambda: env])

    def create_learner_params(self):
        return dict(common=dict(gamma=self.DISCOUNT_FACTOR, seed=self.seed, verbose=0,
                                policy_kwargs=dict(layers=[64, 64], act_fun=F.tanh)),
                    trpo=dict(timesteps_per_batch=self.STEPS_PER_ITER, lam=self.LAM),
                    ppo=dict(n_steps=self.STEPS_PER_ITER, noptepochs=8, nminibatches=32, lam=self.LAM,
                             max_grad_norm=None, vf_coef=1.0, cliprange_vf=-1, ent_coef=0.),
                    sac=dict(learning_rate=3e-4, buffer_size=10000, learning_starts=500, batch_size=64,
                             train_freq=5, target_entropy="auto"))

    def create_experiment(self):
        timesteps = 400 * self.STEPS_PER_ITER
        env, vec_env = self.create_environment(evaluation=False)
        model, interface = self.learner.create_learner(vec_env, self.create_learner_params())

        if isinstance(env.teacher, SelfPacedTeacher) or isinstance(env.teacher, SelfPacedTeacherV2):
            sp_teacher = env.teacher
        else:
            sp_teacher = None

        callback_params = {"learner": interface, "env_wrapper": env, "sp_teacher": sp_teacher, "n_inner_steps": 1,
                           "n_offset": self.OFFSET[self.learner], "save_interval": 5,
                           "step_divider": self.STEPS_PER_ITER if self.learner.sac() else 1}
        return model, timesteps, callback_params

    def create_self_paced_teacher(self):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        if self.curriculum.self_paced():
            alpha_fn = PercentageAlphaFunction(self.ALPHA_OFFSET[self.learner], self.ZETA[self.learner])
            return SelfPacedTeacher(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE.copy(), self.INITIAL_MEAN.copy(),
                                    self.INITIAL_VARIANCE.copy(), bounds, alpha_fn, max_kl=self.MAX_KL,
                                    std_lower_bound=self.STD_LOWER_BOUND, kl_threshold=self.KL_THRESHOLD,
                                    use_avg_performance=True)
        else:
            return SelfPacedTeacherV2(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE.copy(), self.INITIAL_MEAN.copy(),
                                      self.INITIAL_VARIANCE.copy(), bounds, self.PERF_LB[self.learner],
                                      max_kl=self.MAX_KL, std_lower_bound=self.STD_LOWER_BOUND.copy(),
                                      kl_threshold=self.KL_THRESHOLD, use_avg_performance=True)

    def get_env_name(self):
        return "point_mass"

    def evaluate_learner(self, path):
        model_load_path = os.path.join(path, "model.zip")
        model = self.learner.load_for_evaluation(model_load_path, self.vec_eval_env)
        for i in range(0, 50):
            obs = self.vec_eval_env.reset()
            done = False
            while not done:
                action = model.step(obs, state=None, deterministic=False)
                obs, rewards, done, infos = self.vec_eval_env.step(action)

        return self.eval_env.get_statistics()[1]
