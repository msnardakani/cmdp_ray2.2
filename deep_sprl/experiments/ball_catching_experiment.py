import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import numpy as np
import torch.nn as nn

from deep_sprl.experiments.abstract_experiment import AbstractExperiment, Learner
# from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
# from deep_sprl.teachers.goal_gan import GoalGAN, GoalGANWrapper
from deep_sprl.teachers.spl import SelfPacedTeacher, SelfPacedTeacherV2, SelfPacedWrapper
from deep_sprl.teachers.spl.alpha_functions import PercentageAlphaFunction
from deep_sprl.teachers.dummy_teachers import GaussianSampler, UniformSampler
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

from custom_policy import *
from stable_baselines3.common.monitor import Monitor
import json

class DummyBallCatchingExperiment(AbstractExperiment):
    ERROR_MSG = "Mujoco-Py is not installed. Hence running experiments is not available for the environment"

    LOWER_CONTEXT_BOUNDS = np.array([0.125 * np.pi, 0.6, 0.75])
    UPPER_CONTEXT_BOUNDS = np.array([0.5 * np.pi, 1.1, 4.])

    INITIAL_MEAN = np.array([0.68, 0.9, 0.85])
    INITIAL_VARIANCE = np.diag([1e-3, 1e-3, 0.1])

    TARGET_MEAN = np.array([0.3375 * np.pi, 0.85, 2.375])
    TARGET_VARIANCE = np.diag([0.2 * np.pi, 0.15, 1.])

    EVAL_MEAN = np.array([0.3375 * np.pi, 0.85, 2.375])
    EVAL_VAR = np.diag([0.2 * np.pi, 0.15, 1.])

    DISCOUNT_FACTOR = 0.995
    MAX_KL = 0.05

    # These values are set in the constructor as they depend on the experiment type
    ZETA = None
    ALPHA_OFFSET = None
    OFFSET = {Learner.TRPO: 5, Learner.PPO: 5, Learner.SAC: 5}
    PERF_LB = {Learner.TRPO: 42.5, Learner.PPO: 42.5, Learner.SAC: 25.}

    STEPS_PER_ITER = 5000
    LAM = 0.95

    AG_P_RAND = {Learner.TRPO: 0.2, Learner.PPO: 0.3, Learner.SAC: 0.3}
    AG_FIT_RATE = {Learner.TRPO: 200, Learner.PPO: 200, Learner.SAC: 200}
    AG_MAX_SIZE = {Learner.TRPO: 2000, Learner.PPO: 2000, Learner.SAC: 1000}

    GG_NOISE_LEVEL = {Learner.TRPO: 0.1, Learner.PPO: 0.1, Learner.SAC: 0.1}
    GG_FIT_RATE = {Learner.TRPO: 200, Learner.PPO: 200, Learner.SAC: 200}
    GG_P_OLD = {Learner.TRPO: 0.3, Learner.PPO: 0.3, Learner.SAC: 0.3}

    TORCH_ACT= {'nn.Tanh' : nn.Tanh, 'F.relu': nn.ReLU(), 'nn.ReLU()':nn.ReLU()}
    FEATURE_EXTRACTOR = {"StateContextAttnEnc": StateContextAttnEnc,
                            "StateContextDotAttnEnc": StateContextDotAttnEnc,
                            "StateContextEnc": StateContextEnc,
                            "StateEnc": StateEnc}
    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed, **kwargs):
        self.visualize = False
        if "VISUALIZE" in parameters:
            self.visualize = parameters["VISUALIZE"] is True
            del parameters["VISUALIZE"]

        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed, **kwargs)

    def create_experiment(self):
        raise RuntimeError(self.ERROR_MSG)

    def get_env_name(self):
        return "ball_catching"

    def create_self_paced_teacher(self):
        raise RuntimeError(self.ERROR_MSG)

    def evaluate_learner(self, path):
        raise RuntimeError(self.ERROR_MSG)


class BallCatchingExperiment(DummyBallCatchingExperiment):

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed, setup_add,**kwargs):
        if "INIT_CONTEXT" in parameters:
            self.init_context = "True" == parameters["INIT_CONTEXT"]
            del parameters["INIT_CONTEXT"]
        else:
            self.init_context = True

        if "INIT_POLICY" in parameters:
            self.init_policy = "True" == parameters["INIT_POLICY"]
            del parameters["INIT_POLICY"]
        else:
            self.init_policy = True

        if not self.init_context:
            self.ALPHA_OFFSET = {Learner.TRPO: 70, Learner.PPO: 50, Learner.SAC: 60}
            self.ZETA = {Learner.TRPO: 0.4, Learner.PPO: 0.45, Learner.SAC: 0.6}
        else:
            self.ZETA = {Learner.TRPO: 0.425, Learner.PPO: 0.45, Learner.SAC: 0.6}
            self.ALPHA_OFFSET = {Learner.TRPO: 0, Learner.PPO: 0, Learner.SAC: 0}

        if setup_add ==None:
            setup = dict()
        else:
            with open(setup_add, 'r') as f:
                setup = json.load(f)




        # print(self.TORCH_ACT)
        if  'policy'  in setup:
            self.policy = setup['policy']
            self.policy['activation_fn']= self.TORCH_ACT[self.policy['activation_fn']]
            exp_name = setup['name']

            if 'features_extractor_class' in self.policy:
                self.policy['features_extractor_class' ]=self.FEATURE_EXTRACTOR[self.policy['features_extractor_class' ]]
            if 'features_extractor_kwargs' in self.policy:

                if 'stt_enc' in self.policy['features_extractor_kwargs']:
                    self.policy['features_extractor_kwargs']['stt_enc']['act_fn']=self.TORCH_ACT[self.policy['features_extractor_kwargs']['stt_enc']['act_fn']]
                if 'ctx_enc' in self.policy['features_extractor_kwargs']:
                    self.policy['features_extractor_kwargs']['ctx_enc']['act_fn'] = self.TORCH_ACT[self.policy['features_extractor_kwargs']['ctx_enc']['act_fn']]
                if 'attn_nn' in self.policy['features_extractor_kwargs']:
                    self.policy['features_extractor_kwargs']['attn_nn']['act_fn'] = self.TORCH_ACT[self.policy['features_extractor_kwargs']['attn_nn']['act_fn']]
        else:
            self.policy = dict(activation_fn=nn.Tanh,
                                net_arch=[64,64,64] )
            self.name = 'aug_3x64'

        if 'n_epochs' in setup:
            self.epochs = setup['n_epochs']
        else:
            self.epochs = 801

        if 'eval' in setup:
            self.EVAL_MEAN = np.array(setup['eval']['mean'])
            self.EVAL_VAR = np.diag(np.array(setup['eval']['std']))



        # self.log_dir = '../Presentation/PM3/SPDL/S'+str(seed)+'/'+time_stamp +'rand_EXP.99_buff4_stt_ctx_enc_4:2x16_2x4_attn2x16_2x32'

        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed , **kwargs)
        self.name = exp_name
        self.eval_env = self.create_environment(evaluate=True)

        self.log_dir = self.get_log_dir()
    def create_environment(self, evaluate=False):
        if self.visualize:
            env = Monitor(gym.make("fcv',hv -v1"), info_keywords=('success',))
        else:
            env = Monitor(gym.make("ContextualBallCatching-v1"), info_keywords=('success',))
        if self.curriculum.default() :
            teacher = GaussianSampler(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE.copy(),
                                      (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy()))
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif evaluate:
            teacher = GaussianSampler(self.EVAL_MEAN.copy(), self.EVAL_VAR.copy(),
                                      (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy()))
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.self_paced() or self.curriculum.self_paced_v2():
            teacher = self.create_self_paced_teacher()
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True,
                                   max_context_buffer_size=100, reset_contexts=True)

        elif self.curriculum.random():
            teacher = UniformSampler(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible= True)
        else:
            raise RuntimeError("Invalid learning type")

        return env

    def get_other_appendix(self):
        return ("" if self.init_context else "_no_init_con") + ("" if self.init_policy else "_no_init_pol")

    def create_alg_params(self):
        params = dict(common=dict(gamma=self.DISCOUNT_FACTOR, seed=self.seed, verbose=0,
                                  policy_kwargs=self.policy),
                      trpo=dict(max_kl=0.01, n_steps=self.STEPS_PER_ITER, gae_lambda=self.LAM,
                                vf_stepsize=3e-4, tensorboard_log=self.log_dir),
                      ppo=dict(n_steps=self.STEPS_PER_ITER, n_epochs=10, batch_size=25, gae_lambda=self.LAM,# max_grad_norm=None,
                       vf_coef=1.0, clip_range_vf=None, ent_coef=0., tensorboard_log=self.log_dir),
                      sac=dict(learning_rate=3e-4, buffer_size=100000, learning_starts=1000, batch_size=512,
                               train_freq=1, target_entropy="auto", tensorboard_log=self.log_dir))

        if self.init_policy:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            params["pretrain_data_path"] = os.path.join(dir_path, "data", "expert_data.npz")

        return params

    def create_experiment(self):
        env = self.create_environment()
        model, interface = self.learner.create_learner(env, self.create_alg_params())

        timesteps = self.epochs * self.STEPS_PER_ITER
        if isinstance(env.teacher, SelfPacedTeacher) or isinstance(env.teacher, SelfPacedTeacherV2):
            sp_teacher = env.teacher
        else:
            sp_teacher = None

        callback_params = {"learner": interface, "env_wrapper": env, "sp_teacher": sp_teacher, "n_inner_steps": 1,
                           "n_offset": self.OFFSET[self.learner], "save_interval": 5,
                           "step_divider": self.STEPS_PER_ITER }#if self.learner.sac() else 1}
        return model, timesteps, callback_params

    def create_self_paced_teacher(self):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())

        if self.init_context:
            init_mean = self.INITIAL_MEAN.copy()
            init_var = self.INITIAL_VARIANCE.copy()
        else:
            init_mean = self.TARGET_MEAN.copy()
            init_var = self.TARGET_VARIANCE.copy()

        if self.curriculum.self_paced():
            alpha_fn = PercentageAlphaFunction(self.ALPHA_OFFSET[self.learner], self.ZETA[self.learner])
            return SelfPacedTeacher(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE.copy(), init_mean, init_var,
                                    bounds, alpha_fn, max_kl=self.MAX_KL, use_avg_performance=True)
        else:
            return SelfPacedTeacherV2(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE.copy(), init_mean, init_var,
                                      bounds, self.PERF_LB[self.learner], max_kl=self.MAX_KL, use_avg_performance=True)

    def evaluate_learner(self, path):
        model_load_path = os.path.join(path, "model")
        model = self.learner.load_for_evaluation(model_load_path, self.eval_env)

        for i in range(0, 200):
            obs = self.eval_env.reset()
            done = False
            while not done:
                action = model.step(obs, state=None, deterministic=False)
                obs, rewards, done, infos = self.eval_env.step(action)

        return self.eval_env.get_statistics()[1]
