
import os

from imitation.algorithms import bc
from stable_baselines3 import PPO
from sb3_contrib import TRPO
# from deep_sprl.util.funcs import   DnC
import gym
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
from deep_sprl.experiments.abstract_experiment import AbstractExperiment, Learner
# from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
# from deep_sprl.teachers.goal_gan import GoalGAN, GoalGANWrapper
from deep_sprl.teachers.spl import SelfPacedTeacher, SelfPacedTeacherV2, SelfPacedWrapper
from deep_sprl.teachers.spl.alpha_functions import PercentageAlphaFunction
from deep_sprl.teachers.dummy_teachers import GMMSampler, GaussianSampler, UniformSampler
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from custom_policy import *

from stable_baselines3.common.monitor import Monitor
from custom_policy import StateContextAttnEnc

import datetime
import json
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class PointMass2DExperiment(AbstractExperiment):
    # LOWER_CONTEXT_BOUNDS = np.array([-4., 0.5])
    # UPPER_CONTEXT_BOUNDS = np.array([4., 8.])
    #
    # INITIAL_MEAN = np.array([0., 4.25])
    # INITIAL_VARIANCE = np.diag(np.square([2, 1.875]))
    #
    TARGET_MEAN = np.expand_dims(np.array([2.5, 0.5]), axis = 0)
    TARGET_VARIANCE = np.expand_dims(np.square([4e-3, 3.75e-3]), axis =0)
    TARGET_PRIOR = np.array([1])

    LOWER_CONTEXT_BOUNDS = np.array([-4., 0.5])
    UPPER_CONTEXT_BOUNDS = np.array([4., 8.])

    INITIAL_MEAN = np.array([0., 4.25])
    INITIAL_VARIANCE = np.diag(np.square([4, 3]))

    # TARGET_MEAN = np.array([0, 0.5])
    # TARGET_VARIANCE = np.diag(np.square([4, 3.75e-3]))

    EVAL_MEAN = np.expand_dims(np.array([2.5, 0.5]), axis = 0)
    EVAL_VARIANCE = np.expand_dims(np.square([4e-3, 3.75e-3]), axis =0)
    EVAL_PRIOR = np.array([1])
    DISCOUNT_FACTOR = 0.95
    STD_LOWER_BOUND = np.array([0.2, 0.1875])
    KL_THRESHOLD = 8000.
    MAX_KL = 0.05

    ZETA = {Learner.TRPO: 1.6, Learner.PPO: 1.6, Learner.SAC: 1.1, Learner.DnC: 1.6}
    ALPHA_OFFSET = {Learner.TRPO: 20, Learner.PPO: 10, Learner.SAC: 25, Learner.DnC: 10}
    OFFSET = {Learner.TRPO: 5, Learner.PPO: 5, Learner.SAC: 5, Learner.DnC: 5}
    PERF_LB = {Learner.TRPO: 3.5, Learner.PPO: 3.5, Learner.SAC: 3.5, Learner.DnC: 3.5}

    STEPS_PER_ITER = 2048*8
    LAM = 0.99

    AG_P_RAND = {Learner.TRPO: 0.2, Learner.PPO: 0.3, Learner.SAC: 0.2, Learner.DnC: 0.3}
    AG_FIT_RATE = {Learner.TRPO: 200, Learner.PPO: 100, Learner.SAC: 200, Learner.DnC: 100}
    AG_MAX_SIZE = {Learner.TRPO: 2000, Learner.PPO: 500, Learner.SAC: 1000,  Learner.DnC:500}

    GG_NOISE_LEVEL = {Learner.TRPO: 0.05, Learner.PPO: 0.1, Learner.SAC: 0.05}
    GG_FIT_RATE = {Learner.TRPO: 25, Learner.PPO: 50, Learner.SAC: 25}
    GG_P_OLD = {Learner.TRPO: 0.2, Learner.PPO: 0.2, Learner.SAC: 0.1}


    TORCH_ACT= {'nn.Tanh' : nn.Tanh, 'F.relu': nn.ReLU(), 'nn.ReLU()':nn.ReLU()}
    FEATURE_EXTRACTOR = {"StateContextAttnEnc": StateContextAttnEnc,
                            "StateContextDotAttnEnc": StateContextDotAttnEnc,
                            "StateContextEnc": StateContextEnc,
                            "StateEnc": StateEnc}
    def __init__(self, base_log_dir, curriculum_name, learner_name,
                 parameters, seed, setup_add, use_dnc, dnc_args=None, **kwargs):
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
        self.N = 2
        # print(setup_add)
        if dnc_args is None:
            dnc_args = {'save_interval': 5, "distillation_period": 50,
                        'bc_iteration': 5,
                        'bc_period': -1,
                        'last_bc': 5}
        if setup_add is None:
            setup = dict()
        else:
            print(setup_add)
            setup_file = setup_add
            # setup_file = os.path.join(ROOT_DIR, setup_add)
            with open(setup_file, 'r') as f:
                setup = json.load(f)




        # print(self.TORCH_ACT)
        if  'policy'  in setup:
            self.policy = setup['policy']
            self.policy['activation_fn']= self.TORCH_ACT[self.policy['activation_fn']]
            self.name = setup['name']

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
                                net_arch=[64,64] )
            self.name = 'aug_buff4_2x64'

        if 'n_epochs' in setup:
            self.epochs = setup['n_epochs']
        else:
            self.epochs = 501

        if 'target' in setup:
            self.TARGET_MEAN = np.array(setup['target']['mean'])
            self.TARGET_VARIANCE = np.square(np.array(setup['target']['std']))
            if 'prior' in setup['target']:
                self.TARGET_PRIOR = np.array(setup['target']['prior'])
            else:
                self.TARGET_PRIOR = np.array([1])


        if 'eval' in setup:
            self.EVAL_MEAN = np.array(setup['eval']['mean'])
            self.EVAL_VAR = np.square(np.array(setup['eval']['std']))
            if 'prior' in setup['eval']:
                self.EVAL_PRIOR = np.array(setup['eval']['prior'])
            else:
                self.EVAL_PRIOR = np.array([1])
        else:
            self.EVAL_MEAN = self.TARGET_MEAN
            self.EVAL_VAR = self.TARGET_VARIANCE
            self.EVAL_PRIOR = np.array([1 for i in range(self.TARGET_MEAN.shape[0])])

        # self.log_dir = '../Presentation/PM3/SPDL/S'+str(seed)+'/'+time_stamp +'rand_EXP.99_buff4_stt_ctx_enc_4:2x16_2x4_attn2x16_2x32'
        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed, experiment_name= self.name , **kwargs)
        # print(use_dnc)
        self.dnc = use_dnc
        if self.dnc:
            self.dnc_args = dnc_args
            self.distillation_period = dnc_args['distillation_period']
            self.bc_iteration= dnc_args['bc_iteration']
            self.PENALTY= dnc_args['PENALTY']
            self.last_bc= dnc_args['last_bc']
            self.bc_period = dnc_args['bc_period']
            self.bc_samples = dnc_args['bc_samples']
        self.curriculum_name = curriculum_name
        self.log_dir = self.get_log_dir()
        self.curriculum_params = {'TARGET_MEAN': self.TARGET_MEAN.copy(),
                      "TARGET_VARIANCE": self.TARGET_VARIANCE.copy(),
                      'TARGET_PRIORS': self.TARGET_PRIOR.copy(),
                      'EVAL_MEAN': self.EVAL_MEAN.copy(),
                      'EVAL_VARIANCE': self.EVAL_VAR.copy(),
                      'EVAL_PRIORS': self.EVAL_PRIOR.copy(),
                                  'INITIAL_MEAN':self.INITIAL_MEAN.copy(),
                                  'INITIAL_VARIANCE':self.INITIAL_VARIANCE.copy(),
                      'LOWER_CONTEXT_BOUNDS': self.LOWER_CONTEXT_BOUNDS.copy(),
                      'UPPER_CONTEXT_BOUNDS': self.UPPER_CONTEXT_BOUNDS.copy(),
                      'curriculum_type': self.curriculum_name}

        self.learner_params = {'DISCOUNT_FACTOR': self.DISCOUNT_FACTOR,
                  'STD_LOWER_BOUND': self.STD_LOWER_BOUND,
                  'KL_THRESHOLD': self.KL_THRESHOLD,
                  'MAX_KL': self.MAX_KL}
        # os.path.join(base_log_dir,learner_name,curriculum_name,self.name,'/S'+str(seed))
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)
        if not self.dnc:
            dummy_teacher = GMMSampler(self.curriculum_params['TARGET_MEAN'],
                                  self.curriculum_params['TARGET_VARIANCE'],
                                  self.curriculum_params['TARGET_PRIORS'],
                                       (self.curriculum_params['LOWER_CONTEXT_BOUNDS'],self.curriculum_params['UPPER_CONTEXT_BOUNDS']))

            self.curriculum_params['TARGET_MEAN'] = dummy_teacher.mean()
            self.curriculum_params['TARGET_VARIANCE'] = dummy_teacher.covariance_matrix()
            print(self.curriculum_params['TARGET_MEAN'] )
            print(self.curriculum_params['TARGET_VARIANCE'] )
            # self.base_log_dir = self.log_dir


    def create_environment(self, evaluation=False):
        env = Monitor(gym.make("ContextualPointMass2D-v1"), info_keywords=('is_success',))

        if evaluation:
             teacher = GMMSampler(self.curriculum_params['EVAL_MEAN'],
                                  self.curriculum_params['EVAL_VARIANCE'],
                                  self.curriculum_params['EVAL_PRIORS'],
                                       (self.curriculum_params['LOWER_CONTEXT_BOUNDS'],self.curriculum_params['UPPER_CONTEXT_BOUNDS']))
             env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)

        elif self.dnc:


            env = self.DnCWrapper.create_envs()
        elif self.curriculum.default():
            teacher = GMMSampler(self.curriculum_params['TARGET_MEAN'],
                                  self.curriculum_params['TARGET_VARIANCE'],
                                  self.curriculum_params['TARGET_PRIORS'],
                                       (self.curriculum_params['LOWER_CONTEXT_BOUNDS'],self.curriculum_params['UPPER_CONTEXT_BOUNDS']))


            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.self_paced() or self.curriculum.self_paced_v2():
            teacher = self.create_self_paced_teacher()
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.random():
            teacher = UniformSampler(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        else:
            raise RuntimeError("Invalid learning type")

        return env, DummyVecEnv([lambda: env])

    def create_learner_params(self):
        return dict(common=dict(gamma=self.DISCOUNT_FACTOR, seed=self.seed, verbose=0,
                                policy_kwargs= self.policy) ,
                    trpo=dict(n_steps=self.STEPS_PER_ITER, gae_lambda=self.LAM, tensorboard_log=self.log_dir),
                    ppo=dict(n_steps=self.STEPS_PER_ITER, n_epochs=8, batch_size=64, gae_lambda=self.LAM,# max_grad_norm=None,
                              vf_coef=1.0, clip_range_vf=None, ent_coef=0., tensorboard_log=self.log_dir),
                    sac=dict(learning_rate=3e-4, buffer_size=10000, learning_starts=500, batch_size=64,
                             train_freq=5, target_entropy="auto", tensorboard_log=self.log_dir),)

    def create_experiment(self):
        timesteps = self.epochs * self.STEPS_PER_ITER
        # print(self.dnc)
        if self.dnc:
            env = Monitor(gym.make("ContextualPointMass2D-v1"), info_keywords=('is_success',))
            parameters = self.create_learner_params()
            if self.learner.ppo():
                from deep_sprl.util.funcs import DnCPPO as DnC
            else:
                from  deep_sprl.util.modified_algs import DnCTRPO as DnC
            model = DnC(env, curriculum=self.curriculum_params, params=self.learner_params,curriculum_type=self.curriculum, N=self.N,dnc_args= self.dnc_args, **parameters)
            interface = model.get_interface()
            env = model.create_envs()

            callback_params = []
            # print(env)
            for idx ,e in enumerate(env):
                if isinstance(e.teacher, SelfPacedTeacher) or isinstance(e.teacher, SelfPacedTeacherV2):
                    sp_teacher = e.teacher
                else:
                    sp_teacher = None
                callback_params.append({"learner": interface[idx], "env_wrapper": e, "sp_teacher": sp_teacher, "n_inner_steps": 1,
                               "n_offset": self.OFFSET[Learner.PPO], "save_interval": 5,
                               "step_divider": self.STEPS_PER_ITER,
                                        "cluster":model.cluster, "idx": idx})
        else:
            env, vec_env = self.create_environment(evaluation=False)
            model, interface = self.learner.create_learner(vec_env, self.create_learner_params())

            if isinstance(env.teacher, SelfPacedTeacher) or isinstance(env.teacher, SelfPacedTeacherV2):
                sp_teacher = env.teacher
            else:
                sp_teacher = None

            callback_params = {"learner": interface, "env_wrapper": env, "sp_teacher": sp_teacher, "n_inner_steps": 1,
                               "n_offset": self.OFFSET[self.learner], "save_interval": 5,
                               "step_divider": self.STEPS_PER_ITER}# if self.learner.sac() else 1}
        return model, timesteps, callback_params

    def create_self_paced_teacher(self):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        if self.curriculum.self_paced():
            alpha_fn = PercentageAlphaFunction(self.ALPHA_OFFSET[self.learner], self.ZETA[self.learner])
            return SelfPacedTeacher(self.curriculum_params['TARGET_MEAN'].copy(), self.curriculum_params['TARGET_VARIANCE'].copy(), self.INITIAL_MEAN.copy(),
                                    self.INITIAL_VARIANCE.copy(), bounds, alpha_fn, max_kl=self.MAX_KL,
                                    std_lower_bound=self.STD_LOWER_BOUND, kl_threshold=self.KL_THRESHOLD,
                                    use_avg_performance=True)
        else:
            return SelfPacedTeacherV2(self.curriculum_params['TARGET_MEAN'].copy(), self.curriculum_params['TARGET_VARIANCE'].copy(), self.INITIAL_MEAN.copy(),
                                      self.INITIAL_VARIANCE.copy(), bounds, self.PERF_LB[self.learner],
                                      max_kl=self.MAX_KL, std_lower_bound=self.STD_LOWER_BOUND.copy(),
                                      kl_threshold=self.KL_THRESHOLD, use_avg_performance=True)

    def get_env_name(self):
        return "point_mass_2d"

    def evaluate_learner(self, path):


        model_load_path = os.path.join(path, "model.zip")

        model = self.learner.load_for_evaluation(model_load_path, self.vec_eval_env, self.dnc)
        # episode_stat= np.zeros(50)
        # self.vec_eval_env.envs.
        for i in range(0, 50):
            obs = self.vec_eval_env.reset()
            done = False
            while not done:
                action = model.step(obs, state=None, deterministic=False)
                obs, rewards, done, infos = self.vec_eval_env.step(action)

        return self.eval_env.get_episodes_statistic()
