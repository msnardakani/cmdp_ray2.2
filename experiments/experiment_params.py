import numpy as np
import copy
import os
from pathlib import Path

from gymnasium.spaces import flatten_space
from gymnasium.wrappers import TimeLimit
from ray import air, tune, train
from ray.rllib.algorithms.ppo import PPO, PPOConfig
import itertools

from distral.distral_ppo import DistralPPO
from distral.distral_ppo_torch_model import DistralCentralTorchModel, DistralTorchModel
from envs.contextual_env import CtxDictWrapper, ctx_visibility, exp_group, GMMCtxEnvWrapper
from utils.evaluation_fn import DnCCrossEvalSeries, CL_report
from envs.point_mass_2d import PointMassEnv3D
from utils.self_paced_callback import MACL
import ray

from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_torch

from envs.contextual_env import make_multi_agent_divide_and_conquer, make_multi_agent
import numpy as np

from utils.weight_sharing_models import FC_MLP
from utils.ma_policy_config import gen_ppo_distral_policy, dummy_policy_mapping_fn, policy_mapping_fn

torch, nn = try_import_torch()
# torch.autograd.set_detect_anomaly(True)
# torch.set_default_tensor_type(torch.DoubleTensor)
# ctx_norm= None

def ExperimentSetup(env):
    if env.lower() =='PointMass3D'.lower():
        ctx_lb = np.array([-4, .5, 0])
        ctx_ub = np.array([4, 6, 4])
        std_lb = np.array([0.2, 0.1875, .1])
        max_steps = 128
        ctx_dim = 3
        parameters = []
        env_config_pars = list()
        env_config_pars.append({'target_mean' : np.array([2.5, .5, 0.]),
                                'target_var': np.square([4e-3, 3.75e-3,  2e-3]),
                                'init_mean':np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 1]),
                                'prior': None}) # 0 NRR default
        env_config_pars.append({'target_mean' : np.array([3, 2, 0.]),
                                'target_var': np.square([4e-3, 3.75e-3,  2e-3]),
                                'init_mean':np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 1]),
                                'prior': None}) #1  WRR

        env_config_pars.append({'target_mean': np.array([1, .5, 0]),
                                'target_var': np.square([4e-3, 3.75e-3,  2e-3]),
                                'init_mean': np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 1]),
                                'prior': None})  # 2 NR

        env_config_pars.append({'target_mean': np.array([[2.5, .5], [-3, 2]]),
                                'target_var': np.square([[.05, .01], [.5, .2]]),
                                'init_mean': np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 1]),
                                'prior': [.5, .5]})  # 3 GMM

        env_config_pars.append({'target_mean': np.array( [-3, 2, 0.1]),
                                'target_var': np.square([.5, .2, .2]),
                                'init_mean': np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 1]),
                                'prior': None})  # 4 WLL

        env_config_pars.append({'target_mean': np.array( [2.5, .5, 0.1]),
                                'target_var': np.square([.05, .01, .05]),
                                'init_mean': np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 1 ]),
                                'prior': None})  # 5 NRR

        env_config_pars.append({'target_mean': np.array( [[2, 1], [-2, 1]]),
                                'target_var': np.square([[1, .5], [1, .5]]),
                                'init_mean': np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 1]),
                                'prior': [.5, .5]})  # 6 GMM

        env_config_pars.append({'target_mean': np.array([2, 1, 0.5]),
                                'target_var': np.square([1, .5, .5]),
                                'init_mean': np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 2]),
                                'prior': None})  # 7 WRR
        env_config_pars.append({'target_mean': np.array([-2, 1, 0.5]),
                                'target_var': np.square([1, .5, .5]),
                                'init_mean': np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 2]),
                                'prior': None})  # 8 WLL
        env_config_pars.append({'target_mean': np.array([2.5, 2, 0.1]),
                                'target_var': np.square([.05, .01, 0.05]),
                                'init_mean': np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 2]),
                                'prior': None})  # 9 NRR
        env_config_pars.append({'target_mean': np.array([1, .5, 0.1]),
                                'target_var': np.square([.05, .01, 0.05]),
                                'init_mean': np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875,2]),
                                'prior': None})  # 10 NRR
        env_creator = lambda config, ctx_mode: GMMCtxEnvWrapper(
                TimeLimit(PointMassEnv3D(context=np.array([3, .5, .1])), max_episode_steps=max_steps),
                ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_mode=ctx_mode, **config)
        return env_creator, ctx_lb, ctx_ub, max_steps, env_config_pars
