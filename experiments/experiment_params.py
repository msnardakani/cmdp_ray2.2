
from gymnasium.wrappers import TimeLimit
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from envs.contextual_env import CtxDictWrapper, ctx_visibility, exp_group, GMMCtxEnvWrapper

from envs.point_mass_2d import PointMassEnv3D


from ray.rllib.utils.framework import try_import_torch

import numpy as np


torch, nn = try_import_torch()


# torch.autograd.set_detect_anomaly(True)
# torch.set_default_tensor_type(torch.DoubleTensor)
# ctx_norm= None
class ContextSpec:
    def __init__(self, lb, ub, std_lb=0.1):
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)
        if isinstance(std_lb, float):

            self.std_lb = std_lb * np.ones_like(lb)
        else:
            self.std_lb = std_lb


def ExperimentSetup(env):
    if env.lower() == 'PointMass3D'.lower():
        ctx_lb = np.array([-4, .5, 0])
        ctx_ub = np.array([4, 6, 4])
        std_lb = np.array([0.2, 0.1875, .1])
        max_steps = 128
        parameters = []
        env_config_pars = list()
        env_config_pars.append({'target_mean': np.array([2.5, .5, 0.]),
                                'target_var': np.square([4e-3, 3.75e-3, 2e-3]),
                                'init_mean': np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 1]),
                                'prior': None})  # 0 NRR default
        env_config_pars.append({'target_mean': np.array([3, 2, 0.]),
                                'target_var': np.square([4e-3, 3.75e-3, 2e-3]),
                                'init_mean': np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 1]),
                                'prior': None})  # 1  WRR

        env_config_pars.append({'target_mean': np.array([1, .5, 0]),
                                'target_var': np.square([4e-3, 3.75e-3, 2e-3]),
                                'init_mean': np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 1]),
                                'prior': None})  # 2 NR

        env_config_pars.append({'target_mean': np.array([[2.5, .5], [-3, 2]]),
                                'target_var': np.square([[.05, .01], [.5, .2]]),
                                'init_mean': np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 1]),
                                'prior': [.5, .5]})  # 3 GMM

        env_config_pars.append({'target_mean': np.array([-3, 2, 0.1]),
                                'target_var': np.square([.5, .2, .2]),
                                'init_mean': np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 1]),
                                'prior': None})  # 4 WLL

        env_config_pars.append({'target_mean': np.array([2.5, .5, 0.1]),
                                'target_var': np.square([.05, .01, .05]),
                                'init_mean': np.array([0, 6, 2]),
                                'init_var': np.square([2, 1.875, 1]),
                                'prior': None})  # 5 NRR

        env_config_pars.append({'target_mean': np.array([[2, 1], [-2, 1]]),
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
                                'init_var': np.square([2, 1.875, 2]),
                                'prior': None})  # 10 NRR
        ctx = ContextSpec(lb = ctx_lb, ub=ctx_ub, std_lb=std_lb)
        env_creator = lambda config, ctx_mode, room_rew_coeff: GMMCtxEnvWrapper(
            TimeLimit(PointMassEnv3D(context=np.array([3, .5, .1]), rew_coeff= room_rew_coeff), max_episode_steps=max_steps),
            ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_mode=ctx_mode, **config)
        return env_creator, ctx, max_steps, env_config_pars

    elif env.lower() == 'PointMass2D'.lower():
        from envs.point_mass_2d import PointMassEnv

        env_config_pars = list()
        env_config_pars.append({'target_mean': np.array([2.5, .5]),
                                'target_var': np.square([4e-3, 3.75e-3]),
                                'init_mean': np.array([0, 6]),
                                'init_var': np.square([2, 1.875]),
                                'prior': None})  # 0 NRR default
        env_config_pars.append({'target_mean': np.array([3, 2]),
                                'target_var': np.square([4e-3, 3.75e-3]),
                                'init_mean': np.array([0, 6]),
                                'init_var': np.square([2, 1.875]),
                                'prior': None})  # 1  WRR

        env_config_pars.append({'target_mean': np.array([1, .5]),
                                'target_var': np.square([4e-3, 3.75e-3]),
                                'init_mean': np.array([0, 6]),
                                'init_var': np.square([2, 1.875]),
                                'prior': None})  # 2 NR

        env_config_pars.append({'target_mean': np.array([[2.5, .5], [-3, 2]]),
                                'target_var': np.square([[.05, .01], [.5, .2]]),
                                'init_mean': np.array([0, 6]),
                                'init_var': np.square([2, 1.875]),
                                'prior': [.5, .5]})  # 3 GMM

        env_config_pars.append({'target_mean': np.array([-3, 2]),
                                'target_var': np.square([.5, .2]),
                                'init_mean': np.array([0, 6]),
                                'init_var': np.square([2, 1.875]),
                                'prior': None})  # 4 WLL

        env_config_pars.append({'target_mean': np.array([2.5, .5]),
                                'target_var': np.square([.05, .01]),
                                'init_mean': np.array([0, 6]),
                                'init_var': np.square([2, 1.875]),
                                'prior': None})  # 5 NRR

        env_config_pars.append({'target_mean': np.array([[2, 1], [-2, 1]]),
                                'target_var': np.square([[1, .5], [1, .5]]),
                                'init_mean': np.array([0, 6]),
                                'init_var': np.square([2, 1.875]),
                                'prior': [.5, .5]})  # 6 GMM

        env_config_pars.append({'target_mean': np.array([2, 1]),
                                'target_var': np.square([1, .5]),
                                'init_mean': np.array([0, 6]),
                                'init_var': np.square([2, 1.875]),
                                'prior': None})  # 7 WRR
        env_config_pars.append({'target_mean': np.array([-2, 1]),
                                'target_var': np.square([1, .5]),
                                'init_mean': np.array([0, 6]),
                                'init_var': np.square([2, 1.875]),
                                'prior': None})  # 8 WLL
        env_config_pars.append({'target_mean': np.array([2.5, 2]),
                                'target_var': np.square([.05, .01]),
                                'init_mean': np.array([0, 6]),
                                'init_var': np.square([2, 1.875]),
                                'prior': None})  # 9 NRR
        env_config_pars.append({'target_mean': np.array([1, .5]),
                                'target_var': np.square([.05, .01]),
                                'init_mean': np.array([0, 6]),
                                'init_var': np.square([2, 1.875]),
                                'prior': None})  # 10 NRR
        ctx_lb = np.array([-4, .5])
        ctx_ub = np.array([4, 6])
        std_lb = np.array([0.2, 0.1875])
        max_steps = 128
        ctx = ContextSpec(lb = ctx_lb, ub=ctx_ub, std_lb=std_lb)

        env_creator = lambda config, ctx_mode, rew_coeff: GMMCtxEnvWrapper(
            TimeLimit(PointMassEnv(context=np.array([3, .5]),room_rew_coeff = rew_coeff), max_episode_steps=max_steps),
            ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_mode=ctx_mode, **config)


        return env_creator, ctx, max_steps, env_config_pars

