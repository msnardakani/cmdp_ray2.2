
from gymnasium.wrappers import TimeLimit
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from envs.contextual_env import CtxDictWrapper, ctx_visibility, exp_group, GMMCtxEnvWrapper, SquashedGMMCtxEnvWrapper
from envs.bipedal_walker import BipedalWalkerCtx

from envs.point_mass_2d import PointMassEnv3D

from envs.lunar_lander import LunarLanderCtx, LunarLanderCtx5D
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
        max_steps = 100
        parameters = []
        env_config_pars = list()
        env_config_pars.append({'target_mean': np.array([2.5, .6, 0.]),
                                'target_var': np.square([4e-3, 3.75e-3, 2e-3]),
                                'init_mean': np.array([0, 4, 2]),
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

        env_config_pars.append({'target_mean': np.array([2.6, .7, 0.]),
                                'target_var': np.square([.03, .02, .01]),
                                'init_mean': np.array([0, 4, 2]),
                                'init_var': np.square([2, 1.87, 1]),
                                'prior': None})  # 7 WRR

        env_config_pars.append({'target_mean': np.array([2.5, .7, .1]),
                                'target_var': np.square([1, .03, .01]),
                                'init_mean': np.array([0, 4, 2]),
                                'init_var': np.square([2, 1.875, 1]),
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
        env_creator = lambda config, ctx_mode: GMMCtxEnvWrapper(
            TimeLimit(PointMassEnv3D(context=np.array([3, .5, .1])), max_episode_steps=max_steps),
            ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_mode=ctx_mode, **config)
        return env_creator, ctx, max_steps, env_config_pars

    elif env.lower() == 'PointMass2D'.lower():
        from envs.point_mass_2d import PointMassEnv

        env_config_pars = list()
        env_config_pars.append({'target_mean': np.array([2.5, .6]),
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

        env_config_pars.append({'target_mean': np.array([2.5, .75]),
                                'target_var': np.square([.1, .01]),
                                'init_mean': np.array([0, 3]),
                                'init_var': np.square([1, 1]),
                                'prior': None})  # 7 WRR
        env_config_pars.append({'target_mean': np.array([2.5, .75]),
                                'target_var': np.square([.5, .1]),
                                'init_mean': np.array([0, 3]),
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

        # env_config_pars.append({'target_mean': np.array([2.5, .75]),
        #                         'target_var': np.square([.05, .01]),
        #                         'init_mean': np.array([0, 3]),
        #                         'init_var': np.square([.1, .1]),
        #                         'prior': None}) #11  NRR
        ctx_lb = np.array([-4, .5])
        ctx_ub = np.array([4, 6])
        std_lb = np.array([0.2, 0.1875])
        max_steps = 100
        ctx = ContextSpec(lb = ctx_lb, ub=ctx_ub, std_lb=std_lb)

        env_creator = lambda config, ctx_mode, rew_coeff: GMMCtxEnvWrapper(
            TimeLimit(PointMassEnv(context=np.array([3, .5]), room_rew_coeff=rew_coeff), max_episode_steps=max_steps),
            ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_mode=ctx_mode, **config)


        return env_creator, ctx, max_steps, env_config_pars


    elif env.lower() == 'LunarLander'.lower():
        env_config_pars = list()
        env_config_pars.append({'target_mean': np.array([-10., 15., 1.5]),
                                'target_var': np.square([0.01, 0.001, 0.01]),
                                'init_mean': np.array([-10.,15., 1.5]),
                                'init_var': np.square([0.01, 0.01, 0.01]),
                                'prior': None})  # 0  default

        env_config_pars.append({'target_mean': np.array([-10., 15., 1.5]),
                                'target_var': np.square([0.01, 0.01, 0.01]),
                                'init_mean': np.array([-10., 5., .1]),
                                'init_var': np.square([2, 2, 0.3]),
                                'prior': None})  # 1  default
        env_config_pars.append({'target_mean': np.array([-10., 15., 1.5]),
                                'target_var': np.square([1, 1, 0.3]),
                                'init_mean': np.array([-8., 10., .5]),
                                'init_var': np.square([0.001, 0.001, 0.001]),
                                'prior': None}) # 2  default
        env_config_pars.append({'target_mean': np.array([-10., 15., 1.5]),
                                'target_var': np.square([1, 2, 0.2]),
                                'init_mean': np.array([-10., 5., .1]),
                                'init_var': np.square([0.01, 0.01, 0.01]),
                                'prior': None}) # 3  default
        env_config_pars.append({'target_mean': np.array([-10., 2., .1]),
                                'target_var': np.square([0.01, 0.01, 0.01]),
                                'init_mean': np.array([-7., 10., 1]),
                                'init_var': np.square([2, 3, 0.3]),
                                'prior': None}) # 4  default
        env_config_pars.append({'target_mean': np.array([-9., 14., 1.5]),
                                'target_var': np.square([2, 3, 0.3]),
                                'init_mean': np.array([-10., 2., .12]),
                                'init_var': np.square([.1, .1, 0.01]),
                                'prior': None}) # 5  hard target
        
        env_config_pars.append({'target_mean': np.array([-10., 15., 1.5]),
                                'target_var': np.square([1, 3, 0.3]),
                                'init_mean': np.array([-7., 10, 1]),
                                'init_var': np.square([2, 4, 0.4]),
                                'prior': None}) # 6  hard target
        ctx_lb = np.array([-12., 0.0, 0.0, ])
        ctx_ub = np.array([-2., 20.0, 2.0, ])
        std_lb = np.array([0.1, 0.1, .1])
        max_steps = 400
        ctx = ContextSpec(lb=ctx_lb, ub=ctx_ub, std_lb=std_lb)
        env_creator = lambda config, ctx_mode, continuous: GMMCtxEnvWrapper(
            TimeLimit(LunarLanderCtx(context=np.array([-10., 15., 1.5]), continuous=continuous),
                      max_episode_steps=max_steps),
            ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_mode=ctx_mode, **config)
        return env_creator, ctx, max_steps, env_config_pars
    elif env.lower() == 'LunarLander5D'.lower():
        env_config_pars = list()
        env_config_pars.append({'target_mean': np.array([-10., 15., 1.5, 30, 5]),
                                'target_var': np.square([0.01, 0.001, 0.01, 1, .1]),
                                'init_mean': np.array([-10.,15., 1.5, 30, .6]),
                                'init_var': np.square([0.01, 0.01, 0.01, 0.01, 0.01]),
                                'prior': None})  # 0  default

        env_config_pars.append({'target_mean': np.array([-9., 14., 1.5, 25,2]),
                                'target_var': np.square([2, 3, 0.3, 3,1]),
                                'init_mean': np.array([-10., 2., .12, 35, 1]),
                                'init_var': np.square([.1, .1, 0.01, .01, 0.1]),
                                'prior': None}) # 1  hard target
        
        env_config_pars.append({'target_mean': np.array([-10., 15., 1.5, 20, 1]),
                                'target_var': np.square([1, 3, 0.3, 2, .1]),
                                'init_mean': np.array([-7., 10, 1, 40, 1]),
                                'init_var': np.square([2, 4, 0.4, 0.1, 0.1]),
                                'prior': None}) # 2  hard target
        env_config_pars.append({'target_mean': np.array([-9., 14., 1.5, 20, .5]),
                                'target_var': np.square([2., 3, 0.3, 3, .1]),
                                'init_mean': np.array([-7., 10, 1, 40, 1]),
                                'init_var': np.square([2, 4, 0.4, 0.1, 0.1]),
                                'prior': None}) # 3  hard target
        env_config_pars.append({'target_mean': np.array([-9.8, 15., 1.5, 25, .4]),
                                'target_var': np.square([1., 2.5, 0.25, 4, .15]),
                                'init_mean': np.array([-7., 10, 1, 40, 1]),
                                'init_var': np.square([2, 4, 0.4, 0.1, 0.1]),
                                'prior': None}) # 4  hard target
        env_config_pars.append({'target_mean': np.array([-9., 14., 1.5, 27, .4]),
                                'target_var': np.square([1.5, 2., 0.24, 6, .15]),
                                'init_mean': np.array([-7., 10, 1, 40, 1]),
                                'init_var': np.square([2, 4, 0.4, 0.1, 0.1]),
                                'prior': None}) # 5  hard target
        env_config_pars.append({'target_mean': np.array([-9., 18., 1.5, 13, .5]),
                                'target_var': np.square([2, 2., 0.4, 5, .2]),
                                'init_mean': np.array([-7., 10, 1, 40, 1]),
                                'init_var': np.square([2, 4, 0.4, 0.1, 0.1]),
                                'prior': None}) # 6  hard target
        ctx_lb = np.array([-12., 0.0, 0.0, 10, .1])
        ctx_ub = np.array([-2., 20.0, 2.0, 50, 10])
        std_lb = np.array([0.1, 0.1, .1, 0.1,0.1])
        max_steps = 1000
        ctx = ContextSpec(lb=ctx_lb, ub=ctx_ub, std_lb=std_lb)
        env_creator = lambda config, ctx_mode, continuous: GMMCtxEnvWrapper(
            TimeLimit(LunarLanderCtx5D(context=np.array([-10., 15., 1.5, 30, 5]), continuous=continuous),
                      max_episode_steps=max_steps),
            ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_mode=ctx_mode, **config)
        return env_creator, ctx, max_steps, env_config_pars
    elif env.lower()=='BipedalWalker'.lower():
        env_config_pars = list()
        env_config_pars.append({'target_mean' : np.array([80.0, 4.0, 6.0, 5.33, -10.0, 0.0, 2.5]),
                                'target_var': np.square([1, 0.1, .1, 0.1, 0.1, 0.1, 0.1]),
                                'init_mean':np.array([100.0, 7.0, 7.0, 10.0, -10.0, 0.0, 3]),
                                'init_var': np.square([20.0, 2.0, 2.0, 3.0, 2.0, 2. , 2.]),
                                'prior': None}) # 0  default

        env_config_pars.append({'target_mean' : np.array([80.0, 4.0, 6.0, 2.33, -10.0, 0.0, 2.5]),
                                'target_var': np.square([5, 0.5, .5, 0.1, 2., 1., 0.5]),
                                'init_mean':np.array([100.0, 8.0, 8.0, 10., -10.0, 0.0, 5]),
                                'init_var': np.square([20.0, 2.0, 2.0, 3.0, 2.0, 2. , 2.]),
                                'prior': None})   #1
        env_config_pars.append({'target_mean' : np.array([70.0, 4.0, 6.0, 2., -10.0, 0.0, 2.5]),
                                'target_var': np.square([7, 1., 1., 1., 2., 2., 1.]),
                                'init_mean':np.array([100.0, 8.0, 8.0, 10.0, -10.0, 0.0, 5]),
                                'init_var': np.square([20.0, 2.0, 2.0, 3.0, 2.0, 2. , 2.]),
                                'prior': None})   #2
        env_config_pars.append({'target_mean' : np.array([100.0, 6.0, 8.0, 2., -10.0, 0.0, 5]),
                                'target_var': np.square([20, 2., 2., 1., 2., 3., 2.]),
                                'init_mean':np.array([100.0, 8.0, 8.0, 10.0, 0.0, -10.0, 5]),
                                'init_var': np.square([20.0, 2.0, 2.0, 3.0, 2.0, 2. , 2.]),
                                'prior': None})   #3
        env_config_pars.append({'target_mean' : np.array([100.0, 8.0, 8.0, 10., -10.0, 0.0, 5]),
                                'target_var': np.square([30, 3., 3., 3., 3., 3., 2.]),
                                'init_mean':np.array([100.0, 7.0, 7.0, 10.0, -10.0, 0.0, 3]),
                                'init_var': np.square([20.0, 2.0, 2.0, 3.0, 2.0, 2. , 2.]),
                                'prior': None})   #4
        env_config_pars.append({'target_mean' : np.array([90.0, 6.0, 8.0, 2.5, -10.0, 0.0, 5]),
                                'target_var': np.square([10, 2., 2., 1., 2., 3., 2.]),
                                'init_mean':np.array([80.0, 4.0, 6.0, 5.33, -10.0, 0.0, 2.5]),
                                'init_var': np.square([1, 0.1, .1, 0.1, 0.1, 0.1, 0.1]),
                                'prior': None}) 
        
        ctx_lb = np.array([0.01, 0.01, 0.01, 0.01 , -20.0, -10.0, 0.0])
        ctx_ub = np.array([200., 15.0, 15.0, 20.0, -0.01, 10.0, 10.0])
        std_lb = np.array([.1, .1, .1, .1, 0.1, 0.1, 0.1])

        ctx_dim = 6


        max_steps = 1600
        ctx = ContextSpec(lb=ctx_lb, ub=ctx_ub, std_lb=std_lb)
        env_creator = lambda config, ctx_mode: SquashedGMMCtxEnvWrapper(
                TimeLimit(BipedalWalkerCtx(context=np.array([80., 4., 6., 5.33, 0., -10., 2.5])), max_episode_steps=max_steps),
                ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_mode=ctx_mode, **config)
        return env_creator, ctx, max_steps, env_config_pars


    





