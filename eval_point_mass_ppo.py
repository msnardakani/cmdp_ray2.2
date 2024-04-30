# import itertools
#
# from ray.rllib.policy.policy import Policy
# import os
import os
import pickle
from gymnasium.wrappers import TimeLimit

from envs.contextual_env import CtxDictWrapper, ctx_visibility, exp_group, GMMCtxEnvWrapper
import numpy as np
# from ray.rllib.policy.policy import Policy
from distral.distral_ppo_torch_model import DistralCentralTorchModel, DistralTorchModel
from ray.rllib.models import ModelCatalog
# import json
from envs.point_mass_2d import PointMassEnv
from envs.wrappers import TimeLimitRewardWrapper
from utils.policy_visualization_evaluation import ExperimentEvaluation
ModelCatalog.register_custom_model(
    "central",
    DistralCentralTorchModel,
)

ModelCatalog.register_custom_model(
    "local",
    DistralTorchModel,
)

# checkpoint_dir = '../results/GridWorld/V10.1.5/Set0/ctx_hid/baseline/default/PPO_2x64/'
# learner_0 = Policy.from_checkpoint(
#     '../results/GridWorld/V10.1.5/Set0/ctx_hid/baseline/default/PPO_2x64/PPO_MADnCEnv_b16be_00000_0_grad_clip=100,seed=0_2024-01-19_11-06-24/checkpoint_000000/policies/distilled_policy')
ctx_lb = np.array([-4, .5])
ctx_ub = np.array([4, 6])
std_lb = np.array([0.2, 0.1875])
max_steps = 128# ctx_mode = 0
# env_creator = lambda config, ctx_mode: CtxDictWrapper(TimeLimitRewardWrapper(
#                 TaskSettableGridworld(config),max_episode_steps=max_steps, key = 'distance'),
#                 key='region', ctx_visible=ctx_mode
#                 )
env_creator = lambda config, ctx_mode: GMMCtxEnvWrapper(
                TimeLimit(PointMassEnv(context=np.array([3, .5])), max_episode_steps=max_steps),
                ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_mode=ctx_mode, **config)
# learner_0 = Policy.from_checkpoint('./results/GridWorld/V10.1.5/Set0/ctx_hid/baseline/default/PPO_2x64/PPO_MADnCEnv_b16be_00000_0_grad_clip=100,seed=0_2024-01-19_11-06-24/checkpoint_000000/policies/learner_0')
sets = dict(
    # Set0=[0, 0],
    # Set1=[0, ],  # left and right tasks from more complicated ones
    # Set2=[0, 1, 2, ],
    # Set3=[7, 8, ],
    # Set4=[7, 5, 2, ],
    # Set5=[4, 7, 5]
)

parameters = []
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
log_dir = './results/PointMass2D/V10.4.0'
# env = env_creator(config)
# obs = env.reset()
# action = learner_0.compute_single_action(obs[0])
# obs = env.step(action[0])
# distill_dist = learner_0.dist_class(learner_0.model.distill_out())
#
# print(distill_dist.sample()
#       )
func = lambda obs: np.where(obs[2]==1, 1, 0)

def config_trans(config):
    config_out = dict()
    config_out['curriculum'] = 'default'
    config_out['target_mean'] = np.array(config['target_mean'])
    config_out['target_var'] = np.array(config['target_var'])
    config_out['prior'] = None
    return config_out

EV = ExperimentEvaluation(experiment_dir=log_dir )
try:
    EV.load_from_file('result.pkl')
except Exception as e:
    print("file not found starting over")
EV.analyse(env_creator=env_creator, func=func, duration = 100, n_env=20, config_trans = config_trans)
EV.save_to_file('result.pkl')
EV.report()
for setup, idx in sets.items():
    config_lst = [env_config_pars[i] for i in idx]
    try:
        with open(os.path.join(EV.log_dir,'report',setup,'extr_eval.pkl'), 'rb') as file:
        # Call load method to deserialze
            extr_result = pickle.load(file)
        extr_result = EV.update_extr_results(extr_result=extr_result, env_creator=env_creator,
                                             func=func,
                                             duration=100, n_env=20
                                             )


    except Exception:
        print('could not load file')
        extr_result = EV.get_extr_results( setup, env_creator, config_lst, func=None, duration=100, n_env=10,)

    EV.report_extr_eval(extr_results=extr_result)
# print(EV.results)
