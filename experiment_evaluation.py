# import itertools
#
# from ray.rllib.policy.policy import Policy
# import os
import os
import pickle

from envs.gridworld_contextual import TaskSettableGridworld
# from gymnasium.wrappers import TimeLimit, RecordVideo, RecordEpisodeStatistics
from envs.contextual_env import CtxDictWrapper, ctx_visibility, exp_group
import numpy as np
# from ray.rllib.policy.policy import Policy
from distral.distral_ppo_torch_model import DistralCentralTorchModel, DistralTorchModel
from ray.rllib.models import ModelCatalog
# import json

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
max_steps = 15
# ctx_mode = 0
ctx_lb = np.array([-4, .5])
ctx_ub = np.array([4, 6])
std_lb = np.array([0.2, 0.1875])
max_steps = 128
env_creator = lambda config, ctx_mode: CtxDictWrapper(TimeLimitRewardWrapper(
                TaskSettableGridworld(config),max_episode_steps=max_steps, key = 'distance'),
                key='region', ctx_visible=ctx_mode
                )

# learner_0 = Policy.from_checkpoint('./results/GridWorld/V10.1.5/Set0/ctx_hid/baseline/default/PPO_2x64/PPO_MADnCEnv_b16be_00000_0_grad_clip=100,seed=0_2024-01-19_11-06-24/checkpoint_000000/policies/learner_0')

sets = dict(
    Set0=[0,],
    # Set1=[0, 1, 2],
    # Set2=[0, 3],
    # Set3=[0, 1, 2, 3, 4, 5],



)

parameters = []
env_config_pars = list()

size0 = (11, 11)

corridor0 = (1, 3)
env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (1, 1), 1: (-2, -2)}}  # 0 UL -> LR
                       )
env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (5, 1), 1: (-2, -2)}}  # 1 L -> LR
                       )
env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (1, 1), 1: (5, -2)}})  # 2 UL -> R

env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (-2, -2), 1: (1, 1)}}  # 3 LR -> UL
                       )
env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (-2, -2), 1: (5, 1)}})  # 4 LR -> L

env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (5, -2), 1: (1, 1)}}  # 5 R -> UL
                       )

env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (1, -2), 1: (-2, -2)}}  # 6 UR -> LR
                       )
env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (-2, 1), 1: (-2, -2)}})  # 7 LL -> LR

env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (1, -2), 1: (1, 1)}}  # 8 UR -> UL
                       )
env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (-2, 1), 1: (1, 1)}}  # 9 LL -> UL
                       )
log_dir = './results/GridWorld/V11.4.1'
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
    config_out['size'] = tuple(config['size'])
    config_out['corridor'] = tuple(config['corridor'])
    config_out['region'] = {0:tuple(config['region']['0']), 1: tuple(config['region']['1'])}
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

    EV.report_extr_eval(extr_results=extr_result)
# print(EV.results)
