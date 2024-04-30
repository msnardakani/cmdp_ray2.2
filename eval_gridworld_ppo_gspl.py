# import itertools
#
import traceback
import itertools

from envs.gridworld_contextual import TaskSettableGridworld, coordinate_transform

import os
import pickle
# from gymnasium.wrappers import TimeLimit

from envs.contextual_env import CtxDictWrapper, ctx_visibility, exp_group, GMMCtxEnvWrapper, DiscreteCtxDictWrapper, \
    find_category_idx
import numpy as np
# from ray.rllib.policy.policy import Policy
from distral.distral_ppo_torch_model import DistralCentralTorchModel, DistralTorchModel
from ray.rllib.models import ModelCatalog
# import json
# from envs.point_mass_2d import PointMassEnv
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
size0= (11, 11)
corridor0 = (1, 3)
max_steps = 16




### V11.4.1
# embedding_dim = 2
# embedding_map = [{0: (i, j), 1: (9, 9)} for i, j in itertools.product(range(1, 10), range(1, 4))] + \
#                    [{0: (i, j), 1: (9, 9)} for i, j in itertools.product(range(1, 5), range(7, 10))] # LR destination maps
# embedding_transform = [np.array([coordinate_transform(v[0], size0), coordinate_transform(v[1], size0)]) for v in
#                            embedding_map]
# embeddings = np.array(
#     [[2 * v[0][0] / size0[0] - 1, 2 * v[0][1] / size0[1] - 1, ]#2 * v[1][0] / size0[0] - 1, 2 * v[1][1] / size0[1] - 1]
#      for v in
#      embedding_map], dtype=np.float64)

### V11.4.2
embedding_dim = 4
points = [(i, j) for i, j in itertools.product(range(1, 10), range(1, 4))] + [(i, j) for i, j in
                                                                              itertools.product(range(1, 10),
                                                                                                range(7, 10))]


embedding_map = [{0: start, 1: dst} for start, dst in itertools.product(points, points) if start != dst]
embedding_transform = [np.array([coordinate_transform(v[0], size0), coordinate_transform(v[1], size0)]) for v in
                       embedding_map]

embeddings = np.array(
    [[2 * v[0][0] / size0[0] - 1, 2 * v[0][1] / size0[1] - 1, 2 * v[1][0] / size0[0] - 1, 2 * v[1][1] / size0[1] - 1]
     for v in
     embedding_map], dtype=np.float64)


ctx_lb = -2 * np.ones_like(embeddings[0])
ctx_ub = 2 * np.ones_like(embeddings[0])

config0 = {'size': size0,
                        'corridor': corridor0,
                        'region': {0: (1, 1), 1: (-2, -2)}}
env_creator = lambda config, ctx_mode: GMMCtxEnvWrapper(DiscreteCtxDictWrapper(TimeLimitRewardWrapper(
    TaskSettableGridworld(config0), max_episode_steps=max_steps, key='distance'),
    key='region', embedding_dim=embedding_dim, embedding_map=embedding_transform, embeddings=embeddings
), ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_mode=ctx_mode, **config)
# learner_0 = Policy.from_checkpoint('./results/GridWorld/V10.1.5/Set0/ctx_hid/baseline/default/PPO_2x64/PPO_MADnCEnv_b16be_00000_0_grad_clip=100,seed=0_2024-01-19_11-06-24/checkpoint_000000/policies/learner_0')

sets = dict(
    # Set0=[0,],
    # # Set1=[0, 1, 2],
    # # Set2=[0, 3],
    # # Set3=[0, 1, 2, 3, 4, 5],
    Set4=[1,],
    Set7=[1,],


)

parameters = []
env_config_pars = list()

env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (1, 1), 1: (9, 9)}}  # 0 UL -> LR
                       )
env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (5, 1), 1: (9, 9)}}  # 1 L -> LR
                       )
env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (1, 1), 1: (5, 9)}})  # 2 UL -> R

env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (9, 9), 1: (1, 1)}}  # 3 LR -> UL
                       )
env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (9, 9), 1: (5, 1)}})  # 4 LR -> L

env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (5, 9), 1: (1, 1)}}  # 5 R -> UL
                       )

env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (1, 9), 1: (9, 9)}}  # 6 UR -> LR
                       )
env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (9, 1), 1: (9, 9)}})  # 7 LL -> LR

env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (1, 9), 1: (1, 1)}}  # 8 UR -> UL
                       )
env_config_pars.append({'size': size0,
                        'corridor': corridor0,
                        'region': {0: (9, 1), 1: (1, 1)}}  # 9 LL -> UL
                       )


log_dir = './results/GridWorld/V11.4.2'
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
    print("Previously saved file not found,\nRestarting the Evaluation")
EV.analyse(env_creator=env_creator, func=func, duration = 100, n_env=20, config_trans = config_trans)
EV.save_to_file('result.pkl')
EV.report()
for setup, env_idx in sets.items():
    config_lst = [{ 'curriculum': 'default',
                    'target_mean': embeddings[find_category_idx(embedding_map, env_config_pars[idx]['region']), :],
                     'target_var': np.square([.01, ] * embedding_dim),
                     'init_mean': np.zeros(embedding_dim),
                     'init_var': np.square([1, ] * embedding_dim),
                     'prior': None,

                     } for idx in env_idx]
    try:
        with open(os.path.join(EV.log_dir,'report',setup,'extr_eval.pkl'), 'rb') as file:
        # Call load method to deserialze
            extr_result = pickle.load(file)
        extr_result = EV.update_extr_results(extr_result=extr_result, env_creator=env_creator,
                                             func=func,
                                             duration=100, n_env=20
                                             )


    except FileNotFoundError as e:
        print(f'External evaluation result could not be loaded,\nError Message: {e}\nRestarting the external evaluations')

        try:
            extr_result = EV.get_extr_results( setup, env_creator, config_lst, func=func, duration=100, n_env=10,)

        except Exception as e:

            print(f'Error during external {setup} evaluation,\nError Message: {e}\n{traceback.format_exc()}')
            continue


    EV.report_extr_eval(extr_results=extr_result)
# print(EV.results)
