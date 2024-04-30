import numpy as np
from envs.contextual_env import GMMCtxEnvWrapper,DiscreteCtxDictWrapper, find_category_idx
from gaussian_sprl.gaussian_selfpaced_teacher import GaussianSelfPacedTeacher
from utils.ma_policy_config import gen_ppo_distral_policy, dummy_policy_mapping_fn, policy_mapping_fn
from envs.wrappers import TimeLimitRewardWrapper
from envs.contextual_env import CtxDictWrapper, ctx_visibility, exp_group
import itertools
from envs.gridworld_contextual import TaskSettableGridworld, coordinate_transform

# size0 = (11, 11)
# max_steps = 32
# corridor0 = (1, 3)
# embedding_map = [{0:(i,j),1:(9,9)} for i,j in itertools.product(range(1,10),range(1,4))]+\
#                 [{0:(i,j),1:(9,9)} for i,j in itertools.product(range(1,5),range(7,10))]
# embedding_transform = [np.array([coordinate_transform(v[0], size0), coordinate_transform(v[1], size0)]) for v in embedding_map]
# config0 = {'size': size0,
#                         'corridor': corridor0,
#                         'region': {0: (1, 1), 1: (-2, -2)}}  # 0 UL -> LR
# ctx_mode = 1
# embedding_dim = 4
# embeddings = np.random.randn(len(embedding_map), embedding_dim)
#
# ctx_lb = -3*np.ones_like(embeddings[0])
# ctx_ub = 3*np.ones_like(embeddings[0])
#
#
# print(embedding_map)
#
#
# ctx_lb = -3*np.ones_like(embeddings[0])
# ctx_ub = 3*np.ones_like(embeddings[0])
#
# env_config = {'size': size0,
#                         'corridor': corridor0,
#                         'region': {0: (9, 1), 1: (9, 9)}}
# # 0 UL -> LR
#
# env_creator = lambda config: GMMCtxEnvWrapper(DiscreteCtxDictWrapper(TimeLimitRewardWrapper(
#     TaskSettableGridworld(config0),max_episode_steps=max_steps, key = 'distance'),
#     key='region', embedding_dim=embedding_dim, embedding_map= embedding_transform, embeddings = embeddings
#     ), ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_mode=ctx_mode, **config)
#
# env_config_pars={'target_mean':  embeddings[find_category_idx(embedding_map, env_config['region']),:],
#                         'target_var': np.square([.01, ]*embedding_dim),
#                         'init_mean': np.zeros(embedding_dim),
#                         'init_var': np.square([1, ]*embedding_dim),
#                         'prior': None,
#                  'curriculum': 'gaussian_self_paced',
#                 'kl_threshold' : 8000,
#                 'max_kl': 0.05,
#                 'perf_lb': -16,
#                 'std_lb':  np.square([.01, ]*embedding_dim)}
#
# print(env_config_pars)
# env = env_creator(env_config_pars)
# env.reset()
# print(env.get_context())
# print(embeddings[find_category_idx(embeddings,env.get_context()),:])
#
# init_mean = env_config_pars['init_mean'].copy()
#
# target_mean = env_config_pars['target_mean'].copy()
# kl_threshold = env_config_pars.get('kl_threshold', 10000)
# max_kl = env_config_pars.get('max_kl', 0.05)
# perf_lb = env_config_pars.get('perf_lb', 3)
# std_lower_bound = env_config_pars.get('std_lb', 0.1)
#
#     # print('gaussian_teacher_initialization')
# target_var = np.diag(np.clip(env_config_pars['target_var'], a_min=std_lower_bound ** 2, a_max=None)).copy()
# init_var = env_config_pars['init_var'].copy()
#
# init_scale = np.mean(np.array(np.diag(target_var)).reshape(-1) / np.array(init_var).reshape(-1))
# # std_lower_bound = np.mean(np.array(v['target_var']).reshape(-1)/np.array(init_var).reshape(-1))
# teacher = GaussianSelfPacedTeacher(target_mean=target_mean, target_variance=target_var,
#                                        initial_mean=init_mean, init_covar_scale=init_scale,
#                                        context_bounds=(ctx_lb, ctx_ub), perf_lb=perf_lb,
#                                        max_kl=max_kl, std_lower_bound=init_scale,
#                                        kl_threshold=kl_threshold)
#
#
# ctx_config = teacher.export_dist()
# print(ctx_config)
# env.set_distribution(means=ctx_config['target_mean'], sigma2=ctx_config['target_var'], priors=ctx_config['target_priors'])
# obs = env.reset()[0]
#
# print(obs, obs.dtypethe
#       )
#
# print(env.observation_space, env.observation_space.contains(obs))

points = [(i, j) for i, j in itertools.product(range(1, 10), range(1, 4))] + [(i, j) for i, j in
                                                                              itertools.product(range(1, 10),
                                                                                                range(7, 10))]
#
# embedding_map = [{0: (i, j), 1: (9, 9)} for i, j in itertools.product(range(1, 10), range(1, 4))] + \
#                 [{0: (i, j), 1: (9, 9)} for i, j in itertools.product(range(1, 5), range(7, 10))] # LR destination maps
embedding_map = [{0: start, 1: dst} for start, dst in itertools.product(points, points)if start != dst]

print(embedding_map)