import copy
import os
from pathlib import Path
import numpy as np
from gymnasium.wrappers import TimeLimit
from ray import air, tune, train
from ray.rllib.algorithms import PPOConfig
from ray.rllib.algorithms.ppo import PPO, PPOTorchPolicy
import itertools

from distral.distral_ppo import DistralPPO
from distral.distral_ppo_torch_model import DistralCentralTorchModel, DistralTorchModel

from envs.contextual_env import GMMCtxEnvWrapper, DiscreteCtxDictWrapper, ctx_visibility, exp_group, find_category_idx
from gymnasium.spaces.utils import flatten_space
import ray

from ray.rllib.models import ModelCatalog
# from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_torch

from envs.contextual_env import make_multi_agent_divide_and_conquer
from utils.self_paced_callback import MACL

from utils.weight_sharing_models import FC_MLP
from utils.evaluation_fn import DiscreteCL_report, CL_report
from utils.ma_policy_config import gen_ppo_distral_policy, dummy_policy_mapping_fn, policy_mapping_fn
from envs.wrappers import TimeLimitRewardWrapper

torch, nn = try_import_torch()
from envs.gridworld_contextual import TaskSettableGridworld, coordinate_transform




max_steps = 15


if __name__ == "__main__":
    env_name = "GridWorld"
    model = [64, 64,]
    model_name = '2x64'
    version = 'V11.5.0tmp'

    dir_path = os.path.dirname(os.path.realpath(__file__))

    directory = Path(os.path.join(dir_path, 'results'))  # "/results/" + env_name + "/"+ version+ '/'

    # just to control experiments
    #
    DistillCtxAug = True
    ch_freq = 40

    ExpContextModes = [
        0,
                       1,
                       # 2
                       ]

    CL_GSP = True
    CL_SP = False
    CL_Def = False


    BaseLine = True
    Experimental = False
    SingleAgent = True

    debug = False

    ray.init(local_mode=debug, num_gpus=1, num_cpus=16)

    sets = dict(
        # Set0=[2, 1],
        # Set1 = [0,1, 2],
        # Set2=[1, 2, 4, 5],
        # Set3=[0, 1, 2, 3, 4, 5],
        # #
        Set4=[0, 6, 7],  # complex task from basic tasks
        Set5=[0,6,7,3,8,9],
        # Set6 = [0,3],
        # Set7 = [0, 0]

                )

    parameters = []
    env_config_pars = list()

    size0 = (11, 11)

    corridor0 = (1, 3)
    PERF_LB = -30


    embedding_dim = 4

    points = [(i, j) for i, j in itertools.product(range(1, 10), range(1, 4))] + [(i, j) for i, j in itertools.product(range(1, 10), range(7, 10))]
    points_compact = [(i, j) for i, j in itertools.product(range(1, 10), range(1, 7))]
    # embedding_map = [{0: (i, j), 1: (9, 9)} for i, j in itertools.product(range(1, 10), range(1, 4))] + \
    #                 [{0: (i, j), 1: (9, 9)} for i, j in itertools.product(range(1, 5), range(7, 10))] # LR destination maps



    embedding_map = [{0: start, 1: (9,9)} for start in points if start != (9,9)]+\
                    [{0: start, 1: (1,1)} for start in points if start != (1,1)]
    embedding_map_compact = [{0: start, 1: (9, 6)} for start in points_compact if start != (9, 6)] + \
                    [{0: start, 1: (1, 1)} for start in points_compact if start != (1, 1)]

    embedding_transform = [np.array([coordinate_transform(v[0], size0), coordinate_transform(v[1], size0)]) for v in
                           embedding_map]

    # embeddings = np.random.randn(len(embedding_map), embedding_dim)
    embeddings = np.array([[2 * v[0][0] / size0[0] - 1, 2 * v[0][1] / size0[1] - 1, 2 * v[1][0] / size0[0] - 1, 2 * v[1][1] / size0[1] - 1] for v in
                           embedding_map_compact], dtype=np.float64)

    ctx_lb = -1 * np.ones_like(embeddings[0])
    ctx_ub = 1 * np.ones_like(embeddings[0])

    config0 = {'size': size0,
                            'corridor': corridor0,
                            'region': {0: (1, 1), 1: (9, 9)}}

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



    for s, v in sets.items():
        for ctx_mode in ExpContextModes:

            env_creator = lambda config: GMMCtxEnvWrapper(DiscreteCtxDictWrapper(TimeLimitRewardWrapper(
                TaskSettableGridworld(config0), max_episode_steps=max_steps, key='distance'),
                key='region', embedding_dim=embedding_dim, embedding_map=embedding_transform, embeddings=embeddings
            ), ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_mode=ctx_mode, **config)
            MAEnv = make_multi_agent_divide_and_conquer(lambda config: env_creator(config))

            # agent_config = [copy.deepcopy(env_config_pars[i]) for i in v]

            sp_config = {'kl_threshold': 8000,
                             'max_kl': 0.05,
                             'perf_lb': PERF_LB,
                             'std_lb': np.square([.01, ] * embedding_dim)}

            agent_config = [{'target_mean': embeddings[find_category_idx(embedding_map, env_config_pars[idx]['region']), :],
                             'target_var': np.square([.01, ] * embedding_dim),
                             'init_mean': np.zeros(embedding_dim),
                             'init_var': np.square([1, ] * embedding_dim),
                             'prior': None,

                             } for idx in v]

            agent_config_sp = [copy.deepcopy(config)for config in agent_config]
            agent_config_gsp = [copy.deepcopy(config) for config in agent_config]
            agent_config_default = [copy.deepcopy(config) for config in agent_config]
            num_agents = len(v)
            for i in range(num_agents):
                agent_config_default[i]['curriculum'] = 'default'

                agent_config_sp[i]['curriculum'] = 'self_paced'
                agent_config_sp[i].update(sp_config)

                agent_config_gsp[i]['curriculum'] = 'gaussian_self_paced'
                agent_config_gsp[i].update(sp_config)







            iteration_ts = 450* num_agents


            dummy_env = env_creator(config={})
            model_config = {"fcnet_hiddens": model,
                            "fcnet_activation": "relu",

                            }
            dist_class, logit_dim = ModelCatalog.get_action_dist(
                dummy_env.action_space, model_config, framework='torch'
            )

            central_policy, _ = FC_MLP(
                obs_space=flatten_space(dummy_env.observation_space),
                action_space=dummy_env.action_space,
                num_outputs=logit_dim,
                model_config=model_config,
            )

            central_policy_target, _ = FC_MLP(
                obs_space=flatten_space(dummy_env.observation_space),
                action_space=dummy_env.action_space,
                num_outputs=logit_dim,
                model_config=model_config,
            )

            ModelCatalog.register_custom_model(
                "central",
                DistralCentralTorchModel,
            )

            ModelCatalog.register_custom_model(
                "local",
                DistralTorchModel,
            )

            env_config = {"num_agents": num_agents,
                          "agent_config": agent_config_default,
                          }
            eval_env_config = {"num_agents": num_agents,
                               "agent_config": agent_config_default,
                               }
            policies = gen_ppo_distral_policy(N=num_agents, model_config=model_config,
                                              central_policy_target=central_policy_target,
                                              central_policy=central_policy, obs_space=dummy_env.observation_space,
                                              ctx_mode=ctx_mode)
            policy_ids = list(policies.keys())
            ppo_policies = gen_ppo_distral_policy(N=num_agents, model_config=model_config,
                                                  central_policy=central_policy,
                                                  obs_space=dummy_env.observation_space,
                                                  ctx_mode=ctx_mode)
            multiagent_config = {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": list(policies.keys()),
                "count_steps_by": "agent_steps",
            }
            ppo_multiagent_config = {
                "policies": ppo_policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": list(ppo_policies.keys()),
                "count_steps_by": "agent_steps",
            }
            policies_dummy = gen_ppo_distral_policy(N=1, model_config=model_config, central_policy=central_policy,
                                                    obs_space=dummy_env.observation_space, ctx_mode=ctx_mode)
            policy_ids_dummy = list(policies_dummy.keys())
            dummy_multiagent_config = {
                'policies': policies_dummy,
                "policy_mapping_fn": dummy_policy_mapping_fn,
                "count_steps_by": "agent_steps",
                "policies_to_train": policy_ids_dummy,
            }


            config = (PPOConfig().environment(MAEnv, auto_wrap_old_gym_envs=False, env_config=env_config)
                      .training(train_batch_size=4096, sgd_minibatch_size=512, grad_clip=100, )#tune.grid_search([100, ]))
                      .framework('torch')
                      .resources(num_gpus=.12)
                      .rollouts(num_rollout_workers=1, num_envs_per_worker=15)
                      .evaluation(evaluation_interval=10, evaluation_duration=1,
                                  custom_evaluation_function=DiscreteCL_report,
                                  evaluation_num_workers=0)
                      .callbacks(MACL)
                      .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn,
                                   policies_to_train=list(policies.keys()),
                                   count_steps_by='agent_steps', policy_states_are_swappable=True)
                      .reporting(min_sample_timesteps_per_iteration=iteration_ts, )
                      .debugging(seed=tune.grid_search(list(range( 16))))
                      )
            distral_config = {'loss_fn': tune.grid_search([ 31, -2
                                                           ]),
                              "distill_coeff": tune.grid_search([1.,
                                                                 ]),
                              "transfer_coeff": tune.grid_search([1.,
                                                                  ]),
                              "tau": tune.grid_search([.9, .1
                                                       ]),
                                }

            stop = {
                "training_iteration": 100,
            }

            if debug:
                config_debug = config.to_dict()
                env_config.update({"agent_config": agent_config_gsp, })
                config_debug.update({"env_config": env_config, })
                config_debug.update({"multiagent": ppo_multiagent_config, })

                distral_debug = {
                    # 'loss_fn': 31,
                    #              'distill_coeff': 1.0,
                    #              'transfer_coeff': 1.0,
                                 'seed': 0,
                                 'min_sample_timesteps_per_iteration': 300,
                                 'tau': 0.5}

                config_debug.update(distral_debug)
                alg = PPO(config=config_debug)
                for _ in range(25):
                    for _ in range(20):
                        res = alg.train()
                        ev = alg.evaluate()

                        print(res)
                        print(ev)
                break
            config= config.to_dict()

            if CL_Def:
                config.update({"multiagent": multiagent_config, })

                if Experimental:
                    config.update(distral_config)

                    # exp_name = '' if ctx_mode!=2 else ctx_visibility[2]
                    results = tune.Tuner(
                        DistralPPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1, name=env_name + "/" + version + '/' + s +'/' + ctx_visibility[
                            ctx_mode] +  '/' + exp_group[ctx_mode] + "/default/DistralPPO_" + model_name,
                                                 log_to_file=True,
                                                 storage_path=Path(directory),
                                                 checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                        checkpoint_at_end=True))
                    ).fit()

                    [config.pop(k) for k in distral_config.keys()]


                if BaseLine and ctx_mode!=2:
                    config.update({"multiagent": ppo_multiagent_config, })
                    results = tune.Tuner(
                        PPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1,
                                                   name=env_name + '/' + version +  '/' + s +'/' + ctx_visibility[
                                                       ctx_mode] + "/baseline/default/PPO_" + model_name,
                                                   log_to_file=True,
                                                   storage_path=Path(directory),
                                                   checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                          checkpoint_at_end=True),
                                                   )
                    ).fit()


                if SingleAgent and ctx_mode!=2:
                    config.update({"multiagent": dummy_multiagent_config, })
                    results = tune.Tuner(
                        PPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1,
                                                   name=env_name + '/' + version +  '/' + s+'/' + ctx_visibility[
                                                       ctx_mode]  + "/baseline/default/PPO_Central_" + model_name,
                                                   log_to_file=True,
                                                   storage_path=Path(directory),
                                                   checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                          checkpoint_at_end=True))
                    ).fit()



            if CL_SP:
                env_config.update({"agent_config": agent_config_sp, })
                config.update({"env_config": env_config, })
                config.update({"multiagent": multiagent_config, })

                if Experimental:
                    config.update(distral_config)

                    # exp_name = '' if ctx_mode!=2 else ctx_visibility[2]
                    results = tune.Tuner(
                        DistralPPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1,
                                                   name=env_name + "/" + version + '/' + s + '/' +
                                                        ctx_visibility[
                                                            ctx_mode] + '/' + exp_group[
                                                            ctx_mode] + "/selfpaced/DistralPPO_" + model_name,
                                                   log_to_file=True,
                                                   storage_path=Path(directory),
                                                   checkpoint_config=air.CheckpointConfig(
                                                       checkpoint_frequency=ch_freq,
                                                       checkpoint_at_end=True))
                    ).fit()

                    [config.pop(k) for k in distral_config.keys()]


                if BaseLine and ctx_mode != 2:
                    config.update({"multiagent": ppo_multiagent_config, })
                    results = tune.Tuner(
                        PPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1,
                                                   name=env_name + '/' + version + '/' + s + '/' +
                                                        ctx_visibility[
                                                            ctx_mode] + "/baseline/selfpaced/PPO_" + model_name,
                                                   log_to_file=True,
                                                   storage_path=Path(directory),
                                                   checkpoint_config=air.CheckpointConfig(
                                                       checkpoint_frequency=ch_freq,
                                                       checkpoint_at_end=True),
                                                   )
                    ).fit()


                if SingleAgent and ctx_mode != 2:
                    config.update({"multiagent": dummy_multiagent_config, })
                    results = tune.Tuner(
                        PPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1,
                                                   name=env_name + '/' + version + '/' + s + '/' +
                                                        ctx_visibility[
                                                            ctx_mode] + "/baseline/selfpaced/PPO_Central_" + model_name,
                                                   log_to_file=True,
                                                   storage_path=Path(directory),
                                                   checkpoint_config=air.CheckpointConfig(
                                                       checkpoint_frequency=ch_freq,
                                                       checkpoint_at_end=True))
                    ).fit()

            if CL_GSP:
                env_config.update({"agent_config": agent_config_gsp, })
                config.update({"env_config": env_config, })
                config.update({"multiagent": multiagent_config, })

                if Experimental:
                    config.update(distral_config)

                    # exp_name = '' if ctx_mode!=2 else ctx_visibility[2]
                    results = tune.Tuner(
                        DistralPPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1,
                                                   name=env_name + "/" + version + '/' + s + '/' +
                                                        ctx_visibility[
                                                            ctx_mode] + '/' + exp_group[
                                                            ctx_mode] + "/GaussianSP/DistralPPO_" + model_name,
                                                   log_to_file=True,
                                                   storage_path=Path(directory),
                                                   checkpoint_config=air.CheckpointConfig(
                                                       checkpoint_frequency=ch_freq,
                                                       checkpoint_at_end=True))
                    ).fit()

                    [config.pop(k) for k in distral_config.keys()]

                if BaseLine and ctx_mode != 2:
                    config.update({"multiagent": ppo_multiagent_config, })
                    results = tune.Tuner(
                        PPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1,
                                                   name=env_name + '/' + version + '/' + s + '/' +
                                                        ctx_visibility[
                                                            ctx_mode] + "/baseline/GaussianSP/PPO_" + model_name,
                                                   log_to_file=True,
                                                   storage_path=Path(directory),
                                                   checkpoint_config=air.CheckpointConfig(
                                                       checkpoint_frequency=ch_freq,
                                                       checkpoint_at_end=True),
                                                   )
                    ).fit()

                if SingleAgent and ctx_mode != 2:
                    config.update({"multiagent": dummy_multiagent_config, })
                    results = tune.Tuner(
                        PPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1,
                                                   name=env_name + '/' + version + '/' + s + '/' +
                                                        ctx_visibility[
                                                            ctx_mode] + "/baseline/GaussianSP/PPO_Central_" + model_name,
                                                   log_to_file=True,
                                                   storage_path=Path(directory),
                                                   checkpoint_config=air.CheckpointConfig(
                                                       checkpoint_frequency=ch_freq,
                                                       checkpoint_at_end=True))
                    ).fit()

    ray.shutdown()
