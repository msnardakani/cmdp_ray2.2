import copy
import os
from pathlib import Path
import numpy as np
from gymnasium.wrappers import TimeLimit
from ray import air, tune, train
from ray.rllib.algorithms import PPOConfig
from ray.rllib.algorithms.ppo import PPO, PPOTorchPolicy
import itertools
from flatten_dict import flatten, unflatten
from distral.distral_ppo import DistralPPO
from distral.distral_ppo_torch_model import DistralCentralTorchModel, DistralTorchModel

from envs.contextual_env import GMMCtxEnvWrapper, DiscreteCtxDictWrapper, ctx_visibility, exp_group, find_category_idx
from gymnasium.spaces.utils import flatten_space
import ray
import gymnasium as gym
from ray.rllib.models import ModelCatalog
# from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_torch

from envs.contextual_env import make_multi_agent_divide_and_conquer
from utils.self_paced_callback import MACL

from utils.weight_sharing_models import FC_MLP
from utils.evaluation_fn import DiscreteCL_report, CL_report
from utils.ma_policy_config import gen_ppo_distral_policy, dummy_policy_mapping_fn, policy_mapping_fn
from envs.wrappers import TimeLimitRewardWrapper
from experiment_params import ExperimentSetup
import logging

torch, nn = try_import_torch()
from envs.lunar_lander import LunarLanderCtx



max_steps = 500

gym.logger.set_level(logging.ERROR) 

if __name__ == "__main__":
    env_name = "LunarLander5D"
    model = [128, 128,64 ]
    model_name = '3x128'
    version = 'V12.0.1'

    dir_path = os.path.dirname(os.path.realpath(__file__))

    directory = Path(os.path.join(dir_path, 'results'))  # "/results/" + env_name + "/"+ version+ '/'

    # just to control experiments
    #
    step_per_iteration_per_learner = 10000
    seeds = range(0, 6)
    DistillCtxAug = True
    ch_freq = 100
    max_iter = 300
    ExpContextModes = [
        0,
                       # 1,
                       # 2
                       ]
    continuous=True
    CL_GSP = False
    CL_SP = False
    CL_Def = True


    BaseLine = True
    Experimental = False
    SingleAgent = False

    debug = False

    ray.init(local_mode=debug, num_gpus=1, num_cpus=16, logging_level=logging.ERROR)

    sets = dict(
                # Set0 = [0, ],
                # Set3 = [3, ],
                # Set4 = [4, ],
                # Set1 = [1, ],
                # Set2 = [2, ],
        # Set3=[0, 1, 2, 3, 4, 5],
        # #
        # Set4=[0, 6, 7],  # complex task from basic tasks
        # Set5=[5,],
        Set6 = [6,],
        # Set7 = [0, 0]

                )
    ModelCatalog.register_custom_model(
        "central",
        DistralCentralTorchModel,
    )

    ModelCatalog.register_custom_model(
        "local",
        DistralTorchModel,
    )

    PERF_LB = 50.

    #

    ctx_lb = np.array([-12., 0.0, 0.0,10, .1])
    ctx_ub = np.array([0., 20.0, 2.0, 50, 10 ])



    env_creator_, ctx_spec, max_steps, env_config_pars= ExperimentSetup(env=env_name)

    for s, v, in sets.items():
        for ctx_mode in ExpContextModes:
            log_root = f'{env_name}{"Cont" if continuous else ""}/{version}/{s}/{ctx_visibility[ctx_mode]}'
            env_creator = lambda config: env_creator_(config, ctx_mode, continuous)

            # SPEnv = lambda: gymEnvWrapper(spgym())
            MAEnv = make_multi_agent_divide_and_conquer(lambda config: env_creator(config))
            # eval_envs = v[1]

            num_agents = len(v)
            iteration_ts = step_per_iteration_per_learner * num_agents
            # agent_mappings = [list(range(eval_envs)) + list(perm) for perm in
            #                   itertools.permutations(range(eval_envs, num_agents))]
            # # target_means, target_vars, init_mean, init_var, target_prior = parameters[v]
            # log_dir = directory + s
            agent_config_sp = [copy.deepcopy(env_config_pars[i]) for i in v]
            agent_config_gsp = [copy.deepcopy(env_config_pars[i]) for i in v]
            agent_config_default = [copy.deepcopy(env_config_pars[i]) for i in v]
            for i in range(num_agents):
                agent_config_default[i]['curriculum'] = 'default'

                agent_config_sp[i]['curriculum'] = 'self_paced'
                agent_config_sp[i]['kl_threshold'] = 8000
                agent_config_sp[i]['max_kl'] = 0.1
                agent_config_sp[i]['min_episodes'] = 128

                agent_config_sp[i]['update_interval'] = 5

                agent_config_sp[i]['perf_lb'] = PERF_LB
                agent_config_sp[i]['std_lb'] = np.array([0.1, 0.1,.01, 0.1, 0.01])

                agent_config_gsp[i]['curriculum'] = 'gaussian_self_pacedV3'
                agent_config_gsp[i]['kl_threshold'] = 8000
                agent_config_gsp[i]['max_kl'] = 0.1  # tune.grid_search([   0.05,])
                agent_config_gsp[i]['perf_lb'] = PERF_LB
                agent_config_gsp[i]['std_lb'] = tune.grid_search([1, ])
                agent_config_gsp[i]['min_episodes'] = tune.grid_search([128, ])
                agent_config_gsp[i]['update_interval'] = tune.grid_search([5, ])

                agent_config_gsp[i]['perf_slack'] = 1

            dummy_env = env_creator(config={})
            model_config = {"fcnet_hiddens": model,
                            "fcnet_activation": "relu",
                            'vf_share_layers':True,
                            "free_log_std": True,
                            }
            # print(dummy_env.observation_space)
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
            config = (
                PPOConfig().environment(MAEnv, auto_wrap_old_gym_envs=False, env_config=env_config, clip_actions=True)
                .training(lambda_=0.95, train_batch_size=iteration_ts, sgd_minibatch_size=512,
                          grad_clip=.5,num_sgd_iter  = 10, vf_loss_coeff=.5, entropy_coeff=0.001,clip_param = 0.2,
                          lr=3e-4, gamma=.99,vf_clip_param=50, kl_target=0.01,grad_clip_by='norm')  # tune.grid_search([, 1e-5]))
                .framework('torch')
                .callbacks(MACL)
                .resources(num_gpus=.15)
                .rollouts(num_rollout_workers=1, num_envs_per_worker=50, rollout_fragment_length=200, )
                .evaluation(evaluation_interval=20, evaluation_duration=100, 
                            evaluation_config={'env_config': eval_env_config, "explore": False
                            #                   'render_env':True
                                               },
                            evaluation_num_workers=0)
                .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn,
                             policies_to_train=list(policies.keys()),
                             count_steps_by='agent_steps', policy_states_are_swappable=True)
                # .reporting(min_sample_timesteps_per_iteration=iteration_ts, )
                .debugging(seed=tune.grid_search(list(seeds)))
                )

            distral_config = {'loss_fn': tune.grid_search([31, -2
                                                           ]),
                              "distill_coeff": tune.grid_search([1.,
                                                                 ]),
                              "transfer_coeff": tune.grid_search([1.,
                                                                  ]),
                              "tau": tune.grid_search([.9, .1
                                                       ]),

                              }
            distral_config_debug = {'loss_fn': -1,
                                    "distill_coeff": .2
                                    }
            stop = {
                "training_iteration": max_iter,
                # "episode_reward_mean": 200

            }

            if debug:
                config = config.to_dict()
                # config.update(distral_config_debug)
                config.update({"multiagent": ppo_multiagent_config, })
                # config.update({"multiagent": multiagent_config, })
                agent_config_gsp[0]['std_lb'] = 0.01
                agent_config_gsp[0]['min_episodes'] = 64
                agent_config_gsp[0]['update_interval'] = 1


                env_config.update({"agent_config": agent_config_gsp, })
                config.update({"env_config": env_config, })
                config['seed'] = 7
                config['grad_clip'] = 100
                alg = PPO(config=config)
                for _ in range(10):
                    for _ in range(10):
                        # w = alg.get_policy('distilled_policy').get_weights()['_distilled_model.0._model.0.bias'].copy()

                        res = alg.train()
                        ev = alg.evaluate()
                        # print(w, alg.get_policy('distilled_policy').get_weights()['_distilled_model.0._model.0.bias'] - w)

                        # print(ev)
                    print(res)
                break


            config = config.to_dict()
            if CL_Def:

                if Experimental:
                    config.update(distral_config)

                    results = tune.Tuner(
                        DistralPPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1,
                                                   name=f'{log_root}/{exp_group[ctx_mode]}/default/DistralPPO_{model_name}',
                                                   log_to_file=True,
                                                   storage_path=Path(directory),
                                                   checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                          checkpoint_at_end=True))
                    ).fit()

                    [config.pop(k) for k in distral_config.keys()]

                if BaseLine and ctx_mode != 2:
                    config.update({"multiagent": ppo_multiagent_config, })

                    results = tune.Tuner(
                        PPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1,
                                                   name=f"{log_root}/baseline/default/PPO_{model_name}",
                                                   log_to_file=True,
                                                   storage_path=Path(directory),
                                                   checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                          checkpoint_at_end=True),
                                                   )
                    ).fit()

                if SingleAgent and ctx_mode != 2:
                    config.update({"multiagent": dummy_multiagent_config, })
                    results = tune.Tuner(
                        PPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1,
                                                   name=f"{log_root}/baseline/default/PPO_Central_{model_name}",
                                                   log_to_file=True,
                                                   storage_path=Path(directory),
                                                   checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                          checkpoint_at_end=True))
                    ).fit()
                    config.update({"multiagent": multiagent_config, })

            if CL_SP:

                env_config.update({"agent_config": agent_config_sp, })
                config.update({"env_config": env_config, })

                if Experimental:
                    config.update(distral_config)

                    results = tune.Tuner(
                        DistralPPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1,
                                                   name=f'{log_root}/{exp_group[ctx_mode]}/selfpaced/DistralPPO_{model_name}',
                                                   log_to_file=True,
                                                   storage_path=Path(directory),
                                                   checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                          checkpoint_at_end=True))
                    ).fit()

                    [config.pop(k) for k in distral_config.keys()]
                if BaseLine and ctx_mode != 2:
                    config.update({"multiagent": ppo_multiagent_config, })
                    results = tune.Tuner(
                        PPO, param_space=config,
                        run_config=air.RunConfig(stop=stop, verbose=1,
                                                 name=f"{log_root}/baseline/selfpaced/PPO_{model_name}",
                                                 local_dir=Path(directory),
                                                 log_to_file=True,
                                                 checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                        checkpoint_at_end=True))
                    ).fit()
                if SingleAgent and ctx_mode != 2:
                    config.update({"multiagent": dummy_multiagent_config, })
                    results = tune.Tuner(
                        PPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1,
                                                   name=f"{log_root}/baseline/selfpaced/PPO_Central{model_name}",
                                                   log_to_file=True,
                                                   storage_path=Path(directory),
                                                   checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                          checkpoint_at_end=True))
                    ).fit()

            if CL_GSP:
                config.update({"multiagent": multiagent_config, })

                env_config.update({"agent_config": agent_config_gsp, })
                config.update({"env_config": env_config, })

                if Experimental:
                    config.update(distral_config)

                    results = tune.Tuner(
                        DistralPPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1,
                                                   name=f'{log_root}/{exp_group[ctx_mode]}/GaussianSP/DistralPPO_{model_name}',
                                                   log_to_file=True,
                                                   storage_path=Path(directory),
                                                   checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                          checkpoint_at_end=True))
                    ).fit()

                    [config.pop(k) for k in distral_config.keys()]
                if BaseLine and ctx_mode != 2:
                    config.update({"multiagent": ppo_multiagent_config, })
                    results = tune.Tuner(
                        PPO, param_space=config,
                        run_config=air.RunConfig(stop=stop, verbose=1,
                                                 name=f"{log_root}/baseline/GaussianSP/PPO_{model_name}",
                                                 local_dir=Path(directory),
                                                 log_to_file=True,
                                                 checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                        checkpoint_at_end=True))
                    ).fit()
                if SingleAgent and ctx_mode != 2:
                    config.update({"multiagent": dummy_multiagent_config, })
                    results = tune.Tuner(
                        PPO, param_space=config,
                        run_config=train.RunConfig(stop=stop, verbose=1,
                                                   name=f"{log_root}/baseline/GaussianSP/PPO_Central{model_name}",
                                                   log_to_file=True,
                                                   storage_path=Path(directory),
                                                   checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                          checkpoint_at_end=True))
                    ).fit()


