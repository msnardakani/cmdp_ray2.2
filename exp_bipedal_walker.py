import copy
import os
from pathlib import Path

import numpy as np
from gymnasium.spaces import flatten_space
from gymnasium.wrappers import TimeLimit
from ray import air, tune, train
from ray.rllib.algorithms.ppo import PPO, PPOConfig
import itertools

from distral.distral_ppo import DistralPPO
from distral.distral_ppo_torch_model import DistralCentralTorchModel, DistralTorchModel
from envs.contextual_env import  CtxDictWrapper, ctx_visibility, exp_group, GMMCtxEnvWrapper, SquashedGMMCtxEnvWrapper
from utils.evaluation_fn import DnCCrossEvalSeries, CL_report
from envs.bipedal_walker import BipedalWalkerCtx

import ray

from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_torch

from envs.contextual_env import make_multi_agent_divide_and_conquer, make_multi_agent
import numpy as np
from utils.self_paced_callback import MACL

from utils.weight_sharing_models import FC_MLP
from utils.ma_policy_config import gen_ppo_distral_policy, dummy_policy_mapping_fn, policy_mapping_fn

torch, nn = try_import_torch()
# torch.autograd.set_detect_anomaly(True)
# torch.set_default_tensor_type(torch.DoubleTensor)
# ctx_norm= None
ctx_lb = np.array([0.01, 0.01, 0.01, 0.01 , -20.0, -10.0, 0.0])
ctx_ub = np.array([200., 15.0, 15.0, 20.0, -0.01, 10.0, 10.0])
ctx_dim = 6


max_steps = 1600
# from ray.rllib.env.multi_agent_env import make_multi_agent


if __name__ == "__main__":
    env_name = "BipedalWalker"
    model = [256, 256, 128 ]
    model_name = '3x256'
    version = 'V13.1.4'
    # num_agents = 3
    dir_path = os.path.dirname(os.path.realpath(__file__))

    directory = Path(os.path.join(dir_path,'results'))#"/results/" + env_name + "/"+ version+ '/'
    # just to control experiments
    #
    ch_freq = 100
    max_iter = 500

    seeds = range(0,6)
    ctx_vis_list = [0,
                    # 1,
                    # 2,
                    ]
    CL_Def = True
    CL_SP = True
    GAUSS_CL_SP = True

    BaseLine = True
    Experimental = False
    SingleAgent = False

    debug = False

    ray.init(local_mode=debug, num_gpus=1, num_cpus=16)

    ModelCatalog.register_custom_model(
        "central",
        DistralCentralTorchModel,
    )

    ModelCatalog.register_custom_model(
        "local",
        DistralTorchModel,
    )

    # agent_mappings = [[0, ] + list(perm) for perm in itertools.permutations(range(1, num_agents))]
    # agent_mappings_dummy = [0, ] + [1, ] * (num_agents - 1)
    sets = dict(
        #
        # Set1=[ 1, ],  # left and right tasks from more complicated ones

        # Set2=[ 2, ],
        # Set3 = [3, ],
        # Set4= [4, ],
        Set5= [ 5,]
        # Set0=[0, ],

        # Set10 = [1,],
        # Set20=[2, ],
        # Set30=[3, ],

    )
    #context: (phi, r, dx)

    parameters = []
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
                            'prior': None})   #5
    # ctx_lb = np.array([0.01, 0.01, 0.01, 0.01 , -20.0, -10.0, 0.0])
    # ctx_ub = np.array([200., 15.0, 15.0, 20.0, -0.01, 10.0, 10.0])
    
    for s, v, in sets.items():
        for ctx_mode in ctx_vis_list:
            env_creator = lambda config: SquashedGMMCtxEnvWrapper(
                TimeLimit(BipedalWalkerCtx(context=np.array([80., 4., 6., 5.33, 0., -10., 2.5])), max_episode_steps=max_steps),
                ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_mode=ctx_mode, **config)

            # SPEnv = lambda: gymEnvWrapper(spgym())
            MAEnv = make_multi_agent_divide_and_conquer(lambda config: env_creator(config))
            # eval_envs = v[1]

            num_agents = len(v)
            iteration_ts = 10000 * num_agents
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
                # agent_config_sp[i]['kl_threshold'] = 8000
                agent_config_sp[i]['max_kl'] = 0.1
                agent_config_sp[i]['perf_lb'] = 0.
                agent_config_sp[i]['reset'] = True
                
                agent_config_gsp[i]['update_interval'] = tune.grid_search([3, ])


                agent_config_gsp[i]['curriculum'] = 'gaussian_self_pacedV3'
                agent_config_gsp[i]['kl_threshold'] = 8000
                agent_config_gsp[i]['max_kl'] = tune.grid_search([ 0.08,])
                agent_config_gsp[i]['perf_lb'] = tune.grid_search([ 0.,])
                agent_config_gsp[i]['std_lb'] = tune.grid_search([ 1.,])
                agent_config_gsp[i]['reset'] = True
                
                agent_config_gsp[i]['min_episodes'] = tune.grid_search([128, ])
                agent_config_gsp[i]['update_interval'] = tune.grid_search([3, ])


            dummy_env = env_creator(config={})
            model_config = {"fcnet_hiddens": model,
                            "fcnet_activation": "tanh",
                            'vf_share_layers':False,
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
            policies = gen_ppo_distral_policy( N= num_agents ,model_config=model_config,
                                               central_policy_target=central_policy_target,
                                               central_policy=central_policy, obs_space = dummy_env.observation_space,
                                                 ctx_mode=ctx_mode,sample_offset = 10000)
            policy_ids = list(policies.keys())
            ppo_policies = gen_ppo_distral_policy(N=num_agents, model_config=model_config,
                                                  central_policy=central_policy,
                                                  obs_space=dummy_env.observation_space,
                                                  ctx_mode=ctx_mode,sample_offset = 10000)
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
            policies_dummy = gen_ppo_distral_policy(N = 1,model_config=model_config, central_policy=central_policy,
                                                     obs_space=dummy_env.observation_space, ctx_mode=ctx_mode,
                                                     sample_offset = 10000)
            policy_ids_dummy = list(policies_dummy.keys())
            dummy_multiagent_config = {
                'policies': policies_dummy,
                "policy_mapping_fn": dummy_policy_mapping_fn,
                "count_steps_by": "agent_steps",
                "policies_to_train": policy_ids_dummy,
            }
            config = (PPOConfig().environment(MAEnv, auto_wrap_old_gym_envs=False, env_config= env_config)
                      .training(gamma=.99,lambda_=.95,train_batch_size=iteration_ts, sgd_minibatch_size=1024,
                          grad_clip=1.,num_sgd_iter  = 16, vf_loss_coeff=.5, entropy_coeff=0.003,clip_param = 0.2,
                          lr=2e-4,vf_clip_param=100, kl_target=0.01,grad_clip_by='norm',kl_coeff=0.5,
                                )
                      .framework('torch')
                      .callbacks(MACL)
                      .resources(num_gpus=.15)
                      .rollouts(num_rollout_workers=1, num_envs_per_worker=40,rollout_fragment_length=250 )
                      .evaluation(evaluation_interval=50, evaluation_duration=80,
                                  evaluation_config={'env_config': eval_env_config, 'explore': False
                                  #                   'render_env':True
                                                     },
                                  evaluation_num_workers=0)
                      .multi_agent(policies=policies, policy_mapping_fn= policy_mapping_fn, policies_to_train=list(policies.keys()),
                                   count_steps_by='agent_steps',policy_states_are_swappable=True )
                      .reporting(min_sample_timesteps_per_iteration= iteration_ts,)
                      .debugging(seed=tune.grid_search(list(seeds)))
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
            distral_config_debug = {'loss_fn': -1,
                                    "distill_coeff": .2
                                    }
            stop = {
                "training_iteration": max_iter,

            }

            if debug:
                config = config.to_dict()
                # config.update(distral_config_debug)
                config.update({"multiagent": ppo_multiagent_config, })
                # config.update({"multiagent": multiagent_config, })

                env_config.update({"agent_config": agent_config_sp, })
                config.update({"env_config": env_config, })
                config['seed'] = 7
                alg = PPO(config = config)
                for _ in range(25):
                    for _ in range(20):
                        # w = alg.get_policy('distilled_policy').get_weights()['_distilled_model.0._model.0.bias'].copy()

                        res = alg.train()
                        ev= alg.evaluate()
                        # print(w, alg.get_policy('distilled_policy').get_weights()['_distilled_model.0._model.0.bias'] - w)
                        print(res)
                        print(ev)
                break


            # model_config = {"fcnet_hiddens": model,
            #                 "fcnet_activation": "relu",
            #
            #                 }



            #
            config = config.to_dict()
            if CL_Def:

                
                config.update({"multiagent": ppo_multiagent_config, })

                results = tune.Tuner(
                    PPO, param_space=config,
                    run_config=train.RunConfig(stop=stop, verbose=1, name=env_name+'/'+ version +'/'+s+'/'+ctx_visibility[ctx_mode]+"/baseline/default/PPO_" + model_name,
                                                log_to_file=True,
                                                storage_path=Path(directory),
                                                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                    checkpoint_at_end=True),
                                                )
                ).fit()

                
            if CL_SP:

                env_config.update({"agent_config": agent_config_sp, })
                config.update({"env_config": env_config, })

               
                config.update({"multiagent": ppo_multiagent_config, })
                results = tune.Tuner(
                    PPO, param_space=config,
                    run_config=air.RunConfig(stop=stop, verbose=1, name=env_name + "/" + version+'/'+s+'/'+ctx_visibility[ctx_mode]+"/baseline/selfpaced/PPO_" + model_name,
                                                local_dir=Path(directory),
                                                log_to_file=True,
                                                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                    checkpoint_at_end=True))
                ).fit()
               
            if GAUSS_CL_SP:
                config.update({"multiagent": multiagent_config, })

                env_config.update({"agent_config": agent_config_gsp, })
                config.update({"env_config": env_config, })

                
                config.update({"multiagent": ppo_multiagent_config, })
                results = tune.Tuner(
                    PPO, param_space=config,
                    run_config=air.RunConfig(stop=stop, verbose=1, name=env_name + "/" + version+'/'+s+'/'+ctx_visibility[ctx_mode]+"/baseline/GaussianSP/PPO_" + model_name,
                                                local_dir=Path(directory),
                                                log_to_file=True,
                                                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                    checkpoint_at_end=True))
                ).fit()
            




    ray.shutdown()



