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
from envs.contextual_env import  CtxDictWrapper, ctx_visibility, exp_group, GMMCtxEnvWrapper
from utils.evaluation_fn import DnCCrossEvalSeries
from envs.point_mass_2d import PointMassEnv
from utils.self_paced_callback import MACL
import ray

from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_torch

from envs.contextual_env import make_multi_agent_divide_and_conquer, make_multi_agent
import numpy as np

from utils.weight_sharing_models import FC_MLP

torch, nn = try_import_torch()
torch.autograd.set_detect_anomaly(True)
# torch.set_default_tensor_type(torch.DoubleTensor)
ctx_mode =0
# ctx_norm= None
ctx_lb = np.array([-4, .5])
ctx_ub = np.array([4, 4])
std_lb = np.array([0.2, 0.1875])
max_steps = 128
# from ray.rllib.env.multi_agent_env import make_multi_agent

env_creator = lambda config: GMMCtxEnvWrapper(TimeLimit(PointMassEnv(context=np.array([3, .5])), max_episode_steps=max_steps ),
                                        ctx_lb=ctx_lb, ctx_ub = ctx_ub, ctx_mode= ctx_mode, **config )

# SPEnv = lambda: gymEnvWrapper(spgym())
MAEnv = make_multi_agent_divide_and_conquer(lambda config: env_creator(config))

if __name__ == "__main__":
    env_name = "PointMass2D"
    model = [64, 64, 64 ]
    model_name = '3x64'
    version = 'debug_single_env'
    # num_agents = 3
    dir_path = os.path.dirname(os.path.realpath(__file__))

    directory = Path(os.path.join(dir_path,'results',env_name, version ))#"/results/" + env_name + "/"+ version+ '/'
    # just to control experiments
    #
    ch_freq = 100

    CL_SP = False
    CL_Def = True

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
    sets = dict(Set1=([0,  ], 0),  # left and right tasks from more complicated ones
                # Set3=([3, 4, 5], 1),
                # Set4=([0, 7, 5, 2, ], 1),
                # Set2=([0, 1, 2, ], 1),
                # Set0=([0, 0, 0, ], 1)
                )

    parameters = []
    env_config_pars = list()
    env_config_pars.append({'target_mean' : np.array([2.5, .5]),
                            'target_var': np.square([4e-3, 3.75e-3]),
                            'init_mean':np.array([0, 6]),
                            'init_var': np.square([2, 1.875]),
                            'prior': None}) # 0 NRR default
    env_config_pars.append({'target_mean' : np.array([3, 2]),
                            'target_var': np.square([4e-3, 3.75e-3]),
                            'init_mean':np.array([0, 6]),
                            'init_var': np.square([2, 1.875]),
                            'prior': None}) #1  WRR

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

    env_config_pars.append({'target_mean': np.array( [-3, 2]),
                            'target_var': np.square([.5, .2]),
                            'init_mean': np.array([0, 6]),
                            'init_var': np.square([2, 1.875]),
                            'prior': None})  # 4 WLL

    env_config_pars.append({'target_mean': np.array( [2.5, .5]),
                            'target_var': np.square([.05, .01]),
                            'init_mean': np.array([0, 6]),
                            'init_var': np.square([2, 1.875]),
                            'prior': None})  # 5 NRR

    env_config_pars.append({'target_mean': np.array( [[2, 1], [-2, 1]]),
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

    for s, v, in sets.items():
        num_agents = len(v[0])
        eval_envs = v[1]
        iteration_ts = 2048
        agent_mappings = [list(range(eval_envs)) + list(perm) for perm in
                          itertools.permutations(range(eval_envs, num_agents))]
        # target_means, target_vars, init_mean, init_var, target_prior = parameters[v]
        # log_dir = directory + s
        agent_config_sp = [copy.deepcopy(env_config_pars[i]) for i in v[0]]
        agent_config_default = [copy.deepcopy(env_config_pars[i]) for i in v[0]]
        for i in range(num_agents):
            agent_config_default[i]['curriculum'] = 'default'
            if i >eval_envs:
                agent_config_sp[i]['curriculum'] = 'self_paced'
                agent_config_sp[i]['kl_threshold'] = 8000
                agent_config_sp[i]['max_kl'] = 0.05
                agent_config_sp[i]['perf_lb'] = 3.5
                agent_config_sp[i]['std_lb'] = np.array([0.2, 0.1875])


            else:
                agent_config_sp[i]['curriculum'] = 'default'

        dummy_env = env_creator(config={})
        model_config = {"fcnet_hiddens": model,
                        "fcnet_activation": "relu",
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



        def gen_policy(i, obs_space,ctx_mode = 0):
            out = dict()

            ctx_aug = obs_space[0] if ctx_mode==2 else None
            for i in range(i):
                if i == 0:
                    config = {'model': {
                        "custom_model": 'central',
                        "custom_model_config": {"distilled_model": central_policy, 'model': model_config,'ctx_aug': ctx_aug}},

                    }
                    out['distilled_policy'] = PolicySpec(config=config)
                else:
                    config = {'model': {
                        "custom_model": 'local',
                        "custom_model_config": {"distilled_model": central_policy, 'model': model_config,'ctx_aug': ctx_aug}},
                        "num_steps_sampled_before_learning_starts": 500

                    }
                    out["learner_{}".format(i - 1)] = PolicySpec(config=config)

            return out



        #
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            mapping = worker.config['env_config']['env_mapping']
            eval_envs = worker.config['env_config']['non_training_envs']

            if mapping:

                return policy_ids[0 if agent_id < eval_envs else mapping[agent_id] - eval_envs+1]
            else:
                return policy_ids[0 if agent_id < eval_envs else agent_id - eval_envs+1]

        dummy_policy_mapping_fn = lambda agent_id, *args, **kwargs: policy_ids[0 if agent_id < eval_envs else 1]
        # Setup PPO with an ensemble of `num_policies` different policies.
        env_config = {"num_agents": num_agents,
                      "env_mapping": tune.choice(agent_mappings),
                      "agent_config": agent_config_default,
                      'non_training_envs': eval_envs,
                      }
        eval_env_config = {"num_agents": num_agents,
                           "agent_config": agent_config_default,
                           'non_training_envs': eval_envs,
                           }
        policies = gen_policy(num_agents - eval_envs + 1, dummy_env.observation_space, ctx_mode=ctx_mode)
        policy_ids = list(policies.keys())
        multiagent_config = {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": list(policies.keys())[1:],
            "count_steps_by": "agent_steps",
        }
        policies_dummy = gen_policy(2, dummy_env.observation_space, ctx_mode=ctx_mode)
        policy_ids_dummy = list(policies_dummy.keys())
        dummy_multiagent_config = {
            'policies': policies_dummy,
            "policy_mapping_fn": dummy_policy_mapping_fn,
            "count_steps_by": "agent_steps",
            "policies_to_train": policy_ids_dummy[1:],
        }
        config = (PPOConfig().environment(MAEnv, auto_wrap_old_gym_envs=False, env_config= env_config)
                  .training(train_batch_size=4096, sgd_minibatch_size=512)
                  .framework('torch')
                  .callbacks(MACL)
                  .resources(num_gpus=.12)
                  .rollouts(num_rollout_workers=1, num_envs_per_worker=64)
                  .evaluation(evaluation_interval=50, evaluation_duration=128,custom_evaluation_function=DnCCrossEvalSeries,
                              evaluation_config={'env_config': eval_env_config,}, evaluation_num_workers=1)
                  .multi_agent(policies=policies, policy_mapping_fn= policy_mapping_fn, policies_to_train=list(policies.keys())[1:],
                               count_steps_by='env_steps',policy_states_are_swappable=False )
                  .reporting(min_sample_timesteps_per_iteration= iteration_ts,)
                  .debugging(seed=tune.grid_search(list(range(16))))
                  )


        distral_config = {'loss_fn': tune.grid_search([1, 0,-1]),
                          "distill_coeff": tune.grid_search([0.2, ]),
                          'grad_clip': tune.grid_search([20,])

                          }
        distral_config_debug = {'loss_fn': -1,
                                "distill_coeff": .2
                                }
        stop = {
            "training_iteration": 500,

        }

        if debug:
            # config.update(distral_config_debug)
            config['seed'] = 7
            config['env_config']['env_mapping'] = None
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

        if CL_Def:
            if Experimental:
                config.update(distral_config)

                results = tune.Tuner(
                    DistralPPO, param_space=config,
                    run_config=air.RunConfig(stop=stop, verbose=1, name="DistralPPO_" + model_name,
                                             local_dir=directory+ctx_visibility[ctx_mode]+'/'+s+'/'+exp_group[ctx_mode]+'/default/', checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                                       checkpoint_at_end=True))
                ).fit()

                [config.pop(k) for k in distral_config.keys()]

            if BaseLine and ctx_mode!=2:

                results = tune.Tuner(
                    PPO, param_space=config.to_dict(),
                    run_config=train.RunConfig(stop=stop, verbose=1, name="PPO_" + model_name,
                                             log_to_file=True,
                                             storage_path=Path(os.path.join(directory,ctx_visibility[ctx_mode],s,'baseline','default')),
                                             checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                    checkpoint_at_end=True),
                                             )
                ).fit()

            if SingleAgent and ctx_mode!=2:
                config.update({"multiagent": dummy_multiagent_config, })
                results = tune.Tuner(
                    PPO, param_space=config,
                    run_config=train.RunConfig(stop=stop, verbose=1, name="PPO_Central_" + model_name,
                                             local_dir=directory+ctx_visibility[ctx_mode]+'/'+s+'/baseline/default/',
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
                    run_config=air.RunConfig(stop=stop, verbose=1, name="DistralPPO_" + model_name,
                                             local_dir=directory+ctx_visibility[ ctx_mode]+'/'+s+'/'+exp_group[ctx_mode]+'/selfpaced/' ,
                                             checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                    checkpoint_at_end=True))
                ).fit()


                [config.pop(k) for k in distral_config.keys()]
            if BaseLine:
                results = tune.Tuner(
                    PPO, param_space=config,
                    run_config=air.RunConfig(stop=stop, verbose=1, name="PPO_" + model_name,
                                             local_dir=directory+ctx_visibility[ctx_mode]+'/'+s+'/baseline/selfpaced/',
                                             checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                    checkpoint_at_end=True))
                ).fit()
            if SingleAgent:
                config.update({"multiagent": dummy_multiagent_config, })
                results = tune.Tuner(
                    PPO, param_space=config,
                    run_config=air.RunConfig(stop=stop, verbose=1, name="PPO_Central_" + model_name,
                                             local_dir=directory+ctx_visibility[ctx_mode]+'/'+s+'/baseline/selfpaced/',
                                             checkpoint_config=air.CheckpointConfig(checkpoint_frequency=ch_freq,
                                                                                    checkpoint_at_end=True))
                ).fit()






    ray.shutdown()



