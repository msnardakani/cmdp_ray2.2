import numpy as np
# from gym.wrappers import TimeLimit
from ray import air, tune
# from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.sac import SAC
from ray.rllib.algorithms.ppo import PPO
# from ray.rllib.evaluation.metrics import summarize_episodes, collect_episodes

from gym.wrappers.flatten_observation import FlattenObservation
from utils.evaluation_fn import DnCCrossEval, DnCCrossEvalSeries
from utils.ma_policy_config import gen_policy

from envs.brax.half_cheetah_ctx import HalfcheetahCTX
from envs.wrappers import CtxAugmentedObs, BrxEnvObs
from distral.distral import Distral
# from envs.point_mass_2d import PointMassEnv
from envs.utils import make_DnC_env


# import gym.envs.box2d.lunar_lander
import ray

from ray.rllib.models import ModelCatalog, ActionDistribution
# from ray.rllib.policy.policy import PolicySpec

from ray.rllib.utils.framework import try_import_tf, try_import_torch
# from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

from env_funcs import make_multi_agent_divide_and_conquer#, Self_Paced_CL, Self_Paced_MACL
# from envs.point_mass_2d import TaskSettablePointMass2D
from utils.callbacks import MASPCL
from utils.weight_sharing_models import FC_MLP
from gym.wrappers.time_limit import TimeLimit
# from gym.wrappers.normalize import NormalizeObservation
torch, nn = try_import_torch()
# from envs.utils import gymEnvWrapper
# from utils import SPCL_Eval
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

max_steps = 1000
SamplesOffset = 4096
keys = ['torso_mass', 'gravity']
std_lb = .1
ctx_lb = np.array([HalfcheetahCTX.CONTEXT_BOUNDS[k][0] for k in keys] )


ctx_ub = np.array([HalfcheetahCTX.CONTEXT_BOUNDS[k][1] for k in keys] )


env_creator_ctx_vis = lambda : FlattenObservation(CtxAugmentedObs(TimeLimit(HalfcheetahCTX(context_keys =keys),max_episode_steps=max_steps)))
SPEnvVis = make_DnC_env(env_creator_ctx_vis, ctx_lb=ctx_lb, ctx_ub=ctx_ub,  dict_obs=True,
                              ctx_norm= None,
                     std_lb=std_lb, )
MAEnvVis = make_multi_agent_divide_and_conquer(lambda config: SPEnvVis(config))


env_creator = lambda : FlattenObservation(BrxEnvObs(TimeLimit(HalfcheetahCTX(context_keys =keys),max_episode_steps=max_steps)))
SPEnv = make_DnC_env(env_creator, ctx_lb=ctx_lb, ctx_ub=ctx_ub,  dict_obs=True,
                              ctx_norm= None,
                     std_lb=std_lb, )
MAEnv = make_multi_agent_divide_and_conquer(lambda config: SPEnv(config))

#
# SPEnvVisNorm = make_DnC_env(env_creator, ctx_lb=np.array([0.5, 0.05]), ctx_ub=np.array([2, 0.2]), ctx_visible=True,
#                              time_limit=TimeLimit, ctx_norm=ctx_norm)
#
# MAEnvVisNorm = make_multi_agent_divide_and_conquer(lambda config: SPEnvVis(config))




if __name__ == "__main__":
    # args = parser.parse_args()

    ray.init(local_mode=False, num_gpus=1, num_cpus=16)
    # MultiAgentPM2 = make_multi_agent_divide_and_conquer(lambda config: TaskSettablePointMass2D(config))
    env_name = "HalfCheetahTorsoMassGravity"
    model = [128, 128, 128 ]
    model_name = '3x128'
    version = 'V4.1.5'
    num_agents = 3
    iteration_ts = tune.grid_search([ #1024 *num_agents,
                                       2048*num_agents,
                                      ])
    log_dir = "./results/"+env_name+"/" +version +"/Standard"
    BaseLine = True   # just to control experiments
    Experimental = True
    agent_config_sp = [{"target_mean": np.array([9.45, -9.8]),"target_var": np.array([0.04, 0.04])},
     {'curriculum': 'self_paced',
      "target_mean": np.array([15, -12]), "target_var":np.square([0.5, .1]),
      "init_mean": np.array([11, -9]), "init_var": np.square([6, 4])},
     {'curriculum': 'self_paced',
      "target_mean": np.array([7, -6]), "target_var":np.square([0.04, 0.04]),
      "init_mean":np.array([11,-9]), "init_var":np.square([6, 4])}]


    agent_config_eval = [{
        "target_mean": np.array([9.45, -9.8]), "target_var": np.array([0.04,0.04])},
     {
        "target_mean": np.array([15, -12]), "target_var":np.square([0.5, 0.1]),},
     {
       "target_mean": np.array([7, -6]), "target_var":np.square([0.04, 0.04]),}]

    agent_config_default = [{
        "target_mean": np.array([9.45, -9.8]), "target_var": np.array([0.04,0.04])},
     {
        "target_mean": np.array([15, -12]), "target_var":np.square([0.5, 0.1]),},
     {
       "target_mean": np.array([7, -6]), "target_var":np.square([0.04, 0.04]),}]

    dummy_env = SPEnv(config={})
    model_config = {"fcnet_hiddens": model,
                    "fcnet_activation": "relu",
                    }
    dist_class, logit_dim = ModelCatalog.get_action_dist(
        dummy_env.action_space, model_config, framework='torch'
    )

    central_policy, _ = FC_MLP(
        obs_space=dummy_env.observation_space,
        action_space=dummy_env.action_space,
        num_outputs=logit_dim,
        model_config=model_config,
    )




    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == 0:
            return "distilled_policy"
        else:
            return "learner_{}".format(agent_id - 1)



    # Setup PPO with an ensemble of `num_policies` different policies.
    env_config = {"num_agents": num_agents, "agent_config": agent_config_sp}

    policies = gen_policy(num_agents, central_policy=central_policy,model_config=model_config, samples_before_training= SamplesOffset)
    policy_ids = list(policies.keys())
    multiagent_config = {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": list(policies.keys())[1:],
        "count_steps_by": "agent_steps",
    }

    config = {
        "env": MAEnv,
        "env_config": env_config,
        # "env_task_fn": Self_Paced_MACL,
        "disable_env_checking": True,
        "num_workers": 1,
        "num_envs_per_worker":4,
        "callbacks":MASPCL,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": .5,
        "evaluation_interval": 50,
        # Run 10 episodes each time evaluation runs (OR "auto" if parallel to training).
        "evaluation_duration": 64,
        # "evaluation_duration": 4,

        "custom_eval_function": DnCCrossEvalSeries,
        "evaluation_config": {
            "env_config": {"num_agents": num_agents,
                           "agent_config": agent_config_eval}
        },
        "replay_buffer_config" : {
        "_enable_replay_buffer_api": True,
        "type": "MultiAgentReplayBuffer",
        "capacity": int(2e5),
        # How many steps of the model to sample before learning starts.
        "learning_starts": SamplesOffset*(num_agents-1),},
        # "num_steps_sampled_before_learning_starts": 1024*16 * num_agents,
        # "optimization": {
        #     "actor_learning_rate": 5e-4,
        #     "critic_learning_rate": 5e-3,
        #     "entropy_learning_rate": 5e-4,
        # },
        "multiagent": multiagent_config,
        "seed": tune.grid_search([0,1,
                                  # 2,3,4,5,6,7,8,9,10,
                                  # 11,12,13,14,15,
                                  ]),
        # "seed":0,
        # "monitor": True,
        "train_batch_size": 256,
        "evaluation_num_workers": 1,
        "framework": 'torch',
        "min_sample_timesteps_per_iteration": iteration_ts ,
        "clip_actions": False,
        "normalize_actions": True,
        "tau": 0.005,
     "horizon": max_steps,
        # "batch_mode": "complete_episodes",
        "n_step": 1,
        "rollout_fragment_length": 1,
        "no_done_at_end": True,
        "soft_horizon": False

    }

    distral_config = {"distral_alpha": tune.grid_search([.5,1]),
        "distral_beta" : tune.grid_search([ 100, ])}
    # alg = MyAlgo(config=config)
    stop = {
        "training_iteration": 300,

    }
    # alg = SAC(config = config)
    # for _ in range(100):
    #     res = alg.train()
    #     alg.evaluate()
    #     print(res)
    # DnC_PPO, CentralLocalPolicy, name = create_central_local_learner(PPO,config)
    ### 1/6 self paced experiemnts context hidden context
    if BaseLine:
        results = tune.Tuner(
            SAC, param_space=config,
            run_config=air.RunConfig(stop=stop, verbose=1, name="SAC_selfpaced_"+model_name,
                                     local_dir=log_dir,
                                     checkpoint_config=air.CheckpointConfig(checkpoint_frequency=100,
                                                                            checkpoint_at_end=True))
        ).fit()

    if Experimental:
        config.update(distral_config)
        results = tune.Tuner(
            Distral, param_space=config,
            run_config=air.RunConfig(stop=stop, verbose=1, name="Distral_selfpaced_"+model_name,
                                     local_dir=log_dir, checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
                                                                                               checkpoint_at_end=True))
        ).fit()
        [config.pop(k) for k in distral_config.keys()]
    #
    #     ### 2/6 default experiments with hidden context
    env_config.update({"agent_config": agent_config_default, })
    config.update({"env_config": env_config, })
    if BaseLine:
        results = tune.Tuner(
            SAC, param_space=config,
            run_config=air.RunConfig(stop=stop, verbose=1, name="SAC_default_"+model_name,
                                     local_dir=log_dir,
                                     checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
                                                                            checkpoint_at_end=True))
        ).fit()


    if Experimental:
        config.update(distral_config)

        results = tune.Tuner(
            Distral, param_space=config,
            run_config=air.RunConfig(stop=stop, verbose=1, name="Distral_default_"+model_name,
                                     local_dir=log_dir, checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
                                                                                               checkpoint_at_end=True))
        ).fit()
        [config.pop(k) for k in distral_config.keys()]

        ### 3/6 default experiments with visible context

    dummy_env = SPEnvVis(config={})
    model_config = {"fcnet_hiddens": model,
                    "fcnet_activation": "relu",
                    }
    dist_class, logit_dim = ModelCatalog.get_action_dist(
        dummy_env.action_space, model_config, framework='torch'
    )

    central_policy, _ = FC_MLP(
        obs_space=dummy_env.observation_space,
        action_space=dummy_env.action_space,
        num_outputs=logit_dim,
        model_config=model_config,
    )
    num_agents = 3
    policies = gen_policy(num_agents, central_policy=central_policy, model_config=model_config,
                      samples_before_training=SamplesOffset)
    policy_ids = list(policies.keys())
    multiagent_config = {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": list(policies.keys())[1:],
        "count_steps_by": "agent_steps",
    }

    config.update({'env': MAEnvVis, "multiagent": multiagent_config, })

    if BaseLine:
        results = tune.Tuner(
            SAC, param_space=config,
            run_config=air.RunConfig(stop=stop, verbose=1, name="SAC_default_ctxvis_"+model_name,
                                     local_dir=log_dir,
                                     checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
                                                                            checkpoint_at_end=True))
        ).fit()
    # config.update(distral_config)
    if Experimental:
        config.update(distral_config)

        results = tune.Tuner(
            Distral, param_space=config,
            run_config=air.RunConfig(stop=stop, verbose=1, name="Distral_default_ctxvis_"+model_name,
                                     local_dir=log_dir, checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
                                                                                               checkpoint_at_end=True))
        ).fit()
        [config.pop(k) for k in distral_config.keys()]
        ### 4/6 self paced experiments with visible context
    env_config.update({"agent_config": agent_config_sp, })
    config.update({"env_config": env_config, })
    config.update({'env': MAEnvVis, })
    #
    if BaseLine:
        results = tune.Tuner(
            SAC, param_space=config,
            run_config=air.RunConfig(stop=stop, verbose=1, name="SAC_selfpaced_ctxvis_"+model_name,
                                     local_dir=log_dir,
                                     checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
                                                                            checkpoint_at_end=True))
        ).fit()
    if Experimental:
        config.update(distral_config)

        results = tune.Tuner(
            Distral, param_space=config,
            run_config=air.RunConfig(stop=stop, verbose=1, name="Distral_selfpaced_ctxvis_" + model_name,
                                     local_dir=log_dir, checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
                                                                                               checkpoint_at_end=True))
        ).fit()
        [config.pop(k) for k in distral_config.keys()]

        ### 5/6 self paced with observation normalization

        # # env_config.update({"agent_config": agent_config_sp, })
        # # config.update({"env_config": env_config, })
        # config.update({'env': MAEnvVisNorm, })
        # results = tune.Tuner(
        #     SAC, param_space=config,
        #     run_config=air.RunConfig(stop=stop, verbose=1, name="SAC_" + env_name + "_obsnorm_selfpaced_ctxvis_"+model_name+"_" +version,
        #                              local_dir=log_dir,
        #                              checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
        #                                                                     checkpoint_at_end=True))
        # ).fit()
        #
        # results = tune.Tuner(
        #     Distral, param_space=config,
        #     run_config=air.RunConfig(stop=stop, verbose=1, name="Distral_" + env_name + "_obsnorm_selfpaced_ctxvis_"+model_name+"_" +version,
        #                              local_dir=log_dir, checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
        #                                                                                        checkpoint_at_end=True))
        # ).fit()

        ### 6/6 default with obs norm
        # new_envconfig = {"num_agents": num_agents, "agent_config": agent_config_default}
        # config.update({"env_config": new_envconfig, })
        # results = tune.Tuner(
        #     SAC, param_space=config,
        #     run_config=air.RunConfig(stop=stop, verbose=1, name="SAC_" + env_name + "_obsnorm_default_ctxvis_"+model_name+"_" +version,
        #                              local_dir=log_dir,
        #                              checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
        #                                                                     checkpoint_at_end=True))
        # ).fit()
        # results = tune.Tuner(
        #     Distral, param_space=config,
        #     run_config=air.RunConfig(stop=stop, verbose=1, name="Distral_" + env_name + "_obsnorm_default_ctxvis_"+model_name+"_" +version,
        #                              local_dir=log_dir, checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
        #                                                                                        checkpoint_at_end=True))
        # ).fit()

        # # #
        # for _ in range(10):
        #     res = alg.train()
        #     alg.evaluate()
        #     print(res)

    ray.shutdown()