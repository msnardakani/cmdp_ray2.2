
import gym
# from gym.wrappers.step_api_compatibility import StepAPICompatibility
# from gym.envs.box2d.lunar_lander import
import numpy as np
from gym.wrappers import TimeLimit
from ray import air, tune
from ray.rllib.algorithms.sac import SAC
from ray.rllib.evaluation.metrics import summarize_episodes, collect_episodes

"""Simple example of setting up a multi-agent policy mapping.

Control the number of agents and policies via --num-agents and --num-policies.

This works with hundreds of agents and policies, but note that initializing
many TF policies will take some time.

Also, TF evals might slow down with large numbers of policies. To debug TF
execution, set the TF_TIMELINE_DIR environment variable.
"""

# import gym.envs.box2d.lunar_lander
import ray

from ray.rllib.models import ModelCatalog, ActionDistribution
from ray.rllib.policy.policy import PolicySpec

from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

from env_funcs import make_multi_agent_divide_and_conquer, Self_Paced_CL, Self_Paced_MACL
from envs.point_mass_2d import TaskSettablePointMass2D
from utils import FC_MLP, eval_fn, MASPCL
from gym.wrappers.normalize import NormalizeObservation
torch, nn = try_import_torch()
from distral.distral import Distral
from envs.point_mass_2d import PointMassEnv
from envs.utils import make_DnC_env
from envs.utils import gymEnvWrapper
from utils import SPCL_Eval
# gym.make("LunarLander-v2", continuous=True, **config)
# e = gym.make("LunarLander-v2", continuous=True, gravity =-11), False)gym.make("LunarLander-v2", continuous=True, gravity =-11)
# cart_masses = np.random.uniform(low=0.5, high=2.0, size=(n_tasks, 1))
#         pole_masses = np.random.uniform(low=0.05, high=0.2, size=(n_tasks, 1))
ctx_norm = (np.array([1, .1]),np.array([.25, .05]))
# ctx_norm= None
ctx_lb = np.array([-4, .5])
ctx_ub = np.array([4, 4])
std_lb = np.array([0.2, 0.1875])
TimeLimit = 128


env_creator = lambda : PointMassEnv(context=np.array([3, .5]) )
SPEnv = make_DnC_env(env_creator, ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_visible=False,
                             time_limit=TimeLimit, ctx_norm= None,
                     std_lb=std_lb, )
# SPEnv = lambda: gymEnvWrapper(spgym())
MAEnv = make_multi_agent_divide_and_conquer(lambda config: SPEnv(config))

SPEnvVis = make_DnC_env(env_creator, ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_visible=True,
                             time_limit=TimeLimit, ctx_norm=None, std_lb=std_lb,)

MAEnvVis = make_multi_agent_divide_and_conquer(lambda config: SPEnvVis(config))

#
# SPEnvVisNorm = make_DnC_env(env_creator, ctx_lb=np.array([0.5, 0.05]), ctx_ub=np.array([2, 0.2]), ctx_visible=True,
#                              time_limit=TimeLimit, ctx_norm=ctx_norm)
#
# MAEnvVisNorm = make_multi_agent_divide_and_conquer(lambda config: SPEnvVis(config))



if __name__ == "__main__":
    # args = parser.parse_args()

    ray.init(local_mode=False, num_gpus=1, num_cpus=12)
    # MultiAgentPM2 = make_multi_agent_divide_and_conquer(lambda config: TaskSettablePointMass2D(config))
    env_name = "PointMass2D"
    model = [64, 64, 64 ]
    model_name = '3x64'
    version = 'V3.5.1'
    num_agents = 3
    iteration_ts = tune.grid_search([ 1024 *num_agents,
                                      # 2048*num_agents,])
                                      ])
    log_dir = "./results/"+env_name+"/" +version +"/Standard"
    BaseLine = True   # just to control experiments
    Experimental = True
    agent_config_sp = [{"target_mean": np.array([2.5, .5])},
     {'curriculum': 'self_paced',
      "target_mean": np.array([3, 2]), "target_var":np.square([4e-3, 3.75e-3]),
      "init_mean":np.array([0,4.25]), "init_var":np.square([2, 1.875])},
     {'curriculum': 'self_paced',
      "target_mean": np.array([1.5, .5]), "target_var":np.square([4e-3, 3.75e-3]),
      "init_mean":np.array([0,4.25]), "init_var":np.square([2, 1.875])}]


    agent_config_eval = [{"target_mean": np.array([2.5, .5])},
                                            {"target_mean": np.array([3, 2]), },
                                            {"target_mean": np.array([1.5, .5]), }]

    agent_config_default = [{"target_mean": np.array([2.5, .5])},
                                                              {'curriculum': 'default',
                                                               "target_mean": np.array([3, 2]), },
                                                              {'curriculum': 'default',
                                                           "target_mean": np.array([1.5, .5]), }]

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

    def gen_policy(i):
        out = dict()
        for i in range(i):
            if i == 0:
                config = {
                    "custom_model": 'central_model',
                    "custom_model_config": {"distilled_model": central_policy, },
                    "q_model_config": model_config,
                    "policy_model_config": model_config,

                }
                out['distilled_policy'] = PolicySpec(config=config)
            else:
                config = {
                    "custom_model": 'local_model',
                    "custom_model_config": {"distilled_model": central_policy, },
                    "q_model_config": model_config,
                    "policy_model_config": model_config,
                    # # "gamma": random.choice([0.95, 0.99]),
                    "num_steps_sampled_before_learning_starts": 3000

                }
                out["learner_{}".format(i - 1)] = PolicySpec(config=config)

        return out


    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == 0:
            return "distilled_policy"
        else:
            return "learner_{}".format(agent_id - 1)



    # Setup PPO with an ensemble of `num_policies` different policies.
    env_config = {"num_agents": num_agents, "agent_config": agent_config_sp}

    policies = gen_policy(num_agents)
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
        "env_task_fn": Self_Paced_MACL,
        "disable_env_checking": True,
        "num_workers": 1,
        "num_envs_per_worker": 1,
        "callbacks":MASPCL,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": .3,
        "evaluation_interval": 10,
        # Run 10 episodes each time evaluation runs (OR "auto" if parallel to training).
        "evaluation_duration": 64,
        "custom_eval_function": eval_fn,
        "evaluation_config": {
            "env_config": {"num_agents": num_agents,
                           "agent_config": agent_config_eval}
        },
        # "num_steps_sampled_before_learning_starts": 1024*16 * num_agents,
        # "optimization": {
        #     "actor_learning_rate": 5e-4,
        #     "critic_learning_rate": 5e-3,
        #     "entropy_learning_rate": 5e-4,
        # },
        "multiagent": multiagent_config,
        "seed": tune.grid_search([0, 1]),
        # "seed":0,
        # "monitor": True,
        "train_batch_size": 256,
        "evaluation_num_workers": num_agents,
        "framework": 'torch',
        "min_sample_timesteps_per_iteration": iteration_ts ,

    }

    # alg = MyAlgo(config=config)
    stop = {
        "training_iteration": 500,

    }
    alg = SAC(config = config)
    # for _ in range(100):
    #     res = alg.train()
    #     alg.evaluate()
    #     print(res)
    # DnC_PPO, CentralLocalPolicy, name = create_central_local_learner(PPO,config)
    try:
    ### 1/6 self paced experiemnts context hidden context
        if BaseLine:
            results = tune.Tuner(
                SAC, param_space=config,
                run_config=air.RunConfig(stop=stop, verbose=1, name="SAC_selfpaced_"+model_name,
                                         local_dir=log_dir,
                                         checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
                                                                                checkpoint_at_end=True))
            ).fit()

        if Experimental:
            results = tune.Tuner(
                Distral, param_space=config,
                run_config=air.RunConfig(stop=stop, verbose=1, name="Distral_selfpaced_"+model_name,
                                         local_dir=log_dir, checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
                                                                                                   checkpoint_at_end=True))
            ).fit()

            ### 2/6 default experiments with hidden context
        env_config.update({"agent_config": agent_config_default, })
        config.update({"env_config": env_config, })
        if BaseLine:
            results = tune.Tuner(
                SAC, param_space=config,
                run_config=air.RunConfig(stop=stop, verbose=1, name="SAC_default_"+model_name,
                                         local_dir=log_dir,
                                         checkpoint_config=air.CheckpointConfig(checkpoint_frequency=50,
                                                                                checkpoint_at_end=True))
            ).fit()


        if Experimental:
            results = tune.Tuner(
                Distral, param_space=config,
                run_config=air.RunConfig(stop=stop, verbose=1, name="Distral_default_"+model_name,
                                         local_dir=log_dir, checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
                                                                                                   checkpoint_at_end=True))
            ).fit()

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
        policies = gen_policy(num_agents)
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
        if Experimental:

            results = tune.Tuner(
                Distral, param_space=config,
                run_config=air.RunConfig(stop=stop, verbose=1, name="Distral_default_ctxvis_"+model_name,
                                         local_dir=log_dir, checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
                                                                                                   checkpoint_at_end=True))
            ).fit()

            ### 4/6 self paced experiments with visible context
        env_config.update({"agent_config": agent_config_sp, })
        config.update({"env_config": env_config, })
        config.update({'env': MAEnvVis, })
        if BaseLine:
            results = tune.Tuner(
                SAC, param_space=config,
                run_config=air.RunConfig(stop=stop, verbose=1, name="SAC_selfpaced_ctxvis_"+model_name,
                                         local_dir=log_dir,
                                         checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
                                                                                checkpoint_at_end=True))
            ).fit()

        if Experimental:
            results = tune.Tuner(
                Distral, param_space=config,
                run_config=air.RunConfig(stop=stop, verbose=1, name="Distral_selfpaced_ctxvis_" + model_name,
                                         local_dir=log_dir, checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200,
                                                                                                   checkpoint_at_end=True))
            ).fit()

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
    except KeyboardInterrupt:
        ray.shutdown()
    ray.shutdown()