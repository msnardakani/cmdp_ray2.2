
import gym
# from gym.wrappers.step_api_compatibility import StepAPICompatibility
# from gym.envs.box2d.lunar_lander import
from gym.wrappers import TimeLimit
from ray import air, tune
from ray.rllib.algorithms.sac import SAC

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

from env_funcs import make_multi_agent_divide_and_conquer
from envs.point_mass_2d import TaskSettablePointMass2D
from utils import FC_MLP

torch, nn = try_import_torch()
from distral.distral import Distral
from envs.lunar_lander import LunarLanderCtx
# gym.make("LunarLander-v2", continuous=True, **config)
# e = gym.make("LunarLander-v2", continuous=True, gravity =-11), False)gym.make("LunarLander-v2", continuous=True, gravity =-11)
MultiAgentLunarLander = make_multi_agent_divide_and_conquer(lambda config: TimeLimit(LunarLanderCtx( continuous=True, **config), 1000))

if __name__ == "__main__":
    # args = parser.parse_args()

    ray.init(local_mode=False)
    # MultiAgentPM2 = make_multi_agent_divide_and_conquer(lambda config: TaskSettablePointMass2D(config))

    dummy_env = LunarLanderCtx( continuous=True)
    model_config = {"fcnet_hiddens": [128, 128, 128, 128],
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
    # model = ModelCatalog.get_model_v2(
    #     dummy_env.observation_space,
    #     dummy_env.action_space,
    #     logit_dim,
    #    model_config,
    #     framework="torch",
    #     name="distilled_model",
    # # )
    # model = DistilledTorchModel(obs_space=dummy_env.observation_space, action_space=dummy_env.action_space,
    #                             model_config={
    #                                 "distilled_model": central_policy}, num_outputs=None, name="central_policy")


    # Each policy can have a different configuration (including custom model).

    # model.set_action_model(central_policy)
    # print(model.variables())
    def gen_policy(i):
        if False:
            config = {
                "custom_model": 'central_model',
                "custom_model_config": {"distilled_model": central_policy, },
                # # "q_model_config": {
                # #     "fcnet_hiddens": [[64, 64], [32, 32]][i % 2],
                # # },
                # # "gamma": random.choice([0.95, 0.99]),
            }
        else:
            config = {
                "custom_model": 'local_model',
                "custom_model_config": {"distilled_model": central_policy, },
                "q_model_config": model_config,
                "policy_model_config":model_config,
                # # "gamma": random.choice([0.95, 0.99]),
            }
        # config = {
        #         "custom_model": 'local_model',
        #         "custom_model_config": {"distilled_model" : central_policy,},
        #         # # "q_model_config": {
        #         # #     "fcnet_hiddens": [[64, 64], [32, 32]][i % 2],
        #         # # },
        #         # # "gamma": random.choice([0.95, 0.99]),
        #     }
        return PolicySpec(config=config)


    num_agents = 2
    # Setup PPO with an ensemble of `num_policies` different policies.
    policies = {"policy_{}".format(i): gen_policy(i) for i in range(num_agents)}
    policy_ids = list(policies.keys())


    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # if agent_id ==0:
        #     return "central_policy"
        # else:
        return "policy_" + str(agent_id)

    #
    # config = {
    #     "env": MultiAgentCartPole,
    #     "env_config": {
    #         "num_agents": num_agents,
    #     },
    #     # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    #     "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    #     "multiagent": {
    #         "policies": policies,
    #         "policy_mapping_fn": policy_mapping_fn,
    #         "policies_to_train": list(policies.keys())[1:],
    #     },
    #     "framework": "torch",
    #
    # }
    # MultiAgentCartPole = make_multi_agent("CartPole-v0")

    # MultiAgentPM2 = make_multi_agent_divide_and_conquer(lambda config: TaskSettablePointMass2D(config))
    #
    # dummy_env = CartPoleMassEnv()
    # model_config = {"fcnet_hiddens": [64, 64],
    #                 "fcnet_activation": "tanh",
    #                 }
    # dist_class, logit_dim = ModelCatalog.get_action_dist(
    #     dummy_env.action_space, model_config, framework='torch'
    # )
    #
    # central_policy, _ = FC_MLP(
    #     obs_space=dummy_env.observation_space,
    #     action_space=dummy_env.action_space,
    #     num_outputs=logit_dim,
    #     model_config=model_config,
    # )

    # Register the models to use.

    config = {
        "env": MultiAgentLunarLander,
        "env_config": {"num_agents": 2, "agent_config": [{"gravity": -10}, {"gravity": -11.0}]},
        "disable_env_checking": True,
        "num_workers": 12,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 1,
        "evaluation_interval": 10,
        # Run 10 episodes each time evaluation runs (OR "auto" if parallel to training).
        "evaluation_duration": 100,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            # "policies_to_train": list(policies.keys()),
        },
        "seed":  tune.grid_search([0, 1,  ]),
        # "seed":0,
        "evaluation_num_workers": 2,
        "framework": 'torch',


    }

    # alg = MyAlgo(config=config)
    stop = {
        "training_iteration": 1000,

    }
    # alg = Distral(config = config)
    # # DnC_PPO, CentralLocalPolicy, name = create_central_local_learner(PPO,config)
    results = tune.Tuner(
        Distral, param_space=config, run_config=air.RunConfig(stop=stop, verbose=1, name="Distral_LunarLander_4x128_10-11_V4.0",local_dir="./test/", checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10, checkpoint_at_end= True))
    ).fit()
    # results = tune.Tuner(
    #     SAC, param_space=config, run_config=air.RunConfig(stop=stop, verbose=1, name="SAC_LunarLander_4x128_10-11_V3.2",local_dir="./test/",checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10, checkpoint_at_end= True))
    # ).fit()
    # # #
    # for _ in range(10):
    #     res = alg.train()
    #     print(res)
    ray.shutdown()