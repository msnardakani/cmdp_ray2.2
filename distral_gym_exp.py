from typing import Dict, List, Type, Union

import argparse
import os
import random

# __rllib-in-60s-end__
from ray.rllib.env.multi_agent_env import make_multi_agent

from distral.distral import Distral
from distral.distral_torch_model import DistilledTorchModel
from envs.cartpole_mass import CartPoleMassEnv
from utils import FC_MLP

"""Simple example of setting up a multi-agent policy mapping.

Control the number of agents and policies via --num-agents and --num-policies.

This works with hundreds of agents and policies, but note that initializing
many TF policies will take some time.

Also, TF evals might slow down with large numbers of policies. To debug TF
execution, set the TF_TIMELINE_DIR environment variable.
"""

import argparse
import os
import random

import ray
from ray import air, tune
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.examples.models.shared_weights_model import (
    SharedWeightsModel1,
    SharedWeightsModel2,
    TF2SharedWeightsModel,
    TorchSharedWeightsModel,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved

tf1, tf, tfv = try_import_tf()

from ray.rllib.algorithms.sac import SAC
from ray.rllib.policy.sample_batch import SampleBatch

import ray
from ray import air, tune
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPO
# from ray.rllib.examples.env.multi_agent import MultiAgentCartPole as env
# from ray.rllib.examples.models.shared_weights_model import (
#     SharedWeightsModel1,
#     SharedWeightsModel2,
#     TF2SharedWeightsModel,
#     TorchSharedWeightsModel,
# )
from ray.rllib.models import ModelCatalog, ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2, flatten
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

from env_funcs import make_multi_agent_divide_and_conquer
from envs.point_mass_2d import TaskSettablePointMass2D
from utils import FC_MLP
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import TensorType
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.utils.annotations import (
    DeveloperAPI,
    OverrideToImplementCustomLogic,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
    is_overridden,
    override,
 )
torch, nn = try_import_torch()
from distral.distral import Distral

if __name__ == "__main__":
    # args = parser.parse_args()

    ray.init(local_mode=True)
    # MultiAgentPM2 = make_multi_agent_divide_and_conquer(lambda config: TaskSettablePointMass2D(config))

    dummy_env = MultiAgentCartPole({})
    model_config = {"fcnet_hiddens": [64, 64],
                    "fcnet_activation": "tanh",
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
    MultiAgentCartPole = make_multi_agent("CartPole-v0")

    # MultiAgentPM2 = make_multi_agent_divide_and_conquer(lambda config: TaskSettablePointMass2D(config))

    dummy_env = CartPoleMassEnv()
    model_config = {"fcnet_hiddens": [64, 64],
                    "fcnet_activation": "tanh",
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

    # Register the models to use.

    config = {
        "env": MultiAgentCartPole,
        "env_config": {"num_agents": 2,}
                       ,
        "disable_env_checking": True,
        "num_workers":8,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus":0.5,
        "evaluation_interval": 1,
        # Run 10 episodes each time evaluation runs (OR "auto" if parallel to training).
        "evaluation_duration": 20,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            # "policies_to_train": list(policies.keys()),
        },
        "seed":  tune.grid_search([0,1,2,3]),
        # "seed":0,
        "evaluation_num_workers": 2,
        "framework": 'torch',
        "evaluation_config": {
            "env_config": {"num_agents": 2,
},
    },
        "train_batch_size": 2048,
        'lr':0.0003,
        "horizon": 100,

    }

    # alg = MyAlgo(config=config)
    stop = {
        "training_iteration": 400,

    }
    # alg = SAC(config = alg_config)
    # # DnC_PPO, CentralLocalPolicy, name = create_central_local_learner(PPO,config)
    results = tune.Tuner(
        Distral, param_space=config, run_config=air.RunConfig(stop=stop, verbose=1,name="Distral_cartpoleV3.0")
    ).fit()
    # # #
    # for _ in range(10):
    #     res = alg.train()
    #     print(res)
    ray.shutdown()