

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
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
# from ray.rllib.examples.models.shared_weights_model import (
#     SharedWeightsModel1,
#     SharedWeightsModel2,
#     TF2SharedWeightsModel,
#     TorchSharedWeightsModel,
# )
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
torch, nn = try_import_torch()
from envs.contextual_env import  CtxDictWrapper, ctx_visibility, exp_group, GMMCtxEnvWrapper

from envs.contextual_env import make_multi_agent_divide_and_conquer
from envs.point_mass_2d import TaskSettablePointMass2D
from ray.rllib.env.multi_agent_env import make_multi_agent

import ray
# from ray.rllib.algorithms.dqn import DQN, DQNTFPolicy, DQNTorchPolicy
from ray.rllib.algorithms.ppo import (
    PPO, PPOConfig,
    PPOTorchPolicy,
)

from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from gymnasium.spaces import flatten_space
from gymnasium.wrappers import TimeLimit
from envs.point_mass_2d import PointMassEnv
torch.autograd.set_detect_anomaly(True)
# torch.set_default_tensor_type(torch.DoubleTensor)
ctx_mode =1
# ctx_norm= None
ctx_lb = np.array([-4, .5])
ctx_ub = np.array([4, 4])
std_lb = np.array([0.2, 0.1875])
max_steps = 128

env_creator = lambda config: GMMCtxEnvWrapper(TimeLimit(PointMassEnv(context=np.array([3, .5])), max_episode_steps=max_steps ),
                                        ctx_lb=ctx_lb, ctx_ub = ctx_ub,ctx_visibility= ctx_mode, **config )

# SPEnv = lambda: gymEnvWrapper(spgym())
MAEnv = make_multi_agent_divide_and_conquer(lambda config: env_creator(config))
TORCH_GLOBAL_SHARED_LAYER = None
if torch:
    # The global, shared layer to be used by both models.
    TORCH_GLOBAL_SHARED_LAYER = SlimFC(
        6,
        1,
        activation_fn=nn.ReLU,
        initializer=torch.nn.init.xavier_uniform_,
    )




# class CentralCriticPPOPolicy()

class FCSharedVFTorchModel(FullyConnectedNetwork):
    """Example of weight sharing between two different TorchModelV2s.

    The shared (single) layer is simply defined outside of the two Models,
    then used by both Models in their forward pass.
    """

    def __init__(
        self, observation_space, action_space, num_outputs, model_config,  name
    ):

        custom_model_config = model_config.pop('custom_model_config')
        super().__init__(
           observation_space, action_space, num_outputs, model_config, name
        )
        self._global_shared_vf = custom_model_config['value_net']
        # Non-shared initial layer.
        self._output = None



    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self._global_shared_vf(self._last_flat_in).squeeze(1)

num_policies = 2


# Each policy can have a different configuration (including custom model).
def gen_policy(i):

    name = "local_policy_" + str(i)
    config = {
        "model": {

            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "tanh", }
    }

    return name, PolicySpec(config=config)


# Setup PPO with an ensemble of `num_policies` different policies.
policies = {}
# "policy_{}".format(i): gen_policy()
for i in range(num_policies):
    k, v = gen_policy(i)
    policies[k] = v
policy_ids = list(policies.keys())


if __name__ == "__main__":
    # args = parser.parse_args()

    ray.init( )

    # Register the models to use.

    mod1 = FCSharedVFTorchModel
    ModelCatalog.register_custom_model("sharedVF", mod1)
    # ModelCatalog.register_custom_model("model2", mod2)
    num_policies =4
    # Each policy can have a different configuration (including custom model).
    def gen_policy():
        config = PPOConfig.overrides(
           model =  {
                "custom_model": "sharedVF",

                         "fcnet_hiddens": [64, 64 ],
                         "fcnet_activation": "tanh",
                         "custom_model_config": {"value_net": TORCH_GLOBAL_SHARED_LAYER,},}
        )


        return PolicySpec(config=config)

    # Setup PPO with an ensemble of `num_policies` different policies.
    policies = {"local_policy_{}".format(i): gen_policy() for i in range(num_policies)}
    policy_ids = list(policies.keys())


    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "local_policy_" + str(agent_id)

    config = (
        PPOConfig()
        .environment(MAEnv, auto_wrap_old_gym_envs=False, env_config={"num_agents": 2,
                                                                     'non_training_envs': 0,
                                                              "agent_config":
                                                                [{"target_mean": np.array([2.5, 1])},
                                                                {"target_mean": np.array([-2.5, 1])}]},)
        .framework('torch')
        .training(num_sgd_iter=10)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .resources(num_gpus=1)

    )
    # config = {MultiAgentPM2, "disable_env_checking": True,
    #           # "num_workers":
    #
    #           "env": MultiAgentPM2,
    #           "env_config": {"num_agents": 2, "agent_config": [{"target_mean": np.array([2.5, 1])},
    #                                                            {"target_mean": np.array([-2.5, 1])}]},
    #           "multiagent": {
    #               "policies": policies,
    #               "policy_mapping_fn": policy_mapping_fn,
    #               "policies_to_train": list(policies.keys()),
    #           },
    #           "num_workers": 4,
    #           # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    #           "num_gpus": .5,
    #           "evaluation_interval": 1,
    #           # Run 10 episodes each time evaluation runs (OR "auto" if parallel to training).
    #           "evaluation_duration": 10,
    #           # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    #           "seed": tune.grid_search([0, 1, 2, 3]),
    #           "evaluation_num_workers": 2,
    #           "framework": "torch",
    #           "evaluation_config": {
    #               "env_config": {"num_agents": 2,
    #                              "agent_config": [{"target_mean": np.array([[2.5, 1], [-2.5, 1]]),
    #                                                "target_var": np.array([[4e-3, 3.75e-3], [4e-3, 3.75e-3]]),
    #                                                "target_priors": np.array([1, 1])},
    #                                               {"target_mean": np.array([[2.5, 1], [-2.5, 1]]),
    #                                                "target_var": np.array([[4e-3, 3.75e-3], [4e-3, 3.75e-3]]),
    #                                                "target_priors": np.array([1, 1])}, ]},
    #           },
    #           "train_batch_size": 2048,
    #           'lr': 0.0003,
    #
    #           }

    # alg = PPO(config=config)
    #
    #
    # for _ in range(10):
    #     print(alg.train())
    stop = {
        # "episode_reward_mean": args.stop_reward,
        # "timesteps_total": args.stop_timesteps,
        "training_iteration": 50,
    }
    results = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=1),
    ).fit()
    ray.shutdown()