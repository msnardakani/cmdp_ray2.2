import argparse
import gym
import os

import numpy as np
from ray import air
from ray.rllib.policy.policy import PolicySpec
from ray.train.trainer import tune

from env_funcs import make_multi_agent_divide_and_conquer
from envs.point_mass_2d import TaskSettablePointMass2D
from ray.rllib.env.multi_agent_env import make_multi_agent

import ray
# from ray.rllib.algorithms.dqn import DQN, DQNTFPolicy, DQNTorchPolicy
from ray.rllib.algorithms.ppo import (
    PPO,
    PPOTorchPolicy,
)
from ray.rllib.algorithms.ppo.ppo import DEFAULT_CONFIG as PPO_CONFIG

from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env


MultiAgentPM2= make_multi_agent_divide_and_conquer(lambda config: TaskSettablePointMass2D(config))


# env =MultiAgentPM2({"num_agents":2, "agent_config":[{ "init_mean": np.array([0,.5]),"target_mean": np.array([2.5, .5])},{ "init_mean": np.array([0,8]),"target_mean": np.array([-2.5, .5])},]})
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


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "local_policy_" + str(agent_id)


config = {"disable_env_checking":True,
          # "num_workers":

          "env": MultiAgentPM2,
    "env_config":{"num_agents":2, "agent_config": [{ "target_mean": np.array([2.5, 1])},
                                        { "target_mean": np.array([-2.5, 1])}]},
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": ["local_policy_"+str(i) for i in range(2)],
    },
          "num_workers": 4,
          # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
          "num_gpus": .5,
          "evaluation_interval": 1,
          # Run 10 episodes each time evaluation runs (OR "auto" if parallel to training).
          "evaluation_duration": 10,
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
          "seed": tune.grid_search([0, 1, 2, 3]),
          "evaluation_num_workers": 2,
    "framework": "torch" ,
          "evaluation_config": {
              "env_config": {"num_agents": 2,
                             "agent_config": [{"target_mean": np.array([[2.5, 1], [-2.5, 1]]),
                                               "target_var": np.array([[4e-3, 3.75e-3], [4e-3, 3.75e-3]]),
                                               "target_priors": np.array([1, 1])},
                                              {"target_mean": np.array([[2.5, 1], [-2.5, 1]]),
                                               "target_var": np.array([[4e-3, 3.75e-3], [4e-3, 3.75e-3]]),
                                               "target_priors": np.array([1, 1])},]},
          },
          "train_batch_size": 2048,
          'lr': 0.0003,

          }

ray.init()
stop = {
    "training_iteration": 400,

}

results = tune.Tuner(
    "PPO", param_space=config, run_config=air.RunConfig(stop=stop, verbose=1, name="PPO_PM2")
).fit()

#
# algo = PPO(config=config)
#
# # Run it for n training iterations. A training iteration includes
# # parallel sample collection by the environment workers as well as
# # loss calculation on the collected batch and a model update.
# for _ in range(10):
#     print(algo.train())
#
# # Evaluate the trained Trainer (and render each timestep to the shell's
# # output).
# print(algo.evaluate())


# print(env.reset())
# env = MultiAgentTrafficEnv(num_cars=2, num_traffic_lights=1)
#