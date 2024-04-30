# # # __rllib-in-60s-begin__
# # # Import the RL algorithm (Algorithm) we would like to use.
# #
# import argparse
# import numpy as np
# import os
#
# import ray
# from ray import air, tune
# from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
# from ray.rllib.env.env_context import EnvContext
# from ray.rllib.examples.env.curriculum_capable_env import CurriculumCapableEnv
# from ray.rllib.utils.framework import try_import_tf, try_import_torch
# from ray.rllib.utils.test_utils import check_learning_achieved
#
# from ray.rllib.algorithms.ppo import PPO
#
# #
# #
# def curriculum_fn(
#     train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
# ) -> TaskType:
#     """Function returning a possibly new task to set `task_settable_env` to.
#
#     Args:
#         train_results: The train results returned by Algorithm.train().
#         task_settable_env: A single TaskSettableEnv object
#             used inside any worker and at any vector position. Use `env_ctx`
#             to get the worker_index, vector_index, and num_workers.
#         env_ctx: The env context object (i.e. env's config dict
#             plus properties worker_index, vector_index and num_workers) used
#             to setup the `task_settable_env`.
#
#     Returns:
#         TaskType: The task to set the env to. This may be the same as the
#             current one.
#     """
#     # Our env supports tasks 1 (default) to 5.
#     # With each task, rewards get scaled up by a factor of 10, such that:
#     # Level 1: Expect rewards between 0.0 and 1.0.
#     # Level 2: Expect rewards between 1.0 and 10.0, etc..
#     # We will thus raise the level/task each time we hit a new power of 10.0
#     new_task = int(np.log10(train_results["episode_reward_mean"]) + 2.1)
#     # Clamp between valid values, just in case:
#     new_task = max(min(new_task, 5), 1)
#     print(
#         f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
#         f"\nR={train_results['episode_reward_mean']}"
#         f"\nSetting env to task={new_task}"
#     )
#     return new_task
#
# #
# # # Configure the algorithm.
# # config = {
# #     # Environment (RLlib understands openAI gym registered strings).
# #     "env": CurriculumCapableEnv,
# #     # Use 2 environment workers (aka "rollout workers") that parallelly
# #     # collect samples from their own environment clone(s).
# #     "num_workers": 2,
# #     # Change this to "framework: torch", if you are using PyTorch.
# #     # Also, use "framework: tf2" for tf2.x eager execution.
# #     "framework": "torch",
# #     # Tweak the default model provided automatically by RLlib,
# #     # given the environment's observation- and action spaces.
# #     "model": {
# #         "fcnet_hiddens": [64, 64],
# #         "fcnet_activation": "relu",
# #     },
# # "env_task_fn": curriculum_fn,
# #     # Set up a separate evaluation worker set for the
# #     # `algo.evaluate()` call after training (see below).
# #     "evaluation_num_workers": 1,
# #     # Only for evaluation runs, render the env.
# #     "evaluation_config": {
# #         "render_env": True,
# #     },
# # }
# import ray
# import ray.rllib.agents.ppo as ppo
# from ray.tune.logger import pretty_print
#
# # ray.init (num_gpus=1, local_mode = True)
# # config = ppo.DEFAULT_CONFIG.copy()
# #
# # agent = ppo.PPOTrainer(config=config, env="CartPole-v0")
# #
# # for i in range(100):
# #    # Perform one iteration of training the policy with PPO
# #    result = agent.train()
# #    print(pretty_print(result))
# #
# #    if i % 100 == 0:
# #        checkpoint = agent.save()
# #        print("checkpoint saved at", checkpoint)
#
# #
# # from ray.rllib.algorithms.maml import MAMLConfig
# # from ray.tune.registry import register_env
# # from ray.rllib.algorithms.maml import MAML
# # from ray.rllib.examples.env.halfcheetah_rand_direc import HalfCheetahRandDirecEnv
# # register_env("hcd", lambda config: HalfCheetahRandDirecEnv(config))
# #
# # config = {
# #     # Environment (RLlib understands openAI gym registered strings).
# #     "env": "hcd",
# #     # Use 2 environment workers (aka "rollout workers") that parallelly
# #     # collect samples from their own environment clone(s).
# #     "num_workers": 2,
# #     # Change this to "framework: torch", if you are using PyTorch.
# #     # Also, use "framework: tf2" for tf2.x eager execution.
# #     "framework": "torch",
# #     # Tweak the default model provided automatically by RLlib,
# #     # given the environment's observation- and action spaces.
# #     "model": {
# #         "fcnet_hiddens": [64, 64],
# #         "fcnet_activation": "relu",
# #     },
# #     "horizon" : 100,
# #     # Set up a separate evaluation worker set for the
# #     # `algo.evaluate()` call after training (see below).
# #     "evaluation_num_workers": 1,
# #     # Only for evaluation runs, render the env.
# #     "evaluation_config": {
# #         "render_env": True,
# #     },
# # }
# # config = MAMLConfig().training(use_gae=False).resources(num_gpus=0)
# # config.framework('torch')
# # config.horizon = 100
# # # print(config.to_dict())
# # # Build a Algorithm object from the config and run 1 training iteration.
# # # trainer = MAML(config)
# # # print(trainer.config)
# # trainer = config.build(env="hcd")
# # for _ in range(100):
# #     res = trainer.train()
# #     print(pretty_print(res))
# # #
# # trainer.evaluate()
# # # ray.shutdown()



import argparse
import numpy as np
import os

import ray
from ray import air, tune
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext
from ray.rllib.examples.env.curriculum_capable_env import CurriculumCapableEnv
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from envs.point_mass_2d import TaskSettablePointMass2D
# from ray.python.ray.tune.logger.logger import pretty_print

from ray.rllib.algorithms.ppo.ppo import PPO

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()



def Self_Paced_CL(
    train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
    """Function returning a possibly new task to set `task_settable_env` to.

    Args:
        train_results: The train results returned by Algorithm.train().
        task_settable_env: A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx: The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.

    Returns:
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """
    # Our env supports tasks 1 (default) to 5.
    # With each task, rewards get scaled up by a factor of 10, such that:
    # Level 1: Expect rewards between 0.0 and 1.0.
    # Level 2: Expect rewards between 1.0 and 10.0, etc..
    # We will thus raise the level/task each time we hit a new power of 10.0
    # new_task = {'mean':0.95*task_settable_env.get_task()['mean']+ 0.05*np.array([2,0.5]),
    #             'var': 0.95 * task_settable_env.get_task()['var'] + 0.05 * np.array([0.00016, 0.00016])}
    mean_rew, mean_disc_rew, mean_length = task_settable_env.get_statistics()
    vf_inputs, contexts, rewards = task_settable_env.get_context_buffer()
    task_settable_env.teacher.update_distribution(mean_disc_rew, contexts,
                                                        rewards )
    new_task = task_settable_env.get_task()
    # Clamp between valid values, just in case:
    # new_task = max(min(new_task, 5), 1)
    # print(
    #     f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
    #     f"\nR={train_results['episode_reward_mean']}"
    #     f"\nSetting env to task={new_task}"
    # )
    return new_task


if __name__ == "__main__":
    # args = parser.parse_args()
    ray.init()

    # Can also register the env creator function explicitly with:
    # register_env(
    #     "curriculum_env", lambda config: CurriculumCapableEnv(config))

    config = {
        "env": TaskSettablePointMass2D,  # or "curriculum_env" if registered above
         "env_config": {
             "init_mean": np.array([0,4]),
            "target_mean": np.array([-2.5,1]),
             # "curriculum": 'self_paced'
        },
        "horizon": 100,
        "num_workers":4,  # parallelism
        # "env_task_fn": Self_Paced_CL,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 1,
        "framework": 'torch',
        "train_batch_size": 2048,
        'lr': 0.0003,
    "seed":  tune.grid_search([0,1,2,3]),
        # "evaluation_num_workers": 2,
        "evaluation_duration": 20,
        "evaluation_interval": 1,

    }

    # alg = MyAlgo(config=config)
    stop = {
        "training_iteration": 400,

    }

    results = tune.Tuner(
        PPO, param_space=config, run_config=air.RunConfig(stop=stop, verbose=1, name="PPO_default_PM2")
    ).fit()
    # #
    # for _ in range(10):
    #     res = alg.train()
    #     print(res)
    ray.shutdown()