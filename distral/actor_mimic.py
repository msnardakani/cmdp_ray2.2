#The policy is based on SAC

from ray.rllib.algorithms.simple_q import SimpleQTorchPolicy, SimpleQ, SimpleQConfig
from ray.rllib.algorithms.pg import PG
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOTorchPolicy, PPO


from typing import Dict, List, Type, Union
from ray.rllib.algorithms.bc import BC
import argparse
import os
import random
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

from distral.actor_mimic_torch_policy import ActorMimicTorchPolicy
from env_funcs import make_multi_agent_divide_and_conquer
from envs.point_mass_2d import TaskSettablePointMass2D
# from utils import FC_MLP
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import TensorType, nn
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.utils.annotations import (
    DeveloperAPI,
    OverrideToImplementCustomLogic,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
    is_overridden,
    override,
 )
import logging
from typing import Type, Dict, Any, Optional, Union

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.algorithms.sac.sac_tf_policy import SACTFPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
    DEPRECATED_VALUE,
    deprecation_warning,
    Deprecated,
)
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.typing import AlgorithmConfigDict, ResultDict

tf1, tf, tfv = try_import_tf()
tfp = try_import_tfp()

logger = logging.getLogger(__name__)

OPPONENT_OBS = 'opp_obs'
OPPONENT_ACT = 'opp_actions'


class ActorMimicConfig(AlgorithmConfig):
    """Defines a configuration class from which an SAC Algorithm can be built.

    Example:
        >>> config = SACConfig().training(gamma=0.9, lr=0.01)\
        ...     .resources(num_gpus=0)\
        ...     .rollouts(num_rollout_workers=4)
        >>> print(config.to_dict())
        >>> # Build a Algorithm object from the config and run 1 training iteration.
        >>> algo = config.build(env="CartPole-v1")
        >>> algo.train()
    """

    def __init__(self, algo_class=None):
        """Initializes a SimpleQConfig instance."""
        super().__init__(algo_class=algo_class or ActorMimic)

        # Simple Q specific
        # fmt: off
        # __sphinx_doc_begin__
        # self.target_network_update_freq = 500
        self.replay_buffer_config = {
            "_enable_replay_buffer_api": True,
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": int(1e6),
            # How many steps of the model to sample before learning starts.
            "learning_starts": 1500,
            # If True prioritized replay buffer will be used.
            "prioritized_replay": False,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "prioritized_replay_eps": 1e-6,
            # Whether to compute priorities already on the remote worker side.
            "worker_side_prioritization": False,
        }
        self.store_buffer_in_checkpoints = False
        self.lr_schedule = None
        self.adam_epsilon = 1e-8
        self.grad_clip = 40
        # __sphinx_doc_end__
        # fmt: on

        # Overrides of AlgorithmConfig defaults
        # `rollouts()`

        # `training()`
        self.lr = 5e-4
        self.train_batch_size = 256

        # `exploration()`

        # `evaluation()`

        # `reporting()`


    @override(AlgorithmConfig)
    def training(
            self,
            *,
            target_network_update_freq: Optional[int] = None,
            store_buffer_in_checkpoints: Optional[bool] = None,
            lr_schedule: Optional[List[List[Union[int, float]]]] = None,
            replay_buffer_config: Optional[dict] = None,
            adam_epsilon: Optional[float] = None,
            grad_clip: Optional[int] = None,
            **kwargs,
    ) -> "DistralActorMimicConfig":
        """Sets the training related configuration.

        Args:


        Returns:
            This updated AlgorithmConfig object.
        """
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if target_network_update_freq is not None:
            self.target_network_update_freq = target_network_update_freq
        if replay_buffer_config is not None:
            self.replay_buffer_config = replay_buffer_config
        if store_buffer_in_checkpoints is not None:
            self.store_buffer_in_checkpoints = store_buffer_in_checkpoints
        if lr_schedule is not None:
            self.lr_schedule = lr_schedule
        if adam_epsilon is not None:
            self.adam_epsilon = adam_epsilon
        if grad_clip is not None:
            self.grad_clip = grad_clip

        return self

DEFAULT_CONFIG = ActorMimicConfig.to_dict()

class ActorMimic(SimpleQ):
    @classmethod
    @override(SimpleQ)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return DEFAULT_CONFIG

    @override(SimpleQ)
    def validate_config(self, config: AlgorithmConfigDict) -> None:
        # Call super's validation method.
        super().validate_config(config)

        # Update effective batch size to include n-step
        adjusted_rollout_len = max(config["rollout_fragment_length"], config["n_step"])
        config["rollout_fragment_length"] = adjusted_rollout_len

    @override(SimpleQ)
    def get_default_policy_class(
        self, config: AlgorithmConfigDict
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            return ActorMimicTorchPolicy
        else:
            raise ValueError("Only torch framework is implemented")

    @override(SimpleQ)
    def training_step(self) -> ResultDict:
        """ActorMimic training iteration function.

        Each training iteration, we:
        - Sample (MultiAgentBatch) from workers.
        - Store new samples in replay buffer.
        - Sample training batch (MultiAgentBatch) from replay buffer.
        - Learn on training batch.
        - Update remote workers' new policy weights.
        - Update target network every `target_network_update_freq` sample steps.
        - Return all collected metrics for the iteration.

        Returns:
            The results dict from executing the training iteration.
        """
        train_results = {}

        # We alternate between storing new samples and sampling and training
        store_weight, sample_and_train_weight = calculate_rr_weights(self.config)

        for _ in range(store_weight):
            # Sample (MultiAgentBatch) from workers.
            new_sample_batch = synchronous_parallel_sample(
                worker_set=self.workers, concat=True
            )

            # Update counters
            self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
            self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()

            # Store new samples in replay buffer.
            self.local_replay_buffer.add_batch(new_sample_batch)

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }

        for _ in range(sample_and_train_weight):
            # Sample training batch (MultiAgentBatch) from replay buffer.
            train_batch = sample_min_n_steps_from_buffer(
                self.local_replay_buffer,
                self.config["train_batch_size"],
                count_by_agent_steps=self._by_agent_steps,
            )

            # Old-style replay buffers return None if learning has not started
            if train_batch is None or len(train_batch) == 0:
                self.workers.local_worker().set_global_vars(global_vars)
                break

            # Postprocess batch before we learn on it
            post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
            train_batch = post_fn(train_batch, self.workers, self.config)

            # for policy_id, sample_batch in train_batch.policy_batches.items():
            #     print(len(sample_batch["obs"]))
            #     print(sample_batch.count)

            # Learn on training batch.
            # Use simple optimizer (only for multi-agent or tf-eager; all other
            # cases should use the multi-GPU optimizer, even if only using 1 GPU)
            if self.config.get("simple_optimizer") is True:
                train_results = train_one_step(self, train_batch)
            else:
                train_results = multi_gpu_train_one_step(self, train_batch)

            # Update replay buffer priorities.
            update_priorities_in_replay_buffer(
                self.local_replay_buffer,
                self.config,
                train_batch,
                train_results,
            )

            # Update target network every `target_network_update_freq` sample steps.
            cur_ts = self._counters[
                NUM_AGENT_STEPS_SAMPLED
                if self._by_agent_steps
                else NUM_ENV_STEPS_SAMPLED
            ]
            last_update = self._counters[LAST_TARGET_UPDATE_TS]
            if cur_ts - last_update >= self.config["target_network_update_freq"]:
                to_update = self.workers.local_worker().get_policies_to_train()
                self.workers.local_worker().foreach_policy_to_train(
                    lambda p, pid: pid in to_update and p.update_target()
                )
                self._counters[NUM_TARGET_UPDATES] += 1
                self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

            # Update weights and global_vars - after learning on the local worker -
            # on all remote workers.
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(global_vars=global_vars)

        # Return all collected metrics for the iteration.
        return train_results
