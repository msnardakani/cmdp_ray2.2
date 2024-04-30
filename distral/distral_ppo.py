"""
Proximal Policy Optimization (PPO)
==================================

This file defines the distributed Algorithm class for proximal policy
optimization.
See `ppo_[tf|torch]_policy.py` for the definition of the policy loss.

Detailed documentation: https://docs.ray.io/en/master/rllib-algorithms.html#ppo
"""

import dataclasses
import logging
from typing import List, Optional, Type, Union, TYPE_CHECKING

import numpy as np
import tree
from ray.rllib.algorithms import PPOConfig, PPO

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.ppo_learner import (
    PPOLearnerHyperparameters,
    LEARNER_RESULTS_KL_KEY,
)
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.evaluation.postprocessing_v2 import postprocess_episodes_to_sample_batch
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
    synchronous_parallel_sample,
)
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    LAST_TARGET_UPDATE_TS,
    NUM_TARGET_UPDATES,
    SYNCH_WORKER_WEIGHTS_TIMER,
    SAMPLE_TIMER,
    ALL_MODULES,
)
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import ResultDict, AlgorithmConfigDict
from ray.util.debug import log_once

if TYPE_CHECKING:
    from ray.rllib.core.learner.learner import Learner

logger = logging.getLogger(__name__)

class DistralPPOConfig(PPOConfig):
    """Defines a configuration class from which a PPO Algorithm can be built.

    Example:
        >>> from ray.rllib.algorithms.ppo import PPOConfig
        >>> config = PPOConfig().training(gamma=0.9, lr=0.01, kl_coeff=0.3)\
        ...             .resources(num_gpus=0)\
        ...             .rollouts(num_workers=4)
        >>> print(config.to_dict())
        >>> # Build a Algorithm object from the config and run 1 training iteration.
        >>> trainer = config.build(env="CartPole-v1")
        >>> trainer.train()

    Example:
        >>> from ray.rllib.algorithms.ppo import PPOConfig
        >>> from ray import tune
        >>> config = PPOConfig()
        >>> # Print out some default values.
        >>> print(config.clip_param)
        >>> # Update the config object.
        >>> config.training(lr=tune.grid_search([0.001, 0.0001]), clip_param=0.2)
        >>> # Set the config object's env.
        >>> config.environment(env="CartPole-v1")
        >>> # Use to_dict() to get the old-style python config dict
        >>> # when running with tune.
        >>> tune.run(
        ...     "PPO",
        ...     stop={"episode_reward_mean": 200},
        ...     config=config.to_dict(),
        ... )
    """

    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        self._allow_unknown_subkeys = ["distill_coeff" , 'loss_fn', 'ctx_aug_mode']
        super().__init__(algo_class=algo_class or PPO)

        # fmt: off
        # __sphinx_doc_begin__
        # PPO specific settings:
        self.distill_coeff = 0.2
        self.transfer_coeff = 0.2
        self.loss_fn = 0# 0 for added terms and 1 for psuedo PPO
        # self.distral_beta = 1
        self.distilled_model = None
        self.distilled_target_model = None

        self.ctx_aug_mode = False

        self.target_network_update_freq = 0
        self.tau = 1.0

        self.lr_schedule = None
        self.use_critic = True
        self.use_gae = True
        self.lambda_ = 1.0
        self.kl_coeff = 0.2
        self.sgd_minibatch_size = 128
        self.num_sgd_iter = 30
        self.shuffle_sequences = True
        self.vf_loss_coeff = 1.0
        self.entropy_coeff = 0.0
        self.entropy_coeff_schedule = None
        self.clip_param = 0.3
        self.vf_clip_param = 10.0
        self.grad_clip = None
        self.kl_target = 0.01

        # Override some of AlgorithmConfig's default values with PPO-specific values.
        self.rollout_fragment_length = 200
        self.train_batch_size = 4000
        self.lr = 5e-5
        self.model["vf_share_layers"] = False
        # __sphinx_doc_end__
        # fmt: on

        # Deprecated keys.
        self.vf_share_layers = DEPRECATED_VALUE



class DistralPPO(PPO):
    # TODO: Change the return value of this method to return a AlgorithmConfig object
    #  instead.
    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return DistralPPOConfig()


    @override(Algorithm)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            from distral.distral_ppo_torch_policy import DistralPPOTorchPolicy

            return DistralPPOTorchPolicy
        else:

            raise "NotImplemented"
    @override(Algorithm)
    def training_step(self) -> ResultDict:
        use_rollout_worker = self.config.env_runner_cls is None or issubclass(
            self.config.env_runner_cls, RolloutWorker
        )

        # Collect SampleBatches from sample workers until we have a full batch.
        with self._timers[SAMPLE_TIMER]:
            # Old RolloutWorker based APIs (returning SampleBatch/MultiAgentBatch).
            if use_rollout_worker:
                if self.config.count_steps_by == "agent_steps":
                    train_batch = synchronous_parallel_sample(
                        worker_set=self.workers,
                        max_agent_steps=self.config.train_batch_size,
                    )
                else:
                    train_batch = synchronous_parallel_sample(
                        worker_set=self.workers,
                        max_env_steps=self.config.train_batch_size,
                    )
            # New Episode-returning EnvRunner API.
            else:
                if self.workers.num_remote_workers() <= 0:
                    episodes: List[SingleAgentEpisode] = [
                        self.workers.local_worker().sample()
                    ]
                else:
                    episodes: List[SingleAgentEpisode] = self.workers.foreach_worker(
                        lambda w: w.sample(), local_worker=False
                    )
                # Perform PPO postprocessing on a (flattened) list of Episodes.
                postprocessed_episodes: List[
                    SingleAgentEpisode
                ] = self.postprocess_episodes(tree.flatten(episodes))
                # Convert list of postprocessed Episodes into a single sample batch.
                train_batch: SampleBatch = postprocess_episodes_to_sample_batch(
                    postprocessed_episodes
                )

        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Standardize advantages.
        train_batch = standardize_fields(train_batch, ["advantages"])
        # Train
        if self.config._enable_new_api_stack:
            # TODO (Kourosh) Clearly define what train_batch_size
            #  vs. sgd_minibatch_size and num_sgd_iter is in the config.
            # TODO (Kourosh) Do this inside the Learner so that we don't have to do
            #  this back and forth communication between driver and the remote
            #  learner actors.
            # TODO (sven): What's the plan for multi-agent setups when the
            #  policy is gone?
            # TODO (simon): The default method has already this functionality,
            #  but this serves simply as a placeholder until it is decided on
            #  how to replace the functionalities of the policy.

            if (
                self.config.env_runner_cls is None
                or self.config.env_runner_cls.__name__ == "RolloutWorker"
            ):
                is_module_trainable = self.workers.local_worker().is_policy_to_train
                self.learner_group.set_is_module_trainable(is_module_trainable)
            train_results = self.learner_group.update(
                train_batch,
                minibatch_size=self.config.sgd_minibatch_size,
                num_iters=self.config.num_sgd_iter,
            )

        elif self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        if self.config._enable_new_api_stack:
            # The train results's loss keys are pids to their loss values. But we also
            # return a total_loss key at the same level as the pid keys. So we need to
            # subtract that to get the total set of pids to update.
            # TODO (Kourosh): We should also not be using train_results as a message
            #  passing medium to infer which policies to update. We could use
            #  policies_to_train variable that is given by the user to infer this.
            policies_to_update = set(train_results.keys()) - {ALL_MODULES}
        else:
            policies_to_update = list(train_results.keys())

        # TODO (Kourosh): num_grad_updates per each policy should be accessible via
        #  train_results.
        if not use_rollout_worker:
            global_vars = None
        else:
            global_vars = {
                "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
                "num_grad_updates_per_policy": {
                    pid: self.workers.local_worker().policy_map[pid].num_grad_updates
                    for pid in policies_to_update
                },
            }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            if self.workers.num_remote_workers() > 0:
                from_worker_or_learner_group = None
                if self.config._enable_new_api_stack:
                    # sync weights from learner_group to all rollout workers
                    from_worker_or_learner_group = self.learner_group
                self.workers.sync_weights(
                    from_worker_or_learner_group=from_worker_or_learner_group,
                    policies=policies_to_update,
                    global_vars=global_vars,
                )
            elif self.config._enable_new_api_stack:
                weights = self.learner_group.get_weights()
                self.workers.local_worker().set_weights(weights)

        if self.config._enable_new_api_stack:

            kl_dict = {}
            if self.config.use_kl_loss:
                for pid in policies_to_update:
                    kl = train_results[pid][LEARNER_RESULTS_KL_KEY]
                    kl_dict[pid] = kl
                    if np.isnan(kl):
                        logger.warning(
                            f"KL divergence for Module {pid} is non-finite, this will "
                            "likely destabilize your model and the training process. "
                            "Action(s) in a specific state have near-zero probability. "
                            "This can happen naturally in deterministic environments "
                            "where the optimal policy has zero mass for a specific "
                            "action. To fix this issue, consider setting `kl_coeff` to "
                            "0.0 or increasing `entropy_coeff` in your config."
                        )

            # triggers a special update method on RLOptimizer to update the KL values.
            additional_results = self.learner_group.additional_update(
                module_ids_to_update=policies_to_update,
                sampled_kl_values=kl_dict,
                timestep=self._counters[NUM_AGENT_STEPS_SAMPLED],
            )
            for pid, res in additional_results.items():
                train_results[pid].update(res)

            return train_results

        ## Target Network Update Edit
        cur_ts = self._counters[
            NUM_AGENT_STEPS_SAMPLED
            if self.config.count_steps_by == "agent_steps"
            else NUM_ENV_STEPS_SAMPLED
        ]
        last_update = self._counters[LAST_TARGET_UPDATE_TS]
        if cur_ts - last_update >= self.config.target_network_update_freq:
            to_update = self.workers.local_worker().get_policies_to_train()
            self.workers.local_worker().foreach_policy_to_train(
                lambda p, pid: pid in to_update and p.update_target()
            )
            self._counters[NUM_TARGET_UPDATES] += 1
            self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

        ## End Edit


        # Update weights and global_vars - after learning on the local worker -
        # on all remote workers.
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            self.workers.sync_weights(global_vars=global_vars)
        # For each policy: Update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                log_once("ppo_warned_lr_ratio")
                and self.config.get("model", {}).get("vf_share_layers")
                and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                log_once("ppo_warned_vf_clip")
                and mean_reward > self.config.vf_clip_param
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        # TODO (simon): At least in RolloutWorker obsolete I guess as called in
        #  `sync_weights()` called above if remote workers. Can we call this
        #  where `set_weights()` is called on the local_worker?
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results

# DEFAULT_CONFIG = DistralPPOConfig().to_dict()
