#The policy is based on SAC

from ray.rllib.algorithms.sac import SAC, SACTorchPolicy

from typing import Dict, List, Type, Union

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
# # )
# from ray.rllib.models import ModelCatalog, ActionDistribution
# from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
# from ray.rllib.policy.policy import PolicySpec
# from ray.rllib.utils.framework import try_import_tf
# from ray.rllib.utils.test_utils import check_learning_achieved
# import numpy as np
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from ray.rllib.models.modelv2 import ModelV2, flatten
# from ray.rllib.models.torch.misc import SlimFC
# from ray.rllib.utils.annotations import override
# from ray.rllib.utils.framework import try_import_tf, try_import_torch
# from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
#
# from env_funcs import make_multi_agent_divide_and_conquer
# from envs.point_mass_2d import TaskSettablePointMass2D
# # from utils.wei import FC_MLP
# from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from torch import TensorType, nn
# from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
# from ray.rllib.utils.annotations import (
#     DeveloperAPI,
#     OverrideToImplementCustomLogic,
#     OverrideToImplementCustomLogic_CallToSuperRecommended,
#     is_overridden,
#     override,
#  )
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
from ray.rllib.utils.typing import AlgorithmConfigDict

tf1, tf, tfv = try_import_tf()
tfp = try_import_tfp()

logger = logging.getLogger(__name__)

OPPONENT_OBS = 'opp_obs'
OPPONENT_ACT = 'opp_actions'


class DistralConfig(AlgorithmConfig):
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
        super().__init__(algo_class=algo_class or SAC)
        # fmt: off
        # __sphinx_doc_begin__
        # (SAC-specific + distral) config settings.
        self.distral_alpha = None
        self.distral_beta = None
        self.distilled_model = None
        self.distill_coeff =None

        self.twin_q = True
        self.q_model_config = {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "custom_model": None,  # Use this to define custom Q-model(s).
            "custom_model_config": {},
        }
        self.policy_model_config = {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "custom_model": None,  # Use this to define a custom policy model.
            "custom_model_config": {},
        }
        self.clip_actions = False
        self.tau = 5e-3
        self.initial_alpha = 1#(self.distral_alpha - 1)/self.distral_beta
        # self.fixed_alpha= False
        self.target_entropy = "auto"
        self.n_step = 1
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
        self.training_intensity = None
        self.optimization = {
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "entropy_learning_rate": 3e-4,
        }
        self.grad_clip = 10
        self.target_network_update_freq = 0
        self.eval_ctx_config= None
        # .rollout()
        self.rollout_fragment_length = 1
        self.compress_observations = False

        # .training()
        self.train_batch_size = 256

        # .reporting()
        self.min_time_s_per_iteration = 1
        self.min_sample_timesteps_per_iteration = 100
        # __sphinx_doc_end__
        # fmt: on

        self._deterministic_loss = False
        self._use_beta_distribution = False

        self.use_state_preprocessor = DEPRECATED_VALUE
        self.worker_side_prioritization = DEPRECATED_VALUE

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        twin_q: Optional[bool] = None,
        q_model_config: Optional[Dict[str, Any]] = None,
        policy_model_config: Optional[Dict[str, Any]] = None,
        tau: Optional[float] = None,
        initial_alpha: Optional[float] = None,
        target_entropy: Optional[Union[str, float]] = None,
        n_step: Optional[int] = None,
        store_buffer_in_checkpoints: Optional[bool] = None,
        replay_buffer_config: Optional[Dict[str, Any]] = None,
        training_intensity: Optional[float] = None,
        clip_actions: Optional[bool] = None,
        grad_clip: Optional[float] = None,
        optimization_config: Optional[Dict[str, Any]] = None,
        target_network_update_freq: Optional[int] = None,
        _deterministic_loss: Optional[bool] = None,
        _use_beta_distribution: Optional[bool] = None,
        **kwargs,
    ) -> "DistralConfig":
        """Sets the training related configuration.

        Args:
            twin_q: Use two Q-networks (instead of one) for action-value estimation.
                Note: Each Q-network will have its own target network.
            q_model_config: Model configs for the Q network(s). These will override
                MODEL_DEFAULTS. This is treated just as the top-level `model` dict in
                setting up the Q-network(s) (2 if twin_q=True).
                That means, you can do for different observation spaces:
                obs=Box(1D) -> Tuple(Box(1D) + Action) -> concat -> post_fcnet
                obs=Box(3D) -> Tuple(Box(3D) + Action) -> vision-net -> concat w/ action
                -> post_fcnet
                obs=Tuple(Box(1D), Box(3D)) -> Tuple(Box(1D), Box(3D), Action)
                -> vision-net -> concat w/ Box(1D) and action -> post_fcnet
                You can also have SAC use your custom_model as Q-model(s), by simply
                specifying the `custom_model` sub-key in below dict (just like you would
                do in the top-level `model` dict.
            policy_model_config: Model options for the policy function (see
                `q_model_config` above for details). The difference to `q_model_config`
                above is that no action concat'ing is performed before the post_fcnet
                stack.
            tau: Update the target by \tau * policy + (1-\tau) * target_policy.
            initial_alpha: Initial value to use for the entropy weight alpha.
            target_entropy: Target entropy lower bound. If "auto", will be set
                to -|A| (e.g. -2.0 for Discrete(2), -3.0 for Box(shape=(3,))).
                This is the inverse of reward scale, and will be optimized
                automatically.
            n_step: N-step target updates. If >1, sars' tuples in trajectories will be
                postprocessed to become sa[discounted sum of R][s t+n] tuples.
            store_buffer_in_checkpoints: Set this to True, if you want the contents of
                your buffer(s) to be stored in any saved checkpoints as well.
                Warnings will be created if:
                - This is True AND restoring from a checkpoint that contains no buffer
                    data.
                - This is False AND restoring from a checkpoint that does contain
                    buffer data.
            replay_buffer_config: Replay buffer config.
                Examples:
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentReplayBuffer",
                "learning_starts": 1000,
                "capacity": 50000,
                "replay_batch_size": 32,
                "replay_sequence_length": 1,
                }
                - OR -
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 50000,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
                "replay_sequence_length": 1,
                }
                - Where -
                prioritized_replay_alpha: Alpha parameter controls the degree of
                prioritization in the buffer. In other words, when a buffer sample has
                a higher temporal-difference error, with how much more probability
                should it drawn to use to update the parametrized Q-network. 0.0
                corresponds to uniform probability. Setting much above 1.0 may quickly
                result as the sampling distribution could become heavily “pointy” with
                low entropy.
                prioritized_replay_beta: Beta parameter controls the degree of
                importance sampling which suppresses the influence of gradient updates
                from samples that have higher probability of being sampled via alpha
                parameter and the temporal-difference error.
                prioritized_replay_eps: Epsilon parameter sets the baseline probability
                for sampling so that when the temporal-difference error of a sample is
                zero, there is still a chance of drawing the sample.
            training_intensity: The intensity with which to update the model (vs
                collecting samples from the env).
                If None, uses "natural" values of:
                `train_batch_size` / (`rollout_fragment_length` x `num_workers` x
                `num_envs_per_worker`).
                If not None, will make sure that the ratio between timesteps inserted
                into and sampled from th buffer matches the given values.
                Example:
                training_intensity=1000.0
                train_batch_size=250
                rollout_fragment_length=1
                num_workers=1 (or 0)
                num_envs_per_worker=1
                -> natural value = 250 / 1 = 250.0
                -> will make sure that replay+train op will be executed 4x asoften as
                rollout+insert op (4 * 250 = 1000).
                See: rllib/algorithms/dqn/dqn.py::calculate_rr_weights for further
                details.
            clip_actions: Whether to clip actions. If actions are already normalized,
                this should be set to False.
            grad_clip: If not None, clip gradients during optimization at this value.
            optimization_config: Config dict for optimization. Set the supported keys
                `actor_learning_rate`, `critic_learning_rate`, and
                `entropy_learning_rate` in here.
            target_network_update_freq: Update the target network every
                `target_network_update_freq` steps.
            _deterministic_loss: Whether the loss should be calculated deterministically
                (w/o the stochastic action sampling step). True only useful for
                continuous actions and for debugging.
            _use_beta_distribution: Use a Beta-distribution instead of a
                `SquashedGaussian` for bounded, continuous action spaces (not
                recommended; for debugging only).

        Returns:
            This updated AlgorithmConfig object.
        """
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)
        # if se
        if twin_q is not None:
            self.twin_q = twin_q
        if q_model_config is not None:
            self.q_model_config = q_model_config
        if policy_model_config is not None:
            self.policy_model_config = policy_model_config
        if tau is not None:
            self.tau = tau
        if initial_alpha is not None:
            self.initial_alpha = initial_alpha
        if target_entropy is not None:
            self.target_entropy = target_entropy
        if n_step is not None:
            self.n_step = n_step
        if store_buffer_in_checkpoints is not None:
            self.store_buffer_in_checkpoints = store_buffer_in_checkpoints
        if replay_buffer_config is not None:
            self.replay_buffer_config = replay_buffer_config
        if training_intensity is not None:
            self.training_intensity = training_intensity
        if clip_actions is not None:
            self.clip_actions = clip_actions
        if grad_clip is not None:
            self.grad_clip = grad_clip
        if optimization_config is not None:
            self.optimization_config = optimization_config
        if target_network_update_freq is not None:
            self.target_network_update_freq = target_network_update_freq
        if _deterministic_loss is not None:
            self._deterministic_loss = _deterministic_loss
        if _use_beta_distribution is not None:
            self._use_beta_distribution = _use_beta_distribution

        return self

DEFAULT_CONFIG = DistralConfig().to_dict()


class Distral(SAC):
    """Distral Algorithm class
    """

    def __init__(self, *args, **kwargs):
        self._allow_unknown_subkeys += ["distral_alpha", "distral_beta", 'distilled_model', 'eval_ctx_config', 'distill_coeff']
        super().__init__(*args, **kwargs)

    @classmethod
    @override(SAC)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return DistralConfig().to_dict()

    @override(SAC)
    def validate_config(self, config: AlgorithmConfigDict) -> None:
        # Call super's validation method.
        super().validate_config(config)
        if (config["distral_alpha"] is  None or config["distral_beta"] is  None) ^ (config["distill_coeff"]  is not None):
            # print('error')
            raise ValueError("Both distral_alpha/beta and distill_coeff can not be determined or None!")
        if config["distill_coeff"] is None:
            if config["distral_alpha"] is None and config["distral_beta"] is None:
                raise ValueError("Both distral_alpha and distral_beta need to be determined!")

    @override(SAC)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            from distral.distral_torch_policy import DistralTorchPolicy
            # from ray.rllib.algorithms.sac.sac_torch_policy import SACTorchPolicy

            return DistralTorchPolicy
        else:
            raise "NotImplementedError"

#