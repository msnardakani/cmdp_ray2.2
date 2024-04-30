import copy
import logging
from typing import Dict, List, Type, Union, Tuple

import ray
from ray.rllib import TorchPolicy
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.evaluation.postprocessing import (

    compute_gae_for_sample_batch,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    # ValueNetworkMixin,
)
# from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
# from torch.distributions.normal import Normal
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    # explained_variance,
    # sequence_mask,
)
# from ray.rllib.utils.typing import TensorType
# from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
# from torch.distributions. normal import Normal
# from distral.distral_ppo_torch_model import DistralTorchModel, DistralCentralTorchModel
# from distral.distral_torch_policy import _get_dist_class

# import gym
# from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import (
    # AgentID,
    # LocalOptimizer,
    # ModelGradients,
    TensorType,
    # AlgorithmConfigDict,
)

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class Distill(dict):
    DIST_INPUTS = "distill_dist"
    ACTIONS = "distill_actions"
    ACTION_PROB = "distill_action_prob"
    ACTION_LOGP = "distill_action_logp"


class DistilledNetworkMixin:
    """Assigns the `_value()` method to a TorchPolicy.


    """
    def __init__(self, config):
        # When doing GAE, we need the value function estimate on the
        # observation.
        self.update_target(tau=1.0)
        if config["use_gae"]:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.

            def value(**input_dict):
                input_dict = SampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                model_out, _ = self.model(input_dict)
                # [0] = remove the batch dim.
                return self.model.value_function()[0].item()

        # When not doing GAE, we do not require the value function's output.
        else:

            def value(*args, **kwargs):
                return 0.0

        # self.compute_distilled = self.model.central_value_function
        self._value = value

    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        """Defines extra fetches per action computation.

        Args:
            input_dict (Dict[str, TensorType]): The input dict used for the action
                computing forward pass.
            state_batches (List[TensorType]): List of state tensors (empty for
                non-RNNs).
            model (ModelV2): The Model object of the Policy.
            action_dist: The instantiated distribution
                object, resulting from the model's outputs and the given
                distribution class.

        Returns:
            Dict[str, TensorType]: Dict with extra tf fetches to perform per
                action computation.
        """
        # Return value function outputs. VF estimates will hence be added to
        # the SampleBatches produced by the sampler(s) to generate the train
        # batches going into the loss function.
        distill_out = model.distill_out()


        return {
            SampleBatch.VF_PREDS: model.value_function(),
            Distill.DIST_INPUTS: distill_out,
            # Distill.ACTION_LOGP: distill_logp

        }
    def update_target(self, tau=None):
        # Update_target_fn will be called periodically to copy Q network to
        # target Q network, using (soft) tau-synching.
        tau = tau or self.config.get("tau")
        model_state_dict = self.model.state_dict()
        # Support partial (soft) synching.
        # If tau == 1.0: Full sync from Q-model to target Q-model.
        target_state_dict = next(iter(
            self.target_models.values())
                                 ).state_dict()
        model_state_dict = {
            k: tau * model_state_dict[k] + (1 - tau) * v
            for k, v in target_state_dict.items()
        }

        for target in self.target_models.values():
            target.load_state_dict(model_state_dict)

    @override(TorchPolicy)
    def set_weights(self, weights):
        # Makes sure that whenever we restore weights for this policy's
        # model, we sync the target network (from the main model)
        # at the same time.
        TorchPolicy.set_weights(self, weights)
        self.update_target()


class DistralPPOTorchPolicy(
    DistilledNetworkMixin,
    LearningRateSchedule,
    EntropyCoeffSchedule,
    KLCoeffMixin,
    TorchPolicyV2,
):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.algorithms.ppo.ppo.PPOConfig().to_dict(), **config)
        # TODO: Move into Policy API, if needed at all here. Why not move this into
        #  `PPOConfig`?.
        validate_config(config)

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        DistilledNetworkMixin.__init__(self, config)
        # ValueNetworkMixin.__init__(self, config)

        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)
        if config['loss_fn'] ==11 :#0:
            from distral.distral_ppo_loss import loss_ppo_added_terms
            self.distral_loss = loss_ppo_added_terms

        elif  config['loss_fn'] == 21:#1:
            from distral.distral_ppo_loss import loss_psudo_ppo
            self.distral_loss = loss_psudo_ppo

        elif config['loss_fn'] == 12:#2:
            from distral.distral_ppo_loss import loss_ppo_added_terms2
            self.distral_loss = loss_ppo_added_terms2
        elif config['loss_fn'] == 22:#3:
            from distral.distral_ppo_loss import loss_psudo_ppo2
            self.distral_loss = loss_psudo_ppo2

        elif config['loss_fn'] == -2:
            from distral.distral_ppo_loss import loss_ppo_distill3
            self.distral_loss = loss_ppo_distill3

        elif config['loss_fn'] == 13:#4:
            from distral.distral_ppo_loss import loss_ppo_added_terms3
            self.distral_loss = loss_ppo_added_terms3
        elif  config['loss_fn'] == 14:#6:
            from distral.distral_ppo_loss import loss_ppo_added_terms4
            self.distral_loss = loss_ppo_added_terms4

        elif  config['loss_fn'] == 15:#8:
            from distral.distral_ppo_loss import loss_ppo_added_terms5
            self.distral_loss = loss_ppo_added_terms5
        elif  config['loss_fn'] == 16:#10:
            from distral.distral_ppo_loss import loss_ppo_added_terms6
            self.distral_loss = loss_ppo_added_terms6
        elif config['loss_fn'] == 23:#5:
            from distral.distral_ppo_loss import loss_psudo_ppo3
            self.distral_loss = loss_psudo_ppo3
        elif config['loss_fn'] == 24:#7:
            from distral.distral_ppo_loss import loss_psudo_ppo4
            self.distral_loss = loss_psudo_ppo4
        elif config['loss_fn'] == 25:#9:
            from distral.distral_ppo_loss import loss_psudo_ppo5
            self.distral_loss = loss_psudo_ppo5
        elif config['loss_fn'] == 31:  # 9:
            from distral.distral_ppo_loss import dual_distill_regularized_ppo_loss as loss_fn
            self.distral_loss = loss_fn

        else:
            from distral.distral_ppo_loss import loss_ppo_distill
            self.distral_loss = loss_ppo_distill



        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()

    # TODO: Make this an event-style subscription (e.g.:
    #  "after_gradients_computed").
    @override(TorchPolicyV2)
    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    @override(TorchPolicyV2)
    def loss(
            self,
            model: ModelV2,
            dist_class: Type[ActionDistribution],
            train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """

        return self.distral_loss(self,
                                 model,
                                 dist_class,
                                 train_batch, )




    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {"distill_loss": torch.mean(torch.stack(self.get_tower_stats("distill_loss"))),
             "transfer_kl": torch.mean(torch.stack(self.get_tower_stats("transfer_kl"))),
             "ppo_loss": torch.mean(torch.stack(self.get_tower_stats("ppo_loss"))),

             "cur_kl_coeff": self.kl_coeff,
             "cur_lr": self.cur_lr,
             "total_loss": torch.mean(
                 torch.stack(self.get_tower_stats("total_loss"))
             ),
             "policy_loss": torch.mean(
                 torch.stack(self.get_tower_stats("mean_policy_loss"))
             ),
             "vf_loss": torch.mean(
                 torch.stack(self.get_tower_stats("mean_vf_loss"))
             ),
             "vf_explained_var": torch.mean(
                 torch.stack(self.get_tower_stats("vf_explained_var"))
             ),
             "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
             "entropy": torch.mean(
                 torch.stack(self.get_tower_stats("mean_entropy"))
             ),
             "entropy_coeff": self.entropy_coeff,

             }
        )

    @override(TorchPolicyV2)
    def postprocess_trajectory(
            self, sample_batch, other_agent_batches=None, episode=None
    ):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?
        with torch.no_grad():
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )
    @override(TorchPolicyV2)
    def make_model_and_action_dist(self) -> ModelV2:
        dist_class, logit_dim = ModelCatalog.get_action_dist(
            self.action_space, self.config["model"], framework=self.framework
        )

        # self.config["model"] =  {
        #     "custom_model": 'local',
        #     "custom_model_config": {"central": {"distilled_model": central_policy,  'model': model_config, 'ctx_aug': ctx_aug},
        #                             "central_target":{"distilled_model": central_policy_target,  'model': model_config, 'ctx_aug': ctx_aug}  ,}

        # config = dict(custom_model = self.config['model']['custom_model'], custom_model_config =  self.config['model']['custom_model_config']['central'])
        # config_target = copy.deepcopy(self.config['model'])
        # config_target['custom_model_config']['distilled_model'] = self.config['model']['custom_model_config']['target_distilled_model']
        model = ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=logit_dim,
            model_config=self.config['model']['custom_model_config']['central'],
            framework=self.framework,
            name= 'model'
        )
        self.target_model = ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=logit_dim,
            model_config=self.config['model']['custom_model_config']['target_central'],
            framework=self.framework,
            name= 'target_model'
        )
        return model, dist_class
