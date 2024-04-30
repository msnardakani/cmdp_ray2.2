import logging
from typing import Dict, List, Type, Union

import ray
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
)
from ray.rllib.utils.typing import TensorType
from distral.distral_ppo_torch_policy import Distill
torch, nn = try_import_torch()
F = nn.functional

logger = logging.getLogger(__name__)




def loss_psudo_ppo2(
        policy:TorchPolicyV2,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Compute loss for Distral Policy Objective The main difference with PPO is that the central policy takes.
    J = J_clip(pi_i/pi_0 A ) - H(pi_i) + distill_loss(pi_0, pi_i)
    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.

    Returns:
        The PPO loss tensor given the input batch.
    """

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # Edit Distral
    distill_logits = model.distill_out()
    # curr_distill_dist = dist_class(distill_logits, model)



    # End Edit

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean


    # Distral Edit


    prev_action_dist = dist_class(
        train_batch[SampleBatch.ACTION_DIST_INPUTS], model
    )

    # logp_ratio = torch.exp(
    #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    #     - train_batch[SampleBatch.ACTION_LOGP]
    # )

    prev_distill_action_dist = dist_class(
        train_batch[Distill.DIST_INPUTS], model
    )

    prev_distill_action_logp = prev_distill_action_dist.logp(train_batch[SampleBatch.ACTIONS])

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
     -prev_distill_action_logp
        # - train_batch[Distill.ACTION_LOGP]
    )


    # End Edit
    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if policy.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)
        mean_kl_loss = reduce_mean_valid(action_kl)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"]
        ),
    )
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)


    ## DisTral Edit
    # action_dist_updated = dist_class(logits.detach(), model)
    # distill_kl = action_dist_updated.kl(curr_distill_dist)
    # distill_loss = reduce_mean_valid(distill_kl)

    # distill_dist_updated = dist_class(distill_logits.detach(), model)
    # transfer_kl = distill_dist_updated.kl(curr_action_dist)
    # transfer_loss = reduce_mean_valid(transfer_kl)
    #Edit old action
    distill_loss = F.mse_loss(distill_logits, train_batch[SampleBatch.ACTION_DIST_INPUTS])
    transfer_action_kl = curr_action_dist.kl(prev_distill_action_dist)
    transfer_loss = reduce_mean_valid(transfer_action_kl)

    ## End Edit

    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_fn_out = model.value_function()
        vf_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
    # Ignore the value function.
    else:
        value_fn_out = 0
        vf_loss_clipped = mean_vf_loss = 0.0

    ppo_loss = reduce_mean_valid(
        -surrogate_loss
        + policy.config["vf_loss_coeff"] * vf_loss_clipped
        - policy.entropy_coeff * curr_entropy

    )

    # Add mean_kl_loss (already processed through `reduce_mean_valid`),
    # if necessary.
    total_loss = ppo_loss + policy.config['distill_coeff']*(distill_loss )

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["ppo_loss"] = ppo_loss
    model.tower_stats["transfer_kl"] = transfer_loss
    model.tower_stats["distill_loss"] = distill_loss

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
    )
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss




def loss_psudo_ppo(
        policy:TorchPolicyV2,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Compute loss for Distral Policy Objective The main difference with PPO is that the central policy takes.
    J = J_clip(pi_i/pi_0 A ) - H(pi_i) + distill_loss(pi_0, pi_i)
    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.

    Returns:
        The PPO loss tensor given the input batch.
    """

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # Edit Distral
    distill_logits = model.distill_out()
    # curr_distill_dist = dist_class(distill_logits, model)



    # End Edit

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean


    # Distral Edit


    prev_action_dist = dist_class(
        train_batch[SampleBatch.ACTION_DIST_INPUTS], model
    )

    # logp_ratio = torch.exp(
    #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    #     - train_batch[SampleBatch.ACTION_LOGP]
    # )

    prev_distill_action_dist = dist_class(
        train_batch[Distill.DIST_INPUTS], model
    )

    prev_distill_action_logp = prev_distill_action_dist.logp(train_batch[SampleBatch.ACTIONS])

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
     -prev_distill_action_logp
        # - train_batch[Distill.ACTION_LOGP]
    )


    # End Edit
    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if policy.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)
        mean_kl_loss = reduce_mean_valid(action_kl)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"]
        ),
    )
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)


    ## DisTral Edit
    # action_dist_updated = dist_class(logits.detach(), model)
    # distill_kl = action_dist_updated.kl(curr_distill_dist)
    # distill_loss = reduce_mean_valid(distill_kl)

    # distill_dist_updated = dist_class(distill_logits.detach(), model)
    # transfer_kl = distill_dist_updated.kl(curr_action_dist)
    # transfer_loss = reduce_mean_valid(transfer_kl)
    #Edit old action
    distill_loss = F.mse_loss(distill_logits, logits.detach())
    transfer_action_kl = curr_action_dist.kl(prev_distill_action_dist)
    transfer_loss = reduce_mean_valid(transfer_action_kl)

    ## End Edit

    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_fn_out = model.value_function()
        vf_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
    # Ignore the value function.
    else:
        value_fn_out = 0
        vf_loss_clipped = mean_vf_loss = 0.0

    ppo_loss = reduce_mean_valid(
        -surrogate_loss
        + policy.config["vf_loss_coeff"] * vf_loss_clipped
        - policy.entropy_coeff * curr_entropy

    )

    # Add mean_kl_loss (already processed through `reduce_mean_valid`),
    # if necessary.
    total_loss = ppo_loss + policy.config['distill_coeff']*(distill_loss )

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["ppo_loss"] = ppo_loss
    model.tower_stats["transfer_kl"] = transfer_loss
    model.tower_stats["distill_loss"] = distill_loss

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
    )
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss







def loss_ppo_added_terms(
        policy:TorchPolicyV2,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Compute loss for Distral Policy Objective The main difference with PPO is that the central policy takes.
    DisTral: J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i) - MSE(pi_0, pi_i) + H(pi_0) - KL(pi'_i, pi_0)
    PPO:     J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i)
    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.

    Returns:
        The PPO loss tensor given the input batch.
    """

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # Edit Distral
    distill_logits= model.distill_out()
    curr_distill_dist = dist_class(distill_logits, model)



    # End Edit

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean


    # Distral Edit


    prev_action_dist = dist_class(
        train_batch[SampleBatch.ACTION_DIST_INPUTS], model
    )

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
        - train_batch[SampleBatch.ACTION_LOGP]
    )

    prev_distill_action_dist = dist_class(
        train_batch[Distill.DIST_INPUTS], model
    )
    #
    # logp_ratio = torch.exp(
    #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    #     - train_batch[Distill.ACTION_LOGP]
    # )


    # End Edit
    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if policy.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)

        # Default C_kl in PPO: 0.2
        mean_kl_loss = reduce_mean_valid(action_kl)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)


    # Default C_Ent in PPO: 0
    curr_entropy = curr_action_dist.entropy()

    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"]
        ),
    )
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)


    ## DisTral Edit
    # action_dist_updated = dist_class(logits.detach(), model)
    # distill_kl = action_dist_updated.kl(curr_distill_dist)
    # distill_loss = F(distill_kl)

    # distill_dist_updated = dist_class(distill_logits.detach(), model)
    # transfer_kl = distill_dist_updated.kl(curr_action_dist)
    # transfer_loss = reduce_mean_valid(transfer_kl)
    # Edit V7
    distill_loss = F.mse_loss(distill_logits, logits.detach()) #- reduce_mean_valid(curr_distill_dist.entropy())
    transfer_action_kl = curr_action_dist.kl(prev_distill_action_dist)

    # Default C_kl in PPO: 0.2
    transfer_loss = reduce_mean_valid(transfer_action_kl)
    ## End Edit


    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_fn_out = model.value_function()
        vf_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
    # Ignore the value function.
    else:
        value_fn_out = 0
        vf_loss_clipped = mean_vf_loss = 0.0

    ppo_loss = reduce_mean_valid(
        -surrogate_loss
        + policy.config["vf_loss_coeff"] * vf_loss_clipped
        - policy.entropy_coeff * curr_entropy

    )

    # Add mean_kl_loss (already processed through `reduce_mean_valid`),
    # if necessary.
    total_loss = ppo_loss + policy.config['distill_coeff']*(distill_loss + transfer_loss)

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["ppo_loss"] = ppo_loss
    model.tower_stats["transfer_kl"] = transfer_loss
    model.tower_stats["distill_loss"] = distill_loss

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
    )
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss

def loss_ppo_distill(
        policy:TorchPolicyV2,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Compute loss for Distral Policy Objective The main difference with PPO is that the central policy takes.
    DisTral: J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i) - MSE(pi_0, pi_i)
    PPO:     J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i)
    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.

    Returns:
        The PPO loss tensor given the input batch.
    """

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # Edit Distral
    distill_logits= model.distill_out()
    curr_distill_dist = dist_class(distill_logits, model)



    # End Edit

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean


    # Distral Edit


    prev_action_dist = dist_class(
        train_batch[SampleBatch.ACTION_DIST_INPUTS], model
    )

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
        - train_batch[SampleBatch.ACTION_LOGP]
    )

    prev_distill_action_dist = dist_class(
        train_batch[Distill.DIST_INPUTS], model
    )
    #
    # logp_ratio = torch.exp(
    #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    #     - train_batch[Distill.ACTION_LOGP]
    # )


    # End Edit
    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if policy.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)

        # Default C_kl in PPO: 0.2
        mean_kl_loss = reduce_mean_valid(action_kl)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)


    # Default C_Ent in PPO: 0
    curr_entropy = curr_action_dist.entropy()

    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"]
        ),
    )
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)


    ## DisTral Edit
    # action_dist_updated = dist_class(logits.detach(), model)
    # distill_kl = action_dist_updated.kl(curr_distill_dist)
    # distill_loss = F(distill_kl)

    # distill_dist_updated = dist_class(distill_logits.detach(), model)
    # transfer_kl = distill_dist_updated.kl(curr_action_dist)
    # transfer_loss = reduce_mean_valid(transfer_kl)
    # Edit V7
    distill_loss = F.mse_loss(distill_logits, train_batch[SampleBatch.ACTION_DIST_INPUTS]) #- reduce_mean_valid(curr_distill_dist.entropy())
    transfer_action_kl = curr_action_dist.kl(prev_distill_action_dist)

    # Default C_kl in PPO: 0.2
    transfer_loss = reduce_mean_valid(transfer_action_kl)
    ## End Edit


    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_fn_out = model.value_function()
        vf_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
    # Ignore the value function.
    else:
        value_fn_out = 0
        vf_loss_clipped = mean_vf_loss = 0.0

    ppo_loss = reduce_mean_valid(
        -surrogate_loss
        + policy.config["vf_loss_coeff"] * vf_loss_clipped
        - policy.entropy_coeff * curr_entropy

    )

    # Add mean_kl_loss (already processed through `reduce_mean_valid`),
    # if necessary.
    total_loss = ppo_loss + policy.config['distill_coeff']*(distill_loss )

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["ppo_loss"] = ppo_loss
    model.tower_stats["transfer_kl"] = transfer_loss
    model.tower_stats["distill_loss"] = distill_loss

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
    )
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss



def loss_ppo_added_terms2(
        policy:TorchPolicyV2,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Compute loss for Distral Policy Objective The main difference with PPO is that the central policy takes.
    DisTral: J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i) - MSE(pi_0, pi_i) + H(pi_0) - KL(pi'_i, pi_0)
    PPO:     J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i)
    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.

    Returns:
        The PPO loss tensor given the input batch.
    """

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # Edit Distral
    distill_logits= model.distill_out()
    curr_distill_dist = dist_class(distill_logits, model)



    # End Edit

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean


    # Distral Edit


    prev_action_dist = dist_class(
        train_batch[SampleBatch.ACTION_DIST_INPUTS], model
    )

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
        - train_batch[SampleBatch.ACTION_LOGP]
    )

    prev_distill_action_dist = dist_class(
        train_batch[Distill.DIST_INPUTS], model
    )
    #
    # logp_ratio = torch.exp(
    #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    #     - train_batch[Distill.ACTION_LOGP]
    # )


    # End Edit
    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if policy.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)

        # Default C_kl in PPO: 0.2
        mean_kl_loss = reduce_mean_valid(action_kl)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)


    # Default C_Ent in PPO: 0
    curr_entropy = curr_action_dist.entropy()

    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"]
        ),
    )
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)


    ## DisTral Edit
    # action_dist_updated = dist_class(logits.detach(), model)
    # distill_kl = action_dist_updated.kl(curr_distill_dist)
    # distill_loss = F(distill_kl)

    # distill_dist_updated = dist_class(distill_logits.detach(), model)
    # transfer_kl = distill_dist_updated.kl(curr_action_dist)
    # transfer_loss = reduce_mean_valid(transfer_kl)
    # Edit V7
    distill_loss = F.mse_loss(distill_logits, train_batch[SampleBatch.ACTION_DIST_INPUTS]) #- reduce_mean_valid(curr_distill_dist.entropy())
    transfer_action_kl = curr_action_dist.kl(prev_distill_action_dist)

    # Default C_kl in PPO: 0.2
    transfer_loss = reduce_mean_valid(transfer_action_kl)
    ## End Edit


    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_fn_out = model.value_function()
        vf_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
    # Ignore the value function.
    else:
        value_fn_out = 0
        vf_loss_clipped = mean_vf_loss = 0.0

    ppo_loss = reduce_mean_valid(
        -surrogate_loss
        + policy.config["vf_loss_coeff"] * vf_loss_clipped
        - policy.entropy_coeff * curr_entropy

    )

    # Add mean_kl_loss (already processed through `reduce_mean_valid`),
    # if necessary.
    total_loss = ppo_loss + policy.config['distill_coeff']*(distill_loss + transfer_loss)

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["ppo_loss"] = ppo_loss
    model.tower_stats["transfer_kl"] = transfer_loss
    model.tower_stats["distill_loss"] = distill_loss

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
    )
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss






def loss_psudo_ppo3(
        policy:TorchPolicyV2,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Compute loss for Distral Policy Objective The main difference with PPO is that the central policy takes.
    J = J_clip(pi_i/pi_0 A ) - H(pi_i) + distill_loss(pi_0, pi_i)
    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.

    Returns:
        The PPO loss tensor given the input batch.
    """

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # Edit Distral
    distill_logits = model.distill_out()
    curr_distill_dist = dist_class(distill_logits, model)



    # End Edit

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean


    # Distral Edit


    prev_action_dist = dist_class(
        train_batch[SampleBatch.ACTION_DIST_INPUTS], model
    )

    # logp_ratio = torch.exp(
    #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    #     - train_batch[SampleBatch.ACTION_LOGP]
    # )

    prev_distill_action_dist = dist_class(
        train_batch[Distill.DIST_INPUTS], model
    )

    prev_distill_action_logp = prev_distill_action_dist.logp(train_batch[SampleBatch.ACTIONS])

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
     -prev_distill_action_logp
        # - train_batch[Distill.ACTION_LOGP]
    )


    # End Edit
    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if policy.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)
        mean_kl_loss = reduce_mean_valid(action_kl)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"]
        ),
    )
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)


    ## DisTral Edit
    # action_dist_updated = dist_class(logits.detach(), model)
    # distill_kl = action_dist_updated.kl(curr_distill_dist)
    # distill_loss = reduce_mean_valid(distill_kl)

    # distill_dist_updated = dist_class(distill_logits.detach(), model)
    # transfer_kl = distill_dist_updated.kl(curr_action_dist)
    # transfer_loss = reduce_mean_valid(transfer_kl)
    #Edit old action
    #distill_loss = F.mse_loss(distill_logits, logits.detach())
    distill_loss = reduce_mean_valid(curr_distill_dist.kl(prev_action_dist))
    transfer_action_kl = curr_action_dist.kl(prev_distill_action_dist)
    transfer_loss = reduce_mean_valid(transfer_action_kl)

    ## End Edit

    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_fn_out = model.value_function()
        vf_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
    # Ignore the value function.
    else:
        value_fn_out = 0
        vf_loss_clipped = mean_vf_loss = 0.0

    ppo_loss = reduce_mean_valid(
        -surrogate_loss
        + policy.config["vf_loss_coeff"] * vf_loss_clipped
        - policy.entropy_coeff * curr_entropy

    )

    # Add mean_kl_loss (already processed through `reduce_mean_valid`),
    # if necessary.
    total_loss = ppo_loss + policy.config['distill_coeff']*(distill_loss )

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["ppo_loss"] = ppo_loss
    model.tower_stats["transfer_kl"] = transfer_loss
    model.tower_stats["distill_loss"] = distill_loss

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
    )
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss







def loss_ppo_added_terms3(
        policy:TorchPolicyV2,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Compute loss for Distral Policy Objective The main difference with PPO is that the central policy takes.
    DisTral: J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i) - MSE(pi_0, pi_i) + H(pi_0) - KL(pi'_i, pi_0)
    PPO:     J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i)
    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.

    Returns:
        The PPO loss tensor given the input batch.
    """

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # Edit Distral
    distill_logits= model.distill_out()
    curr_distill_dist = dist_class(distill_logits, model)



    # End Edit

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean


    # Distral Edit


    prev_action_dist = dist_class(
        train_batch[SampleBatch.ACTION_DIST_INPUTS], model
    )

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
        - train_batch[SampleBatch.ACTION_LOGP]
    )

    prev_distill_action_dist = dist_class(
        train_batch[Distill.DIST_INPUTS], model
    )
    #
    # logp_ratio = torch.exp(
    #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    #     - train_batch[Distill.ACTION_LOGP]
    # )


    # End Edit
    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if policy.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)

        # Default C_kl in PPO: 0.2
        mean_kl_loss = reduce_mean_valid(action_kl)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)


    # Default C_Ent in PPO: 0
    curr_entropy = curr_action_dist.entropy()

    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"]
        ),
    )
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)


    ## DisTral Edit
    # action_dist_updated = dist_class(logits.detach(), model)
    # distill_kl = action_dist_updated.kl(curr_distill_dist)
    # distill_loss = F(distill_kl)

    # distill_dist_updated = dist_class(distill_logits.detach(), model)
    # transfer_kl = distill_dist_updated.kl(curr_action_dist)
    # transfer_loss = reduce_mean_valid(transfer_kl)
    # Edit V7
    distill_loss = reduce_mean_valid(curr_distill_dist.kl(prev_action_dist))
    # distill_loss = F.mse_loss(distill_logits, logits.detach()) #- reduce_mean_valid(curr_distill_dist.entropy())
    transfer_action_kl = curr_action_dist.kl(prev_distill_action_dist)

    # Default C_kl in PPO: 0.2
    transfer_loss = reduce_mean_valid(transfer_action_kl)
    ## End Edit


    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_fn_out = model.value_function()
        vf_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
    # Ignore the value function.
    else:
        value_fn_out = 0
        vf_loss_clipped = mean_vf_loss = 0.0

    ppo_loss = reduce_mean_valid(
        -surrogate_loss
        + policy.config["vf_loss_coeff"] * vf_loss_clipped
        - policy.entropy_coeff * curr_entropy

    )

    # Add mean_kl_loss (already processed through `reduce_mean_valid`),
    # if necessary.
    total_loss = ppo_loss + policy.config['distill_coeff']*(distill_loss + transfer_loss)

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["ppo_loss"] = ppo_loss
    model.tower_stats["transfer_kl"] = transfer_loss
    model.tower_stats["distill_loss"] = distill_loss

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
    )
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss

def loss_ppo_distill3(
        policy:TorchPolicyV2,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Compute loss for Distral Policy Objective The main difference with PPO is that the central policy takes.
    DisTral: J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i) - MSE(pi_0, pi_i)
    PPO:     J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i)
    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.

    Returns:
        The PPO loss tensor given the input batch.
    """

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # Edit Distral
    distill_logits= model.distill_out()
    curr_distill_dist = dist_class(distill_logits, model)



    # End Edit

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean


    # Distral Edit


    prev_action_dist = dist_class(
        train_batch[SampleBatch.ACTION_DIST_INPUTS], model
    )

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
        - train_batch[SampleBatch.ACTION_LOGP]
    )

    prev_distill_action_dist = dist_class(
        train_batch[Distill.DIST_INPUTS], model
    )
    #
    # logp_ratio = torch.exp(
    #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    #     - train_batch[Distill.ACTION_LOGP]
    # )


    # End Edit
    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if policy.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)

        # Default C_kl in PPO: 0.2
        mean_kl_loss = reduce_mean_valid(action_kl)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)


    # Default C_Ent in PPO: 0
    curr_entropy = curr_action_dist.entropy()

    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"]
        ),
    )
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)


    ## DisTral Edit
    # action_dist_updated = dist_class(logits.detach(), model)
    # distill_kl = action_dist_updated.kl(curr_distill_dist)
    # distill_loss = F(distill_kl)

    # distill_dist_updated = dist_class(distill_logits.detach(), model)
    # transfer_kl = distill_dist_updated.kl(curr_action_dist)
    # transfer_loss = reduce_mean_valid(transfer_kl)
    # Edit V7
    distill_loss = reduce_mean_valid(curr_distill_dist.kl(prev_action_dist))
    transfer_action_kl = curr_action_dist.kl(prev_distill_action_dist)

    # Default C_kl in PPO: 0.2
    transfer_loss = reduce_mean_valid(transfer_action_kl)
    ## End Edit


    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_fn_out = model.value_function()
        vf_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
    # Ignore the value function.
    else:
        value_fn_out = 0
        vf_loss_clipped = mean_vf_loss = 0.0

    ppo_loss = reduce_mean_valid(
        -surrogate_loss
        + policy.config["vf_loss_coeff"] * vf_loss_clipped
        - policy.entropy_coeff * curr_entropy

    )

    # Add mean_kl_loss (already processed through `reduce_mean_valid`),
    # if necessary.
    total_loss = ppo_loss + policy.config['distill_coeff']*(distill_loss )

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["ppo_loss"] = ppo_loss
    model.tower_stats["transfer_kl"] = transfer_loss
    model.tower_stats["distill_loss"] = distill_loss

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
    )
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss




def loss_ppo_added_terms4(
        policy:TorchPolicyV2,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Compute loss for Distral Policy Objective The main difference with PPO is that the central policy takes.
    DisTral: J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i) - MSE(pi_0, pi_i) + H(pi_0) - KL(pi'_i, pi_0)
    PPO:     J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i)
    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.

    Returns:
        The PPO loss tensor given the input batch.
    """

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # Edit Distral
    distill_logits= model.distill_out()
    curr_distill_dist = dist_class(distill_logits, model)



    # End Edit

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean


    # Distral Edit


    prev_action_dist = dist_class(
        train_batch[SampleBatch.ACTION_DIST_INPUTS], model
    )

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
        - train_batch[SampleBatch.ACTION_LOGP]
    )

    prev_distill_action_dist = dist_class(
        train_batch[Distill.DIST_INPUTS], model
    )
    #
    # logp_ratio = torch.exp(
    #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    #     - train_batch[Distill.ACTION_LOGP]
    # )


    # End Edit
    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if policy.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)

        # Default C_kl in PPO: 0.2
        mean_kl_loss = reduce_mean_valid(action_kl)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)


    # Default C_Ent in PPO: 0
    curr_entropy = curr_action_dist.entropy()

    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"]
        ),
    )
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)


    ## DisTral Edit
    # action_dist_updated = dist_class(logits.detach(), model)
    # distill_kl = action_dist_updated.kl(curr_distill_dist)
    # distill_loss = F(distill_kl)

    # distill_dist_updated = dist_class(distill_logits.detach(), model)
    # transfer_kl = distill_dist_updated.kl(curr_action_dist)
    # transfer_loss = reduce_mean_valid(transfer_kl)
    # Edit V7
    distill_loss = reduce_mean_valid(curr_distill_dist.kl(prev_action_dist)) + F.mse_loss(distill_logits, train_batch[SampleBatch.ACTION_DIST_INPUTS])
    # distill_loss = F.mse_loss(distill_logits, logits.detach()) #- reduce_mean_valid(curr_distill_dist.entropy())
    transfer_action_kl = curr_action_dist.kl(prev_distill_action_dist)

    # Default C_kl in PPO: 0.2
    transfer_loss = reduce_mean_valid(transfer_action_kl)
    ## End Edit


    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_fn_out = model.value_function()
        vf_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
    # Ignore the value function.
    else:
        value_fn_out = 0
        vf_loss_clipped = mean_vf_loss = 0.0

    ppo_loss = reduce_mean_valid(
        -surrogate_loss
        + policy.config["vf_loss_coeff"] * vf_loss_clipped
        - policy.entropy_coeff * curr_entropy

    )

    # Add mean_kl_loss (already processed through `reduce_mean_valid`),
    # if necessary.
    total_loss = ppo_loss + policy.config['distill_coeff']*(distill_loss + transfer_loss)

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["ppo_loss"] = ppo_loss
    model.tower_stats["transfer_kl"] = transfer_loss
    model.tower_stats["distill_loss"] = distill_loss

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
    )
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss




def loss_ppo_added_terms5(
        policy:TorchPolicyV2,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Compute loss for Distral Policy Objective The main difference with PPO is that the central policy takes.
    DisTral: J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i) - MSE(pi_0, pi_i) + H(pi_0) - KL(pi'_i, pi_0)
    PPO:     J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i)
    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.

    Returns:
        The PPO loss tensor given the input batch.
    """

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # Edit Distral
    distill_logits= model.distill_out()
    curr_distill_dist = dist_class(distill_logits, model)



    # End Edit

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean


    # Distral Edit


    prev_action_dist = dist_class(
        train_batch[SampleBatch.ACTION_DIST_INPUTS], model
    )

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
        - train_batch[SampleBatch.ACTION_LOGP]
    )

    prev_distill_action_dist = dist_class(
        train_batch[Distill.DIST_INPUTS], model
    )
    #
    # logp_ratio = torch.exp(
    #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    #     - train_batch[Distill.ACTION_LOGP]
    # )


    # End Edit
    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if policy.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)

        # Default C_kl in PPO: 0.2
        mean_kl_loss = reduce_mean_valid(action_kl)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)


    # Default C_Ent in PPO: 0
    curr_entropy = curr_action_dist.entropy()

    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"]
        ),
    )
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)


    ## DisTral Edit
    # action_dist_updated = dist_class(logits.detach(), model)
    # distill_kl = action_dist_updated.kl(curr_distill_dist)
    # distill_loss = F(distill_kl)

    # distill_dist_updated = dist_class(distill_logits.detach(), model)
    # transfer_kl = distill_dist_updated.kl(curr_action_dist)
    # transfer_loss = reduce_mean_valid(transfer_kl)
    # Edit V7
    distill_loss = reduce_mean_valid(curr_distill_dist.kl(prev_action_dist)) + F.mse_loss(distill_logits, train_batch[SampleBatch.ACTION_DIST_INPUTS])
    # distill_loss = F.mse_loss(distill_logits, logits.detach()) #- reduce_mean_valid(curr_distill_dist.entropy())
    transfer_action_kl = prev_distill_action_dist.kl(curr_action_dist)# +  F.mse_loss(train_batch[Distill.DIST_INPUTS], logits)
    # action_kl = prev_action_dist.kl(curr_action_dist)
    # Default C_kl in PPO: 0.2
    transfer_loss = reduce_mean_valid(transfer_action_kl)
    ## End Edit


    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_fn_out = model.value_function()
        vf_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
    # Ignore the value function.
    else:
        value_fn_out = 0
        vf_loss_clipped = mean_vf_loss = 0.0

    ppo_loss = reduce_mean_valid(
        -surrogate_loss
        + policy.config["vf_loss_coeff"] * vf_loss_clipped
        - policy.entropy_coeff * curr_entropy

    )

    # Add mean_kl_loss (already processed through `reduce_mean_valid`),
    # if necessary.
    total_loss = ppo_loss + policy.config['distill_coeff']*(distill_loss + transfer_loss)

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["ppo_loss"] = ppo_loss
    model.tower_stats["transfer_kl"] = transfer_loss
    model.tower_stats["distill_loss"] = distill_loss

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
    )
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss






def loss_ppo_added_terms6(
        policy:TorchPolicyV2,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Compute loss for Distral Policy Objective The main difference with PPO is that the central policy takes.
    DisTral: J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i) - MSE(pi_0, pi_i) + H(pi_0) - KL(pi'_i, pi_0)
    PPO:     J = J_clip(pi'_i/pi_i A ) + H(pi_i) - KL(pi'_i, pi_i)
    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.

    Returns:
        The PPO loss tensor given the input batch.
    """

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # Edit Distral
    distill_logits= model.distill_out()
    curr_distill_dist = dist_class(distill_logits, model)



    # End Edit

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean


    # Distral Edit


    prev_action_dist = dist_class(
        train_batch[SampleBatch.ACTION_DIST_INPUTS], model
    )

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
        - train_batch[SampleBatch.ACTION_LOGP]
    )

    prev_distill_action_dist = dist_class(
        train_batch[Distill.DIST_INPUTS], model
    )
    #
    # logp_ratio = torch.exp(
    #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    #     - train_batch[Distill.ACTION_LOGP]
    # )


    # End Edit
    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if policy.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)

        # Default C_kl in PPO: 0.2
        mean_kl_loss = reduce_mean_valid(action_kl)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)


    # Default C_Ent in PPO: 0
    curr_entropy = curr_action_dist.entropy()

    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"]
        ),
    )
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)


    ## DisTral Edit
    # action_dist_updated = dist_class(logits.detach(), model)
    # distill_kl = action_dist_updated.kl(curr_distill_dist)
    # distill_loss = F(distill_kl)

    # distill_dist_updated = dist_class(distill_logits.detach(), model)
    # transfer_kl = distill_dist_updated.kl(curr_action_dist)
    # transfer_loss = reduce_mean_valid(transfer_kl)
    # Edit V7
    distill_loss = reduce_mean_valid(curr_distill_dist.kl(prev_action_dist)) + F.mse_loss(distill_logits, train_batch[SampleBatch.ACTION_DIST_INPUTS])
    # distill_loss = F.mse_loss(distill_logits, logits.detach()) #- reduce_mean_valid(curr_distill_dist.entropy())
    transfer_action_kl = prev_distill_action_dist.kl(curr_action_dist)
    # action_kl = prev_action_dist.kl(curr_action_dist)
    # Default C_kl in PPO: 0.2
    transfer_loss = reduce_mean_valid(transfer_action_kl) +  F.mse_loss(train_batch[Distill.DIST_INPUTS], logits)
    ## End Edit


    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_fn_out = model.value_function()
        vf_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
    # Ignore the value function.
    else:
        value_fn_out = 0
        vf_loss_clipped = mean_vf_loss = 0.0

    ppo_loss = reduce_mean_valid(
        -surrogate_loss
        + policy.config["vf_loss_coeff"] * vf_loss_clipped
        - policy.entropy_coeff * curr_entropy

    )

    # Add mean_kl_loss (already processed through `reduce_mean_valid`),
    # if necessary.
    total_loss = ppo_loss + policy.config['distill_coeff']*(distill_loss + transfer_loss)

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["ppo_loss"] = ppo_loss
    model.tower_stats["transfer_kl"] = transfer_loss
    model.tower_stats["distill_loss"] = distill_loss

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
    )
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss




def loss_psudo_ppo4(
        policy:TorchPolicyV2,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Compute loss for Distral Policy Objective The main difference with PPO is that the central policy takes.
    J = J_clip(pi_i/pi_0 A ) - H(pi_i) + distill_loss(pi_0, pi_i)
    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.

    Returns:
        The PPO loss tensor given the input batch.
    """

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # Edit Distral
    distill_logits = model.distill_out()
    curr_distill_dist = dist_class(distill_logits, model)



    # End Edit

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean


    # Distral Edit


    prev_action_dist = dist_class(
        train_batch[SampleBatch.ACTION_DIST_INPUTS], model
    )

    # logp_ratio = torch.exp(
    #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    #     - train_batch[SampleBatch.ACTION_LOGP]
    # )

    prev_distill_action_dist = dist_class(
        train_batch[Distill.DIST_INPUTS], model
    )

    prev_distill_action_logp = prev_distill_action_dist.logp(train_batch[SampleBatch.ACTIONS])

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
     -prev_distill_action_logp
        # - train_batch[Distill.ACTION_LOGP]
    )


    # End Edit
    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if policy.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)
        mean_kl_loss = reduce_mean_valid(action_kl)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"]
        ),
    )
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)


    ## DisTral Edit
    # action_dist_updated = dist_class(logits.detach(), model)
    # distill_kl = action_dist_updated.kl(curr_distill_dist)
    # distill_loss = reduce_mean_valid(distill_kl)

    # distill_dist_updated = dist_class(distill_logits.detach(), model)
    # transfer_kl = distill_dist_updated.kl(curr_action_dist)
    # transfer_loss = reduce_mean_valid(transfer_kl)
    #Edit old action
    #distill_loss = F.mse_loss(distill_logits, logits.detach())
    distill_loss = reduce_mean_valid(curr_distill_dist.kl(prev_action_dist))
    transfer_action_kl = prev_distill_action_dist.kl(curr_action_dist)
    transfer_loss = reduce_mean_valid(transfer_action_kl) +  F.mse_loss(train_batch[Distill.DIST_INPUTS], logits)

    ## End Edit

    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_fn_out = model.value_function()
        vf_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
    # Ignore the value function.
    else:
        value_fn_out = 0
        vf_loss_clipped = mean_vf_loss = 0.0

    ppo_loss = reduce_mean_valid(
        -surrogate_loss
        + policy.config["vf_loss_coeff"] * vf_loss_clipped
        - policy.entropy_coeff * curr_entropy

    )

    # Add mean_kl_loss (already processed through `reduce_mean_valid`),
    # if necessary.
    total_loss = ppo_loss + policy.config['distill_coeff']*(distill_loss + transfer_loss)

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["ppo_loss"] = ppo_loss
    model.tower_stats["transfer_kl"] = transfer_loss
    model.tower_stats["distill_loss"] = distill_loss

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
    )
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss



def loss_psudo_ppo5(
        policy:TorchPolicyV2,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Compute loss for Distral Policy Objective The main difference with PPO is that the central policy takes.
    J = J_clip(pi_i/pi_0 A ) - H(pi_i) + distill_loss(pi_0, pi_i)
    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.

    Returns:
        The PPO loss tensor given the input batch.
    """

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # Edit Distral
    distill_logits = model.distill_out()
    curr_distill_dist = dist_class(distill_logits, model)



    # End Edit

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean


    # Distral Edit


    prev_action_dist = dist_class(
        train_batch[SampleBatch.ACTION_DIST_INPUTS], model
    )

    # logp_ratio = torch.exp(
    #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    #     - train_batch[SampleBatch.ACTION_LOGP]
    # )

    prev_distill_action_dist = dist_class(
        train_batch[Distill.DIST_INPUTS], model
    )

    prev_distill_action_logp = prev_distill_action_dist.logp(train_batch[SampleBatch.ACTIONS])

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
     -prev_distill_action_logp
        # - train_batch[Distill.ACTION_LOGP]
    )


    # End Edit
    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if policy.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)
        mean_kl_loss = reduce_mean_valid(action_kl)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"]
        ),
    )
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)


    ## DisTral Edit
    # action_dist_updated = dist_class(logits.detach(), model)
    # distill_kl = action_dist_updated.kl(curr_distill_dist)
    # distill_loss = reduce_mean_valid(distill_kl)

    # distill_dist_updated = dist_class(distill_logits.detach(), model)
    # transfer_kl = distill_dist_updated.kl(curr_action_dist)
    # transfer_loss = reduce_mean_valid(transfer_kl)
    #Edit old action
    #distill_loss = F.mse_loss(distill_logits, logits.detach())
    distill_loss = reduce_mean_valid(curr_distill_dist.kl(prev_action_dist))
    transfer_action_kl = prev_distill_action_dist.kl(curr_action_dist)
    transfer_loss = reduce_mean_valid(transfer_action_kl) #+  F.mse_loss(train_batch[Distill.DIST_INPUTS], logits)

    ## End Edit

    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_fn_out = model.value_function()
        vf_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
    # Ignore the value function.
    else:
        value_fn_out = 0
        vf_loss_clipped = mean_vf_loss = 0.0

    ppo_loss = reduce_mean_valid(
        -surrogate_loss
        + policy.config["vf_loss_coeff"] * vf_loss_clipped
        - policy.entropy_coeff * curr_entropy

    )

    # Add mean_kl_loss (already processed through `reduce_mean_valid`),
    # if necessary.
    total_loss = ppo_loss + policy.config['distill_coeff']*(distill_loss + transfer_loss)

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["ppo_loss"] = ppo_loss
    model.tower_stats["transfer_kl"] = transfer_loss
    model.tower_stats["distill_loss"] = distill_loss

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
    )
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss


# def dual_distill_psudo_ppo_loss(
#         policy:TorchPolicyV2,
#         model: ModelV2,
#         dist_class: Type[ActionDistribution],
#         train_batch: SampleBatch,
# ) -> Union[TensorType, List[TensorType]]:
#     """Compute loss for Distral Policy Objective The main difference with PPO is that the central policy takes.
#     J = J_clip(pi_i/pi_0 A ) - H(pi_i) + distill_loss(pi_0, pi_i)
#     Args:
#         model: The Model to calculate the loss for.
#         dist_class: The action distr. class.
#         train_batch: The training data.
#
#     Returns:
#         The PPO loss tensor given the input batch.
#     """
#     target_model = policy.target_models[model]
#     logits, state = model(train_batch)
#     curr_action_dist = dist_class(logits, model)
#
#     # Edit Distral
#     distill_logits = model.distill_out()
#     curr_distill_dist = dist_class(distill_logits, model)
#
#
#
#     # End Edit
#
#     # RNN case: Mask away 0-padded chunks at end of time axis.
#     if state:
#         B = len(train_batch[SampleBatch.SEQ_LENS])
#         max_seq_len = logits.shape[0] // B
#         mask = sequence_mask(
#             train_batch[SampleBatch.SEQ_LENS],
#             max_seq_len,
#             time_major=model.is_time_major(),
#         )
#         mask = torch.reshape(mask, [-1])
#         num_valid = torch.sum(mask)
#
#         def reduce_mean_valid(t):
#             return torch.sum(t[mask]) / num_valid
#
#     # non-RNN case: No masking.
#     else:
#         mask = None
#         reduce_mean_valid = torch.mean
#
#
#     # Distral Edit
#
#
#     prev_action_dist = dist_class(
#         train_batch[SampleBatch.ACTION_DIST_INPUTS], model
#     )
#
#     # logp_ratio = torch.exp(
#     #     curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
#     #     - train_batch[SampleBatch.ACTION_LOGP]
#     # )
#
#     prev_distill_action_dist = dist_class(
#         train_batch[Distill.DIST_INPUTS], model
#     )
#
#     prev_distill_action_logp = prev_distill_action_dist.logp(train_batch[SampleBatch.ACTIONS])
#
#     logp_ratio = torch.exp(
#         curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
#      -prev_distill_action_logp
#         # - train_batch[Distill.ACTION_LOGP]
#     )
#
#
#     # End Edit
#     # Only calculate kl loss if necessary (kl-coeff > 0.0).
#     if policy.config["kl_coeff"] > 0.0:
#         action_kl = prev_action_dist.kl(curr_action_dist)
#         mean_kl_loss = reduce_mean_valid(action_kl)
#     else:
#         mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)
#
#     curr_entropy = curr_action_dist.entropy()
#     mean_entropy = reduce_mean_valid(curr_entropy)
#
#     surrogate_loss = torch.min(
#         train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
#         train_batch[Postprocessing.ADVANTAGES]
#         * torch.clamp(
#             logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"]
#         ),
#     )
#     mean_policy_loss = reduce_mean_valid(-surrogate_loss)
#
#
#     ## DisTral Edit
#     # action_dist_updated = dist_class(logits.detach(), model)
#     # distill_kl = action_dist_updated.kl(curr_distill_dist)
#     # distill_loss = reduce_mean_valid(distill_kl)
#
#     # distill_dist_updated = dist_class(distill_logits.detach(), model)
#     # transfer_kl = distill_dist_updated.kl(curr_action_dist)
#     # transfer_loss = reduce_mean_valid(transfer_kl)
#     #Edit old action
#     #distill_loss = F.mse_loss(distill_logits, logits.detach())
#     distill_loss = reduce_mean_valid(curr_distill_dist.kl(prev_action_dist))
#     transfer_action_kl = prev_distill_action_dist.kl(curr_action_dist)
#     transfer_loss = reduce_mean_valid(transfer_action_kl) #+  F.mse_loss(train_batch[Distill.DIST_INPUTS], logits)
#
#     ## End Edit
#
#     # Compute a value function loss.
#     if policy.config["use_critic"]:
#         value_fn_out = model.value_function()
#         vf_loss = torch.pow(
#             value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
#         )
#         vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
#         mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
#     # Ignore the value function.
#     else:
#         value_fn_out = 0
#         vf_loss_clipped = mean_vf_loss = 0.0
#
#     ppo_loss = reduce_mean_valid(
#         -surrogate_loss
#         + policy.config["vf_loss_coeff"] * vf_loss_clipped
#         - policy.entropy_coeff * curr_entropy
#
#     )
#
#     # Add mean_kl_loss (already processed through `reduce_mean_valid`),
#     # if necessary.
#     total_loss = ppo_loss + policy.config['distill_coeff']*(distill_loss + transfer_loss)
#
#     if policy.config["kl_coeff"] > 0.0:
#         total_loss += policy.kl_coeff * mean_kl_loss
#
#     # Store values for stats function in model (tower), such that for
#     # multi-GPU, we do not override them during the parallel loss phase.
#     model.tower_stats["ppo_loss"] = ppo_loss
#     model.tower_stats["transfer_kl"] = transfer_loss
#     model.tower_stats["distill_loss"] = distill_loss
#
#     model.tower_stats["total_loss"] = total_loss
#     model.tower_stats["mean_policy_loss"] = mean_policy_loss
#     model.tower_stats["mean_vf_loss"] = mean_vf_loss
#     model.tower_stats["vf_explained_var"] = explained_variance(
#         train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
#     )
#     model.tower_stats["mean_entropy"] = mean_entropy
#     model.tower_stats["mean_kl_loss"] = mean_kl_loss
#
#     return total_loss


#
#
def dual_distill_regularized_ppo_loss(
        policy:TorchPolicyV2,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Compute loss for Distral Policy Objective The main difference with PPO is that the central policy takes.
    J = J_clip(pi_i/pi_0 A ) - H(pi_i) + distill_loss(pi_0, pi_i)
    Args:
        model: The Model to calculate the loss for.
        dist_class: The action distr. class.
        train_batch: The training data.

    Returns:
        The PPO loss tensor given the input batch.
    """



    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # Edit Distral
    distill_logits = model.distill_out()
    curr_distill_dist = dist_class(distill_logits, model)

    target_model = policy.target_models[model]
    target_logits, _ = target_model(train_batch)
    target_distill_logits = target_model.distill_out()
    curr_target_distill_dist = dist_class(target_distill_logits.detach(), target_model)



    # End Edit

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean


    # Distral Edit

    prev_action_dist = dist_class(
        train_batch[SampleBatch.ACTION_DIST_INPUTS], model
    )

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
        - train_batch[SampleBatch.ACTION_LOGP]
    )

    prev_distill_action_dist = dist_class(
        train_batch[Distill.DIST_INPUTS], model
    )


    # End Edit
    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if policy.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)
        mean_kl_loss = reduce_mean_valid(action_kl)
    else:
        mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES]
        * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"]
        ),
    )
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)


    ## DisTral Edit

    distill_loss = reduce_mean_valid(curr_distill_dist.kl(prev_action_dist)) + F.mse_loss(distill_logits, train_batch[SampleBatch.ACTION_DIST_INPUTS])
    # distill_loss = F.mse_loss(distill_logits, logits.detach()) #- reduce_mean_valid(curr_distill_dist.entropy())
    transfer_action_kl = curr_target_distill_dist.kl(curr_action_dist)# +  F.mse_loss(train_batch[Distill.DIST_INPUTS], logits)
    # action_kl = prev_action_dist.kl(curr_action_dist)
    # Default C_kl in PPO: 0.2
    transfer_loss = reduce_mean_valid(transfer_action_kl)
    ## End Edit

    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_fn_out = model.value_function()
        vf_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
    # Ignore the value function.
    else:
        value_fn_out = 0
        vf_loss_clipped = mean_vf_loss = 0.0

    ppo_loss = reduce_mean_valid(
        -surrogate_loss
        + policy.config["vf_loss_coeff"] * vf_loss_clipped
        - policy.entropy_coeff * curr_entropy

    )

    # Add mean_kl_loss (already processed through `reduce_mean_valid`),
    # if necessary.
    total_loss = ppo_loss + policy.config['distill_coeff']*distill_loss + policy.config['transfer_coeff']*transfer_loss

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["ppo_loss"] = ppo_loss
    model.tower_stats["transfer_kl"] = transfer_loss
    model.tower_stats["distill_loss"] = distill_loss

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
    )
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss
#
#
#
# def dual_distill_loss






