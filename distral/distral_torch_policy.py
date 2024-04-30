"""
PyTorch policy class used for SAC.
"""
import copy

import gym
import numpy as np
from gym.spaces import Box, Discrete
import logging
import tree  # pip install dm_tree
from typing import Dict, List, Optional, Tuple, Type, Union

from ray.rllib.algorithms.sac.sac_tf_policy import (
    postprocess_trajectory,
    validate_spaces,
)
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS, postprocess_nstep_and_prio
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.evaluation import Episode
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
    TorchDirichlet,
    TorchSquashedGaussian,
    TorchDiagGaussian,
    TorchBeta,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    minimize_and_clip,
    concat_multi_gpu_td_errors,
    huber_loss,
)
from ray.rllib.utils.typing import (
    LocalOptimizer,
    ModelInputDict,
    TensorType,
    AlgorithmConfigDict, AgentID,
)
# from ray.rllib.algorithms.sac.sac_tf_model import SACTFModel
from distral.distral_torch_model import DistilledTorchModel, DistralTorchModel, DistralCentralTorchModel
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS

# from distral.distral_torch_policy import DistralTorchPolicy
from distral.distral import DEFAULT_CONFIG


torch, nn = try_import_torch()
F = nn.functional

logger = logging.getLogger(__name__)
OPPONENT_OBS = 'opp_obs'
OPPONENT_ACT = 'opp_actions'


def gaussian_kld(
    mean1: TensorType, logvar1: TensorType, mean2: TensorType, logvar2: TensorType
) -> TensorType:
    """Compute KL divergence between a bunch of univariate Gaussian
        distributions with the given means and log-variances.
        ie `KL(N(mean1, logvar1) || N(mean2, logvar2))`

    Args:
        mean1 (TensorType):
        logvar1 (TensorType):
        mean2 (TensorType):
        logvar2 (TensorType):

    Returns:
        TensorType: [description]
    """

    gauss_klds = 0.5 * (
        (logvar2 - logvar1)
        + ((torch.exp(logvar1) + (mean1 - mean2) ** 2.0) / torch.exp(logvar2) )
        # + ((torch.exp(logvar1) + (mean1 - mean2) ** 2.0) / (torch.exp(logvar2)+0.0001)) #reduce the chance of explosive gradinet
        - 1.0
    )
    assert len(gauss_klds.size()) == 2
    return gauss_klds

def categorical_kld(p: TensorType, q: TensorType) -> TensorType:

    return (p* torch.log(p/q)).sum(-1, True)

def _get_dist_class(
    policy: Policy, config: AlgorithmConfigDict, action_space: gym.spaces.Space
) -> Type[TorchDistributionWrapper]:
    """Helper function to return a dist class based on config and action space.

    Args:
        policy: The policy for which to return the action
            dist class.
        config: The Algorithm's config dict.
        action_space (gym.spaces.Space): The action space used.

    Returns:
        Type[TFActionDistribution]: A TF distribution class.
    """
    if hasattr(policy, "dist_class") and policy.dist_class is not None:
        return policy.dist_class
    elif config["policy_model_config"].get("custom_action_dist"):
        action_dist_class, _ = ModelCatalog.get_action_dist(
            action_space, config["policy_model_config"], framework="torch"
        )
        return action_dist_class
    elif isinstance(action_space, Discrete):
        return TorchCategorical
    elif isinstance(action_space, Simplex):
        return TorchDirichlet
    else:
        assert isinstance(action_space, Box)
        if config["normalize_actions"]:
            return (
                TorchSquashedGaussian
                if not config["_use_beta_distribution"]
                else TorchBeta
            )
        else:
            return TorchDiagGaussian

def build_distral_model(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: AlgorithmConfigDict,
) -> ModelV2:
    """Constructs the necessary ModelV2 for the Policy and returns it.

    Args:
        policy: The TFPolicy that will use the models.
        obs_space (gym.spaces.Space): The observation space.
        action_space (gym.spaces.Space): The action space.
        config: The SAC trainer's config dict.

    Returns:
        ModelV2: The ModelV2 to be used by the Policy. Note: An additional
            target model will be created in this function and assigned to
            `policy.target_model`.
    """
    # Force-ignore any additionally provided hidden layer sizes.
    # Everything should be configured using SAC's `q_model_config` and
    # `policy_model_config` config settings.
    distilled_model = config.pop("custom_model_config",{"distilled_model":None})
    policy_model_config = copy.deepcopy(MODEL_DEFAULTS)
    policy_model_config.update(config["policy_model_config"])
    q_model_config = copy.deepcopy(MODEL_DEFAULTS)
    q_model_config.update(config["q_model_config"])

    default_model_cls = DistralTorchModel




    # policy.distilled_model = distilled_model
    # policy.distilled_model = DistilledTorchModel(obs_space=obs_space, action_space=action_space,
    #                            model_config=distilled_model, num_outputs=None, name="central_policy")
    # policy.distilled_model.set_action_model(distilled_model)
    if config['custom_model']=='central_model':
        model = ModelCatalog.get_model_v2(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=None,
            model_config=config["model"],
            framework=config["framework"],
            default_model=DistralCentralTorchModel,
            name="distral_central_model",
            policy_model_config=policy_model_config,
            q_model_config=q_model_config,
            twin_q=config["twin_q"],
            # initial_alpha=config["initial_alpha"],
            # target_entropy=config["target_entropy"],
        )

        assert isinstance(model, DistralCentralTorchModel)

        # Create an exact copy of the model and store it in `policy.target_model`.
        # This will be used for tau-synched Q-target models that run behind the
        # actual Q-networks and are used for target q-value calculations in the
        # loss terms.

        # Create an exact copy of the model and store it in `policy.target_model`.
        # This will be used for tau-synched Q-target models that run behind the
        # actual Q-networks and are used for target q-value calculations in the
        # loss terms.
        policy.target_model = ModelCatalog.get_model_v2(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=None,
            model_config=config["model"],
            framework=config["framework"],
            default_model=default_model_cls,
            name="target_distral_task_model",
            policy_model_config=policy_model_config,
            q_model_config=q_model_config,
            twin_q=config["twin_q"],
            # initial_alpha=config["initial_alpha"],
            # target_entropy=config["target_entropy"],
        )

        model.distilled_model = DistilledTorchModel(obs_space=obs_space, action_space=action_space,
                                                     model_config=distilled_model, num_outputs=None,
                                                     name="central_policy")
        policy.target_model.distilled_model = DistilledTorchModel(obs_space=obs_space, action_space=action_space,
                                                     model_config=distilled_model, num_outputs=None,
                                                     name="central_policy")
    else:
        model = ModelCatalog.get_model_v2(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=None,
            model_config=config["model"],
            framework=config["framework"],
            default_model=default_model_cls,
            name="distral_task_model",
            policy_model_config=policy_model_config,
            q_model_config=q_model_config,
            twin_q=config["twin_q"],
            # initial_alpha=config["initial_alpha"],
            # target_entropy=config["target_entropy"],
        )

        assert isinstance(model, default_model_cls)

        # Create an exact copy of the model and store it in `policy.target_model`.
        # This will be used for tau-synched Q-target models that run behind the
        # actual Q-networks and are used for target q-value calculations in the
        # loss terms.

        # Create an exact copy of the model and store it in `policy.target_model`.
        # This will be used for tau-synched Q-target models that run behind the
        # actual Q-networks and are used for target q-value calculations in the
        # loss terms.
        policy.target_model = ModelCatalog.get_model_v2(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=None,
            model_config=config["model"],
            framework=config["framework"],
            default_model=default_model_cls,
            name="target_distral_task_model",
            policy_model_config=policy_model_config,
            q_model_config=q_model_config,
            twin_q=config["twin_q"],
            # initial_alpha=config["initial_alpha"],
            # target_entropy=config["target_entropy"],
        )


        model.distilled_model = DistilledTorchModel(obs_space=obs_space, action_space=action_space,
                                                     model_config=distilled_model, num_outputs=None,
                                                     name="central_policy")
        policy.target_model.distilled_model = DistilledTorchModel(obs_space=obs_space, action_space=action_space,
                                                     model_config=distilled_model, num_outputs=None,
                                                     name="central_policy")

    assert isinstance(policy.target_model, default_model_cls)

    return model


# def DistralTorchModel


def build_distral_model_and_action_dist(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: AlgorithmConfigDict,
) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
    """Constructs the necessary ModelV2 and action dist class for the Policy.

    Args:
        policy: The TFPolicy that will use the models.
        obs_space (gym.spaces.Space): The observation space.
        action_space (gym.spaces.Space): The action space.
        config: The SAC trainer's config dict.

    Returns:
        ModelV2: The ModelV2 to be used by the Policy. Note: An additional
            target model will be created in this function and assigned to
            `policy.target_model`.
    """
    model = build_distral_model(policy, obs_space, action_space, config)

    # model = build_sac_model(policy, obs_space, action_space, config)

    action_dist_class = _get_dist_class(policy, config, action_space)
    return model, action_dist_class


def action_distribution_fn(
    policy: Policy,
    model: ModelV2,
    input_dict: ModelInputDict,
    *,
    state_batches: Optional[List[TensorType]] = None,
    seq_lens: Optional[TensorType] = None,
    prev_action_batch: Optional[TensorType] = None,
    prev_reward_batch=None,
    explore: Optional[bool] = None,
    timestep: Optional[int] = None,
    is_training: Optional[bool] = None
) -> Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
    """The action distribution function to be used the algorithm.

    An action distribution function is used to customize the choice of action
    distribution class and the resulting action distribution inputs (to
    parameterize the distribution object).
    After parameterizing the distribution, a `sample()` call
    will be made on it to generate actions.

    Args:
        policy: The Policy being queried for actions and calling this
            function.
        model (TorchModelV2): The SAC specific model to use to generate the
            distribution inputs (see sac_tf|torch_model.py). Must support the
            `get_action_model_outputs` method.
        input_dict: The input-dict to be used for the model
            call.
        state_batches (Optional[List[TensorType]]): The list of internal state
            tensor batches.
        seq_lens (Optional[TensorType]): The tensor of sequence lengths used
            in RNNs.
        prev_action_batch (Optional[TensorType]): Optional batch of prev
            actions used by the model.
        prev_reward_batch (Optional[TensorType]): Optional batch of prev
            rewards used by the model.
        explore (Optional[bool]): Whether to activate exploration or not. If
            None, use value of `config.explore`.
        timestep (Optional[int]): An optional timestep.
        is_training (Optional[bool]): An optional is-training flag.

    Returns:
        Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
            The dist inputs, dist class, and a list of internal state outputs
            (in the RNN case).
    """
    # Get base-model output (w/o the SAC specific parts of the network).
    model_out, _ = model(input_dict, [], None)
    # Use the base output to get the policy outputs from the SAC model's
    # policy components.
    action_dist_inputs, _ = model.get_action_model_outputs(model_out)
    # Get a distribution class to be used with the just calculated dist-inputs.
    action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)

    return action_dist_inputs, action_dist_class, []


def postprocess_trajectory(
        policy: Policy,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
        episode: Optional[Episode] = None,
) -> SampleBatch:
    """Postprocesses a trajectory and returns the processed trajectory.

    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.

    Args:
        policy: The Policy used to generate the trajectory
            (`sample_batch`)
        sample_batch: The SampleBatch to postprocess.
        other_agent_batches (Optional[Dict[AgentID, SampleBatch]]): Optional
            dict of AgentIDs mapping to other agents' trajectory data (from the
            same episode). NOTE: The other agents use the same policy.
        episode (Optional[Episode]): Optional multi-agent episode
            object in which the agents operated.

    Returns:
        SampleBatch: The postprocessed, modified SampleBatch (or a new one).
    """
    return postprocess_nstep_and_prio(policy, sample_batch)


def distral_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for the Soft Actor Critic.

    Args:
        policy: The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[TorchDistributionWrapper]: The action distr. class.
        train_batch: The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # Look up the target model (tower) using the model tower.
    target_model = policy.target_models[model]
    # distilled_model = model.distilled_model
    # Should be True only for debugging purposes (e.g. test cases)!
    deterministic = policy.config["_deterministic_loss"]

    model_out_t, _ = model(
        SampleBatch(obs=train_batch[SampleBatch.CUR_OBS], _is_training=True), [], None
    )

    model_out_tp1, _ = model(
        SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True), [], None
    )

    target_model_out_tp1, _ = target_model(
        SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True), [], None
    )

    alpha = torch.exp(model.log_alpha)


    ##Distilled Model

    #### EDIT distilledmodel integrated into the distral model
    # distilled_model_out_tp1, _ = distilled_model(
    #     SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=False), [], None
    # )
    # distilled_model_out_t, _ = distilled_model(
    #     SampleBatch(obs=train_batch[SampleBatch.CUR_OBS], _is_training=False), [], None
    # )

    ## End edit


    # Discrete case.
    if model.discrete:
        # Get all action probs directly from pi and form their logp.
        action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
        log_pis_t = F.log_softmax(action_dist_inputs_t, dim=-1)
        policy_t = torch.exp(log_pis_t)
        action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
        log_pis_tp1 = F.log_softmax(action_dist_inputs_tp1, -1)
        policy_tp1 = torch.exp(log_pis_tp1)

        ## distilled policy
        distilled_action_dist_inputs_tp1, _ = model.get_distilled_model_outputs(model_out_tp1)
        distilled_log_pis_tp1 = F.log_softmax(distilled_action_dist_inputs_tp1, dim=-1)
        # distilled_policy_tp1 = torch.exp(distilled_log_pis_tp1)

        distilled_action_dist_inputs_t, _ = model.get_distilled_model_outputs(model_out_t)
        distilled_log_pis_t = F.log_softmax(distilled_action_dist_inputs_t, dim=-1)
        # distilled_policy_t = torch.exp(distilled_log_pis_t)
        # distill_loss = categorical_kld( policy_t.detach() ,distilled_policy_t )
        #Edit MSE loss  7.0
        distill_loss = F.mse_loss(distilled_log_pis_t, log_pis_tp1.detach())
        ## end edit


        # Q-values.
        q_t, _ = model.get_q_values(model_out_t)
        # Target Q-values.
        q_tp1, _ = target_model.get_q_values(target_model_out_tp1)
        if policy.config["twin_q"]:
            twin_q_t, _ = model.get_twin_q_values(model_out_t)
            twin_q_tp1, _ = target_model.get_twin_q_values(target_model_out_tp1)
            q_tp1 = torch.min(q_tp1, twin_q_tp1)

        # original sac implementation
        # q_tp1 -= alpha * log_pis_tp1

        if policy.config["distill_coeff"]:
            alpha_from_paper = policy.distill_coeff / (policy.distill_coeff + alpha.detach())
            beta_from_paper = 1.0 / (policy.distill_coeff + alpha.detach())

            # Old implementation
            # q_tp1 += (alpha_from_paper*distilled_log_pis_tp1 - log_pis_tp1)/beta_from_paper

            # Version 5 update note:
            q_tp1 += (alpha_from_paper * distilled_log_pis_tp1 - log_pis_tp1) / beta_from_paper

        # distral edit
        # alpha_from_paper = policy.distral_alpha/(policy.distral_alpha + alpha)
        # beta_from_paper = 1.0/(policy.distral_alpha + alpha)
        else:
            q_tp1 += (policy.distral_alpha * distilled_log_pis_tp1 -log_pis_tp1)/policy.distral_beta

        # end edit
        # Actually selected Q-values (from the actions batch).
        one_hot = F.one_hot(
            train_batch[SampleBatch.ACTIONS].long(), num_classes=q_t.size()[-1]
        )
        q_t_selected = torch.sum(q_t * one_hot, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)
        # Discrete case: "Best" means weighted by the policy (prob) outputs.
        q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best
    # Continuous actions case.
    else:
        # Sample single actions from distribution.
        action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
        action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
        # try:
        action_dist_t = action_dist_class(action_dist_inputs_t, model)
        action_dist_t_detach = action_dist_class(action_dist_inputs_t.detach(), model)
        # except Exception:
        #     print('action distribution inputs: ', action_dist_inputs_t)
        #     print('action model input inputs: ', model_out_t)
        #
        #     print("NaN exception in original action model")

        policy_t = (
            action_dist_t.sample()
            if not deterministic
            else action_dist_t.deterministic_sample()
        )
        log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)

        #Edit
        # train_batch[SampleBatch.ACTION_DIST_INPUTS]

        # agent_mean, agent_log_std = torch.chunk(action_dist_inputs_t, 2, dim=1)



        #End Edit

        action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
        action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, model)
        policy_tp1 = (
            action_dist_tp1.sample()
            if not deterministic
            else action_dist_tp1.deterministic_sample()
        )
        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)

        ## distilled policy
        distilled_action_dist_inputs_tp1, _ = model.get_distilled_model_outputs(model_out_tp1)
        # distral_mean, distral_log_std = torch.chunk(distilled_action_dist_inputs_tp1, 2, dim=1)
        # try:
        distilled_action_dist_tp1 = action_dist_class(distilled_action_dist_inputs_tp1, model)
        # except Exception:
        #     print('action distribution inputs: ', distilled_action_dist_inputs_tp1)
        #     print('distilled model input inputs: ', model_out_tp1)
        #
        #     print("NaN exception in distilled model")
        distilled_action_dist_inputs_t, _ = model.get_distilled_model_outputs(model_out_t)
        # distral_mean_t, distral_log_std_t = torch.chunk(distilled_action_dist_inputs_t, 2, dim=1)

        # distilled_action_dist_t = action_dist_class(distilled_action_dist_inputs_t, model)

        ## version notes 2* added to convert std to var
        # distill_loss = gaussian_kld(agent_mean.detach(), 2*agent_log_std.detach(), distral_mean_t, 2*distral_log_std_t)

        ## version 4.3 update note I. detach() for task policy,
        # distill_loss = gaussian_kld(agent_mean.detach(), 2 * agent_log_std.detach() ,distral_mean_t, 2 * distral_log_std_t)

        ## version 5.0 update note: using distill_log_pi (a_t)

        # distill_loss = torch.unsqueeze(distilled_action_dist_tp1.logp(policy_tp1.detach()), -1)

        ## version 6.0 beta distribution is used for kl
        # distill_loss = action_dist_t_detatch.kl( distilled_action_dist_t)

        # distill_loss = distill_loss.sum()/distill_loss.shape[0]
        # Edit V7.0
        distill_loss = F.mse_loss(distilled_action_dist_inputs_t, action_dist_inputs_t.detach())
        ## End edit




        ## distilled policy
        distilled_policy_tp1 = (
            distilled_action_dist_tp1.sample()
            if not deterministic
            else distilled_action_dist_tp1.deterministic_sample()
        )
        distilled_log_pis_tp1 = torch.unsqueeze(distilled_action_dist_tp1.logp(distilled_policy_tp1), -1)
        ## End Edit


        # Q-values for the actually selected actions.
        q_t, _ = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        if policy.config["twin_q"]:
            twin_q_t, _ = model.get_twin_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS]
            )

        # Q-values for current policy in given current state.
        q_t_det_policy, _ = model.get_q_values(model_out_t, policy_t)
        if policy.config["twin_q"]:
            twin_q_t_det_policy, _ = model.get_twin_q_values(model_out_t, policy_t)
            q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)

        # Target q network evaluation.
        q_tp1, _ = target_model.get_q_values(target_model_out_tp1, policy_tp1)
        if policy.config["twin_q"]:
            twin_q_tp1, _ = target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1
            )
            # Take min over both twin-NNs.
            q_tp1 = torch.min(q_tp1, twin_q_tp1)

        q_t_selected = torch.squeeze(q_t, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)


        #distral edit
        if policy.config["distill_coeff"]:

            alpha_from_paper = policy.distill_coeff / (policy.distill_coeff + alpha)
            beta_from_paper = 1.0 / (policy.distill_coeff + alpha)

            #Old implementation
            # q_tp1 += (alpha_from_paper*distilled_log_pis_tp1 - log_pis_tp1)/beta_from_paper

            #Version 5 update note:
            q_tp1 += (alpha_from_paper* distilled_log_pis_tp1 - log_pis_tp1) / beta_from_paper
            # distill_loss = alpha_from_paper * distill_loss_logp
        else:

            q_tp1 += (policy.distral_alpha*distilled_log_pis_tp1 - log_pis_tp1)/policy.distral_beta
       # Distral version 4.2 update note: variable alpha
        # q_tp1 += (policy.distral_alpha*distilled_log_pis_tp1 - log_pis_tp1)/policy.distral_beta
        #     distill_loss = policy.distral_alpha * distill_loss_logp

        # end edit

        q_tp1_best = torch.squeeze(input=q_tp1, dim=-1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best

    # compute RHS of bellman equation
    q_t_selected_target = (
        train_batch[SampleBatch.REWARDS]
        + (policy.config["gamma"] ** policy.config["n_step"]) * q_tp1_best_masked
    ).detach()

    # Compute the TD-error (potentially clipped).
    base_td_error = torch.abs(q_t_selected - q_t_selected_target)
    if policy.config["twin_q"]:
        twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error

    critic_loss = [torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(base_td_error))]
    if policy.config["twin_q"]:
        critic_loss.append(
            torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(twin_td_error))
        )

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
    if model.discrete:
        ## Uncomment for old distral this new version uses fixed alpha
        if policy.config["distill_coeff"]:
            weighted_log_alpha_loss = policy_t.detach() * (
                -model.log_alpha * (log_pis_t + model.target_entropy).detach()
            )
        # # Sum up weighted terms and mean over all batch items.

            alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))
        # Actor loss.
            actor_loss = torch.mean(
                torch.sum(
                    torch.mul(
                        # NOTE: No stop_grad around policy output here
                        # (compare with q_t_det_policy for continuous case).
                        policy_t,
                        alpha.detach() * log_pis_t - q_t.detach(),
                    ),
                    dim=-1,
                )
            )
        else:
            actor_loss = torch.mean(
                torch.sum(
                    torch.mul(
                        # NOTE: No stop_grad around policy output here
                        # (compare with q_t_det_policy for continuous case).
                        policy_t,
                        policy.alpha * log_pis_t - q_t.detach(),
                    ),
                    dim=-1,
                )
            )
            alpha_loss =torch.tensor(0.0, device=actor_loss.device)


    else:
        ## Uncomment for old distral this new version uses fixed alpha
        if policy.config["distill_coeff"]:

            alpha_loss = -torch.mean(
                model.log_alpha * (log_pis_t + model.target_entropy).detach()
            )
            actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_det_policy)
        else:

        # Note: Do not detach q_t_det_policy here b/c is depends partly
        # on the policy vars (policy sample pushed through Q-net).
        # However, we must make sure `actor_loss` is not used to update
        # the Q-net(s)' variables.

            actor_loss = torch.mean(policy.alpha * log_pis_t - q_t_det_policy)
            alpha_loss = torch.tensor(0.0, device=actor_loss.device)
        #

    #distral edit
    if policy.config['distill_coeff']:

        # batch_size = distill_loss.shape[0]

        # Version 5 note torch.clamp added
        distill_loss =torch.mean( distill_loss)*policy.distill_coeff

    else:
        # batch_size = distill_loss.shape[0]

        #Version 5 note torch.clamp added
        distill_loss =  torch.mean(distill_loss)*policy.distral_alpha
    # distill_loss= torch.mean(distill_loss * policy.distral_alpha)
    #end edit

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["q_t"] = q_t
    model.tower_stats["policy_t"] = policy_t
    model.tower_stats["log_pis_t"] = log_pis_t
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss
    model.tower_stats["alpha_loss"] = alpha_loss
    model.tower_stats["distill_loss"] = distill_loss
    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["td_error"] = td_error

    # Return all loss terms corresponding to our optimizers.
    if policy.config["distill_coeff"]:

        return tuple([actor_loss] +[distill_loss]+ critic_loss + [alpha_loss])
    else:
        return tuple([actor_loss ]+ [distill_loss] + critic_loss)


def stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Stats function for SAC. Returns a dict with important loss stats.

    Args:
        policy: The Policy to generate stats for.
        train_batch: The SampleBatch (already) used for training.

    Returns:
        Dict[str, TensorType]: The stats dict.
    """
    q_t = torch.stack(policy.get_tower_stats("q_t"))

    return {
        "actor_loss": torch.mean(torch.stack(policy.get_tower_stats("actor_loss"))),
        "critic_loss": torch.mean(
            torch.stack(tree.flatten(policy.get_tower_stats("critic_loss")))
        ),
        "distill_loss":torch.mean(torch.stack(policy.get_tower_stats("distill_loss"))),
        "alpha_loss": torch.mean(torch.stack(policy.get_tower_stats("alpha_loss"))),
        "alpha_value": torch.exp(policy.model.log_alpha),
        "log_alpha_value": policy.model.log_alpha,
        "target_entropy": policy.model.target_entropy,
        "policy_t": torch.mean(torch.stack(policy.get_tower_stats("policy_t"))),
        "mean_q": torch.mean(q_t),
        "max_q": torch.max(q_t),
        "min_q": torch.min(q_t),
    }


def optimizer_fn(policy: Policy, config: AlgorithmConfigDict) -> Tuple[LocalOptimizer]:
    """Creates all necessary optimizers for SAC learning.

    The 3 or 4 (twin_q=True) optimizers returned here correspond to the
    number of loss terms returned by the loss function.

    Args:
        policy: The policy object to be trained.
        config: The Algorithm's config dict.

    Returns:
        Tuple[LocalOptimizer]: The local optimizers to use for policy training.
    """
    if config['custom_model']=="central_model":
        return tuple()

    else:

        policy.actor_optim = torch.optim.Adam(
            params=policy.model.policy_variables(),#+policy.model.distilled_variables(),
            lr=config["optimization"]["actor_learning_rate"],
            eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
        )
        policy.distill_optim = torch.optim.Adam(
        params=policy.model.distilled_variables(),
        lr=config["optimization"]["actor_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
        )
        critic_split = len(policy.model.q_variables())
        if config["twin_q"]:
            critic_split //= 2

        policy.critic_optims = [
            torch.optim.Adam(
                params=policy.model.q_variables()[:critic_split],
                lr=config["optimization"]["critic_learning_rate"],
                eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
            )
        ]
        if config["twin_q"]:
            policy.critic_optims.append(
                torch.optim.Adam(
                    params=policy.model.q_variables()[critic_split:],
                    lr=config["optimization"]["critic_learning_rate"],
                    eps=1e-7,  # to match tf.keras.optimizers.Adam's eps default
                )
            )
        policy.alpha_optim = torch.optim.Adam(
            params=[policy.model.log_alpha],
            lr=config["optimization"]["entropy_learning_rate"],
            eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
        )
        if policy.config["distill_coeff"]:
            return tuple([policy.actor_optim] + [policy.distill_optim] + policy.critic_optims + [policy.alpha_optim])

        else:
            return tuple([policy.actor_optim] +  [policy.distill_optim ]+ policy.critic_optims )#+ [policy.alpha_optim])



# TODO: Unify with DDPG's ComputeTDErrorMixin when SAC policy subclasses PolicyV2
class ComputeTDErrorMixin:
    """Mixin class calculating TD-error (part of critic loss) per batch item.

    - Adds `policy.compute_td_error()` method for TD-error calculation from a
      batch of observations/actions/rewards/etc..
    """

    def __init__(self):
        def compute_td_error(
            obs_t, act_t, rew_t, obs_tp1, done_mask, importance_weights
        ):
            input_dict = self._lazy_tensor_dict(
                {
                    SampleBatch.CUR_OBS: obs_t,
                    SampleBatch.ACTIONS: act_t,
                    SampleBatch.REWARDS: rew_t,
                    SampleBatch.NEXT_OBS: obs_tp1,
                    SampleBatch.DONES: done_mask,
                    PRIO_WEIGHTS: importance_weights,
                }
            )
            # Do forward pass on loss to update td errors attribute
            # (one TD-error value per item in batch to update PR weights).
            distral_loss(self, self.model, None, input_dict)

            # `self.model.td_error` is set within actor_critic_loss call.
            # Return its updated value here.
            return self.model.tower_stats["td_error"]

        # Assign the method to policy (self) for later usage.
        self.compute_td_error = compute_td_error


# TODO: Unify with DDPG's TargetNetworkMixin when SAC policy subclasses PolicyV2
class TargetNetworkMixin:
    """Mixin class adding a method for (soft) target net(s) synchronizations.

    - Adds the `update_target` method to the policy.
      Calling `update_target` updates all target Q-networks' weights from their
      respective "main" Q-metworks, based on tau (smooth, partial updating).
    """

    def __init__(self):
        # Hard initial update from Q-net(s) to target Q-net(s).
        self.update_target(tau=1.0)

    def update_target(self, tau=None):
        # Update_target_fn will be called periodically to copy Q network to
        # target Q network, using (soft) tau-synching.
        tau = tau or self.config.get("tau")
        model_state_dict = self.model.state_dict()
        # Support partial (soft) synching.
        # If tau == 1.0: Full sync from Q-model to target Q-model.
        target_state_dict = next(iter(self.target_models.values())).state_dict()
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


def setup_late_mixins(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: AlgorithmConfigDict,
) -> None:
    """Call mixin classes' constructors after Policy initialization.

    - Moves the target model(s) to the GPU, if necessary.
    - Adds the `compute_td_error` method to the given policy.
    Calling `compute_td_error` with batch data will re-calculate the loss
    on that batch AND return the per-batch-item TD-error for prioritized
    replay buffer record weight updating (in case a prioritized replay buffer
    is used).
    - Also adds the `update_target` method to the given policy.
    Calling `update_target` updates all target Q-networks' weights from their
    respective "main" Q-metworks, based on tau (smooth, partial updating).

    Args:
        policy: The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config: The Policy's config.
    """

    if config['distill_coeff'] is not None:
        policy.distill_coeff = config['distill_coeff']
    elif config['distral_beta'] is not None and config['distral_alpha'] is not None:
        policy.distral_alpha = config['distral_alpha']
        policy.distral_beta = config['distral_beta']
        policy.alpha = (1 - policy.distral_alpha) / policy.distral_beta


    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy)


# Build a child class of `TorchPolicy`, given the custom functions defined
# above.


DistralTorchPolicy = build_policy_class(
    name="DistralTorchPolicy",
    framework="torch",
    loss_fn=distral_loss,
    get_default_config=lambda: DEFAULT_CONFIG,
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    extra_grad_process_fn=apply_grad_clipping,
    # extra_grad_process_fn= minimize_and_clip,
    optimizer_fn=optimizer_fn,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins,
    make_model_and_action_dist=build_distral_model_and_action_dist,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin],
    action_distribution_fn=action_distribution_fn,
)

