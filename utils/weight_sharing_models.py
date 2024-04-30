import logging
import numpy as np
import gym
# from ray.rllib.algorithms.callbacks import DefaultCallbacks
# from ray.rllib.evaluation.metrics import summarize_episodes, collect_episodes
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
# import ray
# import gc
torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

def FC_MLP( obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
   ):
    """Generic fully connected network."""


    hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
        model_config.get("post_fcnet_hiddens", [])
    )
    activation = model_config.get("fcnet_activation")
    if not model_config.get("fcnet_hiddens", []):
        activation = model_config.get("post_fcnet_activation")
    no_final_linear = model_config.get("no_final_linear")


    layers = []
    prev_layer_size = int(np.product(obs_space.shape))
    # self._logits = None

    # Create layers 0 to second-last.
    for size in hiddens[:-1]:
        layers.append(
            SlimFC(
                in_size=prev_layer_size,
                out_size=size,
                initializer=normc_initializer(1.0),
                activation_fn=activation,
            )
        )
        prev_layer_size = size

    # The last layer is adjusted to be of size num_outputs, but it's a
    # layer with activation.

    layers.append( SlimFC(
                in_size=prev_layer_size,
                out_size=num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None,
            ))
    # Layer to add the log std vars to the state-dependent means.
    # if self.free_log_std and self._logits:
    #     self._append_free_log_std = AppendBiasLayer(num_outputs)

    policy = nn.Sequential(*layers)

    # self._value_branch_separate = None
    # if not self.vf_share_layers:
        # Build a parallel set of hidden layers for the value net.
    prev_vf_layer_size = int(np.product(obs_space.shape))
    vf_layers = []
    for size in hiddens:
        vf_layers.append(
            SlimFC(
                in_size=prev_vf_layer_size,
                out_size=size,
                activation_fn=activation,
                initializer=normc_initializer(1.0),
            )
        )
        prev_vf_layer_size = size


    vf_layers.append(SlimFC(
        in_size=prev_layer_size,
        out_size=1,
        initializer=normc_initializer(0.01),
        activation_fn=None,
        ))
    value_net = nn.Sequential(*vf_layers)


    return policy, value_net


class DnCLocalTorchModel(FullyConnectedNetwork):
    """Example of weight sharing between two different TorchModelV2s.

    The shared (single) layer is simply defined outside of the two Models,
    then used by both Models in their forward pass.
    """

    def __init__(
            self, observation_space, action_space, num_outputs, model_config, name
    ):
        custom_model_config = model_config.pop('custom_model_config')
        super().__init__(
            observation_space, action_space, num_outputs, model_config, name
        )
        self._central_policy = custom_model_config['central_policy']
        self._central_value_net = custom_model_config['central_value_net']
        self._value_branch = None
        self._value_branch_separate = None
        # Non-shared initial layer.
        self._output = None

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self._central_value_net(self._last_flat_in).squeeze(1)

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        """Calculates a custom loss on top of the given policy_loss(es).

        """
        obs = loss_inputs['obs'].float()

        central_policy_output_logits = self._central_policy(obs.reshape(obs.shape[0], -1))

        action_dist = TorchDiagGaussian(central_policy_output_logits, self.model_config)
        self.policy_loss = policy_loss
        self.imitation_loss = torch.mean(-action_dist.logp(loss_inputs["actions"]))

        self.imitation_loss_metric = self.imitation_loss.item()
        self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

        return [loss_ + self.imitation_loss for loss_ in policy_loss]
        # return policy_loss

    def metrics(self):
        return {
            "policy_loss": self.policy_loss_metric,
            "imitation_loss": self.imitation_loss_metric,
        }

class DnCCentralTorchModel(TorchModelV2, nn.Module):
    def __init__(
            self, observation_space, action_space, num_outputs, model_config, name
    ):
        custom_model_config = model_config.pop('custom_model_config')
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self._central_policy = custom_model_config['central_policy']
        self._central_value_net = custom_model_config['central_value_net']
        self._action_dist = custom_model_config['dist_class']
        # Non-shared initial layer.
        self._output = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        return self._central_policy(self._last_flat_in), state

    @override(ModelV2)
    def value_function(self):
        # assert self._features is not None, "must call forward() first"
        return self._central_value_net(self._last_flat_in).squeeze(1)







