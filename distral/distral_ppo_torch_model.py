# import gym
# from gym.spaces import Box, Discrete
# import numpy as np
# import tree  # pip install dm_tree
from typing import Dict, List, Optional

# from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
# from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TensorStructType
# from gym.spaces import Tuple
torch, nn = try_import_torch()


class DistralTorchModel(TorchModelV2, nn.Module):

    def forward_default(self, input_dict, state, seq_lens):
        model_out, _ = self._model(input_dict, state, seq_lens)
        # obs = input_dict["obs_flat"].float()
        # self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._last_flat_in = self._model._last_flat_in
        # self._last_flat_aug_in = self._model._last_flat_in
        return model_out, []

    def forward_tuple_obs(self, input_dict, state, seq_lens):
        model_out, _ = self._model.forward({'obs_flat': input_dict['obs'][0]}, state, seq_lens)
        self._last_flat_in = self._model._last_flat_in
        return model_out, []

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, distilled_model, model, ctx_aug = None):
        # if ctx_aug is none the model assumes full augmentation of the context variable
        # shared_net = model_config.get('distilled_model', None)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        if ctx_aug is None:
            self._model = FullyConnectedNetwork(obs_space, action_space, num_outputs, model, name)
            self.forward_func = self.forward_default
        else:
            self._model = FullyConnectedNetwork(ctx_aug, action_space, num_outputs, model, name)
            self.forward_func = self.forward_tuple_obs

        self._distilled_model = distilled_model

    def distill_out(
            self) -> (TensorType, List[TensorType]):
        assert self.last_ctx_aug_in is not None, "must call forward() first"

        return self._distilled_model(self.last_ctx_aug_in)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):

        obs = input_dict["obs_flat"].float()
        # if any(obs.isnan().flatten()) or not any(obs.isfinite().flatten()):
        #
        #     print('NaN in state', obs)

        self.last_ctx_aug_in = obs.reshape(obs.shape[0], -1)
        # model_out, _ = self._model.forward({'obs_flat':input_dict['obs'][0]}, state, seq_lens)
        # obs = input_dict["obs_flat"].float()
        # self._last_flat_in = obs.reshape(obs.shape[0], -1)
        # # self._last_flat_in = self._model._last_flat_in
        # return model_out, []
        out = self.forward_func(input_dict, state, seq_lens)
        # if any(out[0].isnan().flatten()) or not any(out[0].isfinite().flatten()):
        #     for k, v in self.variables(as_dict=True).items():
        #         if any(v.isnan().flatten()) or not any(v.isfinite().flatten()):
        #             print('nan/inf found in model: ', k, v)

        return out

    @override(TorchModelV2)
    def value_function(self):
        return self._model.value_function()


class DistralCentralTorchModel(DistralTorchModel):
    @override(TorchModelV2)
    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self.last_ctx_aug_in = obs.reshape(obs.shape[0], -1)

        # self._last_flat_in = obs.reshape(obs.shape[0], -1)
        #
        model_out, _ = self.forward_func(input_dict, state, seq_lens)
        self._model._last_flat_in = self._last_flat_in
        return self._distilled_model(self.last_ctx_aug_in), []
