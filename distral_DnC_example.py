from typing import Dict, List, Type, Union

import argparse
import os
import random
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.dqn.dqn import DQN
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

from env_funcs import make_multi_agent_divide_and_conquer
from envs.point_mass_2d import TaskSettablePointMass2D
from utils import FC_MLP
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import TensorType
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.utils.annotations import (
    DeveloperAPI,
    OverrideToImplementCustomLogic,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
    is_overridden,
    override,
 )
torch, nn = try_import_torch()

OPPONENT_OBS = 'opp_obs'
OPPONENT_ACT = 'opp_actions'
def create_central_local_learner(base_learner:Algorithm, config):
    base_policy = base_learner.get_default_policy_class(base_learner,config=config)
    class CentralLocalPolicy(base_policy):

        @override(base_policy)
        def postprocess_trajectory(
            self, sample_batch, other_agent_batches=None, episode=None
        ):
            # assert other_agent_batches is not None
            if other_agent_batches is not None and sample_batch[SampleBatch.AGENT_INDEX][0]==0:
                # if SampleBatch.ACTIONS not in sample_batch.keys():
                #     x=1
                #     print('error')
                # act_dim = (sample_batch[SampleBatch.ACTIONS]).shape[1]
                # obs_dim = (sample_batch[SampleBatch.CUR_OBS]).shape[1]
                opp_acts = None#np.zeros((0,act_dim))
                opp_obs = None#np.zeros((0,obs_dim))

                for k, v in other_agent_batches.items():
                    obs_tmp = v[1][SampleBatch.CUR_OBS]
                    if SampleBatch.ACTIONS not in v[1].keys():
                        # print(v[1][SampleBatch.AGENT_INDEX])
                        # print(v[1][SampleBatch.OBS])

                        continue
                    acts_tmp = v[1][SampleBatch.ACTIONS]
                    if opp_acts is None:
                        opp_obs = obs_tmp
                        opp_acts = acts_tmp
                    else:
                        opp_acts = np.concatenate((opp_acts,acts_tmp))
                        opp_obs = np.concatenate((opp_obs, obs_tmp))


                # also record the opponent obs and actions in the trajectory
                sample_batch[OPPONENT_OBS] = opp_obs
                sample_batch[OPPONENT_ACT] = opp_acts
            return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)

    class DnC(Algorithm):
        @classmethod
        def get_default_policy_class(cls, config):
            return CentralLocalPolicy

    return DnC, CentralLocalPolicy, 'DnC_'+base_learner.__name__
        #
        # """Simple example of setting up a multi-agent policy mapping.
        # Control the number of agents and policies via --num-agents and --num-policies.
        # This works with hundreds of agents and policies, but note that initializing
        # many TF policies will take some time.
        # Also, TF evals might slow down with large numbers of policies. To debug TF
        # execution, set the TF_TIMELINE_DIR environment variable.
        # """


# print(central_value_net)

class DnCLocalTorchModel(FullyConnectedNetwork):
    """Example of weight sharing between two different TorchModelV2s.

    The shared (single) layer is simply defined outside of the two Models,
    then used by both Models in their forward pass.
    """

    def __init__(
        self, observation_space, action_space, num_outputs, model_config,  name
    ):

        custom_model_config = model_config.pop('custom_model_config')
        super().__init__(
           observation_space, action_space, num_outputs, model_config, name
        )
        self._central_policy = custom_model_config['central_policy']
        self._central_value_net = custom_model_config['central_value_net']
        self._value_branch= None
        self._value_branch_separate = None
        # Non-shared initial layer.
        self._output = None



    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self._central_value_net(self._last_flat_in).squeeze(1)


class DnCCentralTorchModel(TorchModelV2, nn.Module):
    def __init__(
        self, observation_space, action_space, num_outputs, model_config,  name
    ):

        custom_model_config = model_config.pop('custom_model_config')
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self._central_policy = custom_model_config['central_policy']
        self._central_value_net = custom_model_config['central_value_net']
        self._dist_class = custom_model_config['dist_class']
        # Non-shared initial layer.
        self._output = None


    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens) :
        obs = flatten(input_dict[SampleBatch.CUR_OBS],framework="torch").float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        return self._central_policy(self._last_flat_in), state

    @override(ModelV2)
    def value_function(self):
        # assert self._features is not None, "must call forward() first"
        return self._central_value_net(self._last_flat_in).squeeze(1)

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        """Calculates a custom loss on top of the given policy_loss(es).

        """

        obs = loss_inputs[OPPONENT_OBS].float()
        # print('central',obs.shape)
        central_policy_out = self._central_policy(obs.reshape(obs.shape[0], -1))

        action_dist = self._dist_class(central_policy_out, self.model_config)
        # self.policy_loss = policy_loss
        imitation_loss = -torch.mean(action_dist.logp(loss_inputs[OPPONENT_ACT]))

        self.imitation_loss_metric = imitation_loss.item()
        return [imitation_loss for _ in policy_loss]


    def metrics(self):
        return {
            # "policy_loss": self.policy_loss_metric,
            "imitation_loss": self.imitation_loss_metric,
        }


if __name__ == "__main__":
    # args = parser.parse_args()


    ray.init( )
    MultiAgentPM2 = make_multi_agent_divide_and_conquer(lambda config: TaskSettablePointMass2D(config))

    dummy_env = TaskSettablePointMass2D({'dummy': True})
    model_config = {"fcnet_hiddens": [64, 64],
                    "fcnet_activation": "tanh",
                    }
    dist_class, logit_dim = ModelCatalog.get_action_dist(
        dummy_env.action_space, model_config, framework='torch'
    )
    model_config = {"fcnet_hiddens": [64, 64],
                    "fcnet_activation": "tanh",
                    }
    central_policy, central_value_net = FC_MLP(
        obs_space=dummy_env.observation_space,
        action_space=dummy_env.action_space,
        num_outputs=logit_dim,
        model_config=model_config,
    )

    # Register the models to use.

    mod1 = DnCLocalTorchModel
    # ModelCatalog.register_custom_model("DnCPol", mod1)
    mod2 = DnCCentralTorchModel

    ModelCatalog.register_custom_model("DnCPolLocal", mod1)
    ModelCatalog.register_custom_model("DnCPolCentral", mod2)
    num_policies =3
    # Each policy can have a different configuration (including custom model).
    def gen_policy(i):
        if i ==0:
            name = "central_policy"
            config = {
                "model": {
                    "custom_model": "DnCPolCentral",

                    "custom_model_config": {"central_policy": central_policy, "central_value_net": central_value_net ,"dist_class":dist_class}, }
            }
        else:
            name = "local_policy_"+str(i-1)
            config = {
            "model": {
                "custom_model": "DnCPolLocal",

                         "fcnet_hiddens": [64, 64 ],
                         "fcnet_activation": "tanh",
                         "custom_model_config": {"central_policy": central_policy, "central_value_net": central_value_net},}
            }


        return name, PolicySpec(config=config)

    # Setup PPO with an ensemble of `num_policies` different policies.
    policies = {}
    # "policy_{}".format(i): gen_policy()
    for i in range(num_policies):
        k,v = gen_policy(i)
        policies[k]=v
    policy_ids = list(policies.keys())

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id ==0:
            return "central_policy"
        else:
            return "local_policy_"+ str(agent_id-1)
    # config= PPO.get_default_config()
    # print(config)
    config = {
        "env": MultiAgentPM2,
        "env_config": {"num_agents": 3,
                       "agent_config": [{ "target_mean": np.array([2.5, 1])},
                                        { "target_mean": np.array([-2.5, 1])},
                                        {"target_mean": np.array([[2.5, 1], [-2.5, 1]]),
                                         "target_var": np.array([[4e-3, 3.75e-3], [4e-3, 3.75e-3]]),
                                         "target_priors": np.array([1, 1])}]},
        "disable_env_checking": True,
        "num_workers":8,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus":1,
        "evaluation_interval": 1,
        # Run 10 episodes each time evaluation runs (OR "auto" if parallel to training).
        "evaluation_duration": 10,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": list(policies.keys()),
        },
        "seed": 0,# tune.grid_search([0,1,2,3]),
        "evaluation_num_workers": 2,
        "framework": 'torch',
        "evaluation_config": {
            "env_config": {"num_agents": 3,
                       "agent_config": [{ "target_mean": np.array([[2.5, 1],[-2.5, 1]]),
                                          "target_var": np.array([[4e-3, 3.75e-3], [4e-3, 3.75e-3]]),
                                          "target_priors": np.array([1,1])},
                                        {"target_mean": np.array([[2.5, 1], [-2.5, 1]]),
                                         "target_var": np.array([[4e-3, 3.75e-3], [4e-3, 3.75e-3]]),
                                         "target_priors": np.array([1, 1])},
                                        { "target_mean": np.array([[2.5, 1],[-2.5, 1]]),
                                          "target_var": np.array([[4e-3, 3.75e-3],[4e-3, 3.75e-3]]),
                                          "target_priors": np.array([1,1])}]},
    },
        "train_batch_size": 2048,
        'lr':0.0003,
        "horizon": 100,

    }
    DnC_PPO, CentralLocalPolicy, name = create_central_local_learner(PPO,config)

    alg = DnC_PPO(config=config)
    stop = {
        "training_iteration": 400,

    }
    # results = tune.Tuner(
    #     DnC_PPO, param_space=config, run_config=air.RunConfig(stop=stop, verbose=1,name=name+"_PM2V3")
    # ).fit()
    # #
    for _ in range(10):
        res = alg.train()
        print(res)
    ray.shutdown()