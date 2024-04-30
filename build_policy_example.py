import numpy as np
from ray.rllib.evaluation import compute_advantages
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_tf, try_import_torch
# import ray
tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()



# def actor_critic_loss(policy, model, dist_class, train_batch):
#     logits, _ = model.from_batch(train_batch)
#     values = model.value_function()
#     action_dist = dist_class(logits)
#     log_probs = action_dist.logp(train_batch[SampleBatch.ACTIONS])
#     policy.entropy = action_dist.entropy().mean()
#     return overall_err
import ray
import argparse
import os

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.algorithms.ppo.ppo import DEFAULT_CONFIG, PPO
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from env_funcs import make_multi_agent_divide_and_conquer
from envs.point_mass_2d import TaskSettablePointMass2D
from utils import FC_MLP
MultiAgentPM2= make_multi_agent_divide_and_conquer(lambda config: TaskSettablePointMass2D(config))

torch, nn = try_import_torch()


dummy_env = TaskSettablePointMass2D({'dummy':True})
model_config = { "fcnet_hiddens": [64, 64 ],
                 "fcnet_activation": "tanh",
                 }
dist_class, logit_dim = ModelCatalog.get_action_dist(
    dummy_env.action_space, model_config, framework='torch'
)

central_policy, central_value_net = FC_MLP(
    obs_space=dummy_env.observation_space,
    action_space=dummy_env.action_space,
    num_outputs=logit_dim,
    model_config=model_config,
)


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

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        """Calculates a custom loss on top of the given policy_loss(es).

        """
        obs = loss_inputs['obs'].float()

        central_policy_output_logits= self._central_policy(obs.reshape(obs.shape[0], -1))

        action_dist = TorchDiagGaussian(central_policy_output_logits, self.model_config)
        self.policy_loss = policy_loss
        self.imitation_loss = torch.mean(-action_dist.logp(loss_inputs["actions"]))

        self.imitation_loss_metric = self.imitation_loss.item()
        self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

        return [loss_ +  self.imitation_loss for loss_ in policy_loss]
        # return policy_loss

    def metrics(self):
        return {
            "policy_loss": self.policy_loss_metric,
            "imitation_loss": self.imitation_loss_metric,
        }

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
        # Non-shared initial layer.
        self._output = None


    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens) :
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        return self._central_policy(self._last_flat_in), state

    @override(ModelV2)
    def value_function(self):
        # assert self._features is not None, "must call forward() first"
        return self._central_value_net(self._last_flat_in).squeeze(1)

@DeveloperAPI
class Postprocessing:
    """Constant definitions for postprocessing."""

    ADVANTAGES = "advantages"
    VALUE_TARGETS = "value_targets"

    CNT_LOGITS = "central_logits"

def model_value_predictions(policy, input_dict, state_batches, model):
    return {SampleBatch.VF_PREDS: model.value_function().cpu().numpy()}


def loss_and_entropy_stats(policy, train_batch):
    return {
        "policy_entropy": policy.entropy.item(),
        "policy_loss": policy.pi_err.item(),
        "vf_loss": policy.value_err.item(),
    }

def policy_gradient_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits)
    log_probs = action_dist.logp(train_batch[SampleBatch.ACTIONS])
    return -train_batch[SampleBatch.REWARDS].dot(log_probs)


def add_advantages(policy,
                   sample_batch,
                   other_agent_batches=None,
                   episode=None):
    # sample_batch[Postprocessing.CNT_LOGITS] = sample_batch
    print(sample_batch)
    return sample_batch

class ValueNetworkMixin(object):
    def _value(self, obs):
        with self.lock:
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            _, _, vf, _ = self.model({"obs": obs}, [])
            return vf.detach().cpu().numpy().squeeze()
# <class 'ray.rllib.policy.torch_policy_template.MyTorchPolicy'>
MyTorchPolicy = build_torch_policy(
    name="MyTorchPolicy",
    loss_fn=policy_gradient_loss)

PPOplusPolicy = build_policy_class(
    name="PPOplus",
    framework="torch",
    get_default_config=lambda: DEFAULT_CONFIG,
    loss_fn=PPOTorchPolicy.loss,
    # stats_fn=loss_and_entropy_stats,
    postprocess_fn=add_advantages,
    # extra_action_out_fn=model_value_predictions,
    # extra_grad_process_fn=apply_grad_clipping,
    # optimizer_fn=torch_optimizer,
    # mixins=[ValueNetworkMixin]
)





if __name__ == "__main__":
    # args = parser.parse_args()


    ray.init( )

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

                    "custom_model_config": {"central_policy": central_policy, "central_value_net": central_value_net }, }
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

    config = {
        "env": MultiAgentPM2,
        # "callbacks": get_central_logits,
        "env_config": {"num_agents": 3,
                       "agent_config": [{ "target_mean": np.array([2.5, 1])},
                                        { "target_mean": np.array([-2.5, 1])},
                                        {"target_mean": np.array([[2.5, 1], [-2.5, 1]]),
                                         "target_var": np.array([[4e-3, 3.75e-3], [4e-3, 3.75e-3]]),
                                         "target_priors": np.array([1, 1])}]},
        "disable_env_checking": True,
        "num_workers":0,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus":0,
        "evaluation_interval": 1,
        # Run 10 episodes each time evaluation runs (OR "auto" if parallel to training).
        "evaluation_duration": 10,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": list(policies.keys())[1:],
        },
        "seed": 0,
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

    }



    alg = PPO(config=config)
    # stop = {
    #     "training_iteration": 400,
    #
    # # }
    # config = {"fcnet_hiddens": [64, 64 ],
    #                      "fcnet_activation": "tanh",
    #                      "custom_model_config": {"central_policy": central_policy, "central_value_net": central_value_net}}
    #
    # model = DnCLocalTorchModel(dummy_env.observation_space, dummy_env.action_space,logit_dim,config, 'tst')

    # print(model.view_requirements)

    #
    results = tune.Tuner(
        "PPO", param_space=config, run_config=air.RunConfig(stop=stop, verbose=1,name="DnC_PM2V2")
    ).fit()
    #
    for _ in range(10):
        res = alg.train()
        print(res)
    ray.shutdown()