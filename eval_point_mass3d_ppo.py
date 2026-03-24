
import os
import pickle

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
# from ray.rllib.policy.policy import Policy
from distral.distral_ppo_torch_model import DistralCentralTorchModel, DistralTorchModel
from ray.rllib.models import ModelCatalog


from utils.policy_evaluation2 import ExperimentEvaluation

ModelCatalog.register_custom_model(
    "central",
    DistralCentralTorchModel,
)
from experiment_params import ExperimentSetup
ModelCatalog.register_custom_model(
    "local",
    DistralTorchModel,
)

sets = dict(
    # Set0=[0, 0],
    # Set1=[0, ],  # left and right tasks from more complicated ones
    # Set2=[0, 1, 2, ],
    # Set3=[7, 8, ],
    # Set4=[7, 5, 2, ],
    # Set5=[4, 7, 5]
)

parameters = []
env = 'PointMass3D'
log_dir = f'./results/{env}/V11.10.0'


env_creator, ctx_spec, max_steps, env_config_pars = ExperimentSetup(env= env)
func = lambda obs: np.where(obs[2]==1, 1, 0)

def config_trans(config):
    config_out = dict()
    config_out['curriculum'] = 'default'
    config_out['target_mean'] = np.array(config['target_mean'])
    config_out['target_var'] = np.array(config['target_var'])
    config_out['prior'] = None
    return config_out

EV = ExperimentEvaluation(experiment_dir=log_dir )

EV.analyse(env_creator=env_creator, func=func, duration = 100, n_env=20, config_trans = config_trans, record=True)
EV.save_to_file('result.pkl')
EV.plot_exps = 'best'
EV.plot_label = 'group'
EV.plot_step = 50

EV.report()

