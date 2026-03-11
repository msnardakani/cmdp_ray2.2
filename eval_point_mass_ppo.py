
import os
import pickle

import numpy as np
from distral.distral_ppo_torch_model import DistralCentralTorchModel, DistralTorchModel
from ray.rllib.models import ModelCatalog


from utils.policy_evaluation2 import ExperimentEvaluation
from experiment_params import ExperimentSetup

ModelCatalog.register_custom_model(
    "central",
    DistralCentralTorchModel,
)

ModelCatalog.register_custom_model(
    "local",
    DistralTorchModel,
)



# learner_0 = Policy.from_checkpoint('./results/GridWorld/V10.1.5/Set0/ctx_hid/baseline/default/PPO_2x64/PPO_MADnCEnv_b16be_00000_0_grad_clip=100,seed=0_2024-01-19_11-06-24/checkpoint_000000/policies/learner_0')
sets = dict(
    # Set0=[0, 0],
    # Set1=[0, ],  # left and right tasks from more complicated ones
    # Set2=[0, 1, 2, ],
    # Set3=[7, 8, ],
    # Set4=[7, 5, 2, ],
    # Set5=[4, 7, 5]
)

parameters = []
room_rew_coeff=1
env = 'PointMass2D'
log_dir = f'./results/{env}/V11.7.3'
# exp_sets= ['Set00', ]
exp_sets= None

env_creator_, ctx_spec, max_steps, env_config_pars = ExperimentSetup(env= env)
env_creator = lambda config, ctx_mode: env_creator_(config, ctx_mode, room_rew_coeff)

func = lambda obs: np.where(obs[2]==1, 1, 0)

def config_trans(config):
    config_out = dict()
    config_out['curriculum'] = 'default'
    config_out['target_mean'] = np.array(config['target_mean'])
    config_out['target_var'] = np.array(config['target_var'])
    config_out['prior'] = None
    return config_out

EV = ExperimentEvaluation(experiment_dir=log_dir )
try:
    EV.load_from_file('result.pkl')
    # for setup in ['Set70', 'Set80']:
    #     for group in [6, 7]:
    #         EV.reset_experiment_group(setup= setup,group = (True, group))
    #
    #         EV.reset_experiment_group(setup= setup,group = (False, group))
    # EV.save_to_file('baseline_result_updated.pkl')

except Exception as e:
    print("file not found starting over")
EV.analyse(env_creator=env_creator, func=func, duration = 100, n_env=50, config_trans = config_trans, record = True)
EV.save_to_file('result.pkl')
EV.plot_exps = 'best'
EV.plot_label = 'group'
EV.plot_step = 50

EV.report(exp_sets=exp_sets)

# for setup, idx in sets.items():
#     config_lst = [env_config_pars[i] for i in idx]
#     try:
#         with open(os.path.join(EV.log_dir,'report', setup, 'extr_eval.pkl'), 'rb') as file:
#         # Call load method to deserialze
#             extr_result = pickle.load(file)
#         extr_result = EV.update_extr_results(extr_result=extr_result, env_creator=env_creator,
#                                              func=func,
#                                              duration=100, n_env=20
#                                              )


#     except Exception:
#         print('could not load file')
#         extr_result = EV.get_extr_results(setup, env_creator, config_lst,
#                                           func=func, duration=100, n_env=20,)

#     EV.report_extr_eval(extr_results=extr_result)
