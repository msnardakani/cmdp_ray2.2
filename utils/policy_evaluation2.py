import itertools
import pickle
import time
from collections import deque
import traceback
from ray.rllib.policy.policy import Policy
import os
import torch
from scipy.stats import ttest_ind
from tqdm import tqdm
from envs.gridworld_contextual import TaskSettableGridworld
from gymnasium.wrappers import TimeLimit, RecordVideo
from utils.vect_wrapper import RecordEpisodeStatistics
from envs.contextual_env import CtxDictWrapper, ctx_visibility, exp_group
import numpy as np
import pandas as pd
from ray.rllib.policy.policy import Policy
from distral.distral_ppo_torch_model import DistralCentralTorchModel, DistralTorchModel
from ray.rllib.models import ModelCatalog
import json
import gymnasium as gym
from matplotlib import pyplot as plt
import pandas as pd
from flatten_dict import flatten
from ray.rllib.utils.spaces import space_utils

gym.logger.set_level(gym.logger.DISABLED)
GreenCell = '\\cellcolor{green!25}'
RedCell = '\\cellcolor{red!25}'
DistilledPolicy = 'distilled_policy'


ModelCatalog.register_custom_model(
    "central",
    DistralCentralTorchModel,
)

ModelCatalog.register_custom_model(
    "local",
    DistralTorchModel,
)



def read_results(file_name, keys_=['info', 'custom_metrics','time_total_s', 'sampler_results', 'policy_reward_mean',]):
    keys=['info', 'custom_metrics','time_total_s', 'sampler_results', 'policy_reward_mean',]+keys_
    try:
        training_data = []
        evaluation_data=[]
        with open(file_name, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())  # Parse each line
                    filtered_obj= {k:obj.get(k, np.nan) for k in keys}
                    training_data.append(flatten(filtered_obj, reducer='slash'))
                    
                    if  'evaluation' in obj:
                        eval_obj={k:obj['evaluation'].get(k, np.nan) for k in keys}
                        evaluation_data.append(flatten(eval_obj, reducer='slash'))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()} - {e}")
        # for obj in data:
        #     print(obj)
    except FileNotFoundError:
        print("Error: 'data_lines.json' not found.")
    return pd.DataFrame.from_records(training_data), pd.DataFrame.from_records(evaluation_data)


class evaluation():
    def __init__(self, returns, lengths, custom_metric= None):
        self.returns = returns
        self.lengths= lengths
        self.custom_metric = custom_metric
        # self.best_idx = best_idx

single_learners = [0, 2, 6]
distill_learners = [ 0, 2, 6, 8, 10, 14, 9, 11, 15]
naive_learners= [1, 3, 7]
sp_learners = [2, 6, 3, 7, 10, 11, 14, 15]
gsp_learners = [6,7,14, 15]
sorted_idx = {0: 'B1',
              2: 'B1S',
              6:'B1G',

              1: 'B0',
              3: 'B0S',
              7: 'B0G',


              8: 'D0',
              9: 'D1',
              10: 'D0S',
              11: 'D1S',
              14: 'D0G',
              15: 'D1G'
              }
curriculum = lambda idx : 'Def' if idx not in sp_learners else ('GSP' if idx in gsp_learners else 'SP')
name_codes = {(True, 0): 'PPO Central',  # ( baseline, default, Central)
              (True, 1): 'PPO',  # (baseline, default, MA)
              (True, 3): 'PPO',  # ( baseline, self_paced, MA)
              (True, 2): 'PPO Central',  # (baseline, self_paced, Central)
              (True, 7): 'PPO',  # ( baseline, GSP, MA)
              (True, 6): 'PPO Central',  # (baseline, GSP, Central)

              (True, 10): 'DistralPPO, Local Ctx Aug',  # (distral, self_paced, task_aug)
              (True, 11): 'DistralPPO, Distill Ctx Aug',  # (distral, self_paced, distill_aug)
              (True, 9): 'DistralPPO, Distill Ctx Aug',  # (distral, default, distill_aug)
              (True, 8): 'DistralPPO, Local Ctx Aug',  # (distral, default, task_aug)

              (True, 14): 'DistralPPO, Local Ctx Aug',  # (distral, GSP, task_aug)
              (True, 15): 'DistralPPO, Distill Ctx Aug',  # (distral, GSP, distill_aug)
              # ctx_hid
              (False, 0): 'PPO Central',  # (baseline, default, Central)
              (False, 1): 'PPO',  # (baseline, default, MA)
              (False, 3): 'PPO',  # (baseline, self_paced, MA)
              (False, 2): 'PPO Central',  # (baseline, self_paced, Central)

              (False, 7): 'PPO',  # (baseline, GSP, MA)
              (False, 6): 'PPO Central',  # (baseline, GSP, Central)

              (False, 10): 'DistralPPO',  # (distral, self_paced, task_aug)
              (False, 11): 'DistralPPO',  # (distral, self_paced, distill_aug)
              (False, 9): 'DistralPPO',  # (distral, default, distill_aug)
              (False, 8): 'DistralPPO',  # (distral, default, task_aug)

              (False, 14): 'DistralPPO',  # (distral, GSP, task_aug)
              (False, 15): 'DistralPPO',  # (distral, GSP, distill_aug)

              }

class trial():
    dir= ''
    policies = list()
    tasks = list()
    reward = dict()


class ExperimentEvaluation():

    def __init__(self, experiment_dir, baselines= None ):
        self.log_dir = experiment_dir
        self.experiment_breakdown = dict()
        self.results = dict()
        self.extrnal_evaluation = dict()
        self.plot_label= 'id'
        self.plot_exps = 'All'

    # def extr_analyse(self, setup, config_lst, env_creator, func = None, duration= 100,n_env= 10,  ):

    def analyse(self, env_creator, func = None, duration= 100,n_env= 10, config_trans = lambda c: c, setup_list = None, record = True):
        self.results, self.experiment_breakdown = self.get_experiments()
        # t0 = time.time()
        # ctr = 1

        if setup_list is None:
            setup_list = list(self.experiment_breakdown.keys())
            # pbar = tqdm(setup_list, position = 0)

            for setup in setup_list:
                self.read_trainig_results(setup=setup)
                # self.get_results(setup=setup, env_creator=env_creator, n_env= n_env, func=func, duration = duration, config_trans=config_trans, record = record)
   
            for setup in setup_list:
                # pbar.set_description("Processing %s" % setup)
                self.best_perf_by_group(setup=setup, ctx_vis=True)
                self.best_perf_by_group(setup=setup, ctx_vis=False)


        return

    def save_to_file(self, name = 'results_summary.pkl'):
        with open(os.path.join(self.log_dir , name), 'wb') as file:
            # A new file will be created
            pickle.dump((self.experiment_breakdown,self.results), file)
        return

    def load_from_file(self, name = 'results_summary.pkl'):
        with open(os.path.join(self.log_dir,name), 'rb') as file:
            # Call load method to deserialze
            self.experiment_breakdown, self.results = pickle.load(file)
        for setup, experiment_set in self.experiment_breakdown.items():
            experiment_set['experiments'] =[]
            groups = [ experiment_set['experiments'].extend(experiment_set[(vis , idx)]['experiments'])  for (vis , idx) in itertools.product([True, False], list(range(7))) if (vis , idx) in experiment_set]
            experiment_set['experiments'] = list(set((experiment_set['experiments'])))

        return

    def reset_experiment_group(self, group , setup):
        if setup not in self.experiment_breakdown:
            print('no such experiment set!')
            return
        if group not in self.experiment_breakdown[setup]:
            print('no such experiment group!')
            return

        experiments = self.experiment_breakdown[setup][group]['experiments']
        for exp in experiments:
            self.results.pop(exp)
        self.experiment_breakdown[setup].pop(group)

        return

    def exp_id2group(self, exp_id): #baseline, default, subgroup)
        experiment_set, visibility, learner_group, curriculum, name, parameters = exp_id
        baseline ='baseline'  in learner_group
        if baseline:
            subgroup = 'Central' not in name
        else:
            subgroup = 'task_aug' not in learner_group

        cl =6 if 'GaussianSP'  in curriculum else 2 if 'selfpaced' in curriculum else 0
        idx = 8*(1- baseline)  +cl + subgroup
        return 'vis' in visibility, idx


    def get_experiments(self):
        #         name_map= {}
        #         param_map ={}
        # baselines ={}
        results = self.results
        files_list = list()
        experiment_breakdown = self.experiment_breakdown
        for (dirpath, dirnames, filenames) in os.walk(self.log_dir):
            files_list += [os.path.join(dirpath, file) for file in filenames if ('result.json' in file

                                                                                 and not os.path.exists(
                        os.path.join(dirpath, 'error.txt')))]

        for f in files_list:
            splts = f.split('/')
            setup = splts[-2]
            name = splts[-3]
            curriculum = splts[-4]
            learner_group = splts[-5]
            experiment_set = splts[-7]
            visibility = splts[-6]
            parameters = '_'.join(setup.split('_')[5:-2])
            # seed = int(re.search('seed=(.\d+?),', parameters).group(1))
            #             print(f)
            seed = int(((parameters.split('seed=')[1]).split(',')[0]).split('_')[0])
            parameters = parameters.replace(f',seed={seed}', '').replace(f'seed={seed}', '')
            # parameters = ','.join(parameters.split(',')[:-1])
            # parameters = ','.join([splt[0], ] + splt[1].split(',')[1:])
                # print(parameters)
            # parameters = parameters.split('min_sample')[0]


            if parameters:
                if parameters[-1] == ',':
                    parameters = parameters[:-1]

                parameters = parameters.replace('distral_', '')
                parameters = parameters.replace(',,', ',')
            experiment_id = ( experiment_set, visibility,  learner_group, curriculum, name, parameters)
            # print(experiment_id)
            # group_id = self.exp_id2group(experiment_id)
            trial_dir = os.path.dirname(f)

            # if 'baseline' in trial_dir and 'distilled_policy' in policies:
            #     policies.remove('distilled_policy')

            if experiment_set not in experiment_breakdown:
                with open(os.path.join(trial_dir, 'params.json'), 'r') as config_file:
                    env_config = json.load(config_file)['env_config']['agent_config']

                experiment_breakdown[experiment_set]={'env_config':env_config,
                                                      'tasks': [f'task_{i}' for i in range(len(env_config))],
                                                      'policies': set(),
                                                      'experiments' : []}
            # else:
            #     experiment_breakdown[experiment_set]['experiments'].append(experiment_id)


            # if experiment_set in results:
            #
            #     if group_id in results[experiment_set]:
            #         if experiment_id in results[experiment_set][group_id]['experiments']:
            #
            if experiment_id in results:
                # if parameters in results[name]['config']:
                if trial_dir not in results[experiment_id]['trial_dir']:
                    results[experiment_id]['trial_dir'].append(trial_dir)
                    results[experiment_id]['seeds'].append(seed)

            else:
                checkpoints = next(os.walk(trial_dir))[1]
                checkpoints.sort(reverse=True)
                policies = next(os.walk(os.path.join(trial_dir, checkpoints[0], 'policies')))[1]
                policies.sort()

                results[experiment_id] = {'trial_dir': [trial_dir, ], 'seeds': [seed, ],
                                          'checkpoints': checkpoints,
                                          'policies': policies,
                                          }

        for exp_id, result in results.items():
            setup = exp_id[0]
            result['runs'] = len(result['seeds'])
            group_id = self.exp_id2group(exp_id=exp_id)
            result['group_id'] = group_id
            if exp_id not in experiment_breakdown[setup]['experiments']:
                experiment_breakdown[setup]['experiments'].append(exp_id)
            experiment_breakdown[setup]['policies'] = experiment_breakdown[setup]['policies'].union(result['policies'])
            if group_id in experiment_breakdown[setup]:
                if exp_id not in experiment_breakdown[setup][group_id]['experiments']:
                    result['ID'] = f'{sorted_idx[group_id[1]]}, {len(experiment_breakdown[setup][group_id]["experiments"])}'
                    # experiment_breakdown[setup]['policies'].union(result['policies'])
                    experiment_breakdown[setup][group_id]['experiments'].append(exp_id)
            else:
                experiment_breakdown[setup][group_id] = {'experiments': [exp_id, ], }
                result['ID'] = f'{sorted_idx[group_id[1]] }, 0'
                # experiment_breakdown[setup][group_id] = {'ex'}

        return results, experiment_breakdown

    
    # def get_results(self, setup, env_creator, func= None,config_trans= lambda c: c, override = False):
    #     env_config = self.experiment_breakdown[setup]['env_config']
    #     # t0 = time.time()
    #     # ctr = 0
    #     # if env_config[0]['curriculum']!='default':
    #     #     return
    #     experiments = self.experiment_breakdown[setup]['experiments']
    #     pbar = tqdm(experiments)
    #     for exp_id in pbar:
    #         result = self.results[exp_id]

    #         pbar.set_description( f'{setup}>> Exp {result["ID"]}, {"Visible Ctx" if result["group_id"][0] else "Hidden Ctx"  }')
    #         # if exp_id[0] != setup:
    #         #     continue
    #         # N = duration // 2
    #         if 'vis' in exp_id[1]:
    #             if 'task_aug' in exp_id[2]:
    #                 ctx_mode = 1
    #             else:
    #                 ctx_mode = 2
    #         else:
    #             ctx_mode = 0
    #         # # t1 = time.time()

    #         # # pc =  (40 * ctr) // len(experiments)

    #         # # print(f'Evaluating {ID_str}, {"Visible Ctx" if result["group_id"][0] else "Hidden Ctx"  } \n{"x"*pc + "-"*(40 - pc) }{ctr}/{len(experiments)}:({time.strftime("%H:%M:%S", time.gmtime(t1- t0))})')
    #         tasks = [f'task_{i}' for i in range(len(env_config))]
    #         policies = result['policies']

    #         # # print(exp_id)


    #         for trial_dir, seed in zip(result['trial_dir'], result['seeds']):
    #             pbar.set_description( f'{setup}>> Exp {result["ID"]}, {"Visible Ctx" if result["group_id"][0] else "Hidden Ctx"  }>> seed {seed}')

    #             if result.get(seed, False) is not False and not override:
    #                 # print(f'{setup}, {"Visible Ctx" if result["group_id"][0] else "Hidden Ctx"  }>>  trials skipped: id: {ID_str}, seed: {seed}', end='\t')

    #                 continue
    #             seed_result = dict()

    #             for policy in policies:
    #                 for task, config in zip(tasks, env_config):
    #                     # task = 'task_'+str(i)
    #                     reward = np.zeros(0)
    #                     length = np.zeros(0)
    #                     custom_result = np.zeros(0)

    #                     for checkpoint in result['checkpoints']:
    #                         # if int(checkpoint.replace('checkpoint_', ''))< 2:
    #                         #     continue
    #                         log_dir = os.path.join(trial_dir, checkpoint, 'policies', policy)
    #                         pbar.set_description( f'{setup}>> Exp {result["ID"]}, {"Visible Ctx" if result["group_id"][0] else "Hidden Ctx"  }>> seed {seed}>> {checkpoint}')

    #                         name_prefix = f'{policy},{task}'
    #                         env_creator_record = lambda cfg: RecordVideo(
    #                             env_creator(cfg, ctx_mode),
    #                             video_folder=log_dir, name_prefix=name_prefix,episode_trigger=lambda x: True, disable_logger=True)

    #                         # env_creator_stat = lambda config: RecordEpisodeStatistics(
    #                         #     gym.vector.SyncVectorEnv([lambda: env_creator(config, ctx_mode) for i in range(n_env)]), deque_size=duration,  gamma =.99)
    #                         # # env_creator_ctx = lambda cfg: RecordVideo(
    #                         #     RecordEpisodeStatistics(env_creator(cfg, ctx_mode), deque_size=duration),
    #                         #     video_folder=log_dir, name_prefix=name_prefix,episode_trigger=lambda x: x%N ==0, disable_logger=True)

    #                         config_ = config_trans(config)

    #                         env = env_creator_record(config_)
    #                         try:

    #                             if record:
    #                                     learner = record_rollouts(log_dir, env, TupleObs= ctx_mode==2)
                                    

    #                             else:
    #                                 learner = Policy.from_checkpoint(log_dir)
    #                         except Exception as e:
    #                             print(f'failed to load checkpoint in: {log_dir}')
    #                             continue

    #                         env = env_creator_stat(config_)

    #                         task_results = evaluate_policy_vec(learner, env, duration = duration, func = func, TupleObs= ctx_mode==2)

    #                         reward = np.append(reward, task_results[0])
    #                         length = np.append(length, task_results[1])
    #                         custom_result = np.append(custom_result, task_results[2])
    #                     seed_result.update({(task, policy): evaluation(returns=reward, lengths=length, custom_metric= custom_result)})

    #             if 'baseline' not in exp_id[2]:
    #                 policy = DistilledPolicy
    #                 for task, config in zip(tasks, env_config):
    #                     # task = 'task_' + str(i)
    #                     reward = np.zeros(0)
    #                     length = np.zeros(0)
    #                     custom_result = np.zeros(0)

    #                     for checkpoint in result['checkpoints']:
    #                         # if int(checkpoint.replace('checkpoint_', ''))< 2:
    #                         #     continue
    #                         log_dir = os.path.join(trial_dir, checkpoint, 'policies', policies[0])
    #                         name_prefix = f'{policy},{task}'

    #                         env_creator_record = lambda cfg: RecordVideo(
    #                             env_creator(cfg, ctx_mode),
    #                             video_folder=log_dir, name_prefix=name_prefix, episode_trigger=lambda x: True,
    #                             disable_logger=True)

    #                         env_creator_stat = lambda config: RecordEpisodeStatistics(
    #                             gym.vector.SyncVectorEnv([lambda: env_creator(config, ctx_mode) for i in range(n_env)]),
    #                             deque_size=duration)
    #                         # env_creator_ctx = lambda cfg: RecordVideo(
    #                         #     RecordEpisodeStatistics(env_creator(cfg, ctx_mode), deque_size=duration),
    #                         #     video_folder=log_dir, name_prefix=name_prefix,episode_trigger=lambda x: x%N ==0, disable_logger=True)

    #                         config_ = config_trans(config)

    #                         env = env_creator_record(config_)
    #                         learner = record_rollouts(log_dir, env, distill=True, TupleObs=ctx_mode == 2)
    #                         env = env_creator_stat(config_)

    #                         task_results = evaluate_policy_vec(learner, env, duration=duration,distill=True, func=func,
    #                                                            TupleObs=ctx_mode == 2)

    #                         reward = np.append(reward, task_results[0])
    #                         length = np.append(length, task_results[1])
    #                         custom_result = np.append(custom_result, task_results[2])

    #                     seed_result.update({(task, policy): evaluation(returns=reward, lengths=length,
    #                                                                         custom_metric=custom_result )})

    #             fields = ['returns', 'lengths', 'custom_metric']
    #             if len(tasks) == len(policies):

    #                 # for attr in ['returns', 'lengths', 'custom']:
    #                 within = [np.sum([seed_result[(task, policy)].__getattribute__(attr)
    #                                       for task, policy in zip(tasks, policies)], axis=0) for attr in fields]
    #                 total = [np.sum([seed_result[(task, policy)].__getattribute__(attr)
    #                                       for task, policy in itertools.product(tasks, policies)], axis=0) for attr in fields]


    #             else:
    #                 policy = policies[0]
    #                 within = [np.sum([seed_result[(task, policy)].__getattribute__(attr)
    #                                       for task in tasks], axis=0)
    #                           for attr in fields]
    #                 total = [len(tasks)*np.sum([seed_result[(task, policy)].__getattribute__(attr)
    #                                  for task, policy in itertools.product(tasks, policies)], axis=0)
    #                          for attr in
    #                          fields]

    #                 for task in tasks:
    #                     seed_result[(task, DistilledPolicy)] = seed_result[(task, policy)]

    #             best_iter = np.argmax(within[0])
    #             seed_result['within'] = evaluation(returns=within[0], lengths=within[1], custom_metric=within[2])
    #             seed_result['total'] = evaluation(returns=total[0], lengths=total[1], custom_metric=total[2])

    #             seed_result['between'] = evaluation(returns=total[0]- within[0], lengths=total[1]- within[1], custom_metric=total[2]- within[2])
    #             seed_result['best_iter'] = best_iter
    #             result[seed] = seed_result

    #         seed_result = result[result['seeds'][0]]
    #         temp = dict()
    #         policy = DistilledPolicy

    #         for task in tasks:
    #             if (task, policy) in seed_result:
    #                 rewards = np.array(
    #                     [result[seed][(task, policy)].returns[result[seed]['best_iter']] for seed in result['seeds']])
    #                 lengths = np.array(
    #                     [result[seed][(task, policy)].lengths[result[seed]['best_iter']] for seed in result['seeds']])
    #                 custom_metric = np.array(
    #                     [result[seed][(task, policy)].custom_metric[result[seed]['best_iter']] for seed in
    #                      result['seeds']])

    #                 temp[(task, policy)] = evaluation(returns=rewards, lengths=lengths, custom_metric=custom_metric)
    #             else:
    #                 break

    #         for task, policy in itertools.product(tasks, policies):
    #             rewards = np.array([result[seed][(task, policy)].returns[result[seed]['best_iter']] for seed in result['seeds']])
    #             lengths = np.array([result[seed][(task, policy)].lengths[result[seed]['best_iter']] for seed in result['seeds']])
    #             custom_metric = np.array(
    #                 [result[seed][(task, policy)].custom_metric[result[seed]['best_iter']] for seed in result['seeds']])

    #             temp[(task, policy)] = evaluation(returns=rewards, lengths = lengths, custom_metric= custom_metric)

    #         for  policy in  policies:
    #             # if (tasks[0], DistilledPolicy)
    #             rewards = np.array([sum([result[seed][(task, policy)].returns[result[seed]['best_iter']] for task in tasks])
    #                                     for seed in result['seeds']])
    #             lengths = np.array(
    #                 [sum([result[seed][(task, policy)].lengths[result[seed]['best_iter']] for task in tasks])
    #                  for seed in result['seeds']])
    #             custom_metric = np.array(
    #                 [sum([result[seed][(task, policy)].returns[result[seed]['best_iter']] for task in tasks])
    #                  for seed in result['seeds']])
    #             temp[('total', policy)] = evaluation(returns=rewards, lengths = lengths, custom_metric= custom_metric)


    #         for policy in [DistilledPolicy,]:
    #             if (tasks[0], policy) in seed_result:
    #                 rewards = np.array(
    #                     [sum([result[seed][(task, policy)].returns[result[seed]['best_iter']] for task in tasks])
    #                      for seed in result['seeds']])
    #                 lengths = np.array(
    #                     [sum([result[seed][(task, policy)].lengths[result[seed]['best_iter']] for task in tasks])
    #                      for seed in result['seeds']])
    #                 custom_metric = np.array(
    #                     [sum([result[seed][(task, policy)].returns[result[seed]['best_iter']] for task in tasks])
    #                      for seed in result['seeds']])
    #                 temp[('total', policy)] = evaluation(returns=rewards, lengths=lengths, custom_metric=custom_metric)

    #         for k in ['within', 'between', 'total']:
    #             rewards = np.array(
    #                 [result[seed][k].returns[result[seed]['best_iter']] for seed in result['seeds']])
    #             lengths = np.array(
    #                 [result[seed][k].lengths[result[seed]['best_iter']] for seed in result['seeds']])
    #             custom_metric = np.array(
    #                 [result[seed][k].custom_metric[result[seed]['best_iter']] for seed in
    #                  result['seeds']])

    #             temp[k] = evaluation(returns= rewards, lengths=lengths, custom_metric=custom_metric)


    #         result['summary'] = temp

    #     return

    def best_perf_by_group(self, setup, ctx_vis = True, iteration= True):
        # single_learners = [0, 2, 6]
        # distill_learners = [0, 2, 6, 8, 10, 14, 9, 11, 15]
        # naive_learners = [1, 3, 7]
        exp_set = self.experiment_breakdown[setup]
        exp_set[('best_exp', ctx_vis)] = dict()
        groups = [(ctx_vis, i) for i in naive_learners + distill_learners if (ctx_vis, i) in exp_set]
        if len(groups) == 0:
            return None, None, None
        group_best_exps=[]
        group_best_vals=[]
        for group in groups:
            experiments = exp_set[group]['experiments']
            average_rewards = [np.mean(self.results[exp_id]['evaluation_best']['reward']) for exp_id in experiments]
            self.experiment_breakdown[setup][group]['best_exp'] = experiments[np.argmax(average_rewards)]
            group_best_exps.append(experiments[np.argmax(average_rewards)])
            group_best_vals.append(np.max(average_rewards))

        exp_set[('best_exp', ctx_vis)] =group_best_exps[np.argmax(group_best_vals)]

        return


    def experiments_txt(self, setup, ctx_vis = True):

        if setup in self.experiment_breakdown:
            exp_set = self.experiment_breakdown[setup]
        else:
            raise Exception(setup +'not in experiment')

        experiments_table = []
        ctx_str = 'Visible Context' if ctx_vis else 'Hidden Context'

        # experiments_table.append('{'+'|c'*5 +'|}\n')
        output = '\\begin{longtable}{'+'|c'*5 + '|}\n\caption{' + setup + ', ' + ctx_str + ' Experiment parameters}\n\label{tab:S' + setup.replace(
            'Set', '') + ('V' if ctx_vis else 'H') + 'Exps}\\\\\n'

        output +='\hline\n'+'&'.join(['ID', 'Name', 'Curriculum', 'Parameters', 'Runs','Training Time' ,]) + '\\\\\n\hline \hline\n\endfirsthead\n'

        output += '\multicolumn{5}{c}{\\bfseries \\tablename\ \\thetable{} -- continued from previous page}\\\\\hline\n'
        output += '\hline\n'+'&'.join(['ID', 'Name', 'Curriculum', 'Parameters', 'Runs','Training Time',]) + '\\\\\n\hline \hline\n\endhead\n'

        output += '\hline \multicolumn{5}{|r|}{{Continued on next page}} \\\\ \hline\n\endfoot\hline\hline\n\endlastfoot\n'

        experiments_table.append(output)
        # results_txt['header'] =  output
        empty = True
        for i in naive_learners + distill_learners:

            if (ctx_vis, i) in exp_set:
                empty = False
                exp_group = exp_set[(ctx_vis, i)]
                # experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )
                for exp_id in exp_group['experiments']:

                    experiment = self.results[exp_id]
                    # learner_group = learner_group

                    parameters = exp_id[5].replace('distill_coeff', '$C_{dstl}$').replace('grad_clip=100,', " ").replace('grad_clip=100', " ")
                    parameters = parameters.replace('.2000', '.2').replace('.0000', '.0').replace('.5000', '.5').replace('.8000', '.8').replace('.4000', '.4')

                    parameters = parameters.replace('loss_fn=11', '$f_{loss}=J_{10}$')
                    parameters = parameters.replace('loss_fn=12', '$f_{loss}=J_{11}$')
                    parameters = parameters.replace('loss_fn=13', '$f_{loss}=J_{12}$')
                    parameters = parameters.replace('loss_fn=14', '$f_{loss}=J_{13}$')
                    parameters = parameters.replace('loss_fn=15', '$f_{loss}=J_{14}$')
                    parameters = parameters.replace('loss_fn=16', '$f_{loss}=J_{15}$')

                    parameters = parameters.replace('loss_fn=21', '$f_{loss}=J_{20}$')
                    parameters = parameters.replace('loss_fn=22', '$f_{loss}=J_{21}$')
                    parameters = parameters.replace('loss_fn=23', '$f_{loss}=J_{22}$')
                    parameters = parameters.replace('loss_fn=24', '$f_{loss}=J_{24}$')
                    parameters = parameters.replace('loss_fn=25', '$f_{loss}=J_{23}$')


                    parameters = parameters.replace('loss_fn=31', '$f_{loss}=J_{14}$')
                    parameters = parameters.replace('loss_fn=32', '$f_{loss}=J_{15}$')

                    parameters = parameters.replace('loss_fn=-1', '$f_{loss}=J_{00}$')
                    parameters = parameters.replace('loss_fn=-3', '$f_{loss}=J_{00}$')
                    parameters = parameters.replace('loss_fn=-2', '$f_{loss}=J_{01}$')

                    parameters = parameters.replace('loss_fn=0', '$f_{loss}=J_{10}$')
                    parameters = parameters.replace('loss_fn=2', '$f_{loss}=J_{11}$')
                    parameters = parameters.replace('loss_fn=4', '$f_{loss}=J_{12}$')
                    parameters = parameters.replace('loss_fn=6', '$f_{loss}=J_{13}$')
                    parameters = parameters.replace('loss_fn=8', '$f_{loss}=J_{14}$')
                    parameters = parameters.replace('loss_fn=10', '$f_{loss}=J_{15}$')


                    parameters = parameters.replace('loss_fn=1', '$f_{loss}=J_{20}$')
                    parameters = parameters.replace('loss_fn=3', '$f_{loss}=J_{21}$')
                    parameters = parameters.replace('loss_fn=5', '$f_{loss}=J_{22}$')
                    parameters = parameters.replace('loss_fn=7', '$f_{loss}=J_{24}$')
                    parameters = parameters.replace('loss_fn=9', '$f_{loss}=J_{23}$')
                    # name = exp_id[4].repalce()
                    parameters = parameters.replace('min_sample_timesteps_per_iteration=', '$\\text{s/itr} =')
                    parameters = parameters.replace('=450', '=450$'
                                                    ).replace('=900', '=900$'
                                                    ).replace('=1350', '=1350$'
                                                    ).replace('=1800', '=1800$'
                                                    ).replace('=2250', '=2250$'
                                                    ).replace('=2700', '=2700$')

                    parameters = parameters.replace(',', ', ')

                    row = [experiment['ID'], name_codes[experiment['group_id']] ,exp_id[3] , parameters, str(experiment['runs']), str(np.round(np.mean(experiment['training_summary']['time_s'])/60, 1))]
                    experiments_table.append('&'.join(row) + '\\\\\n')

        experiments_table.append('\\end{longtable}\n')

        return experiments_table,  empty



    def total_results_txt(self, setup, ctx_vis = True):
        if setup in self.experiment_breakdown:
            exp_set = self.experiment_breakdown[setup]
        else:
            raise Exception(setup +'not in experiment')
        ctx_str = 'Visible Context' if ctx_vis else 'Hidden Context'
        output = '\\begin{longtable}{|c|' +  '|c'*6 + '|}\n\caption{' + setup + ', ' + ctx_str + ' Total rewards }\n\label{tab:S' + setup.replace(
            'Set', '') + ('V' if ctx_vis else 'H') + 'RewTot}\\\\\n'

        output +="\hline\n"+ '&'.join([' ', '\multicolumn{2}{|c|}{Rewards}',
                            '\multicolumn{2}{c|}{Success rate}',
                            '\multicolumn{2}{c|}{Length}']) + '\\\\\n'
        output += '&'.join(['Algorithm', ] + ['mean', 'se' ] * 3 ) + '\\\\\hline\n\endfirsthead\n'

        output += '\multicolumn{7}{c}{\\bfseries \\tablename\ \\thetable{} -- continued from previous page}\\\\\hline\n'
        output += "\hline\n" + '&'.join([' ', '\multicolumn{2}{|c|}{Rewards}',
                            '\multicolumn{2}{c|}{Success rate}',
                            '\multicolumn{2}{c|}{Length}']) + '\\\\\n'
        output += '&'.join(['Algorithm', ] + ['mean', 'se',]*3 ) + '\\\\\hline\n\endhead\n'
        output += '\hline \multicolumn{7}{|r|}{{Continued on next page}} \\\\ \hline\n\endfoot\hline\hline\n\endlastfoot\n'
        # result_tsk_pol.append(output)



        # output += '&'.join(['Algorithm', '\multicolumn{2}{|c|}{ Reward Within }',
        #                    '\multicolumn{2}{c|}{ Rewards Between}',
        #                    '\multicolumn{2}{c|}{Total Rewards}']) + '\\\\\n'
        # output += '&'.join(['ID', 'mean', 'se',
        #                     'mean', 'se',
        #                     'mean', 'se']) + '\\\\\n\hline\hline\n'
        total_results = list()

        total_results.append(output)
        for i in naive_learners + distill_learners:

            if (ctx_vis, i) in exp_set:
                # experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )

                for exp_id in exp_set[(ctx_vis, i)]['experiments']:
                    experiment = self.results[exp_id]
                    n = experiment['runs']
                    row = [experiment['ID'], ]
                    # for k in ['within', 'between', 'total']:
                    returns = experiment['evaluation_best']['reward']
                    row += [str(round(np.mean(returns), 2)), str(round(np.std(returns)/n**.5, 3))]
                    disc_returns = experiment['evaluation_best'][('success_rate','task_0')]
                    row += [str(round(np.mean(disc_returns), 2)), str(round(np.std(disc_returns)/n**.5, 3))]
                    lengths = experiment['evaluation_best']['length']
                    row += [str(round(np.mean(lengths), 2)), str(round(np.std(lengths)/n**.5, 3))]
                    
                    total_results.append('&'.join(row) + '\\\\\n')


        total_results.append('\hline\n\\end{longtable}')
        # total_results[-1] = total_results[-1][:-3]
        return total_results

    


    def statistical_comp(self, setup, ctx_vis= True):
        BlockPerRow = 3
        ctx_str= 'visible context ' if ctx_vis else 'hidden context '

        # for setup,  in self.experiment_breakdown.items():
        exp_set = self.experiment_breakdown[setup]
        # best_performers, groups, best_by_group = self.best_perf_by_key(setup=setup, ctx_vis=ctx_vis)
        groups = [(ctx_vis, i) for i in (naive_learners+ distill_learners) if (ctx_vis, i) in exp_set]
        if len(groups) == 0:
            raise Exception('no experiment')
        # exps = list()

        result_total = list()
       

        result_total.append('\\begin{tabular}{|c|'+'|c'* 6 + '|}\n\hline\n')

        result_total.append( '&'.join(['Algorithm', '\multicolumn{3}{c|}{Reward }',  '\multicolumn{3}{c|}{Discounted Reward }'
                           '\multicolumn{3}{c|}{Success Rate }']) + '\\\\\n')
        result_total.append('&'.join([' ', 'mean', 'se', 'p-value',
                            'mean', 'se', 'p-value',]) + '\\\\\n\hline\n')

        best_experiments = exp_set[('best_exp', ctx_vis)]
        experiments = [exp_set[group]['best_exp'] for group in groups]
        for exp_id in experiments:
            experiment = self.results[exp_id]
            n = experiment['runs']
            row = [experiment['ID'], ]
        

            for k in ['reward', 'disc_reward',('success_rate','task_0'), ]:
                returns = experiment['evaluation_best'][k]
                # if setup=='Set0' and ctx_vis:
                #     print(exp_id,best_experiments[k])
                if exp_id == best_experiments:

                    row += [GreenCell+str(round(np.mean(returns), 4)),
                            GreenCell+str(round(np.std(returns)/n**.5, 4)), ' ']

                else:
                    returns_best = self.results[best_experiments]['evaluation_best'][k]
                    # vals_tst = group_best[k]
                    pvalue = ttest_ind(returns,
                              returns_best,
                              equal_var=False)[1]
                    if pvalue> 0.05 or np.isnan(pvalue):
                        sig= ''
                    else:
                        sig = RedCell

                    row += [ str(round(np.mean(returns), 4)),
                              str(round(np.std(returns)/ n**.5, 4)), sig+ str(round(pvalue, 4))]
            result_total.append('&'.join(row) + '\\\\\n')
        # result_total[-1] = result_total[-1][:-3]
        result_total.append('\\hline\n\\end{tabular}\n')
        
        return result_total

    def report(self, exp_sets= None):

        if exp_sets is None:
            exp_sets= list(self.experiment_breakdown.keys())

        for setup in exp_sets:
            # self.read_trainig_results(setup=setup)

            for ctx_vis in [True, False]:
                folder = 'ctx_vis' if ctx_vis else 'ctx_hid'

                experiments, empty = self.experiments_txt(setup=setup, ctx_vis=ctx_vis)
                if empty:
                    continue

                try:
                    filename = os.path.join(self.log_dir , 'report',setup, folder   ,'experiments.txt')
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, "w") as f:

                        for l in experiments:
                            f.write(l)
                except FileNotFoundError as e:
                    print(f'{setup} folder not found')
                    continue

                total_rewards = self.total_results_txt( setup, ctx_vis=ctx_vis)

                try:
                    with open(os.path.join(self.log_dir , 'report' , setup, folder ,'total_rewards.txt'), "w") as f:

                        for l in total_rewards:
                            f.write(l)
                except FileNotFoundError as e:
                    print(f'{setup} folder not found')



                total = self.statistical_comp(setup, ctx_vis=ctx_vis)
                self.plot_training(setup, ctx_vis=ctx_vis)
          

                try:
                    with open(os.path.join(self.log_dir, 'report'  , setup, folder ,'total_rewards_ttest.txt'), "w") as f:
                        for l in total:
                            f.write(l)

                  
                except FileNotFoundError as e:
                    print(f'{setup} folder not found')

    
    

    def read_trainig_results(self, setup, special_kyes = []):
        experiments = self.experiment_breakdown[setup]['experiments']
        tasks = self.experiment_breakdown[setup]['tasks']

        pbar = tqdm(experiments)
        for exp_id in pbar:
            result = self.results[exp_id]

            rewards = []
            disc_rewards=[]
            losses = []

            rewards_eval=[]
            disc_rewards_eval=[]

            pbar.set_description(
                f'{setup}>> Exp {result["ID"]}, {"Visible Ctx" if result["group_id"][0] else "Hidden Ctx"}')
            # if exp_id[0] != setup:
            #     continue
            trials_lst = result['trial_dir']
            policies = result['policies']
            kl_div = {('kl_div',t):[] for t in tasks}

            v_bar = {('v_bar', t): [] for t in tasks}
            mean_diff = {('mean_diff', t): [] for t in tasks}
            var_diff = {('var_diff', t): [] for t in tasks}
            theta_hat = {('theta_hat', t): [] for t in tasks}
            success_rate={('success_rate', t): [] for t in tasks}
            task_reward = {('reward', t):[] for t in tasks}
            task_disc_reward={('disc_reward', t):[] for t in tasks}

            eval_task_reward={('reward', t):[] for t in tasks}
            eval_task_disc_reward={('disc_reward', t):[] for t in tasks}

            eval_success_rate={('success_rate', t): [] for t in tasks}
            episode_length=[]
            total_time = []
            # runs = len(trials_lst)
            for trial_dir in trials_lst:
                try:
                    training_data, evaluation_data = read_results(os.path.join(trial_dir, 'result.json'), keys_=special_kyes)
                except:
                    print('missing experiment')
                    break
                # for pol in policies:
                if 'policy_reward_mean/learner_0' not in training_data:

                    print(exp_id)
                    continue
                
                episode_length.append([np.mean(row) for row in evaluation_data[ 'sampler_results/hist_stats/episode_lengths']])
                rewards.append(sum([np.array(training_data[f'policy_reward_mean/{pol}'])
                                    for pol in policies])/len(policies))
                rewards_eval.append(sum([np.array(evaluation_data[f'policy_reward_mean/{pol}'])
                                    for pol in policies])/len(policies))
                


                losses.append(sum([np.array(training_data[f'info/learner/{pol}/learner_stats/total_loss'] )
                                  for pol in policies])/len(policies))
                training_time = training_data['time_total_s']
                total_time.append(training_time[training_time.size -1])
                if result["group_id"][1] in sp_learners:
                    for t in tasks:
                        eval_success_rate[('success_rate', t)].append(np.array(evaluation_data[f'custom_metrics/{t}_success_rate_mean']))
                        success_rate[('success_rate', t)].append(np.array(training_data[f'custom_metrics/{t}_success_rate_mean']))
                        kl_div[('kl_div',t)].append(np.array(training_data[f'info/curriculum/{t}/kl_div']))
                        mean_diff[('mean_diff',t)].append(np.array(training_data[f'info/curriculum/{t}/mean_diff']))
                        var_diff[('var_diff',t)].append(np.array(training_data[f'info/curriculum/{t}/var_diff']))
                        v_bar[('v_bar',t)].append(np.array(training_data[f'custom_metrics/{t}_disc_reward_mean']
                                                            if f'custom_metrics/{t}_disc_reward_mean' in training_data
                                                            else [0,]))
                        theta_hat[('theta_hat', t)].append(np.array(training_data[f'info/curriculum/{t}/theta_hat']
                                                            ))
                        task_disc_reward[('disc_reward',t)].append(np.array(training_data[f'custom_metrics/{t}_disc_reward_mean']))
                        eval_task_disc_reward[('disc_reward',t)].append(np.array(evaluation_data[f'custom_metrics/{t}_disc_reward_mean']))
            

                elif result["group_id"][1] in single_learners:
                    for t in tasks:
                        pol = policies[0]
                        eval_success_rate[('success_rate', t)].append(np.array(evaluation_data[f'custom_metrics/{t}_success_rate_mean']))
                        success_rate[('success_rate', t)].append(np.array(training_data[f'custom_metrics/{t}_success_rate_mean']))
                        task_reward[('reward',t)].append(np.array(training_data[f'sampler_results/policy_reward_mean/{pol}']))

                        eval_task_reward[('reward',t)].append(np.array(evaluation_data[f'sampler_results/policy_reward_mean/{pol}']))
                        
                        task_disc_reward[('disc_reward',t)].append(np.array(training_data[f'custom_metrics/{t}_disc_reward_mean']))
                        eval_task_disc_reward[('disc_reward',t)].append(np.array(evaluation_data[f'custom_metrics/{t}_disc_reward_mean']))

                else:
                    for t ,pol in zip(tasks, policies):
                        eval_success_rate[('success_rate', t)].append(np.array(evaluation_data[f'custom_metrics/{t}_success_rate_mean']))
                        success_rate[('success_rate', t)].append(np.array(training_data[f'custom_metrics/{t}_success_rate_mean']))
                        task_reward[('reward', t)].append(
                            np.array(training_data[f'policy_reward_mean/{pol}']))
                        eval_task_reward[('reward', t)].append(
                            np.array(evaluation_data[f'policy_reward_mean/{pol}']))
                        task_disc_reward[('disc_reward',t)].append(np.array(training_data[f'custom_metrics/{t}_disc_reward_mean']))
                        eval_task_disc_reward[('disc_reward',t)].append(np.array(evaluation_data[f'custom_metrics/{t}_disc_reward_mean']))
            

            disc_rewards=sum([np.array(task_disc_reward[('disc_reward',t)]) for t in tasks ])
            disc_rewards_eval=sum([np.array(eval_task_disc_reward[('disc_reward',t)]) for t in tasks ])

            

            result['training_summary'] = dict(reward = np.array(rewards),disc_rewards=np.array(disc_rewards),time_s =np.array(training_time),   loss= np.array(losses))
            result['training_summary'].update(kl_div)
            result['training_summary'].update(mean_diff)
            result['training_summary'].update(var_diff)
            result['training_summary'].update(v_bar)
            result['training_summary'].update(theta_hat)
            result['training_summary'].update(success_rate)
            result['training_summary'].update(task_reward)
            result['training_summary'].update(task_disc_reward)

            
            result['training_summary'].update(kl_div)
            result['training_summary'].update(mean_diff)
            result['training_summary'].update(var_diff)
            result['training_summary'].update(v_bar)
            result['training_summary'].update(theta_hat)

            # result['training_summary'].update(task_reward)
            # seeds = result['seeds']

            # keys = list(result[seeds[0]].keys())
            # keys.remove('best_iter')
            result['evaluation_summary'] = dict(reward = np.array(rewards_eval),disc_rewards=np.array(disc_rewards_eval),time_s =np.array(training_time),   loss= np.array(losses))

            best_iter=[np.argmax(rew) for rew in disc_rewards_eval]

            best_iter_summary=dict(iteration= best_iter, reward= [rew[best_iter[i]] for i, rew in enumerate( rewards_eval)], disc_reward= [rew[best_iter[i]] for i, rew in enumerate( disc_rewards_eval)],length=[l[best_iter[i]] for i, l in enumerate(episode_length)])
            for k,v in eval_task_reward.items():
                best_iter_summary[k]=[rew[best_iter[i]] for i, rew in enumerate(v)]
            for k,v in eval_success_rate.items():
                best_iter_summary[k]=[rew[best_iter[i]] for i, rew in enumerate(v)]
            for k,v in eval_task_reward.items():
                best_iter_summary[k]=[rew[best_iter[i]] for i, rew in enumerate(v)]
            # result['evaluation_summary'] = dict(reward=rewards_eval)
            result['evaluation_summary'].update(eval_task_reward)
            result['evaluation_summary'].update(eval_success_rate)

            result['evaluation_best']=best_iter_summary
            # best_iter = np.arg_max(np.flip(np.array([result[seed]['within'].returns for seed in seeds]), axis =1))
            
        return

    def plot(self, experiment_lst, key, name, folder, x_label='iteration', y_label='', title='', color_str=None,
         labels=None, scale=None,plot_group = 'training_summary', step = 1):
        
        if color_str == None:
            color_str = [f'C{i}' for i, _ in enumerate(experiment_lst)]
        assert len(color_str) == len(experiment_lst)

        if labels == None:
            labels = [self.results[exp_id]['ID'] for exp_id in experiment_lst]
        fig = plt.figure()
        plt.figure(figsize=(8, 4))
        for exp_id, color, label in zip(experiment_lst, color_str, labels):
            result = self.results[exp_id]
            if plot_group in result:
                training_summary = result[plot_group]

            else:
                continue

            if key not in training_summary:
                continue

            y = training_summary[key]
            if isinstance(y, list):
                try:
                    y = np.array(y)
                except Exception:
                    y=np.array([x[:400] for x in y])


            if len(y.shape) < 2:
                # print(result['ID'])
                continue
            
            plt.plot(step+step*np.array(range(0, y.shape[1])), np.mean(y, axis=0), color, label=label)
            plt.fill_between(step +step*np.array(range(0, y.shape[1])),
                             np.mean(y, axis=0) - np.std(y, axis=0) / y.shape[0] ** 0.5,
                             np.mean(y, axis=0) + np.std(y, axis=0) / y.shape[0] ** 0.5,
                             color=color, alpha=0.5, label=
                             None)
            df = pd.DataFrame()
            df['x']=(step+step*np.array(range(0, y.shape[1]))).tolist()
            y_mean=np.mean(y, axis=0)
            y_se=np.std(y, axis=0) / y.shape[0] ** 0.5
            df['y']=(y_mean).tolist()
            df['y_p']=(y_mean+y_se).tolist()
            df['y_m']=(y_mean-y_se).tolist()
            df.to_csv(os.path.join(folder, f'{name}_{self.results[exp_id]["ID"]}.csv'), index=False)
        if scale != None:
            plt.yscale(scale)
        plt.legend()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)

        plt.savefig(os.path.join(folder, f'{name}.pdf'), format='pdf', dpi=300)
        plt.close(fig)

    def plot_training(self, setup, ctx_vis = True, group= None):
        if not hasattr(self, "plot_label"):
            self.plot_label = 'id'
        if not hasattr(self, "plot_exps"):
            self.plot_exps = 'all'
        if not hasattr(self, "plot_step"):
            self.plot_step = 1
        vis_dir = 'ctx_vis' if ctx_vis else 'ctx_hid'

        exp_set = self.experiment_breakdown[setup]
        # best_performers, groups, best_by_group = self.best_perf_by_key(setup=setup, ctx_vis=ctx_vis)
        groups = [(ctx_vis, i) for i in (naive_learners+ distill_learners) if (ctx_vis, i) in exp_set]
        if len(groups) == 0:
            raise Exception('no experiment')

        experiments = self.plot_exps
        assert (experiments.lower() in ['all', 'a',  'best', 'b',  'group', 'g', ]) , "experiments to be plotted must be in ['all', 'best', 'group']!"
        # best_experiments = exp_set[('best_exp', ctx_vis)]

        if experiments.lower().startswith('b'):
            experiment_lst = [exp_set[group]['best_exp'] for group in groups]
            exps_tag = 'best'
        elif experiments.lower().startswith('g'):

            assert group in groups, f"specified group: {group} does not exist in the experiment setup"
            experiment_lst = exp_set[group]['experiments']
            exps_tag = f'{group}'
        else:
            experiment_lst = []
            for group in groups:
                experiment_lst+=exp_set[group]['experiments']
            exps_tag = 'all'
        folder = os.path.join(self.log_dir,'report', setup, vis_dir)
        if self.plot_label.lower() =='id':
            labels = [f'{self.results[exp_id]["ID"]}' for exp_id in experiment_lst]

        elif self.plot_label.lower() =='full':
            labels = [f'{self.results[exp_id]["ID"]}, {exp_id[5]}' for exp_id in experiment_lst]

        else:
            labels = [f'{curriculum(self.results[exp_id]["group_id"][1])}' for exp_id in experiment_lst]
            labels = ['Default' if lab =='Def' else lab for lab in labels]
            labels = ['Self-paced' if lab == 'SP' else lab for lab in labels]
            labels = ['Gaussian Self-paced' if lab == 'GSP' else lab for lab in labels]
        # experiment_lst, key, name, folder, x_label = 'iteration', y_label = '', title = '', color_str = None, labels = None
        self.plot(experiment_lst=experiment_lst, key ='reward', name = f'total_rewards_{exps_tag}',
                  title= '', y_label='Average Collected Reward', labels= labels, folder= folder)

        self.plot(experiment_lst=experiment_lst, key='loss', name=f'loss_{exps_tag}',
                  title='', y_label='Total Loss', labels=labels,folder= folder)
        tasks = self.experiment_breakdown[setup]['tasks']
        for tsk in tasks:
            self.plot(experiment_lst=experiment_lst, key=('kl_div', tsk), name=f'kl_div_{tsk}_{exps_tag}',
                  title='', y_label='KL divergence', labels=labels, folder= folder, scale= 'log')
            self.plot(experiment_lst=experiment_lst, key=('theta_hat', tsk), name=f'theta_{tsk}_{exps_tag}',
                      title='', y_label='theta', labels=labels, folder=folder, scale='log')
            self.plot(experiment_lst=experiment_lst, key=('mean_diff', tsk), name=f'mean_diff_{tsk}_{exps_tag}',
                      title='', y_label='avg mean difference', labels=labels, folder=folder)

            self.plot(experiment_lst=experiment_lst, key=('var_diff', tsk), name=f'var_diff_{tsk}_{exps_tag}',
                      title='', y_label='avg var difference', labels=labels, folder=folder)

            self.plot(experiment_lst=experiment_lst, key=('v_bar', tsk), name=f'v_bar_{tsk}_{exps_tag}',
                      title='', y_label='CL V bar', labels=labels, folder=folder)
            self.plot(experiment_lst=experiment_lst, key=('success_rate', tsk), name=f'success_{tsk}_{exps_tag}',
                      title='', y_label='success', labels=labels, folder=folder)
            self.plot(experiment_lst=experiment_lst, key=('reward', tsk), name=f'reward_{tsk}_{exps_tag}',
                      title='', y_label='Average Collected Reward', labels=labels, folder=folder)
        # print(experiment_lst)

            self.plot(experiment_lst=experiment_lst, key=('reward', tsk), name=f'eval_reward_{tsk}_{exps_tag}',
                      title='', y_label='Average Collected Reward', labels=labels, folder=folder, plot_group='evaluation_summary',
                      step=self.plot_step)
            self.plot(experiment_lst=experiment_lst, key=('success_rate', tsk), name=f'eval_success_rate_{tsk}_{exps_tag}',
                      title='', y_label='success', labels=labels, folder=folder, plot_group='evaluation_summary',
                      step=self.plot_step)

        # tasks = self.experiment_breakdown[setup]['tasks']
    
