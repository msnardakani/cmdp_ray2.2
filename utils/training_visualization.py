import itertools
import pickle
import time
from collections import deque
from matplotlib import pyplot as plt
from ray.rllib.policy.policy import Policy
import os
import pandas as pd
import torch
from scipy.stats import ttest_ind
from tqdm import tqdm
from envs.gridworld_contextual import TaskSettableGridworld
from gymnasium.wrappers import TimeLimit, RecordVideo, RecordEpisodeStatistics
from envs.contextual_env import CtxDictWrapper, ctx_visibility, exp_group
import numpy as np
from ray.rllib.policy.policy import Policy
from distral.distral_ppo_torch_model import DistralCentralTorchModel, DistralTorchModel
from ray.rllib.models import ModelCatalog
import json
import gymnasium as gym
from .policy_visualization_evaluation import ExperimentEvaluation, distill_learners, single_learners, naive_learners, sp_learners, name_codes, curriculum

# from gymnasium.envs.mujoco.pusher_v4 import  PusherEnv
# class iterations_reuslt():
#     def __init__(self, returns, lengths, custom_metric= None):
#         self.returns = returns
#         self.lengths= lengths
#         self.custom_metric = custom_metric

class ExperimentResult(ExperimentEvaluation):
    def read_trainig_results(self, setup, special_kyes = None):
        experiments = self.experiment_breakdown[setup]['experiments']
        tasks = self.experiment_breakdown[setup]['tasks']

        pbar = tqdm(experiments)
        for exp_id in pbar:
            result = self.results[exp_id]

            rewards = []
            losses = []
            pbar.set_description(
                f'{setup}>> Exp {result["ID"]}, {"Visible Ctx" if result["group_id"][0] else "Hidden Ctx"}')
            # if exp_id[0] != setup:
            #     continue
            trials_lst = result['trial_dir']
            policies = result['policies']
            kl_div = {('kl_div',t):[] for t in tasks}
            task_reward = {('reward', t):[] for t in tasks}
            total_time = []
            # runs = len(trials_lst)
            for trial_dir in trials_lst:
                try:
                    training_data = pd.read_csv(os.path.join(trial_dir, 'progress.csv'))
                except:
                    print('missing experiment')
                    break
                # for pol in policies:
                rewards.append(sum([np.array(training_data[f'sampler_results/policy_reward_mean/{pol}'])
                                    for pol in policies])/len(policies))
                losses.append(sum([np.array(training_data[f'info/learner/{pol}/learner_stats/total_loss'] )
                                  for pol in policies])/len(policies))
                training_time = training_data['time_total_s']
                total_time.append(training_time[training_time.size -1])
                if result["group_id"][1] in sp_learners:
                    for t in tasks:

                        kl_div[('kl_div',t)].append(np.array(training_data[f'info/curriculum/{t}/kl_div']))

                if result["group_id"][1] in single_learners:
                    for t in tasks:
                        pol = policies[0]
                        task_reward[('reward',t)].append(np.array(training_data[f'sampler_results/policy_reward_mean/{pol}']))
                else:
                    for t ,pol in zip(tasks, policies):

                        task_reward[('reward', t)].append(
                            np.array(training_data[f'sampler_results/policy_reward_mean/{pol}']))

            result['training_summary'] = dict(reward = np.array(rewards),time_s =np.array(training_time),   loss= np.array(losses))
            result['training_summary'].update(kl_div)

            result['training_summary'].update(task_reward)

    def plot(self, experiment_lst, key, name, folder, x_label = 'iteration', y_label = '', title = '',color_str=None, labels = None):
        if color_str == None:
            color_str = [f'C{i}' for i,_ in enumerate(experiment_lst)]
        assert len(color_str) == len(experiment_lst)

        if labels ==None:
            labels = [ self.results[exp_id]['ID'] for exp_id in experiment_lst]
        fig = plt.figure()
        plt.figure(figsize=(10, 4))
        for exp_id, color, label in zip(experiment_lst, color_str, labels) :
            result = self.results[exp_id]
            if 'training_summary' in result:
                training_summary = result['training_summary']

            else:
                continue

            # if len(training_summary[key].shape) == 1:
            #     print(result['ID'])
            #     continue

            # legends.append(result['ID'])
            # legends.append(None)
            if key not in training_summary:
                continue


            y = training_summary[key]
            if isinstance(y, list):
                y = np.array(y)


            if len(y.shape) <2:
                print(result['ID'])
                continue
            plt.plot(np.mean(y, axis=0), color, label=label)
            plt.fill_between(range(y.shape[1]),
                             np.mean(y, axis=0) - np.std(y, axis=0) / y.shape[0] ** 0.5,
                             np.mean(y, axis=0) + np.std(y, axis=0) / y.shape[0] ** 0.5,
                             color=color, alpha=0.5, label=
                             None)

        plt.legend()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)

        plt.savefig(os.path.join(folder, f'{name}.pdf'), format='pdf')
        # plt.show()

    def plot_training(self, setup, experiments = 'All', ctx_vis = True, group= None):

        vis_dir = 'ctx_vis' if ctx_vis else 'ctx_hid'

        exp_set = self.experiment_breakdown[setup]
        # best_performers, groups, best_by_group = self.best_perf_by_key(setup=setup, ctx_vis=ctx_vis)
        groups = [(ctx_vis, i) for i in (naive_learners+ distill_learners) if (ctx_vis, i) in exp_set]
        if len(groups) == 0:
            raise Exception('no experiment')


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
        labels = [f'{name_codes[self.results[exp_id]["group_id"]]}, {curriculum(self.results[exp_id]["group_id"][1])}' for exp_id in experiment_lst]
        # experiment_lst, key, name, folder, x_label = 'iteration', y_label = '', title = '', color_str = None, labels = None
        self.plot(experiment_lst=experiment_lst, key ='reward', name = f'rewards_{exps_tag}',
                  title= '', y_label='Average Collected Reward', labels= labels, folder= folder)

        self.plot(experiment_lst=experiment_lst, key='loss', name=f'loss_{exps_tag}',
                  title='', y_label='Total Loss', labels=labels,folder= folder)
        tasks = self.experiment_breakdown[setup]['tasks']
        for tsk in tasks:
            self.plot(experiment_lst=experiment_lst, key=('kl_div', tsk), name=f'kl_div_{tsk}_{exps_tag}',
                  title='', y_label='KL divergence', labels=labels, folder= folder)
        # print(experiment_lst)


    def report(self):
        super().report()
        for setup in self.experiment_breakdown.keys():
            self.read_trainig_results(setup=setup)
            for ctx_vis in [True, False]:

                self.plot_training(setup, experiments='best',ctx_vis = ctx_vis)
