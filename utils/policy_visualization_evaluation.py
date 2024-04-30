import itertools
import pickle
import time
from collections import deque

from ray.rllib.policy.policy import Policy
import os
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


def record_rollouts(checkpoint_dir, env, trajectories= 2, distill= False, TupleObs = False):
    env.episode_id = 0
    if TupleObs:
        preprocessor = lambda x: np.append(x[0], x[1])
    else:
        preprocessor = lambda x: x

    learner = Policy.from_checkpoint(checkpoint_dir)
    # custom_result = []

    for _ in range(trajectories):
        obs = env.reset()
        done = False
        while not done:
            # print(obs)
            action = learner.compute_single_action(input_dict=dict(obs=preprocessor(obs[0])))[0]
            if distill:
                action = learner.dist_class(learner.model.distill_out()).sample().detach().to('cpu').numpy()
            obs = env.step(action)

            done = obs[2] or obs[3]
        # if func is not None:
        #     custom_result.append(func(obs))
    return learner


def evaluate_policy_vec(learner, env, duration= 128, distill= False, func = None, TupleObs = False):
    if TupleObs:
        preprocessor = lambda x: np.concatenate(x, axis =1)

    else:
        preprocessor = lambda x: x
    custom_metric = deque([], maxlen=duration)
    # learner = Policy.from_checkpoint(checkpoint_dir)

    if func is not None and not isinstance(func, list):
        func = [func,]
    if func is None:
        func = []
    while len(env.return_queue)<duration:
        obs = env.reset()
        done = False
        while (not done) and len(env.return_queue)<duration:
            # print(obs)
            action = learner.compute_actions(obs_batch=preprocessor(obs[0]))[0]
            if distill:
                action = learner.dist_class(learner.model.distill_out()).sample().detach().to('cpu').numpy()
            obs = env.step(action)
            terminals = np.logical_or(obs[2], obs[3])
            done = np.all(terminals)
            if np.any(terminals):
                metrics = np.array([f(obs) for f in func])
                custom_metric.extend(list(metrics.T[terminals]))

    return np.mean(np.array(env.return_queue).reshape(-1)), np.mean(np.array(env.length_queue).reshape(-1)), np.mean(custom_metric, axis= 0).reshape(-1)


def evaluate_policy(checkpoint_dir, env, duration= 128, distill= False, func = None, TupleObs = False):
    env.episode_id = 0
    if TupleObs:
        preprocessor = lambda x : np.append(x[0], x[1])
    else:
        preprocessor = lambda x: x

    learner = Policy.from_checkpoint(checkpoint_dir)
    custom_result = []

    for _ in range(duration):
        obs = env.reset()
        done = False
        while not done:
            # print(obs)
            action = learner.compute_single_action(input_dict=dict(obs = preprocessor(obs[0])))[0]
            if distill:
                action = learner.dist_class(learner.model.distill_out()).sample().detach().to('cpu').numpy()
            obs = env.step(action)

            done = obs[2] or obs[3]
        if func is not None:
            custom_result.append(func(obs))

    return np.mean(np.array(env.return_queue).reshape(-1)), np.mean(np.array(env.length_queue).reshape(-1)), np.mean(np.array(custom_result), axis= 0).reshape(-1)

class evaluation():
    def __init__(self, returns, lengths, custom_metric= None):
        self.returns = returns
        self.lengths= lengths
        self.custom_metric = custom_metric
        # self.best_idx = best_idx

single_learners = [0, 2, 6]
distill_learners = [ 0, 2, 6, 8, 10, 14, 9, 11, 15]
naive_learners= [1, 3, 7]
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

    def __init__(self, experiment_dir ):
        self.log_dir = experiment_dir
        self.experiment_breakdown = dict()
        self.results = dict()
        self.extrnal_evaluation = dict()

    # def extr_analyse(self, setup, config_lst, env_creator, func = None, duration= 100,n_env= 10,  ):

    def analyse(self, env_creator, func = None, duration= 100,n_env= 10, config_trans = lambda c: c, setup_list = None):
        self.results, self.experiment_breakdown = self.get_experiments()
        # t0 = time.time()
        # ctr = 1

        if setup_list is None:
            setup_list = list(self.experiment_breakdown.keys())
            # pbar = tqdm(setup_list, position = 0)

            for setup in setup_list:
                # pbar.set_description("Evaluating %s" % setup)
                # start = time.time()
                # print(f'Started evaluating {setup} ({ctr}/{len(self.experiment_breakdown)})')
                self.get_results(setup=setup, env_creator=env_creator, n_env= n_env, func=func, duration = duration, config_trans=config_trans)
                # end = time.time()
                # ctr+=1
                # print(f'Finished evaluating {setup}\nSegment completion time: {time.strftime("%H:%M:%S", time.gmtime(end- start))},\tTotal elapsed time: {time.strftime("%H:%M:%S", time.gmtime(end- t0))}')
                self.save_to_file('temp_result.pkl')
            # setup_list = list(self.experiment_breakdown.keys())
            # pbar = tqdm(setup_list, position=0)
            for setup in setup_list:
                # pbar.set_description("Processing %s" % setup)
                self.best_perf_by_group(setup=setup, ctx_vis=True)
                self.best_perf_by_group(setup=setup, ctx_vis=False)
                # end = time.time()
                # print(
                #     f'Finished processing {setup} evaluation\nSegment completion time: {time.strftime("%H:%M:%S", time.gmtime(end- start))},\tTotal elapsed time: {time.strftime("%H:%M:%S", time.gmtime(end- t0))}')
            self.save_to_file('temp_result.pkl')

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

    def get_extr_results(self, setup, env_creator, env_config_lst, func=None, duration=100, n_env=10,):
        # env_config = self.experiment_breakdown[setup]['env_config']
        # t0 = time.time()
        # ctr = 0
        experiments = self.experiment_breakdown[setup]['experiments']
        pbar = tqdm(experiments)
        tasks = [f'eval_task_{i}' for i in range(len(env_config_lst))]

        extr_result = dict(tasks = tasks, env_config = env_config_lst, setup = setup)

        for exp_id in pbar:
            result = self.results[exp_id]

            extr_result[exp_id] = dict()
            pbar.set_description(
                f'{setup}>> Exp {result["ID"]}, {"Visible Ctx" if result["group_id"][0] else "Hidden Ctx"}')


            if 'vis' in exp_id[1]:
                if 'task_aug' in exp_id[2]:
                    ctx_mode = 1
                else:
                    ctx_mode = 2
            else:
                ctx_mode = 0
            # t1 = time.time()

            # pc =  (40 * ctr) // len(experiments)

            # print(f'Evaluating {ID_str}, {"Visible Ctx" if result["group_id"][0] else "Hidden Ctx"  } \n{"x"*pc + "-"*(40 - pc) }{ctr}/{len(experiments)}:({time.strftime("%H:%M:%S", time.gmtime(t1- t0))})')
            policies = result['policies']

            # print(exp_id)

            for trial_dir, seed in zip(result['trial_dir'], result['seeds']):
                assert 'best_iter' in result[seed], "Must run internal evaluations first!"
                best_iter = result[seed]['best_iter']
                seed_result = dict()
                checkpoint = result['checkpoints'][best_iter]
                for policy in policies:
                    for task, config in zip(tasks, env_config_lst):
                        # task = 'task_'+str(i)
                        reward = np.zeros(0)
                        length = np.zeros(0)
                        custom_result = np.zeros((0, duration))


                        # if int(checkpoint.replace('checkpoint_', ''))< 2:
                        #     continue
                        log_dir = os.path.join(trial_dir, checkpoint, 'policies', policy)
                        name_prefix = f'{policy},{task}'
                        env_creator_record = lambda cfg: RecordVideo(
                            env_creator(cfg, ctx_mode),
                            video_folder=log_dir, name_prefix=name_prefix, episode_trigger=lambda x: True,
                            disable_logger=True)

                        env_creator_stat = lambda config: RecordEpisodeStatistics(
                            gym.vector.SyncVectorEnv([lambda: env_creator(config, ctx_mode) for i in range(n_env)]),
                            deque_size=duration)


                        env = env_creator_record(config)
                        learner = record_rollouts(log_dir, env, TupleObs= ctx_mode == 2)
                        env = env_creator_stat(config)

                        task_results = evaluate_policy_vec(learner, env, duration=duration, func=func,
                                                           TupleObs= ctx_mode == 2)

                        reward = np.append(reward, task_results[0])
                        length = np.append(length, task_results[1])
                        custom_result = np.append(custom_result, task_results[2]).reshape(-1, task_results[2].size)

                        seed_result.update({(task, policy): evaluation(returns=reward, lengths=length,
                                                                       custom_metric=custom_result if func is not None else None)})
                if len(policies) ==1:
                    policy = DistilledPolicy
                    for task, config in zip(tasks, env_config_lst):
                        seed_result.update({(task, policy): seed_result[(task, policies[0])]})

                if 'baseline' not in exp_id[2]:
                    policy = DistilledPolicy
                    for task, config in zip(tasks, env_config_lst):
                        # task = 'task_' + str(i)
                        reward = np.zeros(0)
                        length = np.zeros(0)
                        custom_result = np.zeros((0, duration))

                        # if int(checkpoint.replace('checkpoint_', ''))< 2:
                        #     continue
                        log_dir = os.path.join(trial_dir, checkpoint, 'policies', policies[0])
                        name_prefix = f'{policy},{task}'

                        env_creator_record = lambda cfg: RecordVideo(
                            env_creator(cfg, ctx_mode),
                            video_folder=log_dir, name_prefix=name_prefix, episode_trigger=lambda x: True,
                            disable_logger=True)

                        env_creator_stat = lambda config: RecordEpisodeStatistics(
                            gym.vector.SyncVectorEnv([lambda: env_creator(config, ctx_mode) for i in range(n_env)]),
                            deque_size=duration)


                        env = env_creator_record(config)
                        learner = record_rollouts(log_dir, env, distill=True, TupleObs=ctx_mode == 2)
                        env = env_creator_stat(config)

                        task_results = evaluate_policy_vec(learner, env, duration=duration, distill=True, func=func,
                                                           TupleObs=ctx_mode == 2)

                        reward = np.append(reward, task_results[0])
                        length = np.append(length, task_results[1])
                        custom_result = np.append(custom_result, task_results[2]).reshape(-1,
                                                                                          task_results[2].size)

                        seed_result.update({(task, policy): evaluation(returns=reward, lengths=length,
                                                                       custom_metric=custom_result if func is not None else np.zeros_like(
                                                                           reward))})


                extr_result[exp_id][seed] = seed_result
            seed_result = extr_result[exp_id][result['seeds'][0]]
            temp = dict()
            policy = DistilledPolicy

            for task in tasks:
                if (task, policy) in seed_result:
                    rewards = np.array(
                        [extr_result[exp_id][seed][(task, policy)].returns for seed in
                         result['seeds']]).reshape(-1)
                    lengths = np.array(
                        [extr_result[exp_id][seed][(task, policy)].lengths for seed in
                         result['seeds']]).reshape(-1)
                    custom_metric = np.array(
                        [extr_result[exp_id][seed][(task, policy)].custom_metric for seed in
                         result['seeds']]).T

                    temp[(task, policy)] = evaluation(returns=rewards, lengths=lengths, custom_metric=custom_metric)
                else:
                    break

            for task, policy in itertools.product(tasks, policies):
                rewards = np.array(
                    [extr_result[exp_id][seed][(task, policy)].returns for seed in result['seeds']]).reshape(-1)
                lengths = np.array(
                    [extr_result[exp_id][seed][(task, policy)].lengths for seed in result['seeds']]).reshape(-1)
                custom_metric = np.array(
                    [extr_result[exp_id][seed][(task, policy)].custom_metric for seed in
                     result['seeds']]).T

                temp[(task, policy)] = evaluation(returns=rewards, lengths=lengths, custom_metric=custom_metric)

            extr_result[exp_id]['summary'] = temp
        with open(os.path.join(self.log_dir ,'report', setup,'extr_eval.pkl'), 'wb') as file:
            # A new file will be created
            pickle.dump(extr_result, file)

        return extr_result

    def update_extr_results(self, extr_result, env_creator, func=None, duration=100, n_env=10,):
        # with open(os.path.join(self.log_dir,setup,name), 'rb') as file:
        # # Call load method to deserialze
        #     extr_result = pickle.load(file)

        # setup = extr_result['setup']
        setup = extr_result['setup']
        experiments = self.experiment_breakdown[setup]['experiments']
        pbar = tqdm(experiments)
        tasks = extr_result['tasks']
        env_config_lst = extr_result['env_config']


        for exp_id in pbar:
            result = self.results[exp_id]
            if exp_id not in extr_result:
                extr_result[exp_id] = dict()
            pbar.set_description(
                f'{setup}>> Exp {result["ID"]}, {"Visible Ctx" if result["group_id"][0] else "Hidden Ctx"}')


            if 'vis' in exp_id[1]:
                if 'task_aug' in exp_id[2]:
                    ctx_mode = 1
                else:
                    ctx_mode = 2
            else:
                ctx_mode = 0
            # t1 = time.time()

            # pc =  (40 * ctr) // len(experiments)

            # print(f'Evaluating {ID_str}, {"Visible Ctx" if result["group_id"][0] else "Hidden Ctx"  } \n{"x"*pc + "-"*(40 - pc) }{ctr}/{len(experiments)}:({time.strftime("%H:%M:%S", time.gmtime(t1- t0))})')
            policies = result['policies']

            # print(exp_id)

            for trial_dir, seed in zip(result['trial_dir'], result['seeds']):

                assert 'best_iter' in result[seed], "Must run internal evaluations first!"
                if seed in extr_result[exp_id]:
                    continue

                best_iter = result[seed]['best_iter']
                seed_result = dict()
                checkpoint = result['checkpoints'][best_iter]
                for policy in policies:
                    for task, config in zip(tasks, env_config_lst):
                        # task = 'task_'+str(i)
                        reward = np.zeros(0)
                        length = np.zeros(0)
                        custom_result = np.zeros((0, duration))


                        # if int(checkpoint.replace('checkpoint_', ''))< 2:
                        #     continue
                        log_dir = os.path.join(trial_dir, checkpoint, 'policies', policy)
                        name_prefix = f'{policy},{task}'
                        env_creator_record = lambda cfg: RecordVideo(
                            env_creator(cfg, ctx_mode),
                            video_folder=log_dir, name_prefix=name_prefix, episode_trigger=lambda x: True,
                            disable_logger=True)

                        env_creator_stat = lambda config: RecordEpisodeStatistics(
                            gym.vector.SyncVectorEnv([lambda: env_creator(config, ctx_mode) for i in range(n_env)]),
                            deque_size=duration)


                        env = env_creator_record(config)
                        learner = record_rollouts(log_dir, env, TupleObs= ctx_mode == 2)
                        env = env_creator_stat(config)

                        task_results = evaluate_policy_vec(learner, env, duration=duration, func=func,
                                                           TupleObs= ctx_mode == 2)

                        reward = np.append(reward, task_results[0])
                        length = np.append(length, task_results[1])
                        custom_result = np.append(custom_result, task_results[2]).reshape(-1, task_results[2].size)

                        seed_result.update({(task, policy): evaluation(returns=reward, lengths=length,
                                                                       custom_metric=custom_result if func is not None else None)})
                if len(policies) ==1:
                    policy = DistilledPolicy
                    for task, config in zip(tasks, env_config_lst):
                        seed_result.update({(task, policy): seed_result[(task, policies[0])]})

                if 'baseline' not in exp_id[2]:
                    policy = DistilledPolicy
                    for task, config in zip(tasks, env_config_lst):
                        # task = 'task_' + str(i)
                        reward = np.zeros(0)
                        length = np.zeros(0)
                        custom_result = np.zeros((0, duration))

                        # if int(checkpoint.replace('checkpoint_', ''))< 2:
                        #     continue
                        log_dir = os.path.join(trial_dir, checkpoint, 'policies', policies[0])
                        name_prefix = f'{policy},{task}'

                        env_creator_record = lambda cfg: RecordVideo(
                            env_creator(cfg, ctx_mode),
                            video_folder=log_dir, name_prefix=name_prefix, episode_trigger=lambda x: True,
                            disable_logger=True)

                        env_creator_stat = lambda config: RecordEpisodeStatistics(
                            gym.vector.SyncVectorEnv([lambda: env_creator(config, ctx_mode) for i in range(n_env)]),
                            deque_size=duration)


                        env = env_creator_record(config)
                        learner = record_rollouts(log_dir, env, distill=True, TupleObs=ctx_mode == 2)
                        env = env_creator_stat(config)

                        task_results = evaluate_policy_vec(learner, env, duration=duration, distill=True, func=func,
                                                           TupleObs=ctx_mode == 2)

                        reward = np.append(reward, task_results[0])
                        length = np.append(length, task_results[1])
                        custom_result = np.append(custom_result, task_results[2]).reshape(-1,
                                                                                          task_results[2].size)

                        seed_result.update({(task, policy): evaluation(returns=reward, lengths=length,
                                                                       custom_metric=custom_result if func is not None else np.zeros_like(
                                                                           reward))})


                extr_result[exp_id][seed] = seed_result
            seed_result = extr_result[exp_id][result['seeds'][0]]
            temp = dict()
            policy = DistilledPolicy

            for task in tasks:
                if (task, policy) in seed_result:
                    rewards = np.array(
                        [extr_result[exp_id][seed][(task, policy)].returns for seed in
                         result['seeds']]).reshape(-1)
                    lengths = np.array(
                        [extr_result[exp_id][seed][(task, policy)].lengths for seed in
                         result['seeds']]).reshape(-1)
                    custom_metric = np.array(
                        [extr_result[exp_id][seed][(task, policy)].custom_metric  for seed in
                         result['seeds']]).T

                    temp[(task, policy)] = evaluation(returns=rewards, lengths=lengths, custom_metric=custom_metric)
                else:
                    break

            for task, policy in itertools.product(tasks, policies):
                rewards = np.array(
                    [extr_result[exp_id][seed][(task, policy)].returns for seed in result['seeds']]).reshape(-1)
                lengths = np.array(
                    [extr_result[exp_id][seed][(task, policy)].lengths for seed in result['seeds']]).reshape(-1)
                custom_metric = np.array(
                    [extr_result[exp_id][seed][(task, policy)].custom_metric  for seed in
                     result['seeds']]).T

                temp[(task, policy)] = evaluation(returns=rewards, lengths=lengths, custom_metric=custom_metric)

            extr_result[exp_id]['summary'] = temp
        with open(os.path.join(self.log_dir , 'report',setup, 'extr_eval.pkl'), 'wb') as file:
            # A new file will be created
            pickle.dump(extr_result, file)
        return extr_result

    def get_results(self, setup, env_creator, func= None, duration = 100, n_env = 10, config_trans= lambda c: c, override = False):
        env_config = self.experiment_breakdown[setup]['env_config']
        # t0 = time.time()
        # ctr = 0
        experiments = self.experiment_breakdown[setup]['experiments']
        pbar = tqdm(experiments)
        for exp_id in pbar:
            result = self.results[exp_id]

            pbar.set_description( f'{setup}>> Exp {result["ID"]}, {"Visible Ctx" if result["group_id"][0] else "Hidden Ctx"  }')
            # if exp_id[0] != setup:
            #     continue
            # N = duration // 2
            if 'vis' in exp_id[1]:
                if 'task_aug' in exp_id[2]:
                    ctx_mode = 1
                else:
                    ctx_mode = 2
            else:
                ctx_mode = 0
            # t1 = time.time()

            # pc =  (40 * ctr) // len(experiments)

            # print(f'Evaluating {ID_str}, {"Visible Ctx" if result["group_id"][0] else "Hidden Ctx"  } \n{"x"*pc + "-"*(40 - pc) }{ctr}/{len(experiments)}:({time.strftime("%H:%M:%S", time.gmtime(t1- t0))})')
            tasks = [f'task_{i}' for i in range(len(env_config))]
            policies = result['policies']

            # print(exp_id)


            for trial_dir, seed in zip(result['trial_dir'], result['seeds']):
                if result.get(seed, False) is not False and not override:
                    # print(f'{setup}, {"Visible Ctx" if result["group_id"][0] else "Hidden Ctx"  }>>  trials skipped: id: {ID_str}, seed: {seed}', end='\t')

                    continue
                seed_result = dict()

                for policy in policies:
                    for task, config in zip(tasks, env_config):
                        # task = 'task_'+str(i)
                        reward = np.zeros(0)
                        length = np.zeros(0)
                        custom_result = np.zeros((0,duration))

                        for checkpoint in result['checkpoints']:
                            # if int(checkpoint.replace('checkpoint_', ''))< 2:
                            #     continue
                            log_dir = os.path.join(trial_dir, checkpoint, 'policies', policy)
                            name_prefix = f'{policy},{task}'
                            env_creator_record = lambda cfg: RecordVideo(
                                env_creator(cfg, ctx_mode),
                                video_folder=log_dir, name_prefix=name_prefix,episode_trigger=lambda x: True, disable_logger=True)

                            env_creator_stat = lambda config: RecordEpisodeStatistics(
                                gym.vector.SyncVectorEnv([lambda: env_creator(config, ctx_mode) for i in range(n_env)]), deque_size=duration)
                            # env_creator_ctx = lambda cfg: RecordVideo(
                            #     RecordEpisodeStatistics(env_creator(cfg, ctx_mode), deque_size=duration),
                            #     video_folder=log_dir, name_prefix=name_prefix,episode_trigger=lambda x: x%N ==0, disable_logger=True)

                            config_ = config_trans(config)

                            env = env_creator_record(config_)
                            learner = record_rollouts(log_dir, env, TupleObs= ctx_mode==2)
                            env = env_creator_stat(config_)

                            task_results = evaluate_policy_vec(learner, env, duration = duration, func = func, TupleObs= ctx_mode==2)

                            reward = np.append(reward, task_results[0])
                            length = np.append(length, task_results[1])
                            custom_result = np.append(custom_result, task_results[2]).reshape(-1, task_results[2].size)

                        seed_result.update({(task, policy): evaluation(returns=reward, lengths=length, custom_metric= custom_result if func is not None else None)})

                if 'baseline' not in exp_id[2]:
                    policy = DistilledPolicy
                    for task, config in zip(tasks, env_config):
                        # task = 'task_' + str(i)
                        reward = np.zeros(0)
                        length = np.zeros(0)
                        custom_result = np.zeros((0, duration))

                        for checkpoint in result['checkpoints']:
                            # if int(checkpoint.replace('checkpoint_', ''))< 2:
                            #     continue
                            log_dir = os.path.join(trial_dir, checkpoint, 'policies', policies[0])
                            name_prefix = f'{policy},{task}'

                            env_creator_record = lambda cfg: RecordVideo(
                                env_creator(cfg, ctx_mode),
                                video_folder=log_dir, name_prefix=name_prefix, episode_trigger=lambda x: True,
                                disable_logger=True)

                            env_creator_stat = lambda config: RecordEpisodeStatistics(
                                gym.vector.SyncVectorEnv([lambda: env_creator(config, ctx_mode) for i in range(n_env)]),
                                deque_size=duration)
                            # env_creator_ctx = lambda cfg: RecordVideo(
                            #     RecordEpisodeStatistics(env_creator(cfg, ctx_mode), deque_size=duration),
                            #     video_folder=log_dir, name_prefix=name_prefix,episode_trigger=lambda x: x%N ==0, disable_logger=True)

                            config_ = config_trans(config)

                            env = env_creator_record(config_)
                            learner = record_rollouts(log_dir, env, distill=True, TupleObs=ctx_mode == 2)
                            env = env_creator_stat(config_)

                            task_results = evaluate_policy_vec(learner, env, duration=duration,distill=True, func=func,
                                                               TupleObs=ctx_mode == 2)

                            reward = np.append(reward, task_results[0])
                            length = np.append(length, task_results[1])
                            custom_result = np.append(custom_result, task_results[2]).reshape(-1,
                                                                                              task_results[2].size)

                        seed_result.update({(task, policy): evaluation(returns=reward, lengths=length,
                                                                            custom_metric=custom_result if func is not None else np.zeros_like(reward))})

                fields = ['returns', 'lengths', 'custom_metric']
                if len(tasks) == len(policies):

                    # for attr in ['returns', 'lengths', 'custom']:
                    within = [np.sum([seed_result[(task, policy)].__getattribute__(attr)
                                          for task, policy in zip(tasks, policies)], axis=0) for attr in fields]
                    total = [np.sum([seed_result[(task, policy)].__getattribute__(attr)
                                          for task, policy in itertools.product(tasks, policies)], axis=0) for attr in fields]


                else:
                    policy = policies[0]
                    within = [np.sum([seed_result[(task, policy)].__getattribute__(attr)
                                          for task in tasks], axis=0)
                              for attr in fields]
                    total = [len(tasks)*np.sum([seed_result[(task, policy)].__getattribute__(attr)
                                     for task, policy in itertools.product(tasks, policies)], axis=0)
                             for attr in
                             fields]

                    for task in tasks:
                        seed_result[(task, DistilledPolicy)] = seed_result[(task, policy)]

                best_iter = np.argmax(within[0])
                seed_result['within'] = evaluation(returns=within[0], lengths=within[1], custom_metric=within[2])
                seed_result['total'] = evaluation(returns=total[0], lengths=total[1], custom_metric=total[2])

                seed_result['between'] = evaluation(returns=total[0]- within[0], lengths=total[1]- within[1], custom_metric=total[2]- within[2])
                seed_result['best_iter'] = best_iter
                result[seed] = seed_result

            seed_result = result[result['seeds'][0]]
            temp = dict()
            policy = DistilledPolicy

            for task in tasks:
                if (task, policy) in seed_result:
                    rewards = np.array(
                        [result[seed][(task, policy)].returns[result[seed]['best_iter']] for seed in result['seeds']])
                    lengths = np.array(
                        [result[seed][(task, policy)].lengths[result[seed]['best_iter']] for seed in result['seeds']])
                    custom_metric = np.array(
                        [result[seed][(task, policy)].custom_metric[result[seed]['best_iter']] for seed in
                         result['seeds']])

                    temp[(task, policy)] = evaluation(returns=rewards, lengths=lengths, custom_metric=custom_metric)
                else:
                    break

            for task, policy in itertools.product(tasks, policies):
                rewards = np.array([result[seed][(task, policy)].returns[result[seed]['best_iter']] for seed in result['seeds']])
                lengths = np.array([result[seed][(task, policy)].lengths[result[seed]['best_iter']] for seed in result['seeds']])
                custom_metric = np.array(
                    [result[seed][(task, policy)].custom_metric[result[seed]['best_iter']] for seed in result['seeds']])

                temp[(task, policy)] = evaluation(returns=rewards, lengths = lengths, custom_metric= custom_metric)

            for  policy in  policies:
                # if (tasks[0], DistilledPolicy)
                rewards = np.array([sum([result[seed][(task, policy)].returns[result[seed]['best_iter']] for task in tasks])
                                        for seed in result['seeds']])
                lengths = np.array(
                    [sum([result[seed][(task, policy)].lengths[result[seed]['best_iter']] for task in tasks])
                     for seed in result['seeds']])
                custom_metric = np.array(
                    [sum([result[seed][(task, policy)].returns[result[seed]['best_iter']] for task in tasks])
                     for seed in result['seeds']])
                temp[('total', policy)] = evaluation(returns=rewards, lengths = lengths, custom_metric= custom_metric)


            for policy in [DistilledPolicy,]:
                if (tasks[0], policy) in seed_result:
                    rewards = np.array(
                        [sum([result[seed][(task, policy)].returns[result[seed]['best_iter']] for task in tasks])
                         for seed in result['seeds']])
                    lengths = np.array(
                        [sum([result[seed][(task, policy)].lengths[result[seed]['best_iter']] for task in tasks])
                         for seed in result['seeds']])
                    custom_metric = np.array(
                        [sum([result[seed][(task, policy)].returns[result[seed]['best_iter']] for task in tasks])
                         for seed in result['seeds']])
                    temp[('total', policy)] = evaluation(returns=rewards, lengths=lengths, custom_metric=custom_metric)

            for k in ['within', 'between', 'total']:
                rewards = np.array(
                    [result[seed][k].returns[result[seed]['best_iter']] for seed in result['seeds']])
                lengths = np.array(
                    [result[seed][k].lengths[result[seed]['best_iter']] for seed in result['seeds']])
                custom_metric = np.array(
                    [result[seed][k].custom_metric[result[seed]['best_iter']] for seed in
                     result['seeds']])

                temp[k] = evaluation(returns= rewards, lengths=lengths, custom_metric=custom_metric)


            result['summary'] = temp

        return

    def best_perf_by_group(self, setup, ctx_vis = True):
        # single_learners = [0, 2, 6]
        # distill_learners = [0, 2, 6, 8, 10, 14, 9, 11, 15]
        # naive_learners = [1, 3, 7]
        exp_set = self.experiment_breakdown[setup]
        exp_set[('best_exp', ctx_vis)] = dict()
        groups = [(ctx_vis, i) for i in naive_learners + distill_learners if (ctx_vis, i) in exp_set]
        if len(groups) == 0:
            return None, None, None
        for group in groups:
            experiments = exp_set[group]['experiments']
            average_rewards = [np.mean(self.results[exp_id]['summary']['within'].returns) for exp_id in experiments]
            self.experiment_breakdown[setup][group]['best_exp'] = experiments[np.argmax(average_rewards)]

        experiments = [self.experiment_breakdown[setup][group]['best_exp'] for group in groups]
        for p, t in itertools.product(exp_set['policies'], ['total', ]+ exp_set['tasks']):
            vals = [np.mean(self.results[exp_id]['summary'][(t, p)].returns) if len(self.results[exp_id]['policies'])!= 1 else
                    np.mean(self.results[exp_id]['summary'][(t, self.results[exp_id]['policies'][0])].returns)
                    for exp_id in experiments]
            exp_set[('best_exp', ctx_vis)][(t, p)] = experiments[np.argmax(vals)]




        for k in [ 'within', 'between', 'total']:
            vals = [np.mean(self.results[exp_id]['summary'][k].returns) for exp_id in experiments]
            exp_set[('best_exp', ctx_vis)][k] = experiments[np.argmax(vals)]

        p = DistilledPolicy
        experiments  = [exp_id for exp_id in experiments if self.results[exp_id]['group_id'][1] not in naive_learners]
        for t in  ['total', ]+exp_set['tasks']:
            vals = [np.mean(self.results[exp_id]['summary'][(t, p)].returns) if len(self.results[exp_id]['policies'])!= 1 else
                    np.mean(self.results[exp_id]['summary'][(t, self.results[exp_id]['policies'][0])].returns)
                    for exp_id in experiments]
            exp_set[('best_exp', ctx_vis)][(t, p)] = experiments[np.argmax(vals)]

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

        output +='\hline\n'+'&'.join(['ID', 'Name', 'Curriculum', 'Parameters', 'Runs',]) + '\\\\\n\hline \hline\n\endfirsthead\n'

        output += '\multicolumn{5}{c}{\\bfseries \\tablename\ \\thetable{} -- continued from previous page}\\\\\hline\n'
        output += '\hline\n'+'&'.join(['ID', 'Name', 'Curriculum', 'Parameters', 'Runs',]) + '\\\\\n\hline \hline\n\endhead\n'

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

                    row = [experiment['ID'], name_codes[experiment['group_id']] ,exp_id[3] , parameters, str(experiment['runs'])]
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

        output +="\hline\n"+ '&'.join([' ', '\multicolumn{2}{|c|}{Reward Within}',
                            '\multicolumn{2}{c|}{Reward Between}',
                            '\multicolumn{2}{c|}{Total Reward}']) + '\\\\\n'
        output += '&'.join(['Algorithm', ] + ['mean', 'se' ] * 3 ) + '\\\\\hline\n\endfirsthead\n'

        output += '\multicolumn{7}{c}{\\bfseries \\tablename\ \\thetable{} -- continued from previous page}\\\\\hline\n'
        output += "\hline\n" + '&'.join([' ', '\multicolumn{2}{|c|}{Reward Within}',
                                         '\multicolumn{2}{c|}{Reward Between}',
                                         '\multicolumn{2}{c|}{Total Reward}']) + '\\\\\n'
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
                    for k in ['within', 'between', 'total']:
                        returns = experiment['summary'][k].returns
                        row += [str(round(np.mean(returns), 2)), str(round(np.std(returns)/n**.5, 3))]
                    total_results.append('&'.join(row) + '\\\\\n')


        total_results.append('\hline\n\\end{longtable}')
        # total_results[-1] = total_results[-1][:-3]
        return total_results

    def task_results_txt(self, setup, ctx_vis = True):
        BlockPerRow = 4

        if setup in self.experiment_breakdown:
            exp_set = self.experiment_breakdown[setup]
        else:
            raise Exception(setup +'not in experiment')
        # for setup, exp_set in self.experiment_breakdown.items():
        policies = list(exp_set['policies'])
        policies.sort()
        policies_ = [DistilledPolicy,] + policies

        train_tasks = exp_set['tasks']


        learner_num = len(train_tasks) +1
        policies.sort()
        output = list()
        if learner_num in [3, 5, 6]:
            BlockPerRow =3

        FullLines = learner_num// BlockPerRow
        rest = learner_num % BlockPerRow

        ctx_str= 'visible context ' if ctx_vis else 'hidden context '
        output.append('\\begin{longtable}{|c|'+'|c'* (BlockPerRow*2)+ '|}\n\\caption{'+setup+', ' +ctx_str +' task policy rewards}\n\\label{tab:S'+setup.replace('Set', '') +'TP}\\\\\n\hline\n')


        output.append('&'.join(['Algorithm', ] + ['mean', 'se'] * BlockPerRow) + '\\\\\n \hline\endfirsthead\n')


        output.append('\multicolumn{'+str(BlockPerRow*2 +1 ) +'}{c}{\\bfseries \\tablename\ \\thetable{} -- continued from previous page}\\\\\hline\n')


        output.append('&'.join(['Algorithm', ] + ['mean', 'se'] * BlockPerRow) + '\\\\\n \hline\endhead\n')

        output.append('\hline \multicolumn{'+str(BlockPerRow*2 +1 )+'}{|r|}{{Continued on next page}} \\\\ \hline\n\endfoot\hline\hline\n\endlastfoot\n')

        ### totals
        for  tsk in ['total rewards', ]:
            output.append(
                '\\nopagebreak[4]\hline\n\multicolumn{' + str(2 * BlockPerRow + 1) + '}{|c|}{' + tsk.replace('_',
                                                                                                        ' ') + '}\\\\\n\hline\n')


            for line_ctr in range(FullLines):
                pols = policies_[BlockPerRow*line_ctr: BlockPerRow*(line_ctr+1)]
                header = '\hline\n\\pagebreak[2]' + '&'.join(
                    [' ', '\multicolumn{2}{|c|}{' + pols[0].replace('_', ' ') + '}', ] + [
                        '\multicolumn{2}{c|}{' + p.replace('_', ' ') + '}' for p in pols[1:]]) + '\\\\\hline\n'
                output.append(header)
                # Multiagent baseline
                for i in naive_learners:

                    if (ctx_vis, i) in exp_set:
                        # experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )

                        for exp_id in exp_set[(ctx_vis, i)]['experiments']:
                            experiment = self.results[exp_id]
                            n = experiment['runs']
                            row = ['\\nopagebreak[4] ' + experiment['ID'], ]
                            # for tsk in train_tasks:
                            # row += [' ', ' ']
                            for pol in pols:
                                if pol == DistilledPolicy:
                                    row += [' ', ' ']
                                else:
                                    returns = sum([experiment['summary'][(t, pol)].returns for t in train_tasks])

                                    # experiment['summary'][('total', pol)] = returns
                                    row += [str(round(np.mean(returns), 2)),
                                            str(round(np.std(returns) / n ** .5, 3))]

                            output.append('&'.join(row) + '\\\\\n')

                # pagebreak = False
                for i in distill_learners:
                    pagebreak = i > 2
                    if (ctx_vis, i) in exp_set:

                        for exp_id in exp_set[(ctx_vis, i)]['experiments']:
                            experiment = self.results[exp_id]
                            n = experiment['runs']
                            if pagebreak:
                                row = ['\\pagebreak[1] ' + experiment['ID'], ]
                                pagebreak = False

                            else:
                                row = ['\\nopagebreak[4] ' + experiment['ID'], ]



                            for pol in pols:
                                if (tsk, pol) in experiment['summary']:

                                    returns = sum([experiment['summary'][(t, pol)].returns for t in train_tasks])
                                else:
                                    returns = sum([experiment['summary'][(t, DistilledPolicy)].returns for t in train_tasks])

                                # experiment['summary'][('total', pol)] = returns

                                row += [str(round(np.mean(returns), 2)),
                                        str(round(np.std(returns) / n ** .5, 3))]

                            output.append('&'.join(row) + '\\\\\n')


            if rest>0:
                filler = '&\multicolumn{' + str(2 * (BlockPerRow - rest)) + '}{c|}{ }'

                pols = policies_[-rest:]
                header = '\hline\n ' + '\\pagebreak[2] ' + '&'.join(
                    [' ', '\multicolumn{2}{|c|}{' + pols[0].replace('_', ' ') + '}', ] + [
                        '\multicolumn{2}{c|}{' + p.replace('_', ' ') + '}' for p in pols[1:]]) + filler + '\\\\\hline\n'
                output.append(header)
                # Multiagent baseline
                for i in naive_learners:

                    if (ctx_vis, i) in exp_set:
                        # experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )

                        for exp_id in exp_set[(ctx_vis, i)]['experiments']:
                            experiment = self.results[exp_id]
                            n = experiment['runs']
                            row = ['\\nopagebreak[4] ' + experiment['ID'], ]
                            # for tsk in train_tasks:

                            for pol in pols:
                                if pol == DistilledPolicy:
                                    row+= [" ", ' ']
                                    continue
                                returns = sum([experiment['summary'][(t, pol)].returns for t in train_tasks])
                                # experiment['summary'][('total', pol)] = returns

                                row += [str(round(np.mean(returns), 2)),
                                        str(round(np.std(returns) / n ** .5, 3))]

                            output.append('&'.join(row) + filler+'\\\\\n')

                # pagebreak = False
                for i in distill_learners:
                    pagebreak = i > 2
                    if (ctx_vis, i) in exp_set:
                        # experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )

                        for exp_id in exp_set[(ctx_vis, i)]['experiments']:
                            experiment = self.results[exp_id]
                            n = experiment['runs']
                            if pagebreak:
                                row = ['\\pagebreak[2] ' + experiment['ID'], ]
                                pagebreak = False

                            else:
                                row = ['\\nopagebreak[4] ' + experiment['ID'], ]


                            for pol in pols:
                                if (tsk, pol) in experiment['summary']:

                                    returns = sum([experiment['summary'][(t, pol)].returns for t in train_tasks])
                                else:
                                    returns = sum(
                                        [experiment['summary'][(t, DistilledPolicy)].returns for t in train_tasks])

                                # experiment['summary'][('total', pol)] = evaluation(returns = returns, lengths= None, custom_metric= None)

                                row += [str(round(np.mean(returns), 2)),
                                        str(round(np.std(returns) / n ** .5, 3))]

                            output.append('&'.join(row)  + filler+ '\\\\\n')

        ### tasks


        for tsk in train_tasks:
            output.append('\\pagebreak[3]\hline\n\multicolumn{' + str(2*BlockPerRow  +1) + '}{|c|}{' + tsk.replace('_', ' ') + '}\\\\\n\hline\n')
            for line_ctr in range(FullLines):
                pols = policies_[BlockPerRow * line_ctr: BlockPerRow * (line_ctr + 1)]
        # Multiagent baseline
                header = '\hline\n\\pagebreak[2]' + '&'.join(
                    [' ', '\multicolumn{2}{|c|}{' + pols[0].replace('_', ' ') + '}', ] + [
                        '\multicolumn{2}{c|}{' + p.replace('_', ' ') + '}' for p in pols[1:]]) + '\\\\\hline\n'
                output.append(header)
                for i in naive_learners:

                    if (ctx_vis, i) in exp_set:
                        # experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )

                        for exp_id in exp_set[(ctx_vis, i)]['experiments']:
                            experiment = self.results[exp_id]
                            n = experiment['runs']
                            row = ['\\nopagebreak[4] '+experiment['ID'], ]
                            # for tsk in train_tasks:
                            for pol in pols:
                                if pol == DistilledPolicy:
                                    row += [' ', ' ']
                                    continue

                                returns = experiment['summary'][(tsk, pol)].returns
                                row += [str(round(np.mean(returns), 2)),
                                        str(round(np.std(returns)/n**.5, 3))]

                            output.append( '&'.join(row) + '\\\\\n')

                # pagebreak = False
                for i in distill_learners:
                    pagebreak = i>2
                    if (ctx_vis, i) in exp_set:
                        # experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )

                        for exp_id in exp_set[(ctx_vis, i)]['experiments']:
                            experiment = self.results[exp_id]
                            n = experiment['runs']
                            if pagebreak:
                                row = ['\\pagebreak[1] ' + experiment['ID'], ]
                                pagebreak = False

                            else:
                                row = ['\\nopagebreak[4] '+experiment['ID'], ]
                            # for tsk in train_tasks:

                            # returns = experiment['summary'][(tsk, DistilledPolicy)].returns
                            #
                            # row += [str(round(np.mean(returns), 2)),
                            #             str(round(np.std(returns) / n ** .5, 3))]

                            for pol in pols:
                                if (tsk, pol) in experiment['summary']:

                                    returns = experiment['summary'][(tsk, pol)].returns
                                else:
                                    returns = experiment['summary'][(tsk, DistilledPolicy)].returns
                                row += [str(round(np.mean(returns), 2)),
                                        str(round(np.std(returns) / n ** .5, 3))]

                            output.append('&'.join(row) + '\\\\\n')

            if rest>0:
                filler = '&\multicolumn{' + str(2 * (BlockPerRow - rest)) + '}{c|}{ }'

                pols = policies_[-rest:]
                header = '\hline\n ' + '\\pagebreak[2] ' + '&'.join(
                    [' ', '\multicolumn{2}{|c|}{' + pols[0].replace('_', ' ') + '}', ] + [
                        '\multicolumn{2}{c|}{' + p.replace('_', ' ') + '}' for p in
                        pols[1:]]) + filler + '\\\\\hline\n'
                output.append(header)
                for i in naive_learners:

                    if (ctx_vis, i) in exp_set:
                        # experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )

                        for exp_id in exp_set[(ctx_vis, i)]['experiments']:
                            experiment = self.results[exp_id]
                            n = experiment['runs']
                            row = ['\\nopagebreak[4] ' + experiment['ID'], ]
                            # for tsk in train_tasks:
                            for pol in pols:
                                if pol == DistilledPolicy:
                                    row+=[' ', ' ']
                                    continue

                                returns = sum([experiment['summary'][(t, pol)].returns for t in train_tasks])

                                row += [str(round(np.mean(returns), 2)),
                                        str(round(np.std(returns) / n ** .5, 3))]

                            output.append('&'.join(row) +  filler+'\\\\\n')

                # pagebreak = False
                for i in distill_learners:
                    pagebreak = i > 2
                    if (ctx_vis, i) in exp_set:
                        # experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )

                        for exp_id in exp_set[(ctx_vis, i)]['experiments']:
                            experiment = self.results[exp_id]
                            n = experiment['runs']
                            if pagebreak:
                                row = ['\\pagebreak[2] ' + experiment['ID'], ]
                                pagebreak = False

                            else:
                                row = ['\\nopagebreak[4] ' + experiment['ID'], ]
                            # for tsk in train_tasks:

                            # returns = sum([experiment['summary'][(t, DistilledPolicy)].returns for t in train_tasks])
                            #
                            # row += [str(round(np.mean(returns), 2)),
                            #         str(round(np.std(returns) / n ** .5, 3))]

                            for pol in pols:
                                if (tsk, pol) in experiment['summary']:

                                    returns = sum([experiment['summary'][(t, pol)].returns for t in train_tasks])
                                else:
                                    returns = sum(
                                        [experiment['summary'][(t, DistilledPolicy)].returns for t in train_tasks])
                                row += [str(round(np.mean(returns), 2)),
                                        str(round(np.std(returns) / n ** .5, 3))]

                            output.append('&'.join(row) + filler + '\\\\\n')
        output.append('\hline\n\end{longtable}')

        return output


    def statistical_comp(self, setup, ctx_vis= True):
        BlockPerRow = 3
        ctx_str= 'visible context ' if ctx_vis else 'hidden context '

        # for setup,  in self.experiment_breakdown.items():
        exp_set = self.experiment_breakdown[setup]
        # best_performers, groups, best_by_group = self.best_perf_by_key(setup=setup, ctx_vis=ctx_vis)
        groups = [(ctx_vis, i) for i in (naive_learners+ distill_learners) if (ctx_vis, i) in exp_set]
        if len(groups) == 0:
            raise Exception('no experiment')
        result_tsk_pol = list()
        # exps = list()

        result_total = list()
        # print('Context Visible')
        result_total.append('\\begin{tabular}{|c|'+'|c'* 9 + '|}\n\hline\n')

        result_total.append( '&'.join(['Algorithm', '\multicolumn{3}{c|}{Reward Within}',
                           '\multicolumn{3}{c|}{Reward Between}',
                           '\multicolumn{3}{c|}{Total Rewards}']) + '\\\\\n')
        result_total.append('&'.join([' ', 'mean', 'se', 'p-value',
                            'mean', 'se', 'p-value',
                            'mean', 'se','p-value',]) + '\\\\\n\hline\n')

        best_experiments = exp_set[('best_exp', ctx_vis)]
        experiments = [exp_set[group]['best_exp'] for group in groups]
        for exp_id in experiments:
            experiment = self.results[exp_id]
            n = experiment['runs']
            row = [experiment['ID'], ]
            for k in ['within', 'between', 'total']:
                returns = experiment['summary'][k].returns
                # if setup=='Set0' and ctx_vis:
                #     print(exp_id,best_experiments[k])
                if exp_id == best_experiments[k]:

                    row += [GreenCell+str(round(np.mean(returns), 2)),
                            GreenCell+str(round(np.std(returns)/n**.5, 3)), ' ']

                else:
                    returns_best = self.results[best_experiments[k]]['summary'][k].returns
                    # vals_tst = group_best[k]
                    pvalue = ttest_ind(returns,
                              returns_best,
                              equal_var=False)[1]
                    if pvalue> 0.05 or np.isnan(pvalue):
                        sig= ''
                    else:
                        sig = RedCell

                    row += [ str(round(np.mean(returns), 2)),
                              str(round(np.std(returns)/ n**.5, 3)), sig+ str(round(pvalue, 3))]
            result_total.append('&'.join(row) + '\\\\\n')
        # result_total[-1] = result_total[-1][:-3]
        result_total.append('\\hline\n\\end{tabular}\n')
        policies = list(exp_set['policies'])
        policies.sort()
        tasks = ['total', ]+ exp_set['tasks']
        if len(tasks) in [2, 4]:
            BlockPerRow = 2

        FullLines = len(tasks)// BlockPerRow

        output = '\\begin{longtable}{' + '|c|'+'|c'* 3*BlockPerRow + '|}\n\caption{'+setup+', '+ctx_str+' task policy rewards significance test  }\n\label{tab:S'+ setup.replace('Set', '')+ 'TPtst}\\\\\n\hline\n '
        # output += '&'.join(['Algorithm', '\multicolumn{2}{c}{Reward Within}',
        #                    '\multicolumn{2}{c}{Reward Between}',
        #                    '\multicolumn{2}{c}{Total Reward}']) + '\\\\\n'
        output += '&'.join(['Algorithm',] + ['mean', 'se', 'p-value']*BlockPerRow) + '\\\\\hline\n\endfirsthead\n'

        output+= '\multicolumn{'+str(3*BlockPerRow +1)+'}{c}{\\bfseries \\tablename\ \\thetable{} -- continued from previous page}\\\\\hline\n'
        output += '&'.join(['Algorithm',] + ['mean', 'se', 'p-value']*BlockPerRow) + '\\\\\hline\n\endhead\n'
        output += '\hline \multicolumn{'+str(3*BlockPerRow+1)+'}{|r|}{{Continued on next page}} \\\\ \hline\n\endfoot\hline\hline\n\endlastfoot\n'
        output += '\hline\hline\n\multicolumn{'+str(3*BlockPerRow+1)+'}{|c|}{Distilled Policy}\\\\*\n'
        result_tsk_pol.append(output)
        # policy_task_best = best_performers['policy_task_rewards']
        policy = DistilledPolicy

        for i in range(FullLines):

            # for tsk in tasks[i * BlockPerRow: (i + 1) * BlockPerRow]:
            tsk = tasks[i * BlockPerRow: (i + 1) * BlockPerRow]
            header = '\hline\n\\pagebreak[1]'+'&'.join([' ', '\multicolumn{3}{|c|}{'+tsk[0].replace('_', ' ')+'}', ] +['\multicolumn{3}{c|}{'+t.replace('_', ' ')+'}' for t in tsk[1:]]) + '\\\\*\hline\n'
            result_tsk_pol.append(header)
            for exp_id in experiments:

                experiment = self.results[exp_id]
                if experiment['group_id'][1] in naive_learners:
                    continue
                row = ['\\nopagebreak ' + experiment['ID'], ]
                for t in tsk:
                    best_id = best_experiments[(t, policy)]



                    returns = experiment['summary'][(t, policy)].returns

                    if exp_id == best_id:
                        row += [GreenCell + str(round(np.mean(returns), 2)),
                                GreenCell + str(round(np.std(returns) / n ** .5, 3)), ' ']
                    else:
                        best_exp = self.results[best_id]
                        returns_best = best_exp['summary'][(t, policy)].returns
                        # vals_tst = group_best[k]
                        pvalue = ttest_ind(returns,
                                           returns_best,
                                           equal_var=False)[1]
                        if np.isnan(pvalue):
                            pvalue = 1
                        if pvalue > 0.05 :
                            sig = ''
                        else:
                            sig = RedCell

                        row += [str(round(np.mean(returns), 2)),
                                str(round(np.std(returns) / n ** .5, 3)), sig + str(round(pvalue, 3))]

                result_tsk_pol.append('&'.join(row) +'\\\\\n')

        rest =  len(tasks)% BlockPerRow
        if rest >0 :

            filler = '&\multicolumn{'+str(3*(BlockPerRow-rest)) +'}{c|}{ }'
            tsk =tasks[-rest:]
            header = '\hline\n ' + '\\nopagebreak[4] ' + '&'.join([' ','\multicolumn{3}{|c|}{' + tsk[0].replace('_', ' ') + '}', ] + [
                '\multicolumn{3}{c|}{' + t.replace('_', ' ') + '}' for t in tsk[1:]]  ) + filler + '\\\\*\hline\n'
            result_tsk_pol.append(header)
            for exp_id in experiments:

                experiment = self.results[exp_id]
                if experiment['group_id'][1] in naive_learners:
                    continue
                row = ['\\nopagebreak[4] ' + experiment['ID'], ]
                for t in tsk:
                    best_id = best_experiments[(t, policy)]

                    returns = experiment['summary'][(t, policy)].returns

                    if exp_id == best_id:
                        row += [GreenCell + str(round(np.mean(returns), 2)),
                                GreenCell + str(round(np.std(returns) / n ** .5, 3)), ' ']
                    else:
                        best_exp = self.results[best_id]
                        returns_best = best_exp['summary'][(t, policy)].returns
                        # vals_tst = group_best[k]
                        pvalue = ttest_ind(returns,
                                           returns_best,
                                           equal_var=False)[1]
                        if np.isnan(pvalue):
                            pvalue = 1
                        if pvalue > 0.05:
                            sig = ''
                        else:
                            sig = RedCell

                        row += [str(round(np.mean(returns), 2)),
                                str(round(np.std(returns) / n ** .5, 3)), sig + str(round(pvalue, 3))]

                result_tsk_pol.append('&'.join(row) + filler +'\\\\\n')
        for policy in policies:
            result_tsk_pol.append( '\hline\hline\n \\pagebreak[1]  \multicolumn{' + str(3 * BlockPerRow + 1) + '}{|c|}{'+policy.replace('_', ' ' )+'}\\\\\n')
            for i in range(FullLines):

                # for tsk in tasks[i * BlockPerRow: (i + 1) * BlockPerRow]:
                tsk = tasks[i * BlockPerRow: (i + 1) * BlockPerRow]
                header = '\hline\n'+  '\\nopagebreak[4] ' + '&'.join([' ', '\multicolumn{3}{|c|}{' + tsk[0].replace('_', ' ') + '}', ] + [
                    '\multicolumn{3}{c|}{' + t.replace('_', ' ') + '}' for t in tsk[1:]]) + '\\\\\hline\n'
                result_tsk_pol.append(header)
                for exp_id in experiments:

                    experiment = self.results[exp_id]


                    row = ['\\nopagebreak[4] ' + experiment['ID'], ]
                    for t in tsk:

                        best_id = best_experiments[(t, policy)]
                        if experiment['group_id'][1] in single_learners:
                            returns = experiment['summary'][(t, DistilledPolicy)].returns
                        else:
                            returns = experiment['summary'][(t, policy)].returns

                        if exp_id == best_id:
                            row += [GreenCell + str(round(np.mean(returns), 2)),
                                    GreenCell + str(round(np.std(returns) / n ** .5, 3)), ' ']
                        else:
                            best_exp = self.results[best_id]
                            if best_exp['group_id'][1] in single_learners:
                                returns_best = best_exp['summary'][(t, DistilledPolicy)].returns
                            else:
                                returns_best = best_exp['summary'][(t, policy)].returns


                            # returns_best = best_exp['summary'].get((t, policy) , best_exp['summary'][(t, DistilledPolicy)]).returns
                            # vals_tst = group_best[k]
                            pvalue = ttest_ind(returns,
                                               returns_best,
                                               equal_var=False)[1]
                            if np.isnan(pvalue):
                                pvalue = 1
                            if pvalue > 0.05:
                                sig = ''
                            else:
                                sig = RedCell

                            row += [str(round(np.mean(returns), 2)),
                                    str(round(np.std(returns) / n ** .5, 3)), sig + str(round(pvalue, 3))]

                    result_tsk_pol.append('&'.join(row) + '\\\\\n')

            rest = len(tasks) % BlockPerRow
            if rest>0:


                filler = '&\multicolumn{' + str(3 * (BlockPerRow - rest)) + '}{c|}{ }'
                tsk = tasks[-rest:]
                header =  '\hline\n'+ '\\nopagebreak[4] ' + '&'.join([' ', '\multicolumn{3}{|c|}{' + tsk[0].replace('_', ' ') + '}', ] + [
                    '\multicolumn{3}{c|}{' + t.replace('_', ' ') + '}' for t in tsk[1:]]) + filler + '\\\\\hline\n'
                result_tsk_pol.append(header)
                for exp_id in experiments:

                    experiment = self.results[exp_id]

                    row = ['\\nopagebreak [4]' +experiment['ID'], ]
                    for t in tsk:
                        best_id = best_experiments[(t, policy)]

                        if experiment['group_id'][1] in single_learners:
                            returns = experiment['summary'][(t, DistilledPolicy)].returns
                        else:
                            returns = experiment['summary'][(t, policy)].returns
                        # returns = experiment['summary'][(t, policy)].returns

                        if exp_id == best_id:
                            row += [GreenCell + str(round(np.mean(returns), 2)),
                                    GreenCell + str(round(np.std(returns) / n ** .5, 3)), ' ']
                        else:
                            best_exp = self.results[best_id]
                            if (t, policy) in best_exp['summary']:
                                returns_best = best_exp['summary'][(t, policy)].returns
                            else:
                                returns_best = best_exp['summary'][(t, DistilledPolicy)].returns
                            # vals_tst = group_best[k]
                            pvalue = ttest_ind(returns,
                                               returns_best,
                                               equal_var=False)[1]
                            if np.isnan(pvalue):
                                pvalue = 1
                            if pvalue > 0.05:
                                sig = ''
                            else:
                                sig = RedCell

                            row += [str(round(np.mean(returns), 2)),
                                    str(round(np.std(returns) / n ** .5, 3)), sig + str(round(pvalue, 3))]

                    result_tsk_pol.append('&'.join(row) + filler + '\\\\\n')
        result_tsk_pol.append('\hline\n\end{longtable}')

        return result_total, result_tsk_pol

    def report(self):
        for setup in self.experiment_breakdown.keys():
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

                task_results = self.task_results_txt( setup,  ctx_vis = ctx_vis)

                try:
                    with open(os.path.join(self.log_dir, 'report' , setup , folder ,'policy_task_rewards.txt'), "w") as f:

                        for l in task_results:
                            f.write(l)
                except FileNotFoundError as e:
                    print(f'{setup} folder not found')

                total, policy_task = self.statistical_comp(setup, ctx_vis=ctx_vis)
                # try:
                #     total, policy_task = self.statistical_comp( setup, ctx_vis= ctx_vis)
                # except Exception as e:
                #     print('skipping empty ' + setup + ' with ctx_vis = ' + str(ctx_vis) + '\n', e)
                #     continue

                try:
                    with open(os.path.join(self.log_dir, 'report'  , setup, folder ,'total_rewards_ttest.txt'), "w") as f:
                        for l in total:
                            f.write(l)

                    with open(os.path.join(self.log_dir, 'report' , setup, folder  ,'policy_task_rewards_ttest.txt'), "w") as f:
                        for l in policy_task:
                            f.write(l)
                except FileNotFoundError as e:
                    print(f'{setup} folder not found')

    def extr_task_results_txt(self, extr_result, ctx_vis=True):
        # self.rewards_table = dict()
        setup = extr_result['setup']
        exp_set = self.experiment_breakdown[setup]

        policies = list(exp_set['policies'])
        tasks = extr_result['tasks']

        learner_num = len(exp_set['tasks']) + 1
        policies.sort()
        output = list()

        ctx_str = 'visible context ' if ctx_vis else 'hidden context '
        output.append('\\begin{longtable}{|c|' + '|c' * (
                    learner_num * 2) + '|}\n\\caption{' + setup + ', ' + ctx_str + ' evaluation-tasks policy rewards breakdown}\n\\label{tab:S' + setup.replace(
            'Set', '') + 'ETP}\\\\\n\hline\n')

        output.append('&'.join(
            [' ', '\multicolumn{2}{|c|}{distilled policy}', ] + ['\multicolumn{2}{c|}{' + pol.replace('_', ' ') + '}'
                                                                 for pol in policies]) + '\\\\\n')
        output.append('&'.join(['Algorithm', ] + ['mean', 'se'] * learner_num) + '\\\\\n \hline\endfirsthead\n')

        output.append('\multicolumn{' + str(
            learner_num * 2 + 1) + '}{c}{\\bfseries \\tablename\ \\thetable{} -- continued from previous page}\\\\\hline\n')
        output.append('&'.join(
            [' ', '\multicolumn{2}{|c|}{distilled policy}', ] + ['\multicolumn{2}{c|}{' + pol.replace('_', ' ') + '}'
                                                                 for pol in policies]) + '\\\\\n')
        output.append('&'.join(['Algorithm', ] + ['mean', 'se'] * learner_num) + '\\\\\n \hline\endhead\n')

        output.append('\hline \multicolumn{' + str(
            learner_num * 2 + 1) + '}{|r|}{{Continued on next page}} \\\\ \hline\n\endfoot\hline\hline\n\endlastfoot\n')



        ### tasks
        for tsk in tasks:
            output.append(
                '\\pagebreak[1]\hline\n\multicolumn{' + str(2 * learner_num + 1) + '}{|c|}{' + tsk.replace('_',
                                                                                                           ' ') + '}\\\\\n\hline\n')

            # Multiagent baseline
            for i in naive_learners:

                if (ctx_vis, i) in exp_set:
                    # experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )

                    for exp_id in exp_set[(ctx_vis, i)]['experiments']:
                        experiment = self.results[exp_id]
                        n = experiment['runs']
                        row = ['\\nopagebreak[4] ' + experiment['ID'], ]
                        # for tsk in train_tasks:
                        row += [' ', ' ']
                        for pol in policies:
                            returns = extr_result[exp_id]['summary'][(tsk, pol)].returns
                            row += [str(round(np.mean(returns), 2)),
                                    str(round(np.std(returns) / n ** .5, 3))]

                        output.append('&'.join(row) + '\\\\\n')

            # pagebreak = False
            for i in distill_learners:
                pagebreak = i > 2
                if (ctx_vis, i) in exp_set:
                    # experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )

                    for exp_id in exp_set[(ctx_vis, i)]['experiments']:
                        experiment = self.results[exp_id]
                        n = experiment['runs']
                        if pagebreak:
                            row = ['\\pagebreak[2] ' + experiment['ID'], ]
                            pagebreak = False

                        else:
                            row = ['\\nopagebreak[4] ' + experiment['ID'], ]
                        # for tsk in train_tasks:
                        if  (tsk, DistilledPolicy) in extr_result[exp_id]['summary']:

                            returns = extr_result[exp_id]['summary'][(tsk, DistilledPolicy)].returns
                        else:
                            print(exp_id)

                        row += [str(round(np.mean(returns), 2)),
                                str(round(np.std(returns) / n ** .5, 3))]

                        for pol in policies:
                            if (tsk, pol) in experiment['summary']:

                                returns = extr_result[exp_id]['summary'][(tsk, pol)].returns
                            else:
                                returns = extr_result[exp_id]['summary'][(tsk, DistilledPolicy)].returns
                            row += [str(round(np.mean(returns), 2)),
                                    str(round(np.std(returns) / n ** .5, 3))]

                        output.append('&'.join(row) + '\\\\\n')
        output.append('\hline\n\end{longtable}')

        return output


    def extr_statistical_comp(self, extr_result, ctx_vis= True):
        BlockPerRow = 3
        ctx_str= 'visible context ' if ctx_vis else 'hidden context '
        setup = extr_result['setup']
        # for setup,  in self.experiment_breakdown.items():
        exp_set = self.experiment_breakdown[setup]
        tasks = extr_result['tasks']
        # best_performers, groups, best_by_group = self.best_perf_by_key(setup=setup, ctx_vis=ctx_vis)
        groups = [(ctx_vis, i) for i in naive_learners + distill_learners if (ctx_vis, i) in exp_set]

        if len(groups) == 0:
            raise Exception('no experiment')

        # best_experiments = exp_set[('best_exp', ctx_vis)]
        experiments = [exp_set[group]['best_exp'] for group in groups]

        result_tsk_pol = list()

        policies = list(exp_set['policies'])
        policies.sort()
        if len(policies) in [1,3]:
            BlockPerRow = 2

        FullLines = (len(policies)+1)// BlockPerRow

        output = '\\begin{longtable}{' + '|c|'+'|c'* 3*BlockPerRow + '|}\n\caption{'+setup+', '+ctx_str+' evaluation task policy rewards significance test  }\n\label{tab:S'+ setup.replace('Set', '')+ 'ETPtst}\\\\\n\hline\n '
        # output += '&'.join(['Algorithm', '\multicolumn{2}{c}{Reward Within}',
        #                    '\multicolumn{2}{c}{Reward Between}',
        #                    '\multicolumn{2}{c}{Total Reward}']) + '\\\\\n'
        output += '&'.join(['Algorithm',] + ['mean', 'se', 'p-value']*BlockPerRow) + '\\\\\hline\n\endfirsthead\n'

        output += '\multicolumn{'+str(3*BlockPerRow +1)+'}{c}{\\bfseries \\tablename\ \\thetable{} -- continued from previous page}\\\\\hline\n'
        output += '&'.join(['Algorithm',] + ['mean', 'se', 'p-value']*BlockPerRow) + '\\\\\hline\n\endhead\n'
        output += '\hline \multicolumn{'+str(3*BlockPerRow+1)+'}{|r|}{{Continued on next page}} \\\\ \hline\n\endfoot\hline\hline\n\endlastfoot\n'
        result_tsk_pol.append(output)
        policies_ = [DistilledPolicy, ] + policies

        best_exps_idx = {(tsk, pol): np.argmax([np.mean(extr_result[exp_id]['summary'].get((tsk, pol), extr_result[exp_id]['summary'].get((tsk, DistilledPolicy), -1000)).returns)
                                            for exp_id in experiments if (pol!= DistilledPolicy or (self.results[exp_id]['group_id'][1] not in  naive_learners))])
                     for tsk ,pol in  itertools.product(tasks, policies_)}

        best_exps ={k: [exp_id for exp_id in experiments if (k[1]!=DistilledPolicy or (self.results[exp_id]['group_id'][1] not in naive_learners))][idx] for k, idx in best_exps_idx.items()}


        for tsk in tasks:
            result_tsk_pol.append( '\hline\hline\n\multicolumn{'+str(3*BlockPerRow+1)+'}{|c|}{'+tsk.replace('_', ' ')+'}\\\\*\n')

        # policy_task_best = best_performers['policy_task_rewards']
        # policy =

            for i in range(FullLines):

                # for tsk in tasks[i * BlockPerRow: (i + 1) * BlockPerRow]:
                pols = policies_[i * BlockPerRow: (i + 1) * BlockPerRow]
                header = '\hline\n\\pagebreak[1]'+'&'.join([' ', '\multicolumn{3}{|c|}{'+pols[0].replace('_', ' ')+'}', ] +['\multicolumn{3}{c|}{'+p.replace('_', ' ')+'}' for p in pols[1:]]) + '\\\\*\hline\n'
                result_tsk_pol.append(header)
                for exp_id in experiments:

                    experiment = self.results[exp_id]
                    n = experiment['runs']

                    row = ['\\nopagebreak ' + experiment['ID'], ]
                    for pol in pols:

                        best_id = best_exps[(tsk, pol)]
                        if experiment['group_id'][1] in naive_learners and pol == DistilledPolicy:
                            row += [' ', ' ', ' ']
                            continue

                        if (tsk, pol) in extr_result[exp_id]['summary']:
                            returns = extr_result[exp_id]['summary'][(tsk, pol)].returns
                        else:
                            returns = extr_result[exp_id]['summary'][(tsk, DistilledPolicy)].returns


                        if exp_id == best_id:
                            row += [GreenCell + str(round(np.mean(returns), 2)),
                                    GreenCell + str(round(np.std(returns) / n ** .5, 3)), ' ']
                        else:
                            best_exp = extr_result[best_id]
                            if (tsk, pol) in best_exp['summary']:
                                returns_best = best_exp['summary'][(tsk, pol) ].returns
                            else:
                                returns_best = best_exp['summary'][(tsk, DistilledPolicy)].returns
                            # vals_tst = group_best[k]
                            pvalue = ttest_ind(returns,
                                               returns_best,
                                               equal_var=False)[1]
                            if np.isnan(pvalue):
                                pvalue = 1
                            if pvalue > 0.05 :
                                sig = ''
                            else:
                                sig = RedCell

                            row += [str(round(np.mean(returns), 2)),
                                    str(round(np.std(returns) / n ** .5, 3)), sig + str(round(pvalue, 3))]

                    result_tsk_pol.append('&'.join(row) +'\\\\\n')

            rest =  len(policies_)% BlockPerRow
            if rest >0 :

                filler = '&\multicolumn{'+str(3*(BlockPerRow-rest)) +'}{c|}{ }'
                pols =policies_[-rest:]
                header = '\hline\n ' + '\\nopagebreak[4] ' + '&'.join([' ','\multicolumn{3}{|c|}{' + pols[0].replace('_', ' ') + '}', ] + [
                    '\multicolumn{3}{c|}{' + p.replace('_', ' ') + '}' for p in pols[1:]]  ) + filler + '\\\\\hline\n'
                result_tsk_pol.append(header)
                for exp_id in experiments:

                    experiment = self.results[exp_id]
                    n = experiment['runs']

                    row = ['\\nopagebreak[4] ' + experiment['ID'], ]
                    for pol in pols:
                        best_id = best_exps[(tsk, pol)]
                        if experiment['group_id'][1] in naive_learners and pol == DistilledPolicy:
                            row +=[' ', ' ', ' ']
                            continue

                        if (tsk, pol) in extr_result[exp_id]['summary']:
                            returns = extr_result[exp_id]['summary'][(tsk, pol)].returns
                        else:
                            returns = extr_result[exp_id]['summary'][(tsk, DistilledPolicy)].returns

                        if exp_id == best_id:
                            row += [GreenCell + str(round(np.mean(returns), 2)),
                                    GreenCell + str(round(np.std(returns) / n ** .5, 3)), ' ']
                        else:
                            best_exp = extr_result[best_id]
                            if (tsk, pol) in best_exp['summary']:
                                returns_best = best_exp['summary'][(tsk, pol)].returns
                            else:
                                returns_best = best_exp['summary'][(tsk, DistilledPolicy)].returns
                            # vals_tst = group_best[k]

                            # vals_tst = group_best[k]
                            pvalue = ttest_ind(returns,
                                               returns_best,
                                               equal_var=False)[1]
                            if np.isnan(pvalue):
                                pvalue = 1
                            if pvalue > 0.05:
                                sig = ''
                            else:
                                sig = RedCell

                            row += [str(round(np.mean(returns), 2)),
                                    str(round(np.std(returns) / n ** .5, 3)), sig + str(round(pvalue, 3))]

                    result_tsk_pol.append('&'.join(row) + filler +'\\\\\n')

        result_tsk_pol.append('\hline\n\end{longtable}')

        return  result_tsk_pol


    def report_extr_eval(self, extr_results):
        # for setup in self.experiment_breakdown.keys():
        setup = extr_results['setup']
        for ctx_vis in [True, False]:
            folder = 'ctx_vis' if ctx_vis else 'ctx_hid'


            task_results = self.extr_task_results_txt( extr_result=extr_results,  ctx_vis = ctx_vis)

            try:
                with open(os.path.join(self.log_dir, 'report' , setup , folder ,'eval_tasks_policy_task_rewards.txt'), "w") as f:

                    for l in task_results:
                        f.write(l)
            except FileNotFoundError as e:
                print(f'{setup} folder not found')
                continue

            policy_task = self.extr_statistical_comp(extr_result=extr_results, ctx_vis=ctx_vis)


            try:

                with open(os.path.join(self.log_dir, 'report' , setup, folder  ,'eval_task_policy_rewards_ttest.txt'), "w") as f:
                    for l in policy_task:
                        f.write(l)
            except FileNotFoundError as e:
                print(f'{setup} folder not found')


