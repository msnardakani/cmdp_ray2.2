import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import json
# import pandas as pd
import random
import itertools
import pickle


def exp_id2group(exp_id):  # baseline, default, subgroup)
    visibility, experiment_set, learner_group, curriculum, name, parameters = exp_id
    baseline = 'baseline' in learner_group
    if baseline:
        subgroup = 'Central' in name
    else:
        subgroup = 'task_aug' in learner_group
    cl = 'default' in curriculum
    idx = 7 - baseline * 4 - cl * 2 - subgroup
    return (baseline, cl, subgroup), ('vis' in visibility, idx)

class performance:
    def __init__(self, mean, std,n):
        self.mean = mean
        self.std = std
        self.n = n

class ExperimentAnalysis:
    # (visible ,baseline, default, subgroup)
    sorted_idx = {0: 'B1',
                  2: 'B1S',
                  1: 'B0',
                  3: 'B0S',
                  4: 'D0',
                  5: 'D1',
                  6: 'D0S',
                  7: 'D1S'}

    name_codes = {(True, 0): 'PPO Central', # ( baseline, default, Central)
                         (True, 1): 'PPO', # (baseline, default, MA)
                         (True, 3): 'PPO',  # ( baseline, self_paced, MA)
                         (True, 2): 'PPO Central',  # (baseline, self_paced, Central)
                         (True, 6): 'DistralPPO, Context Augmentation: Learner',  # (distral, self_paced, task_aug)
                         (True, 7): 'DistralPPO, Context Augmenttaion: Distillation',  # (distral, self_paced, distill_aug)
                         (True, 5): 'DistralPPO, Context Augmenttaion: Distillation',  # (distral, default, distill_aug)
                         (True, 4): 'DistralPPO, Context Augmenttaion: Learner',  # (distral, default, task_aug)

                         #ctx_hid
                         (False, 0): 'PPO Central',  # (baseline, default, Central)
                         (False, 1): 'PPO',  # (baseline, default, MA)
                         (False, 3): 'PPO',  # (baseline, self_paced, MA)
                         (False, 2): 'PPO Central',  # (baseline, self_paced, Central)
                         (False, 6): 'DistralPPO',  # (distral, self_paced, task_aug)
                         (False, 7): 'DistralPPO',  # (distral, self_paced, distill_aug)
                         (False, 5): 'DistralPPO',  # (distral, default, distill_aug)
                         (False, 4): 'DistralPPO',  # (distral, default, task_aug)
}
    experiment_groups = {
                         #ctx_vis
                         (True, 0): (True, True, True), # ( baseline, default, Central)
                         (True, 1): (True, True, False), # (baseline, default, MA)
                         (True, 3): (True, False, False),  # ( baseline, self_paced, MA)
                         (True, 2): (True, False, True),  # (baseline, self_paced, Central)
                         (True, 6): (False, False, True),  # (distral, self_paced, task_aug)
                         (True, 7): (False, False, False),  # (distral, self_paced, distill_aug)
                         (True, 5): (False, True, False),  # (distral, default, distill_aug)
                         (True, 4): (False, True, True),  # (distral, default, task_aug)

                         #ctx_hid
                         (False, 0): (True, True, True),  # (baseline, default, Central)
                         (False, 1): (True, True, False),  # (baseline, default, MA)
                         (False, 3): (True, False, False),  # (baseline, self_paced, MA)
                         (False, 2): (True, False, True),  # (baseline, self_paced, Central)
                         (False, 6): (False, False, True),  # (distral, self_paced, task_aug)
                         (False, 7): (False, False, False),  # (distral, self_paced, distill_aug)
                         (False, 5): (False, True, False),  # (distral, default, distill_aug)
                         (False, 4): (False, True, True),  # (distral, default, task_aug)

                         }
    def __init__(self, log_dir, env_map = False):
        # if not isinstance(version, list):
        #     version = [version]
        self.log_dir = log_dir
        self.env_map = env_map
        self.experiments = None
        self.summary = None
        self.experiment_breakdown = None
        self.best_performers = dict()
        # self.tasks = tasks
        # self.policies = policies

        # self.summary = self.experiment_summary()
        # self.best_performers = self.get_best_results()
    def analyse(self):
        self.experiments = self.get_experiments( self.env_map)
        self.summary, self.experiment_breakdown = self.experiment_results()

        for setup in self.experiment_breakdown.keys():
            best_performers, groups, best_by_group = self.best_perf_by_key(setup=setup, ctx_vis=True)
            if groups is not None:
                self.best_performers[setup] = {True: {'best_performers':best_performers, "groups": groups,'best_by_group': best_by_group},}

            best_performers, groups, best_by_group = self.best_perf_by_key(setup=setup, ctx_vis=False)
            if groups is not None:
                self.best_performers[setup].update( {
                    False: {'best_performers': best_performers, "groups": groups, 'best_by_group': best_by_group}, })
        return

    def get_experiments(self, env_map = False):
        #         name_map= {}
        #         param_map ={}
        # baselines ={}
        results = {}
        files_list = list()
        experiment_sets = list()
        for (dirpath, dirnames, filenames) in os.walk(self.log_dir):
            files_list += [os.path.join(dirpath, file) for file in filenames if ('result.json' in file

                                                                                             and not os.path.exists(
                        os.path.join(dirpath, 'error.txt')))]

        for f in files_list:
            setup = f.split('/')[-2]
            name = f.split('/')[-3]
            curriculum = f.split('/')[-4]
            learner_group = f.split('/')[-5]
            experiment_set = f.split('/')[-6]
            visibility = f.split('/')[-7]
            parameters = '_'.join(setup.split('_')[5:-2])
            # seed = int(re.search('seed=(.\d+?),', parameters).group(1))
            #             print(f)
            seed = int(parameters.split('seed=')[1])
            parameters = ','.join(parameters.split(',')[:-1])
            if env_map:
                if parameters.startswith('env_mapping='):
                    parameters = ','.join(parameters.split(',')[1:])
                else:
                    splt = parameters.split('env_mapping=')
                    parameters = ','.join([splt[0],]+splt[1].split(',')[1:])
                # print(parameters)
            # parameters = parameters.split('min_sample')[0]
            if experiment_set not in experiment_sets:
                experiment_sets.append(experiment_set)
            if parameters:
                if parameters[-1] == ',':
                    parameters = parameters[:-1]

                parameters = parameters.replace('distral_', '')
                parameters = parameters.replace(',,', ',')
            experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )
            # print(experiment_id)
            if experiment_id in results:
                # if parameters in results[name]['config']:
                results[experiment_id]['files'].append(f)
                results[experiment_id]['seeds'].append(seed)

            else:
                results[experiment_id] = {'files': [f, ], 'seeds': [seed, ]}
                # results[name]['config'].append(parameters)
            #                 results[name]['learner'] =
            #                 results[name]['curriculum'] = name.split('_')[1]
            #                 results[name]['context_visible'] = 'ctxvis' in name

            # else:
            #     results[name] = {parameters: {'files': [f, ], 'seeds': [seed, ]},
            #                      'config': [parameters, ],
            #                      'learner': name.split('_')[0],
            #                      'curriculum': name.split('_')[1],
            #                      'context_visible': 'ctxvis' in name}

        self.experiment_sets = experiment_sets
        return results

    #     def get_experiments(self, env_name, )


    def experiment_results(self):
        # tasks = self.tasks
        # policies = self.policies
        # res = self.experiments
        experiment_by_group = {}
        results_summary = {}
        for experiment_id, data in self.experiments.items():
            tasks = list()
            policies = list()
            # print(experiment_id)
            with open( data['files'][0], 'r') as f:
                for l in f.readlines():
                    rec = json.loads(l)
                    # print(rec)
                    if 'evaluation' in rec:
                        # print(rec['evaluation'])
                        for k in rec['evaluation'].keys():
                            # print(k)
                            if 'task' in k and k not in tasks:
                                tasks.append(k)


                        # for k in rec['evaluation'][tasks[0]]['episode_reward_mean']:
                        for k in rec['evaluation'][tasks[0]]['policy_reward_mean'].keys():
                            # print(k)
                            if 'baseline' in experiment_id[2] and 'distilled' in k :
                                distilled_policy = k
                                continue
                            elif k not in policies:
                                policies.append(k)

                        # break
                    # break




            f.close()
            data['policies'] = policies
            data['tasks'] = tasks
            # print(policies, tasks)

                #             if 'evaluation' in res[name][params]:
                #             print(name, params)

            rew_within = np.zeros(0)
            rew_between = np.zeros(0)
            rew_total = np.zeros(0)
            all_rewards = {(p,t):  np.zeros(0) for p, t  in itertools.product(policies, tasks) }

            for file, seed in zip(data['files'], data['seeds']):

                seed_results = {(p, t):  np.zeros(0) for p, t  in itertools.product(policies, tasks) }
                seed_results['total_reward_between'] = np.zeros(0)
                seed_results['total_reward_within'] = np.zeros(0)
                seed_results['total_reward'] = np.zeros(0)
                with open(file, 'r') as f:
                    for ctr, l in enumerate(f.readlines()):
                        #                         print(ctr)
                        try:
                            rec = json.loads(l)
                        except:
                            print('tapan 2: ', file, ctr, l)
                            continue
                        if 'evaluation' in rec:
                            for p, t in itertools.product(policies, tasks):
                                seed_results[(p, t)] = np.append(seed_results[(p, t)],
                                                               rec['evaluation'][t]['policy_reward_mean'][p] )




                            train_tasks =  [t for t in tasks if 'train' in t]
                            learners = [ p for p in policies if 'distilled' not in p]
                            # print(tasks, policies)
                            if len(learners)==len(train_tasks):
                                total_train_rew = [rec['evaluation'][t]['policy_reward_mean'][p]  for p, t in
                                         zip(learners,train_tasks)]
                                total_rew = np.sum([rec['evaluation'][t]['policy_reward_mean'][p] for p, t in
                                                     itertools.product(learners, train_tasks)])

                            else:
                                total_train_rew = [rec['evaluation'][t]['policy_reward_mean'][learners[0]]  for t in
                                         train_tasks]
                                total_rew = len(train_tasks)*np.sum([rec['evaluation'][t]['policy_reward_mean'][p] for p, t in
                                                     itertools.product(learners, train_tasks)])

                            #                                 print(total_rew))


                            seed_results['total_reward_within'] = np.append(seed_results['total_reward_within'], np.sum(total_train_rew))


                            seed_results['total_reward_between'] = np.append(seed_results['total_reward_between'],
                                                                             total_rew - np.sum(total_train_rew))
                            seed_results['total_reward'] = np.append(seed_results['total_reward'],
                                                                             total_rew )

                seed_results['best_itr'] = max(seed_results['total_reward_within'].argmax(), seed_results['total_reward'].argmax())

                data[seed] = seed_results
                rew_total = np.append(rew_total, seed_results['total_reward'][seed_results['best_itr']])

                rew_within = np.append(rew_within, seed_results['total_reward_within'][seed_results['best_itr']])
                rew_between = np.append(rew_between, seed_results['total_reward_between'][seed_results['best_itr']])
                for p, t in itertools.product(policies, tasks):
                    all_rewards[(p,t)] =np.append(all_rewards[(p,t)], seed_results[(p,t)][seed_results['best_itr']])

                if 'Central' in experiment_id[4]:
                    for  t in  tasks:
                        all_rewards[(distilled_policy, t)] = all_rewards[learners[0], t]


            group, group_id = exp_id2group(experiment_id)
            results_summary[experiment_id] = {'n': len(rew_total), 'total_reward_within': rew_within,
                                              'total_reward_between': rew_between, 'total_reward':rew_total,
                                              'eval_tasks': [t for t in tasks if 'eval' in t],
                                              'train_tasks': [t for t in tasks if 'train' in t],
                                              'policies': policies,
                                              'policy_task_reward': all_rewards,
                                              'group_id': group_id,
                                              'group': group,
                                              'exp_id': experiment_id}


            if experiment_id[1] in experiment_by_group: #experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )
                if group_id in experiment_by_group[experiment_id[1]]:
                    experiment_by_group[experiment_id[1]][group_id]['experiments'].append(experiment_id)
                else:
                    experiment_by_group[experiment_id[1]][group_id] = {'experiments':[experiment_id,]}
                    if len(policies)> len(experiment_by_group[experiment_id[1]]['policies']):
                        experiment_by_group[experiment_id[1]]['policies']= policies

            else:
                experiment_by_group[experiment_id[1]] = {group_id:{ 'experiments':[ experiment_id,],},
                                                         'policies': policies,
                                                         'train_tasks':results_summary[experiment_id]['train_tasks'],
                                                         'eval_tasks': results_summary[experiment_id]['eval_tasks']}

            # results_summary.update(all_rewards)

        # print(results_summary)
        # print(experiment_by_group)
        for setup, exp_set in experiment_by_group.items():
            # print(setup)

            for group_id, data in exp_set.items():
                if not isinstance(group_id, tuple):
                    continue
                # print(data['experiments'])
                tmp = data['experiments'][0]
                # print(tmp)
                exp_result = results_summary[tmp]
                # print(exp_result)
                policies = exp_result['policies']
                tasks = exp_result['train_tasks'] + exp_result['eval_tasks']
                # total_rewards = {'mean':list(), 'sd': list()}
                # for exp in data['experiments']:

                all_total_reward_within = {'mean': [np.mean(results_summary[exp]['total_reward_within']) for exp in data['experiments']],
                                           'std': [np.std(results_summary[exp]['total_reward_within']) for exp in data['experiments']]}
                all_total_reward_between = {'mean': [np.mean(results_summary[exp]['total_reward_between']) for exp in
                                           data['experiments']],
                                            'std': [np.std(results_summary[exp]['total_reward_between']) for exp in
                                           data['experiments']]}

                all_total_reward = {'mean': [np.mean(results_summary[exp]['total_reward']) for exp in
                                           data['experiments']],
                                    'std': [np.std(results_summary[exp]['total_reward']) for exp in
                                           data['experiments']]}
                all_rewards = {}
                for p, t in itertools.product(policies, tasks):
                    all_rewards[(p, t)]= {'mean':[np.mean(results_summary[exp]['policy_task_reward'][(p,t)]) for exp in
                                           data['experiments']],
                                          'std': [np.std(results_summary[exp]['policy_task_reward'][(p,t)]) for exp in
                                           data['experiments']]}


                best_idx = np.argmax(all_total_reward_within['mean'])
                best_exp = data['experiments'][best_idx]

                data.update({'reward_within': all_total_reward_within,
                             'reward_between': all_total_reward_between,
                             'total': all_total_reward,
                             'policy_task': all_rewards,
                             'best_idx': best_idx,
                             'best_exp_id': best_exp,
                             'best_performance': results_summary[best_exp],
                             'eval_tasks': exp_result['eval_tasks'],
                             'train_tasks': exp_result['train_tasks'],
                             'policies': policies})


        return results_summary, experiment_by_group

    # results_summary[experiment_id] = {'n': len(rew_total), 'total_reward_within': rew_within,
    #                                   'total_reward_between': rew_between, 'total_reward': rew_total,
    #                                   'eval_tasks': [t for t in tasks if 'eval' in t],
    #                                   'train_tasks': [t for t in tasks if 'train' in t],
    #                                   'policies': policies,
    #                                   'policy_task_reward': all_rewards,
    #                                   'group_id': group_id,
    #                                   'group': group,
    #                                   'exp_id': experiment_id}

    def best_perf_by_key(self, setup, ctx_vis = True):

        exp_set = self.experiment_breakdown[setup]
        groups = [(ctx_vis, i) for i in [1, 3, 0, 2, 4, 6, 5, 7] if (ctx_vis, i) in  exp_set]
        if len(groups)==0:
            return None, None, None
        best_by_group = [exp_set[group_id]['best_performance'] for group_id in groups]
        best_performers = dict()
        total_rewards = dict()
        policy_task_rewards = dict()
        for p, t in itertools.product(exp_set['policies'], exp_set['train_tasks'] + exp_set['eval_tasks']):
            vals = [np.mean(performer['policy_task_reward'].get((p,t),
                                          (performer['policy_task_reward'][(performer['policies'][0],t) ] if len(performer['policies'] )== 1
                                           else -np.infty)) )
                    for performer in best_by_group]
            policy_task_rewards[(p,t)] = best_by_group[np.argmax(vals)]

        for k in [ 'total_reward_within', 'total_reward_between', 'total_reward']:
            vals = [np.mean(performer.get(k, -np.infty)) for performer in best_by_group]

            total_rewards[k] = best_by_group[np.argmax(vals)]

        best_performers['total_rewards'] = total_rewards
        best_performers['policy_task_rewards'] = policy_task_rewards
        return best_performers, groups, best_by_group








    def save_to_file(self, name = 'results_summary.pkl'):
        with open(self.log_dir + name, 'wb') as file:
            # A new file will be created
            pickle.dump((self.experiment_breakdown, self.summary,self.experiments), file)
        return

    def load_from_file(self, name = 'results_summary.pkl'):

        with open(self.log_dir+name, 'rb') as file:
            # Call load method to deserialze
            self.experiment_breakdown, self.summary, self.experiments = pickle.load(file)
            for setup in self.experiment_breakdown.keys():
                best_performers, groups, best_by_group = self.best_perf_by_key(setup=setup, ctx_vis=True)
                if groups is not None:
                    self.best_performers[setup] = {
                        True: {'best_performers': best_performers, "groups": groups, 'best_by_group': best_by_group}, }

                best_performers, groups, best_by_group = self.best_perf_by_key(setup=setup, ctx_vis=False)
                if groups is not None:
                    self.best_performers[setup].update({
                        False: {'best_performers': best_performers, "groups": groups,
                                'best_by_group': best_by_group}, })

        return

    def total_rewards_txt(self, setup, ctx_vis = True):
        # self.rewards_table = dict()
        # for setup, exp_set in self.experiment_breakdown.items():
        if setup in self.experiment_breakdown:
            exp_set = self.experiment_breakdown[setup]
        else:
            raise Exception(setup +'not in experiment')
        results_txt = dict()
        exp_results = list()
        exps = list()

        # print('Context Visible')
        experiments = []
        experiments.append('{'+'|c'*5 +'|}\n')
        experiments.append('\hline\n'+'&'.join(['ID', 'name', 'curriculum', 'parameters', 'runs',]) + '\\\\\n\hline \hline\n')
        output = '{|c|' + '|c'*6 +'|}\n\hline\n'
        output += '&'.join(['Algorithm', '\multicolumn{2}{c|}{ Reward Within }',
                           '\multicolumn{2}{c|}{ Rewards Between}',
                           '\multicolumn{2}{c|}{Total Rewards}']) + '\\\\\n'
        output += '&'.join(['ID', 'mean', 'se',
                            'mean', 'se',
                            'mean', 'se']) + '\\\\\n\hline\hline\n'
        results_txt['header'] =  output
        for i in [1, 3, 0, 2, 4, 6, 5, 7]:

            if (ctx_vis, i) in exp_set:
                exp_group = exp_set[(ctx_vis, i)]
                # experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )
                ctr = 0
                for j, exp_id in enumerate(exp_group['experiments']):
                    n = self.summary[exp_id]['n']
                    exps.append(exp_id)
                    ID = self.sorted_idx[i]+',' + str(ctr)
                    self.summary[exp_id]['ID'] = ID
                    ctr+=1

                    row = [ID, self.name_codes[(ctx_vis, i)], exp_id[5].replace('distill_coeff', '$C_{dstl}$').replace('loss_fn', '$f_{loss}$') , exp_id[3], str(n)]
                    experiments.append('&'.join(row) + '\\\\\n')
                    row = [ID, ]
                    for k in ['reward_within', 'reward_between', 'total']:
                        row += [str(round(exp_group[k]['mean'][j], 2)), str(round(exp_group[k]['std'][j]/n**.5, 3))]
                    exp_results.append('&'.join(row) + '\\\\\n')
        exp_results.append('\hline\n')
        experiments.append('\hline\n')
        results_txt['experiments'] = exps
        results_txt['texts'] = exp_results

        return results_txt, experiments
            # self.rewards_table[setup] = {'totals': results_txt}

    def task_results(self, setup, tasks = None, ctx_vis = True):
        # self.rewards_table = dict()
        if setup in self.experiment_breakdown:
            exp_set = self.experiment_breakdown[setup]
        else:
            raise Exception(setup +'not in experiment')
        # for setup, exp_set in self.experiment_breakdown.items():
        policies = exp_set['policies']
        if tasks is None:
            train_tasks = exp_set['train_tasks'] + exp_set['eval_tasks']
        else:
            train_tasks = tasks
        # eval_tasks = exp_set['eval_tasks']
        exp_results = list()
        results_txt = dict()
        learner_num = len(policies)
        task_num = len(train_tasks)
        exps = list()
        policies.sort()
        # cols_num = 5 + train_tasks* len(policies)*2
        output = '{|c|'+'|c'* (learner_num)*2 + '|}\n\hline\n'
        output += '&'.join(['Algorithm',] + ['\multicolumn{2}{|c}{' + pol.replace('_', ' ') + '}'
                                         for pol in policies] ) + '\\\\\n'
        output += '&'.join([' ', ] + ['mean', 'se'] * learner_num) + '\\\\\n \hline'
        results_txt['header'] = output

        for tsk in train_tasks:
            output = '\hline\n'+'&'.join([' ', '\multicolumn{' + str((learner_num ) * 2) + '}{|c|}{' + tsk.replace('_', ' ') + '}'])  + '\\\\\n\hline\n'


            exp_results.append(output)
        # results_txt
        # Multiagent baseline
            for i in [1, 3]:

                if (ctx_vis, i) in exp_set:
                    exp_group = exp_set[(ctx_vis, i)]
                    # experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )

                    for j, exp_id in enumerate(exp_group['experiments']):
                        exps.append(exp_id)
                        n = self.summary[exp_id]['n']
                        id= self.summary[exp_id]['ID']
                        row = [id, ]
                        # for tsk in train_tasks:
                        for pol in policies:
                            if (pol, tsk) in exp_group['policy_task']:
                                row += [str(round(exp_group['policy_task'][(pol, tsk)]['mean'][j], 2)),
                                        str(round(exp_group['policy_task'][(pol, tsk)]['std'][j]/n**.5, 3))]
                            else:
                                row += [' ', ' ']
                        exp_results.append( '&'.join(row) + '\\\\\n')


            for i in [0, 2]:

                if (ctx_vis, i) in exp_set:
                    exp_group = exp_set[(ctx_vis, i)]
                    # experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )

                    for j, exp_id in enumerate(exp_group['experiments']):
                        exps.append(exp_id)
                        n = self.summary[exp_id]['n']
                        id = self.summary[exp_id]['ID']

                        row = [id, ]
                        # for tsk in train_tasks:
                        for pol in policies:
                            if (pol, tsk) in exp_group['policy_task']:
                                row += [str(round(exp_group['policy_task'][(pol, tsk)]['mean'][j], 2)),
                                        str(round(exp_group['policy_task'][(pol, tsk)]['std'][j]/n**.5, 3))]

                            elif (exp_group['policies'][0], tsk) in exp_group['policy_task']:
                                row += [str(round(exp_group['policy_task'][(exp_group['policies'][0], tsk)]['mean'][j], 2)),
                                        str(round(exp_group['policy_task'][(exp_group['policies'][0], tsk)]['std'][j]/n**.5, 3))]
                            # elif (policies[1], tsk) in exp_group['policy_task']:
                            #     row += [str(round(exp_group['policy_task'][(policies[1], tsk)]['mean'][j], 2)),
                            #             str(round(exp_group['policy_task'][(policies[1], tsk)]['std'][j] / n ** .5, 3))]
                            else:
                                row += [' ', ' ']
                        exp_results.append( '&'.join(row) + '\\\\\n')

            for i in [4, 6, 5, 7]:

                if (ctx_vis, i) in exp_set:
                    exp_group = exp_set[(ctx_vis, i)]
                    # experiment_id = (visibility, experiment_set, learner_group,curriculum, name, parameters )


                    for j, exp_id in enumerate(exp_group['experiments']):
                        exps.append(exp_id)
                        n = self.summary[exp_id]['n']

                        row = [self.summary[exp_id]['ID'], ]
                        # for tsk in train_tasks:
                        for pol in policies:
                            if (pol, tsk) in exp_group['policy_task']:
                                row += [str(round(exp_group['policy_task'][(pol, tsk)]['mean'][j], 2)),
                                        str(round(exp_group['policy_task'][(pol, tsk)]['std'][j]/n**.5, 3))]
                            else:
                                row += [' ', ' ']

                    exp_results.append('&'.join(row) + '\\\\\n')

            results_txt['experiments'] = exps
            results_txt['texts'] = exp_results

        return results_txt
                # self.total_rewards_table[(setup, ctx_vis)].update({name:results_txt})

    def best_results(self, setup, ctx_vis= True):
        # self.rewards_table = dict()
        # for setup,  in self.experiment_breakdown.items():
        # exp_set = self.experiment_breakdown[setup]
        best_performers, groups, best_by_group = self.best_perf_by_key(setup=setup, ctx_vis=ctx_vis)
        if groups is None:
            raise Exception('no experiment')
        results_txt = dict()
        exp_results = list()
        # exps = list()

        results_total = dict()
        # print('Context Visible')
        output = '{|c|'+'|c'* 9 + '|}\n\hline\n'

        output += '&'.join(['Algorithm', '\multicolumn{3}{c|}{Reward Within}',
                           '\multicolumn{3}{c|}{Reward Between}',
                           '\multicolumn{3}{c|}{Total Rewards}']) + '\\\\\n'
        output += '&'.join([' ', 'mean', 'se', 'p-value',
                            'mean', 'se', 'p-value',
                            'mean', 'se''p-value',]) + '\\\\\n\hline\n'
        results_total['header'] = output
        totals_best = best_performers['total_rewards']
        for group, group_best in zip(groups, best_by_group):
            exp_id = group_best['exp_id']
            n = group_best['n']
            row = [self.summary[exp_id]['ID'], ]
            for k, best in totals_best.items():
                if exp_id == best['exp_id']:
                    vals = group_best[k]
                    row += ['\\cellcolor{green!25}'+str(round(np.mean(vals), 2)),
                            '\\cellcolor{green!25}'+str(round(np.std(vals)/n**.5, 3)), '\\cellcolor{green!25}'+' ']

                else:
                    vals_best = best[k]
                    vals_tst = group_best[k]
                    pvalue = ttest_ind(vals_tst,
                              vals_best,
                              equal_var=False)[1]
                    if pvalue< 0.05:
                        sig = '\\cellcolor{red!25}'
                    else:
                        sig = ''
                    row += [ str(round(np.mean(vals_tst), 2)),
                              str(round(np.std(vals_tst)/ n**.5, 3)), sig+ str(round(pvalue, 3))]
            exp_results.append('&'.join(row) + '\\\\\n')
        results_total['texts'] = exp_results
        # exp_set = self.experiment_breakdown[setup]
        # policies = exp_set['policies']
        policy_task_best = best_performers['policy_task_rewards']
        # tasks = list({k[1] for k in policy_task_best.keys()})
        # policies = list({k[0] for k in policy_task_best.keys()})
        exp_results = list()
        # exps = list()

        results_policy_task = dict()
        output = '{' + '|c|'+'|c'* 3 + '|}\n\hline\n'
        # output += '&'.join(['Algorithm', '\multicolumn{2}{c}{Reward Within}',
        #                    '\multicolumn{2}{c}{Reward Between}',
        #                    '\multicolumn{2}{c}{Total Reward}']) + '\\\\\n'
        output += '&'.join(['Algorithm', 'mean', 'se', 'p-value\\\\\n'])
        results_policy_task['header'] = output
        # policy_task_best = best_performers['policy_task_rewards']

        for k, best in policy_task_best.items():

            body = '\hline\n\multicolumn{4}{|c|}{'+ k[0].replace('_', ' ') + ', '+ k[1].replace('_', ' ') +'} \\\\\n\hline\n'
            for group, group_best in zip(groups, best_by_group):
                if (group[1]==1 or group[1]==3) and 'distill' in k[0]:
                    continue
                exp_id = group_best['exp_id']
                n = group_best['n']
                row = [self.summary[exp_id]['ID'], ]
            # for k, best in policy_task_best.items():
                if k in group_best['policy_task_reward'] or (group_best['policies'][0], k[1]) in group_best['policy_task_reward'] :

                    if exp_id == best['exp_id']:

                        vals = best['policy_task_reward'].get(k, (best['policy_task_reward'][
                                                                 (best['policies'][0], k[1])] if len(best['policies']) == 1 else 0))
                        # vals = best['policy_task_reward'][k]
                        row += ['\\cellcolor{green!25}' + str(round(np.mean(vals), 2)),
                                '\\cellcolor{green!25}' + str(round(np.std(vals) / n ** .5, 3)),
                                '\\cellcolor{green!25}' + ' ']

                    else:

                        vals_best = best['policy_task_reward'].get(k, (best['policy_task_reward'][
                                                             (best['policies'][0], k[1])] ))
                        vals_tst = group_best['policy_task_reward'].get(k, (group_best['policy_task_reward'][
                                                             (group_best['policies'][0], k[1])]))
                        pvalue = ttest_ind(vals_tst,
                                       vals_best,
                                       equal_var=False)[1]
                        if pvalue < 0.05:
                            sig = '\\cellcolor{red!25}'
                        else:
                            sig = ''
                        row += [str(round(np.mean(vals_tst), 2)),
                                str(round(np.std(vals_tst) / n ** .5, 3)), sig + str(round(pvalue, 3))]
                else:
                    continue
                body += '&'.join(row) + '\\\\\n'
            exp_results.append(body)
        exp_results.append('\hline')
        results_policy_task['texts'] = exp_results
        return results_total, results_policy_task

    def report(self):
        for setup in self.experiment_breakdown.keys():
            for ctx_vis in [True, False]:
                folder = 'ctx_vis/' if ctx_vis else 'ctx_hid/'
                total, experiments  = self.total_rewards_txt(setup=setup, ctx_vis= ctx_vis)
                with open(self.log_dir+folder+setup+'/experiments.txt', "w") as f:

                    for l in experiments:
                        f.write(l)


                with open(self.log_dir+folder+setup+'/total_rewards.txt', "w") as f:
                    f.write(total['header'])
                    for l in total['texts']:
                        f.write(l)

                total = self.task_results(setup=setup, ctx_vis=ctx_vis)
                with open(self.log_dir + folder + setup + '/policy_task_rewards.txt', "w") as f:
                    f.write(total['header'])
                    for l in total['texts']:
                        f.write(l)

                try:
                    total, policy_task = self.best_results(setup=setup, ctx_vis=ctx_vis)
                except Exception as e:
                    print('skipping empty '+ setup+' with ctx_vis = ' + str(ctx_vis) +'\n', e)
                    continue
                with open(self.log_dir + folder + setup + '/total_rewards_ttest.txt', "w") as f:
                    f.write(total['header'])
                    for l in total['texts']:
                        f.write(l)

                with open(self.log_dir + folder + setup + '/policy_task_rewards_ttest.txt', "w") as f:
                    f.write(policy_task['header'])
                    for l in policy_task['texts']:
                        f.write(l)



