#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import json
import pandas as pd
import random
import itertools
from scipy.stats import ttest_ind
from utils.stats_analysis import ExperimentAnalysis


# In[2]:


# res = experiment_results
# log_dir = './results/'
# tasks = ['base_task', 'subtask_0', 'subtask_1']
# policies = ['distilled_policy', 'learner_0', 'learner_1']
# analysis = ExperimentAnalysis(log_dir, tasks, policies,
#                                 env_name = 'PointMass2D',
#                                 version = ['4.2.0/','4.1.5/'], env_map=True)


# In[2]:


def print_summary(analysis):
    summary = analysis.summary
    
    print('****** VISIBLE CONTEXT *****')
    print('task '.ljust(30), end ='\t| ')
    
    N = len(analysis.tasks)
    
    tasks = ['base',]
    pols = ['ctr',]
    tasks += [str(i) for i in range(N-1)]
    pols += [str(i) for i in range(N-1)]
    
    for p,t in itertools.product(pols, tasks):
        print (t.ljust(6), end = ' | ')

    print('\npolicy '.ljust(30), end ='\t| ')

    for p,t in itertools.product(pols, tasks):
        print (p.ljust(6), end = ' | ')
    print('\n','================'*7)
    # print('Learner\t\t| seeds | distill interval  ||  seeds  |    mean    | standard error | ')
    for k , v in summary.items():
        if 'SAC' not in k[0] or 'ctxvis' not in k[0]:
            continue

        curr = ', Default' if 'default' in k[0] else ', SP'    

        learner = 'SAC_Central' if 'Central' in k[0] else 'SAC'
        print( (learner + curr).ljust(30),end ='\t| ')

        for p,t in itertools.product(analysis.policies, analysis.tasks):

            print(str(np.round(v[(p,t)]['mean'],2)).ljust(6), end = ' | ')

        print('')


    print('----------------'*7)
    # print('Learner\t\t| seeds | distill interval  ||  seeds  |    mean    | standard error | ')
    for k , v in summary.items():
        if 'Distral' not in k[0] or 'ctxvis' not in k[0]:
            continue
        if 'default' in k[0]:    
            curr = "Default,"
        else:
             curr = "SP,"   

        print( (curr +k[1].replace('distral_', '')).ljust(30),end ='\t| ')

        for p,t in itertools.product(analysis.policies, analysis.tasks):

            print(str(np.round(v[(p,t)]['mean'],2)).ljust(6), end = ' | ')

        print('')



    print('\n\n\n')
    print('****** HIDDEN CONTEXT *****')
    print('================'*7)
    # print('Learner\t\t| seeds | distill interval  ||  seeds  |    mean    | standard error | ')
    for k , v in summary.items():
        if 'SAC' not in k[0] or 'ctxvis'  in k[0]:
            continue
        print( k[0].ljust(30),end ='\t| ')

        for p,t in itertools.product(analysis.policies, analysis.tasks):
            print(str(np.round(v[(p,t)]['mean'],2)).ljust(6), end = ' | ')        
        print('')


    print('----------------'*7)
    # print('Learner\t\t| seeds | distill interval  ||  seeds  |    mean    | standard error | ')
    for k , v in summary.items():
        if 'Distral' not in k[0] or 'ctxvis'  in k[0]:
            continue
        print( k[0].ljust(30),end ='\t| ')

        for p,t in itertools.product(analysis.policies, analysis.tasks):

            print(str(np.round(v[(p,t)]['mean'], 2)).ljust(6), end = ' | ')

        print('')


# In[4]:


def print_independent(analysis):
    win = '\033[44m'
    sig = '\033[42m'
    normal = '\033[0m'

    print('\n\nIndependent Single Task Comparisons\n')
    print('****** VISIBLE CONTEXT *****')

    print('DisTral'.ljust(10),'|', 'learner'.center(10), '|  CL  | ', 'parameters'.center(21),'|',
          'mean'.center(10),'|', 'se'.center(10),  '| ', 'runs', ' | p-value |'  )
    print('='*101)
    for k, v in analysis.best_performers[0].items():
        curriculum = 'D'
        if 'selfpaced' in v[0]:
            curriculum = 'SP'

        learner = 'DisTral'
        parameters = v[1].replace('distral_', '')
    #     for task in analysis.tasks:
        m = np.round(analysis.summary[v][k]['mean'],2)
        if m > analysis.summary[analysis.best_performers[1][k]][k]['mean']:
            m = win+str(m).center(10)+ normal
        else:
            m = str(m).center(10)
        se = np.round(analysis.summary[v][k]['se'],2)
        runs = np.round(analysis.summary[v][k]['runs'],2)
        pvalue = ttest_ind(analysis.summary[v][k]['rewards'], 
                           analysis.summary[analysis.best_performers[1][k]][k]['rewards'],
                           equal_var=False)[1]
        if pvalue<0.05:
            pvalue = sig+str(np.round(pvalue,3)).center(7)+normal
        else:
            pvalue = str(np.round(pvalue,3)).center(7)

        print(k[1].ljust(10),'|',learner.center(10), '| ', curriculum.ljust(3), '| ' ,
              parameters.ljust(21),'|', m, '|', str(se).center(10), '|' ,
              str(runs).center(6), '|',
             pvalue, '|')
    print('~'*101)
    print('Baseline'.ljust(10),'|', 'learner'.center(10), '|  CL  | ', 'parameters'.center(21),'|','mean'.center(10),'|', 'se'.center(10),  '| ', 'runs', ' |'  )
    print('========'*11+'===')
    for k, v in analysis.best_performers[1].items():
        curriculum = 'D'
        learner = 'SAC_Cntr' if 'Central' in v[0] else 'SAC'
        if 'selfpaced' in v[0]:
            curriculum = 'SP'

    #     for task in analysis.tasks:
        m = np.round(analysis.summary[v][k]['mean'],2)
        se = np.round(analysis.summary[v][k]['se'],2)
        runs = np.round(analysis.summary[v][k]['runs'],2)
        print(k[1].ljust(10),'|',learner.center(10), '| ', curriculum.ljust(3), '| ' ,
              'NA'.center(21),'|', str(m).center(10), '|', str(se).center(10), '|' ,
              str(runs).center(6), '|')

    print('~'*91)

    print('\n\n\n')
    print('****** HIDDEN CONTEXT *****')


# In[3]:


def print_best(analysis):
    win = '\033[44m'
    sig = '\033[42m'
    normal = '\033[0m'
    distral_vis_best = analysis.get_overall_best(distral = True, ctx= True)
    SAC_vis_best = analysis.get_overall_best(distral = False, ctx= True)


    distral_hid_best = analysis.get_overall_best(distral = True, ctx= False)
    SAC_hid_best = analysis.get_overall_best(distral = False, ctx= False)



    print('****** Best DisTral Vs Best Baseline Comparison (VISIBLE CONTEXT) *****')
    print('================'*7)

    print('Learner'.ljust(10),'|', 'Parameters'.center(21), '|  CL  |' ,
          'Across Tasks'.center(32), '|', 'Within Tasks'.center(32), '|')
    print(''.ljust(10),'|', ''.center(21), '|      |' ,
          'mean'.center(9), '|','se'.center(8), '|','p-value'.center(9), '|', 
         'mean'.center(9), '|','se'.center(8), '|','p-value'.center(9), '|')
    print('='*112)
    exp , params = distral_vis_best[0]
    curriculum = 'D'
    if 'selfpaced' in exp:
        curriculum = 'SP'

    learner = 'DisTral'
    parameters = params.replace('distral_', '')
    #     for task in analysis.tasks:
    summary = analysis.summary[(exp, params)]

    distral_inter_r = summary['intertask']['rewards']
    distral_inter_m = np.round(summary['intertask']['mean'],2)
    distral_inter_se = np.round(summary['intertask']['se'],2)

    distral_intra_r = summary['intratask']['rewards']
    distral_intra_m = np.round(summary['intratask']['mean'],2)
    distral_intra_se = np.round(summary['intratask']['se'],2)


    exp , params = SAC_vis_best[0]
    summary = analysis.summary[(exp, params)]

    bl_inter_r = summary['intertask']['rewards']
    bl_inter_m = np.round(summary['intertask']['mean'],2)
    bl_inter_se = np.round(summary['intertask']['se'],2)

    bl_intra_r = summary['intratask']['rewards']
    bl_intra_m = np.round(summary['intratask']['mean'],2)
    bl_intra_se = np.round(summary['intratask']['se'],2)



    if distral_inter_m > bl_inter_m:
        distral_inter_m = win+str(distral_inter_m).center(9)+ normal
    else:
        distral_inter_m = str(distral_inter_m).center(9)


    if distral_intra_m > bl_intra_m:
        distral_intra_m = win+str(distral_intra_m).center(9)+ normal
    else:
        distral_intra_m = str(distral_intra_m).center(9)



    pvalue_inter = ttest_ind(distral_inter_r, 
                       bl_inter_r,
                       equal_var=False)[1]

    pvalue_intra = ttest_ind(distral_intra_r, 
                       bl_intra_r,
                       equal_var=False)[1]


    if pvalue_inter<0.05:
        pvalue_inter = sig+str(np.round(pvalue_inter,3)).center(9)+normal
    else:
        pvalue_inter = str(np.round(pvalue_inter,3)).center(9)

    if pvalue_intra<0.05:
        pvalue_intra = sig+str(np.round(pvalue_intra,3)).center(9)+normal
    else:
        pvalue_intra = str(np.round(pvalue_intra,3)).center(9)


    print('DisTral'.ljust(10),'|',parameters.center(21), '| ', curriculum.ljust(3), '|' ,
           distral_intra_m, '|', str(distral_intra_se).center(8), '|' ,
          pvalue_intra.center(9), '|',
          distral_inter_m, '|', str(distral_inter_se).center(8), '|' ,
          pvalue_inter.center(9), '|')

    curriculum = 'SP' if 'selfpaced' in exp else 'D'
    learner = 'SAC_cntr' if 'Central' in exp else 'SAC'
    parameters = params
    print(learner.ljust(10),'|',parameters.center(21), '| ', curriculum.ljust(3), '|' ,
           str(bl_intra_m).center(9), '|', str(bl_intra_se).center(8), '|' ,
          ''.center(9), '|',
          str(bl_inter_m).center(9), '|', str(bl_inter_se).center(8), '|' ,
          ''.center(9), '|')

    print('~'*112)







    print('\n\n\n****** Best DisTral Vs Best Baseline Comparison (Hidden CONTEXT) *****')
    print('================'*7)

    print('Learner'.ljust(10),'|', 'Parameters'.center(21), '|  CL  |' ,
          'Across Tasks'.center(32), '|', 'Within Tasks'.center(32), '|')
    print(''.ljust(10),'|', ''.center(21), '|      |' ,
          'mean'.center(9), '|','se'.center(8), '|','p-value'.center(9), '|', 
         'mean'.center(9), '|','se'.center(8), '|','p-value'.center(9), '|')
    print('='*112)


    if  distral_hid_best[0]:
        exp , params = distral_hid_best[0]
        curriculum = 'D'
        if 'selfpaced' in exp:
            curriculum = 'SP'

        learner = 'DisTral'
        parameters = params.replace('distral_', '')
        #     for task in analysis.tasks:
        summary = analysis.summary[(exp, params)]

        distral_inter_r = summary['intertask']['rewards']
        distral_inter_m = np.round(summary['intertask']['mean'],2)
        distral_inter_se = np.round(summary['intertask']['se'],2)

        distral_intra_r = summary['intratask']['rewards']
        distral_intra_m = np.round(summary['intratask']['mean'],2)
        distral_intra_se = np.round(summary['intratask']['se'],2)

    if  SAC_hid_best[0]:

        exp , params = SAC_hid_best[0]
        summary = analysis.summary[(exp, params)]

        bl_inter_r = summary['intertask']['rewards']
        bl_inter_m = np.round(summary['intertask']['mean'],2)
        bl_inter_se = np.round(summary['intertask']['se'],2)

        bl_intra_r = summary['intratask']['rewards']
        bl_intra_m = np.round(summary['intratask']['mean'],2)
        bl_intra_se = np.round(summary['intratask']['se'],2)



        if distral_inter_m > bl_inter_m:
            distral_inter_m = win+str(distral_inter_m).center(9)+ normal
        else:
            distral_inter_m = str(distral_inter_m).center(9)


        if distral_intra_m > bl_intra_m:
            distral_intra_m = win+str(distral_intra_m).center(9)+ normal
        else:
            distral_intra_m = str(distral_intra_m).center(9)



        pvalue_inter = ttest_ind(distral_inter_r, 
                           bl_inter_r,
                           equal_var=False)[1]

        pvalue_intra = ttest_ind(distral_intra_r, 
                           bl_intra_r,
                           equal_var=False)[1]


        if pvalue_inter<0.05:
            pvalue_inter = sig+str(np.round(pvalue_inter,3)).center(9)+normal
        else:
            pvalue_inter = str(np.round(pvalue_inter,3)).center(9)

        if pvalue_intra<0.05:
            pvalue_intra = sig+str(np.round(pvalue_intra,3)).center(9)+normal
        else:
            pvalue_intra = str(np.round(pvalue_intra,3)).center(9)


        print('DisTral'.ljust(10),'|',parameters.center(21), '| ', curriculum.ljust(3), '|' ,
               distral_intra_m, '|', str(distral_intra_se).center(8), '|' ,
              pvalue_intra.center(9), '|',
              distral_inter_m, '|', str(distral_inter_se).center(8), '|' ,
              pvalue_inter.center(9), '|')

        curriculum = 'SP' if 'selfpaced' in exp else 'D'
        learner = 'SAC_cntr' if 'Central' in exp else 'SAC'
        parameters = params
        print(learner.ljust(10),'|',parameters.center(21), '| ', curriculum.ljust(3), '|' ,
               str(bl_intra_m).center(9), '|', str(bl_intra_se).center(8), '|' ,
              ''.center(9), '|',
              str(bl_inter_m).center(9), '|', str(bl_inter_se).center(8), '|' ,
              ''.center(9), '|')

    print('~'*112)





# In[7]:

log_dir = './results/'
tasks = ['base_task', 'subtask_0', 'subtask_1']
policies = ['distilled_policy', 'learner_0', 'learner_1']
analysis0 = ExperimentAnalysis(log_dir, tasks, policies,
                                env_name = 'PointMass2D',
                                version = ['5.3.0/Set0'], env_map=True)

print('Summary for set 0\n')

print('Context dist.: base [2.5, 0.5],  subtask0 [2.5, 0.5],  subtask1 [2.5, 0.5]\n\n')
print_summary(analysis0)

# print_independent(analysis0)


# print('\noverall comparisons\n')

# print_best(analysis0)


# In[22]:

#
# log_dir = './results/'
# tasks = ['base_task', 'subtask_0', 'subtask_1']
# policies = ['distilled_policy', 'learner_0', 'learner_1']
# analysis1 = ExperimentAnalysis(log_dir, tasks, policies,
#                                 env_name = 'PointMass2D',
#                                 version = ['5.2.1/Set1'])
#
# print('Summary for set 1\n')
# print('Context dist.: base [2.5, 0.5],  subtask0 [3, 2],  subtask1 [1, 0.5]\n\n')
# print_summary(analysis1)
#
# # print_independent(analysis1)
#
#
# # print('\noverall comparisons\n')
#
# # print_best(analysis1)
#
#
# # In[25]:
#
#
# log_dir = './results/'
# tasks = ['base_task', 'subtask_0', 'subtask_1']
# policies = ['distilled_policy', 'learner_0', 'learner_1']
# analysis2 = ExperimentAnalysis(log_dir, tasks, policies,
#                                 env_name = 'PointMass2D',
#                                 version = ['5.2.1/Set2'])
#
# print('Summary for set 2\n')
# print('Context dist.: base [(2.5,-3), (0.5, 2)],   subtask0 [2.5(5e-2), 0.5(1e-2)],   subtask1 [-3(0.5), 2(0.5)]\n\n')
# print_summary(analysis2)
#
# # print_independent(analysis2)
#
#
# # print('\noverall comparisons\n')
#
# # print_best(analysis2)
#
#
# # In[ ]:
#
#
#
#
#
# # In[24]:
#
#
# log_dir = './results/'
# tasks = ['base_task', 'subtask_0', 'subtask_1']
# policies = ['distilled_policy', 'learner_0', 'learner_1']
# analysis3 = ExperimentAnalysis(log_dir, tasks, policies,
#                                 env_name = 'PointMass2D',
#                                 version = ['5.2.1/Set3'])
#
# print('Summary for set 3\n')
# print('Context dist.: base [(2,-2), (1, 1)],  subtask0 [2(1), 1(0.5)],  subtask1 [-2(1), 1(0.5)]\n\n')
# print_summary(analysis3)

# print_independent(analysis3)


# print('\noverall comparisons\n')

# print_best(analysis3)

