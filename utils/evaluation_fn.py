import logging
import numpy as np
import gym
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.metrics import summarize_episodes, collect_episodes
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
import ray
import gc
torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

def cross_eval(w, timeout,cons):
    i = w.worker_index
    episodes, _ = collect_episodes(
        remote_workers=[w, ], timeout_seconds=timeout
    )
    # all_episodes += episodes
    index = 'base_task' if i == 0 else 'subtask_' + str(i - 1)
    results = summarize_episodes(episodes)
    sp_progress = dict()
    for j in range(len(cons[i]['mean'])):
        sp_progress['ctx_' + str(j) + '_mean'] = cons[i]['mean'][j]
        sp_progress['ctx_' + str(j) + '_var'] = cons[i]['var'][j]

    # context_evals[index] = sp_progress
    return {'index': index, 'sp_progress': sp_progress, 'evals': results}

#New version not working yet
def DnCCrossEval1(algorithm, eval_workers):
    """Example of a custom evaluation function.

    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.

    Returns:
        metrics: Evaluation metrics dict.
    """
    context_evals = dict()
    cons = algorithm.workers.foreach_env(lambda env: env.report_task())[1][0]
    all_episodes = list()
    task_metrics = dict()
    # for
    # We configured 2 eval workers in the training config.
    # workers = eval_workers.remote_workers()
    # # assert len(workers) ==algorithm.con
    # funcs = [lambda w: w.foreach_env(lambda env: env.copy_task(env.get_task()[i])) for i in range(algorithm.config['evaluation_num_workers'])]
    # eval_workers.foreach_worker(func=funcs)
    for _ in range(algorithm.config['evaluation_duration']):
        # print("Custom evaluation round", i)
        # Calling .sample() runs exactly one episode per worker due to how the
        # eval workers are configured.
        eval_workers.foreach_worker(func=lambda w: w.sample(), local_worker=False)

    evaluations = eval_workers.foreach_worker(func = lambda w: cross_eval(w, algorithm.config["evaluation_sample_timeout_s"], cons))

    # for i, w in enumerate(eval_workers):
        # w.foreach_env.remote(lambda env: env.copy_task(env.get_task()[i]))

            # eval_workers.foreach
    # workers# workers = algorithm.workers.remote_workers()
    # Set different env settings for each worker. Here we use a fixed config,
    # which also could have been computed in each worker by looking at
    # env_config.worker_index (printed in SimpleCorridor class above).
    # workers[0].foreach_env.remote(lambda env: env.set_corridor_length(4))
    # worker_2.foreach_env.remote(lambda env: env.set_corridor_length(7))


    # for i, w in enumerate(workers):
    #     episodes, _ = collect_episodes(
    #         remote_workers=[w, ], timeout_seconds=algorithm.config["evaluation_sample_timeout_s"]
    #     )
    #     all_episodes += episodes
    #     index = 'base_task' if i == 0 else 'subtask_' + str(i-1)
    #     task_metrics[index] = summarize_episodes(episodes)
    #     sp_progress = dict()
    #     for j in range(len(cons[i]['mean'])):
    #         sp_progress['ctx_'+str(j)+'_mean'] = cons[i]['mean'][j]
    #         sp_progress['ctx_'+ str(j)+'_var' ] = cons[i]['var'][j]
    #
    #     context_evals[index] = sp_progress
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    # metrics = summarize_episodes(all_episodes)
    # # Note that the above two statements are the equivalent of:
    # # metrics = collect_metrics(eval_workers.local_worker(),
    # #                           eval_workers.remote_workers())
    #
    # # You can also put custom values in the metrics dict.
    # metrics.update(task_metrics)
    # metrics['curriculum'] = context_evals
    return {}


#working not optimized
def DnCCrossEval(algorithm, eval_workers):
    """Example of a custom evaluation function.

    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.

    Returns:
        metrics: Evaluation metrics dict.
    """
    context_evals = dict()
    cons = algorithm.workers.foreach_env(lambda env: env.report_task())[1][0]
    all_episodes = list()
    task_metrics = dict()
    # for
    # We configured 2 eval workers in the training config.
    workers = eval_workers.remote_workers()
    # # assert len(workers) ==algorithm.con
    # funcs = [lambda w: w.foreach_env(lambda env: env.copy_task(env.get_task()[i])) for i in range(algorithm.config['evaluation_num_workers'])]
    # eval_workers.foreach_worker(func=funcs)

    for i, w in enumerate(workers):
        # w.foreach_env.remote(lambda env: env.copy_task(env.get_task()[i]))
        for _ in range(algorithm.config['evaluation_duration']):
            # print("Custom evaluation round", i)
            # Calling .sample() runs exactly one episode per worker due to how the
            # eval workers are configured.
            ray.get(w.sample.remote() )
            # eval_workers.foreach
    # workers# workers = algorithm.workers.remote_workers()
    # Set different env settings for each worker. Here we use a fixed config,
    # which also could have been computed in each worker by looking at
    # env_config.worker_index (printed in SimpleCorridor class above).
    # workers[0].foreach_env.remote(lambda env: env.set_corridor_length(4))
    # worker_2.foreach_env.remote(lambda env: env.set_corridor_length(7))


    # for i, w in enumerate(workers):
        episodes, _ = collect_episodes(
            remote_workers=[w, ], timeout_seconds=algorithm.config["evaluation_sample_timeout_s"]
        )
        all_episodes += episodes
        index = 'base_task' if i == 0 else 'subtask_' + str(i-1)
        task_metrics[index] = summarize_episodes(episodes)
        sp_progress = dict()
        for j in range(len(cons[i]['mean'])):
            sp_progress['ctx_'+str(j)+'_mean'] = cons[i]['mean'][j]
            sp_progress['ctx_'+ str(j)+'_var' ] = cons[i]['var'][j]

        context_evals[index] = sp_progress
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = summarize_episodes(all_episodes)
    # Note that the above two statements are the equivalent of:
    # metrics = collect_metrics(eval_workers.local_worker(),
    #                           eval_workers.remote_workers())

    # You can also put custom values in the metrics dict.
    metrics.update(task_metrics)
    metrics['curriculum'] = context_evals
    return metrics

def CL_report(algorithm, eval_workers):
    """Example of a custom evaluation function.

    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.

    Returns:
        metrics: Evaluation metrics dict.
    """
    context_evals = dict()
    # all_episodes = list()
    # task_metrics = dict()
    sp_progress = dict()
    cons = algorithm.workers.foreach_env(lambda env: env.report_task())[1][0]

    for i  in range(algorithm.config['env_config']['num_agents']):

        index = 'task_' + str(i)



        # else:
        for j in range(len(cons[i]['mean'])):
            sp_progress['ctx_' + str(j) + '_mean'] = cons[i]['mean'][j]
            sp_progress['ctx_' + str(j) + '_var'] = cons[i]['var'][j]

        context_evals[index] = sp_progress

    return dict(curriculum=context_evals)


def DiscreteCL_report(algorithm, eval_workers):
    """Example of a custom evaluation function.

    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.

    Returns:
        metrics: Evaluation metrics dict.
    """
    context_evals = dict()
    # all_episodes = list()
    # task_metrics = dict()
    sp_progress = dict()
    cons = algorithm.workers.foreach_env(lambda env: env.report_task())[1][0]

    for i  in range(algorithm.config['env_config']['num_agents']):

        index = 'task_' + str(i)



        # else:
        for j in range(len(cons[i]['mean'])):
            sp_progress['ctx_' + str(j) + '_mean'] = cons[i]['mean'][j]
            # sp_progress['ctx_' + str(j) + '_var'] = cons[i]['var'][j]

        context_evals[index] = sp_progress

    return dict(curriculum=context_evals)



def DnCCrossEvalSeries(algorithm, eval_workers):
    """Example of a custom evaluation function.

    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.

    Returns:
        metrics: Evaluation metrics dict.
    """
    context_evals = dict()
    all_episodes = list()
    task_metrics = dict()
    sp_progress = dict()
    # for
    # We configured 2 eval workers in the training config.
    # workers = if algorithm.config["evaluation_num_workers"] == 0 else eval_workers.remote_workers()
    # if workers
    # if 'env_mapping' in algorithm.config['env_config']:
    #     env_mapping = algorithm.config['env_config']['env_mapping']
    #     sp_progress['env_mapping'] = env_mapping
    # else:
    #     env_mapping = None
    # env_mapping = algorithm.config['env_config'].get('env_mapping', None)
    eval_task_n = algorithm.config['env_config'].get('non_training_envs', 1)
    # print(workers)
    # # assert len(workers) ==algorithm.con
    # funcs = [lambda w: w.foreach_env(lambda env: env.copy_task(env.get_task()[i])) for i in range(algorithm.config['evaluation_num_workers'])]
    # eval_workers.foreach_worker(func=funcs)
    # eval_workers.foreach_env(lambda env: env.get_context_buffer())
    for i, config in enumerate(algorithm.config['evaluation_config']['env_config']['agent_config']):
        eval_workers.foreach_env(lambda env: env.reconfig_all(config))
        eval_workers.foreach_env(lambda env: env.training_mode(False))
        # eval_workers.foreach_env(lambda env: env.report_task())
        # eval_workers.foreach_env(lambda env: env.reset())
        for _ in range(algorithm.config['evaluation_duration']):
            eval_workers.foreach_worker(lambda w: w.sample(), local_worker=False)
            # eval_workers.foreach_worker(lambda w: w.sample())

        episodes = collect_episodes(
                workers=eval_workers, timeout_seconds=algorithm.config["evaluation_sample_timeout_s"]
            )

        # ctx_buffer = eval_workers.foreach_env(lambda env: env.get_ctx_hist())

        all_episodes += episodes
        index = 'eval_task_' + str(i) if i < eval_task_n else 'train_task_' + str(i-eval_task_n)
        task_metrics[index] = summarize_episodes(episodes)


        cons = algorithm.workers.foreach_env(lambda env: env.report_task())[1][0]

        # if env_mapping:
        #     for j in range(len(cons[env_mapping[i]]['mean'])):
        #         sp_progress['ctx_'+str(j)+'_mean'] = cons[env_mapping[i]]['mean'][j]
        #         sp_progress['ctx_'+ str(j)+'_var' ] = cons[env_mapping[i]]['var'][j]
        # else:
        for j in range(len(cons[i]['mean'])):
            sp_progress['ctx_' + str(j) + '_mean'] = cons[i]['mean'][j]
            sp_progress['ctx_' + str(j) + '_var'] = cons[i]['var'][j]
        context_evals[index] = sp_progress
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = summarize_episodes(all_episodes)
    # Note that the above two statements are the equivalent of:
    # metrics = collect_metrics(eval_workers.local_worker(),
    #                           eval_workers.remote_workers())

    # You can also put custom values in the metrics dict.
    metrics.update(task_metrics)
    metrics['curriculum'] = context_evals
    eval_workers.foreach_env(lambda env: env.training_mode(True))
    return metrics


def DnCCLCrossEvalSeries(algorithm, eval_workers):
    """Example of a custom evaluation function.

    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.

    Returns:
        metrics: Evaluation metrics dict.
    """
    context_evals = dict()
    cons = algorithm.workers.foreach_env(lambda env: env.report_task())[1][0]
    all_episodes = list()
    task_metrics = dict()
    # for
    # We configured 2 eval workers in the training config.
    workers = eval_workers.remote_workers()
    env_mapping = algorithm.config['env_config'].get('env_mapping', None)
    # # assert len(workers) ==algorithm.con
    # funcs = [lambda w: w.foreach_env(lambda env: env.copy_task(env.get_task()[i])) for i in range(algorithm.config['evaluation_num_workers'])]
    # eval_workers.foreach_worker(func=funcs)
    # eval_workers.foreach_env(lambda env: env.get_context_buffer())
    for i, config in enumerate(algorithm.config['evaluation_config']['env_config']['agent_config']):
        eval_workers.foreach_env(lambda env: env.reconfig_all(config))
        eval_workers.foreach_env(lambda env: env.training_mode(True))

        eval_workers.foreach_env(lambda env: env.reset())
        for _ in range(algorithm.config['evaluation_duration']):
            # eval_workers.foreach_worker(lambda w: w.sample(), local_worker=False)
            eval_workers.foreach_worker(lambda w: w.sample())

        episodes, _ = collect_episodes(
                remote_workers=workers, timeout_seconds=algorithm.config["evaluation_sample_timeout_s"]
            )

        # ctx_buffer = eval_workers.foreach_env(lambda env: env.get_ctx_hist())
        all_episodes += episodes
        index = 'base_task' if i == 0 else 'subtask_' + str(i-1)

        task_metrics[index] = summarize_episodes(episodes)
        sp_progress = dict()
        if env_mapping:
            for j in range(len(cons[env_mapping[i]]['mean'])):
                sp_progress['ctx_'+str(j)+'_mean'] = cons[env_mapping[i]]['mean'][j]
                sp_progress['ctx_'+ str(j)+'_var' ] = cons[env_mapping[i]]['var'][j]
        else:
            for j in range(len(cons[i]['mean'])):
                sp_progress['ctx_' + str(j) + '_mean'] = cons[i]['mean'][j]
                sp_progress['ctx_' + str(j) + '_var'] = cons[i]['var'][j]

        context_evals[index] = sp_progress
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = summarize_episodes(all_episodes)
    # Note that the above two statements are the equivalent of:
    # metrics = collect_metrics(eval_workers.local_worker(),
    #                           eval_workers.remote_workers())

    # You can also put custom values in the metrics dict.
    metrics.update(task_metrics)
    metrics['curriculum'] = context_evals
    eval_workers.foreach_env(lambda env: env.training_mode(True))
    return metrics

def DnCCrossEval(algorithm, eval_workers):
    """Example of a custom evaluation function.

    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.

    Returns:
        metrics: Evaluation metrics dict.
    """
    context_evals = dict()
    cons = algorithm.workers.foreach_env(lambda env: env.report_task())[1][0]
    all_episodes = list()
    task_metrics = dict()
    # for
    # We configured 2 eval workers in the training config.
    workers = eval_workers.remote_workers()
    # # assert len(workers) ==algorithm.con
    # funcs = [lambda w: w.foreach_env(lambda env: env.copy_task(env.get_task()[i])) for i in range(algorithm.config['evaluation_num_workers'])]
    # eval_workers.foreach_worker(func=funcs)

    for i, w in enumerate(workers):
        # w.foreach_env.remote(lambda env: env.copy_task(env.get_task()[i]))
        for _ in range(algorithm.config['evaluation_duration']):
            # print("Custom evaluation round", i)
            # Calling .sample() runs exactly one episode per worker due to how the
            # eval workers are configured.
            ray.get(w.sample.remote() )
            # eval_workers.foreach
    # workers# workers = algorithm.workers.remote_workers()
    # Set different env settings for each worker. Here we use a fixed config,
    # which also could have been computed in each worker by looking at
    # env_config.worker_index (printed in SimpleCorridor class above).
    # workers[0].foreach_env.remote(lambda env: env.set_corridor_length(4))
    # worker_2.foreach_env.remote(lambda env: env.set_corridor_length(7))


    # for i, w in enumerate(workers):
        episodes, _ = collect_episodes(
            remote_workers=[w, ], timeout_seconds=algorithm.config["evaluation_sample_timeout_s"]
        )
        all_episodes += episodes
        index = 'base_task' if i == 0 else 'subtask_' + str(i-1)
        task_metrics[index] = summarize_episodes(episodes)
        sp_progress = dict()
        for j in range(len(cons[i]['mean'])):
            sp_progress['ctx_'+str(j)+'_mean'] = cons[i]['mean'][j]
            sp_progress['ctx_'+ str(j)+'_var' ] = cons[i]['var'][j]

        context_evals[index] = sp_progress
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = summarize_episodes(all_episodes)
    # Note that the above two statements are the equivalent of:
    # metrics = collect_metrics(eval_workers.local_worker(),
    #                           eval_workers.remote_workers())

    # You can also put custom values in the metrics dict.
    metrics.update(task_metrics)
    metrics['curriculum'] = context_evals
    return metrics