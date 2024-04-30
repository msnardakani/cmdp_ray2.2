import logging
from typing import Union, Dict, Tuple

import numpy as np
# import gym
from ray.rllib import Policy, BaseEnv, SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.postprocessing import discount_cumsum

from deep_sprl.teachers.spl.self_paced_teacher_v2 import SelfPacedTeacherV2
from gaussian_sprl.gaussian_selfpaced_teacher import GaussianSelfPacedTeacher
import random
logger = logging.getLogger(__name__)

UPDATE_INTERVAL = 5
MIN_BUFFER_SIZE = 64


class Buffer:

    def __init__(self, n_elements, max_buffer_size, reset_on_query):
        self.reset_on_query = reset_on_query
        self.max_buffer_size = max_buffer_size
        self.buffers = [list() for i in range(0, n_elements)]

    def update_buffer(self, datas):
        if isinstance(datas[0], list):
            for buffer, data in zip(self.buffers, datas):
                buffer.extend(data)
        else:
            for buffer, data in zip(self.buffers, datas):
                buffer.append(data)

        while len(self.buffers[0]) > self.max_buffer_size:
            for buffer in self.buffers:
                del buffer[0]

    def merge_buffer(self, addition_buffer):
        for buffer, addition in zip(self.buffers, addition_buffer.buffers):
            buffer +=addition

        while len(self.buffers[0]) > self.max_buffer_size:
            for buffer in self.buffers:
                del buffer[0]

    def read_buffer(self, reset=None):
        if reset is None:
            reset = self.reset_on_query

        res = tuple([buffer for buffer in self.buffers])

        if reset:
            for i in range(0, len(self.buffers)):
                self.buffers[i] = []

        return res

    def __len__(self):
        return len(self.buffers[0])





class MACL(DefaultCallbacks):
    iteration=0
    sp_teachers = dict()
    def on_train_result(self, *, algorithm, result: dict, **kwargs):

        # id = random.choice(list(self.sp_teachers.keys()))

        # print('task id: ', id,' buffer length: ', len(self.sp_teachers[id]['buffer']) )
        data = algorithm.workers.foreach_env(lambda env: env.read_buffer())

        for buffer in data[1]:
            for entry in buffer[0]:
                if entry[0] in self.sp_teachers:
                    # rew =
                    self.sp_teachers[entry[0]]['buffer'].update_buffer((entry[1], entry[2], entry[3]))
        # self.data.clear()
        # agent = task if self.env_mapping is None else self.env_mapping[task]
        #     disc_rew = discount_cumsum(rewards[agent], self.gamma)[0]
        #     rew = episode.agent_rewards[agent]
        #     self.sp_teachers[task]['buffer'].update_buffer((contexts[agent], rew, disc_rew))
        self.iteration += 1
        if self.iteration % UPDATE_INTERVAL ==0:# and self.iteration > 9:
            # print(self.iteration)
            # you can mutate the result dict to add new fields to return

            for tsk_num, v in self.sp_teachers.items():
                buffer = v['buffer']
                teacher = v['teacher']

                if len(buffer)<MIN_BUFFER_SIZE:

                    continue
                idx = tsk_num
                cons, rews, disc_rews = buffer.read_buffer()
                # stats = algorithm.workers.foreach_env(lambda env: env.get_env_episodes_statistics(idx))
                # buffers = algorithm.workers.foreach_env(lambda env: env.get_env_context_buffer(idx))


                # vf_inputs = np.array(ins)
                contexts = np.array(cons)
                discounted_rewards = np.array(disc_rews)
                avg_perf = np.mean(rews)
                teacher.update_distribution(avg_performance=avg_perf, contexts=contexts, values=discounted_rewards)

                ctx_config = teacher.export_dist()
                algorithm.workers.foreach_env(lambda env: env.set_sampler_dist(idx, means = ctx_config['target_mean'],
                                                                               sigma2= ctx_config['target_var'],
                                                                               w = ctx_config['target_priors']))

                # kl = teacher.target_context_kl(True)
        kl_results = dict()

        for tsk_num, v in self.sp_teachers.items():
            sp_progress = dict()
            teacher = v['teacher']
            ctx_config = teacher.get_context()
            for j, (m, s2) in enumerate(zip(ctx_config['mean'], list(ctx_config['var']))):
                sp_progress['ctx_' + str(j) + '_mean'] = m
                sp_progress['ctx_' + str(j) + '_var'] = s2
            sp_progress['kl_div'] = ctx_config['kl_div']

            status = teacher.get_status()

            sp_progress.update(status)
            kl_results['task_' + str(tsk_num)] = sp_progress

        # print(kl_results)
        result['info'].update(dict(curriculum = kl_results))

        return

    # def on_sample_end(
    #     self, *, worker: "RolloutWorker", samples: SampleBatch, **kwargs
    # ) -> None:
    #     return
    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies ,
        episode: Union[Episode, EpisodeV2, Exception],
        **kwargs,
    ) -> None:
        # agent_cl = dict()
        # if base_env.get_sub_environments()[0].TRAINING ==False:
        #     # print('eval:', base_env.get_sub_environments()[episode.env_id].get_context())
        #
        #     return
        rewards = episode._agent_reward_history
        contexts = base_env.get_sub_environments()[episode.env_id].get_context()

        gamma = worker.config['gamma']
        # print('train:', base_env.get_sub_environments()[episode.env_id].get_context())

        for task in rewards.keys():
            ctx = contexts[task]
            disc_rew = discount_cumsum(rewards[task], gamma)[0]
            rew = sum(rewards[task])

            base_env.get_sub_environments()[episode.env_id].update_buffer( ((task, ctx, rew, disc_rew),))
            # self.sp_teachers[task]['buffer'].update_buffer((ctx, rew, disc_rew))

        # episode.custom_metrics['context_data'] = data
        # print('episode end new_data: ', len(base_env.get_sub_environments()[episode.env_id].buffer))

        return

    def on_algorithm_init(
            self,
            *,
            algorithm: "Algorithm",
            **kwargs,
    ) -> None:
        """Callback run when a new algorithm instance has finished setup.

        This method gets called at the end of Algorithm.setup() after all
        the initialization is done, and before actually training starts.

        Args:
            algorithm: Reference to the trainer instance.
            kwargs: Forward compatibility placeholder.
        """
        self.iteration = 0
        # workers = algorithm.evaluation_workers.remote_workers()
        # for i, w in enumerate(workers):
        #     w.foreach_env.remote(lambda env: env.copy_task(env.get_task()[i]))
        # algorithm.evaluation_workers.foreach_env(lambda env: env.get_env_teacher(1))
        self.sp_teachers = dict()
        env_config =algorithm.config['env_config']
        self.num_agents = env_config['num_agents']
        # self.eval_configs = algorithm.config['']
        # self.env_mapping = env_config.get('env_mapping', None)
        # self.non_training_envs = env_config.get('non_training_envs', 1)
        curriculum = env_config['agent_config']
        self.gamma = algorithm.config['gamma']
        context_space = algorithm.workers.foreach_env(lambda env: env.get_context_space())
        ctx_lb = context_space[1][0].low
        ctx_ub = context_space[1][0].high
        # if self.env_mapping:
        #     for i, v in enumerate(curriculum):
        #         if v.get('curriculum', 'default') =='self_paced':
        #             self.sp_teachers[self.env_mapping[i]] = Buffer(n_elements=3, max_buffer_size=1000, reset_on_query=True)
        # else:
        for idx, v in enumerate(curriculum):
            if 'self_paced' in v.get('curriculum', 'default') :
                init_mean = v['init_mean'].copy()

                target_mean = v['target_mean'].copy()
                kl_threshold = v.get('kl_threshold', 10000)
                max_kl = v.get('max_kl', 0.05)
                perf_lb = v.get('perf_lb', 3)
                std_lower_bound = v.get('std_lb', 0.1)
                if 'gaussian' in v.get('curriculum', 'default'):
                    # print('gaussian_teacher_initialization')
                    target_var = np.diag(np.clip(v['target_var'], a_min=std_lower_bound**2, a_max=None)).copy()
                    init_var = v['init_var'].copy()
                    if len(init_var) == 1:
                        init_scale= init_var
                    elif len(init_var) ==len(v['target_var']):
                        init_scale= np.mean(np.array(np.diag(target_var)).reshape(-1)/np.array(init_var).reshape(-1))
                    # std_lower_bound = np.mean(np.array(v['target_var']).reshape(-1)/np.array(init_var).reshape(-1))
                    teacher = GaussianSelfPacedTeacher(target_mean=target_mean, target_variance=target_var,
                                                       initial_mean=init_mean, init_covar_scale=init_scale,
                                                       context_bounds=(ctx_lb, ctx_ub), perf_lb=perf_lb,
                                                       max_kl=max_kl, std_lower_bound=init_scale,
                                                       kl_threshold=kl_threshold)


                else:
                    target_var = np.diag(v['target_var']).copy()
                    init_var = np.diag(v['init_var']).copy()
                    teacher = SelfPacedTeacherV2(target_mean=target_mean,target_variance=target_var,
                                                 initial_mean=init_mean, initial_variance= init_var,
                                           context_bounds=(ctx_lb, ctx_ub), perf_lb=perf_lb,
                                          max_kl=max_kl, std_lower_bound=std_lower_bound,
                                          kl_threshold=kl_threshold, use_avg_performance=True)
                self.sp_teachers.update({idx: dict(buffer=Buffer(n_elements=3, max_buffer_size=1000, reset_on_query=True),
                                          teacher = teacher)})
                ctx_config = teacher.export_dist()
                # idx = i
                algorithm.workers.foreach_env(lambda env: env.set_sampler_dist(idx, means = ctx_config['target_mean'],
                                                                               sigma2= ctx_config['target_var'],
                                                                               w = ctx_config['target_priors']))

        return


