import logging
from typing import Union, Dict, Tuple

import numpy as np
# import gym
from ray.rllib import Policy, BaseEnv, SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
# from ray.rllib.evaluation.metrics import summarize_episodes, collect_episodes
# from ray.rllib.models import ModelV2
# from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
# from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
#
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
# from ray.rllib.utils.annotations import override
# from ray.rllib.utils.framework import try_import_torch
# from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
# import ray
import gc
# torch, nn = try_import_torch()
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import AgentID, EnvID, EpisodeID, PolicyID

logger = logging.getLogger(__name__)



class MASPCL(DefaultCallbacks):
    iteration=0

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        self.iteration += 1
        if self.iteration % 5 ==0 and self.iteration > 9:
            # print(self.iteration)
            # you can mutate the result dict to add new fields to return
            kl_results = dict()
            for tsk_num in self.sp_teachers:
                idx = self.env_mapping[tsk_num] if self.env_mapping else tsk_num
                buff_size = algorithm.workers.foreach_env(lambda env: env.get_env_buffer_size(idx))
                if sum(buff_size[0]+buff_size[1]) < 128:
                    continue
                # stats = algorithm.workers.foreach_env(lambda env: env.get_env_episodes_statistics(idx))
                buffers = algorithm.workers.foreach_env(lambda env: env.get_env_context_buffer(idx))

                ins, cons, rewards = [], [], []

                for buff in buffers[0]+ buffers[1]:
                    # ins += buff[0]
                    cons += buff[1]
                    rewards += buff[2]


                # vf_inputs = np.array(ins)
                contexts = np.array(cons)
                rewards = np.array(rewards)
                teachers = algorithm.workers.foreach_env(lambda env: env.get_env_teacher(idx))
                for t in teachers[0]+teachers[1]:
                    if t:
                        t.update_distribution(0, contexts, rewards)
                        weights = t.get_task()
                        kl = t.target_context_kl(True)
                        kl_results['subtask_'+str(tsk_num-1)]= np.mean(kl)
                        break
                algorithm.workers.foreach_env(lambda env: env.update_env_teacher(weights=weights, idx = idx))

                # cl  ={'ctx_0': weights[0],'ctx_1': weights[1],'ctx_2': weights[2],'ctx_3': weights[3] }
                # result['curriculum'] = cl
                del teachers, buffers,
                del ins, cons, rewards, contexts

                gc.collect()
                result['target_ctx_kl'] = kl_results

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
        self.iteration =0
        # workers = algorithm.evaluation_workers.remote_workers()
        # for i, w in enumerate(workers):
        #     w.foreach_env.remote(lambda env: env.copy_task(env.get_task()[i]))
        # algorithm.evaluation_workers.foreach_env(lambda env: env.get_env_teacher(1))
        self.sp_teachers = set()
        env_config =algorithm.config['env_config']
        self.num_agents = env_config['num_agents']
        # self.eval_configs = algorithm.config['']
        self.env_mapping = env_config.get('env_mapping', None)
        curriculum = env_config['agent_config']
        if self.env_mapping:
            for i, v in enumerate(curriculum):
                if v.get('curriculum', 'default') =='self_paced':
                    self.sp_teachers.add(self.env_mapping[i])
        else:
            for i, v in enumerate(curriculum):
                if v.get('curriculum', 'default') =='self_paced':
                    self.sp_teachers.add(i)

        return

class SPCL_Eval(DefaultCallbacks):
    iteration=0

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        self.iteration += 1
        if self.iteration % 5 ==0 and self.iteration>20:
            print(self.iteration)
            # you can mutate the result dict to add new fields to return
            stats = algorithm.workers.foreach_env(lambda env: env.get_episodes_statistics())
            buffers = algorithm.workers.foreach_env(lambda env: env.get_context_buffer())
            rewards, disc_rewards, steps, n = [], [], [], 0
            for st in stats[0]+stats[1]:
                rewards += st[0]
                disc_rewards += st[1]
                steps += st[2]
                n += st[3]

            # mean_rew=np.mean(rewards)
            mean_disc_rew= np.mean(disc_rewards)
            # mean_length = np.mean(steps)
            ins, cons, rewards = [], [], []
            for buff in buffers[0]+ buffers[1]:
                # ins += buff[0]
                cons += buff[1]
                rewards += buff[2]


            # vf_inputs = np.array(ins)
            contexts = np.array(cons)
            rewards = np.array(rewards)
            teachers = algorithm.workers.foreach_env(lambda env: env.get_teacher())
            for t in teachers[0]+teachers[1]:
                if t:
                    t.update_distribution(mean_disc_rew, contexts, rewards)
                    weights = t.get_task()

                    break
            algorithm.workers.foreach_env(lambda env: env.update_teacher(weights=weights))
            # cl  ={'ctx_0': weights[0],'ctx_1': weights[1],'ctx_2': weights[2],'ctx_3': weights[3] }
            # result['curriculum'] = cl
            del teachers, stats, buffers,
            del ins, cons, rewards, contexts, disc_rewards

            gc.collect()
        else:
            teachers = algorithm.workers.foreach_env(lambda env: env.get_teacher())
            for t in teachers[0] + teachers[1]:
                if t:
                    weights = t.get_task()
                    cl = {'ctx_0': weights[0], 'ctx_1': weights[1]}
                    result['curriculum'] = cl
                    del teachers
                    gc.collect()
                    break

        # # mean_rew, mean_disc_rew, mean_length = task_settable_env.get_statistics()
       #  buffers = algorithm.workers.foreach_env(lambda env: env.get_context_buffer())
       #  # vf_inputs, contexts, rewards = task_settable_env.get_context_buffer()
       #  task_settable_env.teacher.update_distribution(mean_disc_rew, contexts,
       #                                                rewards)
       #  new_task = task_settable_env.get_task()
       #  print('callback done!')
        return



