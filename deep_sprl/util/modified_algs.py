import collections
import copy
import os
import pickle
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from imitation.algorithms import bc
from sklearn.mixture import GaussianMixture
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

import gym
import numpy as np
import sb3_contrib
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from torch import nn
from torch.distributions import kl_divergence
from torch.nn import functional as F

from sb3_contrib.common.utils import conjugate_gradient_solver, flat_grad
from sb3_contrib.trpo import TRPO, MlpPolicy

from deep_sprl.experiments import CurriculumType
from deep_sprl.experiments.abstract_experiment import MAPPOInterface, PPOTRPOEvalWrapper
from deep_sprl.teachers.abstract_teacher import DummyWrapper, BaseWrapper, DNCWrapper
from deep_sprl.teachers.dummy_teachers import GMMSampler, GaussianSampler
from deep_sprl.teachers.spl import SelfPacedTeacher, SelfPacedWrapper
from deep_sprl.teachers.spl.alpha_functions import PercentageAlphaFunction
from deep_sprl.util.funcs import RolloutBufferPlus, RolloutBufferPlusSamples, Metrics, sb3_roll2trans


class TRPO_loss(TRPO):
    """
    Trust Region Policy Optimization (TRPO)

    Paper: https://arxiv.org/abs/1502.05477
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    and Stable Baselines (TRPO from https://github.com/hill-a/stable-baselines)

    Introduction to TRPO: https://spinningup.openai.com/en/latest/algorithms/trpo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate for the value function, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size for the value function
    :param gamma: Discount factor
    :param cg_max_steps: maximum number of steps in the Conjugate Gradient algorithm
        for computing the Hessian vector product
    :param cg_damping: damping in the Hessian vector product computation
    :param line_search_shrinking_factor: step-size reduction factor for the line-search
        (i.e., ``theta_new = theta + alpha^i * step``)
    :param line_search_max_iter: maximum number of iteration
        for the backtracking line-search
    :param n_critic_updates: number of critic updates per policy update
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param normalize_advantage: Whether to normalize or not the advantage
    :param target_kl: Target Kullback-Leibler divergence between updates.
        Should be small for stability. Values like 0.01, 0.05.
    :param sub_sampling_factor: Sub-sample the batch to make computation faster
        see p40-42 of John Schulman thesis http://joschu.net/docs/thesis.pdf
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def _setup_model(self) -> None:
        super(TRPO, self)._setup_model()
        self.rollout_buffer = RolloutBufferPlus(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBufferPlus,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
                dist = self.policy.get_distribution(obs_tensor).distribution

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs, dist.loc,
                               dist.scale)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self, agents=None, idx=0, penalty=0, dnc=False) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        policy_objective_values = []
        kl_divergences = []
        line_search_results = []
        value_losses = []
        all_trajectories = []
        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            for i in range(1, len(agents)):
                agent = agents[i]
                if i != idx:
                    trj = next(agent.rollout_buffer.get(batch_size=None))
                    all_trajectories.append(trj)
            actions = rollout_data.actions
            # Optional: sub-sample data for faster computation
            if self.sub_sampling_factor > 1:
                rollout_data = RolloutBufferPlusSamples(
                    rollout_data.observations[:: self.sub_sampling_factor],
                    rollout_data.actions[:: self.sub_sampling_factor],
                    rollout_data.old_values[:: self.sub_sampling_factor],  # old values, not used here
                    rollout_data.old_log_prob[:: self.sub_sampling_factor],
                    rollout_data.advantages[:: self.sub_sampling_factor],
                    rollout_data.returns[:: self.sub_sampling_factor],  # returns, not used here
                    rollout_data.means[:: self.sub_sampling_factor],
                    rollout_data.stds[:: self.sub_sampling_factor],
                )

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            # Re-sample the noise matrix because the log_std has changed
            if self.use_sde:
                # batch_size is only used for the value function
                self.policy.reset_noise(actions.shape[0])

            with th.no_grad():
                # Note: is copy enough, no need for deepcopy?
                # If using gSDE and deepcopy, we need to use `old_distribution.distribution`
                # directly to avoid PyTorch errors.
                old_distribution = copy.copy(self.policy.get_distribution(rollout_data.observations))

            distribution = self.policy.get_distribution(rollout_data.observations)
            log_prob = distribution.log_prob(actions)

            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (rollout_data.advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = th.exp(log_prob - rollout_data.old_log_prob)
            additional_loss = 0
            if penalty >0:
                additional_loss = Metrics.kl_on_others(rollout_data, all_trajectories)

            # surrogate policy objective
            policy_objective = (advantages * ratio).mean()  + penalty * additional_loss

            # KL divergence
            kl_div = kl_divergence(distribution.distribution, old_distribution.distribution).mean()

            # Surrogate & KL gradient
            self.policy.optimizer.zero_grad()

            actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(kl_div, policy_objective)

            # Hessian-vector dot product function used in the conjugate gradient step
            hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)

            # Computing search direction
            search_direction = conjugate_gradient_solver(
                hessian_vector_product_fn,
                policy_objective_gradients,
                max_iter=self.cg_max_steps,
            )

            # Maximal step length
            line_search_max_step_size = 2 * self.target_kl
            line_search_max_step_size /= th.matmul(
                search_direction, hessian_vector_product_fn(search_direction, retain_graph=False)
            )
            line_search_max_step_size = th.sqrt(line_search_max_step_size)

            line_search_backtrack_coeff = 1.0
            original_actor_params = [param.detach().clone() for param in actor_params]

            is_line_search_success = False
            with th.no_grad():
                # Line-search (backtracking)
                for _ in range(self.line_search_max_iter):

                    start_idx = 0
                    # Applying the scaled step direction
                    for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
                        n_params = param.numel()
                        param.data = (
                            original_param.data
                            + line_search_backtrack_coeff
                            * line_search_max_step_size
                            * search_direction[start_idx : (start_idx + n_params)].view(shape)
                        )
                        start_idx += n_params

                    # Recomputing the policy log-probabilities
                    distribution = self.policy.get_distribution(rollout_data.observations)
                    log_prob = distribution.log_prob(actions)

                    # New policy objective
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    new_policy_objective = (advantages * ratio).mean()

                    # New KL-divergence
                    kl_div = kl_divergence(distribution.distribution, old_distribution.distribution).mean()

                    # Constraint criteria:
                    # we need to improve the surrogate policy objective
                    # while being close enough (in term of kl div) to the old policy
                    if (kl_div < self.target_kl) and (new_policy_objective > policy_objective):
                        is_line_search_success = True
                        break

                    # Reducing step size if line-search wasn't successful
                    line_search_backtrack_coeff *= self.line_search_shrinking_factor

                line_search_results.append(is_line_search_success)

                if not is_line_search_success:
                    # If the line-search wasn't successful we revert to the original parameters
                    for param, original_param in zip(actor_params, original_actor_params):
                        param.data = original_param.data.clone()

                    policy_objective_values.append(policy_objective.item())
                    kl_divergences.append(0)
                else:
                    policy_objective_values.append(new_policy_objective.item())
                    kl_divergences.append(kl_div.item())

        # Critic update
        for _ in range(self.n_critic_updates):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values_pred = self.policy.predict_values(rollout_data.observations)
                value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
                value_losses.append(value_loss.item())

                self.policy.optimizer.zero_grad()
                value_loss.backward()
                # Removing gradients of parameters shared with the actor
                # otherwise it defeats the purposes of the KL constraint
                for param in actor_params:
                    param.grad = None
                self.policy.optimizer.step()

        self._n_updates += 1
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/policy_objective", np.mean(policy_objective_values))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/is_line_search_success", np.mean(line_search_results))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


class MATRPOwrapper:  # A multi agent version of TRPO
    PENALTY = 0.01

    def __init__(self,
                 policy: Union[str, Type[ActorCriticPolicy]],
                 envs,
                 learning_rate: Union[float, Schedule] = 1e-3,
                 n_steps: int = 2048,
                 n_epochs = 10,
                 batch_size: int = 128,
                 gamma: float = 0.99,
                 cg_max_steps: int = 15,
                 cg_damping: float = 0.1,
                 line_search_shrinking_factor: float = 0.8,
                 line_search_max_iter: int = 10,
                 n_critic_updates: int = 10,
                 gae_lambda: float = 0.95,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 normalize_advantage: bool = True,
                 target_kl: float = 0.01,
                 sub_sampling_factor: int = 1,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True,
                 ):
        self.log_dir = tensorboard_log
        print(self.log_dir)
        # self.clusters = None
        self.logger = configure(self.log_dir, ["tensorboard"])
        self.TRPO_agents = []
        agent_global = TRPO(policy,
                           envs[0],
                           learning_rate=learning_rate,
                           n_steps=n_steps,
                           gamma=gamma,
                           gae_lambda=gae_lambda,
                            use_sde=use_sde,
                            sde_sample_freq=sde_sample_freq,
                            tensorboard_log=tensorboard_log,
                            policy_kwargs=policy_kwargs,
                            verbose=verbose,
                            device=device,
                            create_eval_env=create_eval_env,
                            seed=seed,
                            _init_setup_model=_init_setup_model,
                           )
        agent_global.set_logger(self.logger)
        self.TRPO_agents.append(agent_global)
        for env in envs[1:]:
            agent = TRPO_loss(
                policy,
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                device=device,
                create_eval_env=create_eval_env,
                seed=seed,
                _init_setup_model=_init_setup_model,
            )
            agent.set_logger(self.logger)
            self.TRPO_agents.append(agent)
        self.cluster = None
        # print(self.PPO_agents[0].policy)
        # self.reset_agents()

        # self.logger = Logger(log_dir=self.log_dir)

    def reset_agents(self):
        for agent in self.TRPO_agents:
            agent.set_parameters(self.TRPO_agents[0].get_parameters())

    def learn(self,
              total_timesteps: int,
              callback: MaybeCallback = None,
              log_interval: int = 1,
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "TRPODnC",
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True, ):
        if callback == None:
            setup = [agent._setup_learn(
                total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
                tb_log_name
            ) for i, agent in enumerate(self.PPO_agents)]

        else:
            setup = [agent._setup_learn(
                total_timesteps, eval_env, callback[i], eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
                tb_log_name
            ) for i, agent in enumerate(self.PPO_agents)]
        iteration = 0
        self.reset_agents()
        for total_timesteps, callback in setup:
            callback.on_training_start(locals(), globals())
        total_timesteps = setup[0][0]
        while self.TRPO_agents[0].num_timesteps < total_timesteps:
            for i, agent in enumerate(self.PPO_agents):

                continue_training = agent.collect_rollouts(agent.env, setup[i][1], agent.rollout_buffer,
                                                           n_rollout_steps=agent.n_steps)

                if continue_training is False:
                    break

                agent._update_current_progress_remaining(agent.num_timesteps, total_timesteps)

            iteration += 1
            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:

                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                # fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                for i, agent in enumerate(self.TRPO_agents):

                    if len(agent.ep_info_buffer) > 0 and len(agent.ep_info_buffer[0]) > 0:
                        self.logger.record("rollout/ep_rew_mean_" + str(i),
                                           safe_mean([ep_info["r"] for ep_info in agent.ep_info_buffer]))
                        self.logger.record("rollout/ep_len_mean_" + str(i),
                                           safe_mean([ep_info["l"] for ep_info in agent.ep_info_buffer]))
                    # self.logger.record("time/fps", fps)
                    # self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                    # self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                    self.logger.dump(step=agent.num_timesteps)
            for i, agent in enumerate(self.PPO_agents):
                agent.train(self.PPO_agents, i, penalty=self.PENALTY)

        for total_timesteps, callback in setup:
            callback.on_training_end()

        return self

    # def train(self):

    def set_clustering(self, clustering):
        self.cluster = clustering


class MAPPOEvalWrapper:

    def __init__(self, model, clusters):
        self.model = model
        self.clusters = clusters
        self.ctx_dim = 0

    def step(self, observation, state=None, deterministic=False):

        if len(observation.shape) == 1:
            observation = observation[None, :]
            i = self.clusters.predict(observation[:, -self.ctx_dim:])
            return self.model[i].predict(observation, state=state, deterministic=deterministic)[0][0, :]
        else:
            # idx = self.clusters.predict(observation[:, -self.ctx_dim:])
            idx = self.clusters.predict(observation[:, -self.ctx_dim:])

            return np.array(
                [self.model[i].predict(observation, state=state, deterministic=deterministic)[0] for i in idx])

class DnCTRPO(MATRPOwrapper):
    save_interval = 5

    def __init__(self, env, params, curriculum=None, N=3, curriculum_type=CurriculumType.Default,
                 clusters=None, dnc_args=None, dummy_eval=False,
                 **parameters):
        # print(parameters)
        # print(env.observation_space)
        if dnc_args is None:
            dnc_args = {'save_interval': 5, "distillation_period": 50,
                        'bc_iteration': 30,
                        'bc_period': 1,
                        'last_bc': 5,
                        'PENALTY': 0.0,
                        'bc_samples': 1}
        self.distillation_period = dnc_args['distillation_period']
        self.bc_iteration = dnc_args['bc_iteration']
        self.PENALTY = dnc_args['PENALTY']
        self.last_bc = dnc_args['last_bc']
        self.bc_period = dnc_args['bc_period']
        self.bc_samples = dnc_args['bc_samples']
        self.curriculum_type = curriculum_type
        # self.cluster = None
        # if clustering == 'kmeans':
        #     self.clusters = KMeans(n_clusters=N, random_state=0).fit(s)
        # else:
        #     self.clusters = AgglomerativeClustering(n_clusters=N ).fit(s)
        if clusters == None:
            self.teacher = GMMSampler(curriculum['TARGET_MEAN'].copy(),
                                      curriculum['TARGET_VARIANCE'].copy(),
                                      curriculum['TARGET_PRIORS'].copy(),
                                      (curriculum['LOWER_CONTEXT_BOUNDS'].copy(),
                                       curriculum['UPPER_CONTEXT_BOUNDS'].copy()))

            self.s = [self.teacher.sample() for i in range(1000)]
            self.cluster = GaussianMixture(n_components=N, random_state=0).fit(self.s)
        else:
            self.cluster = clusters
        self.model = None
        # print(self.cluster.means_)
        # print(self.cluster.covariances_)
        self.envs = []
        env_global = copy.deepcopy(env)
        if dummy_eval:
            print(params)
            env_global = DummyWrapper(env_global, discount_factor=params['DISCOUNT_FACTOR'])
        else:
            teacher_new = GMMSampler(curriculum['EVAL_MEAN'].copy(),
                                     curriculum['EVAL_VARIANCE'].copy(),
                                     curriculum['EVAL_PRIORS'].copy(),
                                     (curriculum['LOWER_CONTEXT_BOUNDS'].copy(),
                                      curriculum['UPPER_CONTEXT_BOUNDS'].copy()))
            env_global = BaseWrapper(env_global, teacher_new, params['DISCOUNT_FACTOR'], context_visible=True)
            # for i in range(len(curriculum['TARGET_PRIORS'])):
        self.envs.append(env_global)
        means = self.cluster.means_
        vars = np.array([np.diag(cov) for cov in self.cluster.covariances_])
        for i in range(N):
            env_new = copy.deepcopy(env)
            if curriculum_type.value == 4:
                mu = means[i]
                print(mu)
                var = np.diag(vars[i])
                teacher_new = GaussianSampler(mu,
                                              var,
                                              (curriculum['LOWER_CONTEXT_BOUNDS'].copy(),
                                               curriculum['UPPER_CONTEXT_BOUNDS'].copy()))
                env_new = DNCWrapper(env_new, teacher_new, params['DISCOUNT_FACTOR'])
                self.envs.append(env_new)
            else:
                alpha_fn = PercentageAlphaFunction(10, 1.6)
                # teacher_new= SelfPacedTeacher(curriculum['TARGET_MEAN'][i].copy(),
                #                   np.diag(curriculum['TARGET_VARIANCE'][i]).copy(),
                #                               curriculum['INITIAL_MEAN'].copy(),
                #                   curriculum['INITIAL_VARIANCE'].copy(), (curriculum['LOWER_CONTEXT_BOUNDS'].copy(), curriculum['UPPER_CONTEXT_BOUNDS'].copy()), alpha_fn, max_kl=params['MAX_KL'],
                #                         std_lower_bound=params['STD_LOWER_BOUND'], kl_threshold=params['KL_THRESHOLD'],
                #                         use_avg_performance=True)

                teacher_new = SelfPacedTeacher(self.cluster.means_[i].copy(),
                                               self.cluster.covariances_[i].copy(),
                                               curriculum['INITIAL_MEAN'].copy(),
                                               curriculum['INITIAL_VARIANCE'].copy(), (
                                                   curriculum['LOWER_CONTEXT_BOUNDS'].copy(),
                                                   curriculum['UPPER_CONTEXT_BOUNDS'].copy()), alpha_fn,
                                               max_kl=params['MAX_KL'],
                                               std_lower_bound=params['STD_LOWER_BOUND'],
                                               kl_threshold=params['KL_THRESHOLD'],
                                               use_avg_performance=True)

                env_new = SelfPacedWrapper(env_new, teacher_new, params['DISCOUNT_FACTOR'], context_visible=True)
                self.envs.append(env_new)

        env_new = copy.deepcopy(env)
        # teacher_new = GMMSampler(curriculum['EVAL_MEAN'].copy(),
        #                          curriculum['EVAL_VARIANCE'].copy(),
        #                          curriculum['EVAL_PRIORS'].copy(),
        #                          (curriculum['LOWER_CONTEXT_BOUNDS'].copy(), curriculum['UPPER_CONTEXT_BOUNDS'].copy()))
        env_new = DummyWrapper(env_new, params['DISCOUNT_FACTOR'])

        self.eval_env = env_new
        # self.eval_vec_env =
        bc_policy = MlpPolicy(
            observation_space=env_global.observation_space,
            action_space=env_global.action_space,
            # Set lr_schedule to max value to force error if policy.optimizer
            # is used by mistake (should use self.optimizer instead).
            lr_schedule=lambda _: th.finfo(th.float32).max,
            **parameters["common"]["policy_kwargs"],
        )
        # print(self.eval_env.observation_space)
        # print(bc_policy)

        # self.bc_agent.logger.output_formats

        super().__init__('MlpPolicy', self.envs, **parameters["common"], **parameters["trpo"])
        self.bc_agent = bc.BC(
            observation_space=env_global.observation_space,
            action_space=env_global.action_space,
            demonstrations=None,
            policy=bc_policy,
            batch_size=parameters['trpo']['n_steps']-1,
            custom_logger=self.logger
        )
        # self.agent_global = self.PPO_agents.pop(0)

        self.reset_agents()

        # self.bc_agent.save_policy()
        # print(self.envs)

    def learn(self,
              total_timesteps: int,
              callback: MaybeCallback = None,
              log_interval: int = 1,
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "PPODnC",
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True, ):
        if callback == None:
            setup = [agent._setup_learn(
                total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
                tb_log_name
            ) for i, agent in enumerate(self.TRPO_agents)]

        else:
            setup = [agent._setup_learn(
                total_timesteps, eval_env, callback[i], eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
                tb_log_name
            ) for i, agent in enumerate(self.TRPO_agents)]
        iteration = 0

        for total_timesteps, callback in setup:
            callback.on_training_start(locals(), globals())
        total_timesteps = setup[0][0]
        while self.TRPO_agents[0].num_timesteps < total_timesteps:
            # for i, agent in enumerate(self.PPO_agents):
            for i in range(1, len(self.TRPO_agents)):
                agent = self.TRPO_agents[i]
                continue_training = agent.collect_rollouts(agent.env, setup[i][1], agent.rollout_buffer,
                                                           n_rollout_steps=agent.n_steps)

                if continue_training is False:
                    break

                agent._update_current_progress_remaining(agent.num_timesteps, total_timesteps)

            iteration += 1
            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:

                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                # fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                for i, agent in enumerate(self.TRPO_agents[1:]):

                    if len(agent.ep_info_buffer) > 0 and len(agent.ep_info_buffer[0]) > 0:
                        self.logger.record("rollout/ep_rew_mean_" + str(i),
                                           safe_mean([ep_info["r"] for ep_info in agent.ep_info_buffer]))
                        self.logger.record("rollout/ep_len_mean_" + str(i),
                                           safe_mean([ep_info["l"] for ep_info in agent.ep_info_buffer]))
                    # self.logger.record("time/fps", fps)
                    # self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                    # self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                    self.logger.dump(step=agent.num_timesteps)
            # for i, agent in enumerate(self.PPO_agents[1:]):

            for i in range(1, len(self.TRPO_agents)):
                self.TRPO_agents[i].train(self.TRPO_agents, i, penalty=self.PENALTY)

            if (iteration - 1) % self.distillation_period == 0:
                self.distill(self.bc_iteration)
            else:
                self.distill(1)

            # agent = self.PPO_agents[0]
            # env = self.envs[0]
            # env.teacher.set_means([e.teacher.get_mean()])
            if self.curriculum_type.value !=4:
                context_means = [self.envs[i].teacher.context_dist.mean() for i in range(1, len(self.envs))]
                context_vars = [self.envs[i].teacher.context_dist.covariance_matrix() for i in range(1, len(self.envs))]
                self.envs[0].teacher.set_means(context_means)
                self.envs[0].teacher.set_vars(context_vars)
            # print(context_mean)
            # print(context_std)

            continue_training = self.TRPO_agents[0].collect_rollouts(self.TRPO_agents[0].env, setup[0][1],
                                                                    self.TRPO_agents[0].rollout_buffer,
                                                                    n_rollout_steps=self.TRPO_agents[0].n_steps)
            #
            if continue_training is False:
                break

            self.TRPO_agents[0]._update_current_progress_remaining(self.TRPO_agents[0].num_timesteps, total_timesteps)
            #
            # iteration += 1
            # # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                #
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                # fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                #
                if len(self.TRPO_agents[0].ep_info_buffer) > 0 and len(self.TRPO_agents[0].ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean_" + str(0),
                                       safe_mean([ep_info["r"] for ep_info in self.TRPO_agents[0].ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean_" + str(0),
                                       safe_mean([ep_info["l"] for ep_info in self.TRPO_agents[0].ep_info_buffer]))
                #         # self.logger.record("time/fps", fps)
                #         # self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                #         # self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=agent.num_timesteps)

            if (iteration - 1) % self.distillation_period == 0 and iteration > 5:
                self.reset_agents()

            if (iteration - 1) % self.save_interval == 0:

                iter_log_dir = os.path.join(self.log_dir, "iteration-" + str(iteration - 1))
                os.makedirs(iter_log_dir, exist_ok=True)
                self.bc_agent.save_policy(os.path.join(iter_log_dir, 'model.zip'))
                for i, agent in enumerate(self.TRPO_agents):
                    agent.save(os.path.join(iter_log_dir, 'model_dist_' + str(i) + '.zip'))

        for total_timesteps, callback in setup:
            callback.on_training_end()

        return self

    def distill(self, iterations=10):

        rollouts = []

        rollouts = [next(expert.rollout_buffer.get(expert.rollout_buffer.buffer_size)) for expert in self.TRPO_agents[1:]]
            # env = expert.env.envs[0]
            # rollouts += rollout.rollout(
            #     expert,
            #     DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
            #     rollout.make_sample_until(min_timesteps=self.bc_samples * self.bc_agent.batch_size, min_episodes=None),
            # )
            # rollouts+=
        # print(rollouts)
        transitions = sb3_roll2trans(rollouts)
        self.bc_agent.set_demonstrations(transitions)
        self.bc_agent.train(n_epochs=iterations, progress_bar=False)

    def reset_agents(self):
        self.bc_agent.policy.optimizer.state = collections.defaultdict(dict)
        state_dict = self.bc_agent.policy.state_dict()
        for k in self.bc_agent.policy.state_dict().keys():
            if 'value' in k or 'log_std' in k:
                state_dict.pop(k)
        for agent in self.TRPO_agents:
            # state_dict['log_std'] = agent.policy.get_parameter('log_std')
            agent.policy.load_state_dict(state_dict, strict=False)
            agent.policy.optimizer.state = collections.defaultdict(dict)

    def get_interface(self):
        return [MAPPOInterface(self.TRPO_agents[i], self.envs[i].observation_space.shape[0], idx=i) for i in
                range(len(self.envs))]

    def create_envs(self):
        return self.envs

    def get_policy(self, ctx):
        return self.clusters.predict(ctx)

    #
    # def create_learner(self):
    #     return self.model, interface
    def get_eval_env(self):
        return self.eval_env, DummyVecEnv([lambda: self.eval_env])

    def save(self, cluster_path, model_dir):
        with open(cluster_path, "wb") as f:
            pickle.dump(self.clusters, f)

        for i, agent in enumerate(self.PPO_agents):
            agent.save(os.path.join(model_dir, "model" + str(i)))

        self.bc_agent.save_policy(os.path.join(model_dir, "model.zip"))

    def load(self, cluster_path, model_dir, env):
        with open(cluster_path, "rb") as f:
            clusters = pickle.load(f)

        if (self.trpo() or self.ppo()) and not issubclass(type(env), VecEnv):
            env = DummyVecEnv([lambda: env])

        PPO_agents = []
        for i in range(clusters.n_clusters):
            PPO_agents.append(TRPO.load(env=env, path=os.path.join(model_dir, "model" + str(i) + ".zip")))

        return MAPPOEvalWrapper(model=PPO_agents, clusters=clusters)
        # model = self.load(path, env)

    # def evaluate_learner(self, iteration_log_dir):
    #     model = self.load(os.path.join(iteration_log_dir,"clusters.pkl"), iteration_log_dir, self.get_eval_env())
    #
    #
    #     # model_load_path = os.path.join(path, "model.zip")
    #     # model = self.learner.load_for_evaluation(model_load_path, self.vec_eval_env)
    #     for i in range(0, 50):
    #         obs = self.vec_eval_env.reset()
    #         done = False
    #         while not done:
    #             action = model.step(obs, state=None, deterministic=False)
    #             obs, rewards, done, infos = self.vec_eval_env.step(action)
    #
    #     return self.eval_env.get_statistics()[1]

    def evaluate_learner(self, iteration_log_dir):
        model_load_path = os.path.join(iteration_log_dir, "model.zip")
        model = MAPPOEvalWrapper(bc.reconstruct_policy(model_load_path))
        for i in range(0, 50):
            obs = self.vec_eval_env.reset()
            done = False
            while not done:
                action = model.step(obs, state=None, deterministic=False)
                obs, rewards, done, infos = self.vec_eval_env.step(action)

        return self.eval_env.get_episodes_statistic()

    def evaluate(self):
        log_dir = self.get_log_dir()

        iteration_dirs = [d for d in os.listdir(log_dir) if d.startswith("iteration-")]
        unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
        idxs = np.argsort(unsorted_iterations)
        sorted_iteration_dirs = np.array(iteration_dirs)[idxs].tolist()
        # print(sorted_iteration_dirs)
        # First evaluate the KL-Divergences if Self-Paced learning was used
        if (self.curriculum.self_paced() or self.curriculum.self_paced_v2()) and not \
                os.path.exists(os.path.join(log_dir, "kl_divergences.pkl")):
            kl_divergences = []
            for iteration_dir in sorted_iteration_dirs:
                teacher = self.create_self_paced_teacher()
                iteration_log_dir = os.path.join(log_dir, iteration_dir)
                teacher.load(os.path.join(iteration_log_dir, "context_dist.npy"))
                kl_divergences.append(teacher.target_context_kl())

            kl_divergences = np.array(kl_divergences)
            with open(os.path.join(log_dir, "kl_divergences.pkl"), "wb") as f:
                pickle.dump(kl_divergences, f)
        # print(log_dir)
        if not os.path.exists(os.path.join(log_dir, "performance.pkl")):
            # print('elav')
            seed_performance = []
            for iteration_dir in sorted_iteration_dirs:
                iteration_log_dir = os.path.join(log_dir, iteration_dir)
                perf = self.evaluate_learner(iteration_log_dir)
                print("Evaluated " + iteration_dir + ": " + str(perf))
                seed_performance.append(perf)

            seed_performance = np.array(seed_performance)
            with open(os.path.join(log_dir, "performance.pkl"), "wb") as f:
                pickle.dump(seed_performance, f)

