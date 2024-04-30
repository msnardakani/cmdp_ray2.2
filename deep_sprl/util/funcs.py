import collections
import copy
import os
import pickle
import warnings
from typing import Any, Dict, Optional, Type, Union, NamedTuple, Generator

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from sklearn.cluster import KMeans, AgglomerativeClustering
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize
from sklearn.mixture import GaussianMixture

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None
from stable_baselines3.common.logger import configure

from gym import spaces
from torch.nn import functional as F
# from deep_sprl.experiments.abstract_experiment import CurriculumType
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, \
    MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.ppo import PPO, MlpPolicy
from sb3_contrib.trpo import TRPO
# from torch.utils.tensorboard import SummaryWriter
import time
from stable_baselines3.common.buffers import BaseBuffer
import torch as th

import gym
import numpy as np
from gym import spaces

from deep_sprl.experiments.abstract_experiment import MAPPOInterface, PPOTRPOEvalWrapper, CurriculumType
from deep_sprl.teachers.spl import SelfPacedTeacher, SelfPacedTeacherV2, SelfPacedWrapper
from deep_sprl.teachers.spl.alpha_functions import PercentageAlphaFunction
from deep_sprl.teachers.dummy_teachers import GaussianSampler, UniformSampler, GMMSampler
from deep_sprl.teachers.abstract_teacher import BaseWrapper, DummyWrapper, DNCWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.logger import Logger

TensorDict = Dict[Union[str, int], th.Tensor]


class RolloutBufferPlusSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    means: th.Tensor
    stds: th.Tensor


class DictRolloutBufferPlusSamples(RolloutBufferPlusSamples):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    means: th.Tensor
    stds: th.Tensor


# import numpy as np
from imitation.data import types


def sb3_roll2trans(rollouts):
    """Flatten a series of trajectory dictionaries into arrays.
    Args:
        trajectories: list of trajectories.
    Returns:
        The trajectories flattened into a single batch of Transitions.
        :param rollouts:
    """
    keys = ["obs", "next_obs", "acts", "dones", "infos"]
    parts = {key: [] for key in keys}
    for traj in rollouts:
        parts["acts"].append(traj.actions.cpu().detach().numpy()[:-1])

        obs = traj.observations.cpu().detach().numpy()
        parts["obs"].append(obs[:-1])
        parts["next_obs"].append(obs[1:])

        dones = np.zeros(len(traj.actions) - 1, dtype=bool)
        #         dones[-1] = traj.terminal
        parts["dones"].append(dones)

        #         if traj.infos is None:
        #             infos = np.array([{}] * len(traj))
        #         else:
        #             infos = traj.infos
        infos = np.array([{}] * (len(traj.actions) - 1))
        parts["infos"].append(infos)

    cat_parts = {
        key: np.concatenate(part_list, axis=0) for key, part_list in parts.items()
    }
    lengths = set(map(len, cat_parts.values()))
    assert len(lengths) == 1, f"expected one length, got {lengths}"
    return types.Transitions(**cat_parts)


class RolloutBufferPlus(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):

        super(RolloutBufferPlus, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages, self.stds, self.means = None, None, None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.stds = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.means = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBufferPlus, self).reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
            mean: th.Tensor,
            std: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        # print(action)
        # print(mean.clone().cpu().numpy())
        self.stds[self.pos] = std.clone().cpu().numpy()
        self.means[self.pos] = mean.clone().cpu().numpy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "means",
                "stds",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferPlusSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.means[batch_inds],
            self.stds[batch_inds]
        )
        return RolloutBufferPlusSamples(*tuple(map(self.to_torch, data)))


class PPO_loss(PPO):  # KL pentalty loss added to PPO

    def _setup_model(self) -> None:
        super(PPO, self)._setup_model()
        self.rollout_buffer = RolloutBufferPlus(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

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
                actions, values, log_probs = self.policy(obs_tensor)
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

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts,
                               values, log_probs, dist.loc,
                               dist.scale)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self, agents=None, idx=0, penalty=0, dnc=False):
        """
                Update policy using the currently gathered rollout buffer.
                """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                all_trajectories = []
                # for i, agent in enumerate(agents):
                for i in range(1, len(agents)):
                    agent = agents[i]
                    if i != idx:
                        trj = next(agent.rollout_buffer.get(self.batch_size))
                        all_trajectories.append(trj)
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                additional_loss = 0

                if penalty > 0:
                    additional_loss = Metrics.kl_on_others(rollout_data, all_trajectories)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + penalty * additional_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        if penalty > 0:
            self.logger.record("train/kl_on_others_loss_" + str(idx), np.mean(additional_loss.cpu().numpy()))

        self.logger.record("train/entropy_loss_" + str(idx), np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss_" + str(idx), np.mean(pg_losses))
        self.logger.record("train/value_loss_" + str(idx), np.mean(value_losses))
        self.logger.record("train/approx_kl_" + str(idx), np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction_" + str(idx), np.mean(clip_fractions))
        self.logger.record("train/loss_" + str(idx), loss.item())
        self.logger.record("train/explained_variance_" + str(idx), explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates_" + str(idx), self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range_" + str(idx), clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf_" + str(idx), clip_range_vf)


#
class MAPPOwrapper:  # A multi agent version of PPO
    PENALTY = 0.01

    def __init__(self,
                 policy: Union[str, Type[ActorCriticPolicy]],
                 envs,
                 learning_rate: Union[float, Schedule] = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: Union[float, Schedule] = 0.2,
                 clip_range_vf: Union[None, float, Schedule] = None,
                 normalize_advantage: bool = True,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 target_kl: Optional[float] = None,
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
        self.PPO_agents = []
        agent_global = PPO(policy,
                           envs[0],
                           learning_rate=learning_rate,
                           n_steps=n_steps,
                           gamma=gamma,
                           gae_lambda=gae_lambda,
                           ent_coef=ent_coef,
                           vf_coef=vf_coef,
                           max_grad_norm=max_grad_norm,
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
        self.PPO_agents.append(agent_global)
        for env in envs[1:]:
            agent = PPO_loss(
                policy,
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
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
            self.PPO_agents.append(agent)
        self.cluster = None
        # print(self.PPO_agents[0].policy)
        # self.reset_agents()

        # self.logger = Logger(log_dir=self.log_dir)

    def reset_agents(self):
        for agent in self.PPO_agents:
            agent.set_parameters(self.PPO_agents[0].get_parameters())

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
        while self.PPO_agents[0].num_timesteps < total_timesteps:
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
                for i, agent in enumerate(self.PPO_agents):

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


class DnCPPO(MAPPOwrapper):
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
                                     self.cluster.weights_,
                                     (curriculum['LOWER_CONTEXT_BOUNDS'].copy(),
                                      curriculum['UPPER_CONTEXT_BOUNDS'].copy()))
            env_global = BaseWrapper(env_global, teacher_new, params['DISCOUNT_FACTOR'], context_visible=True)
            # for i in range(len(curriculum['TARGET_PRIORS'])):
        self.envs.append(env_global)
        means = env.retrieve_centers(self.cluster.means_)
        vars = env.retrieve_centers(np.array([np.diag(cov) for cov in clusters.covariances_]))
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
            observation_space=self.eval_env.observation_space,
            action_space=self.eval_env.action_space,
            # Set lr_schedule to max value to force error if policy.optimizer
            # is used by mistake (should use self.optimizer instead).
            lr_schedule=lambda _: th.finfo(th.float32).max,
            **parameters["common"]["policy_kwargs"],
        )
        # print(self.eval_env.observation_space)
        # print(bc_policy)

        # self.bc_agent.logger.output_formats

        super().__init__('MlpPolicy', self.envs, **parameters["common"], **parameters["ppo"])
        self.bc_agent = bc.BC(
            observation_space=self.eval_env.observation_space,
            action_space=self.eval_env.action_space,
            demonstrations=None,
            policy=bc_policy,
            batch_size=parameters['ppo']['n_steps'],
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
            ) for i, agent in enumerate(self.PPO_agents)]

        else:
            setup = [agent._setup_learn(
                total_timesteps, eval_env, callback[i], eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
                tb_log_name
            ) for i, agent in enumerate(self.PPO_agents)]
        iteration = 0

        for total_timesteps, callback in setup:
            callback.on_training_start(locals(), globals())
        total_timesteps = setup[0][0]
        while self.PPO_agents[0].num_timesteps < total_timesteps:
            # for i, agent in enumerate(self.PPO_agents):
            for i in range(1, len(self.PPO_agents)):
                agent = self.PPO_agents[i]
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
                for i, agent in enumerate(self.PPO_agents[1:]):

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

            for i in range(1, len(self.PPO_agents)):
                self.PPO_agents[i].train(self.PPO_agents, i, penalty=self.PENALTY)

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

            continue_training = self.PPO_agents[0].collect_rollouts(self.PPO_agents[0].env, setup[0][1],
                                                                    self.PPO_agents[0].rollout_buffer,
                                                                    n_rollout_steps=self.PPO_agents[0].n_steps)
            #
            if continue_training is False:
                break

            self.PPO_agents[0]._update_current_progress_remaining(self.PPO_agents[0].num_timesteps, total_timesteps)
            #
            # iteration += 1
            # # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                #
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                # fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                #
                if len(self.PPO_agents[0].ep_info_buffer) > 0 and len(self.PPO_agents[0].ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean_" + str(0),
                                       safe_mean([ep_info["r"] for ep_info in self.PPO_agents[0].ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean_" + str(0),
                                       safe_mean([ep_info["l"] for ep_info in self.PPO_agents[0].ep_info_buffer]))
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
                for i, agent in enumerate(self.PPO_agents):
                    agent.save(os.path.join(iter_log_dir, 'model_dist_' + str(i) + '.zip'))

        for total_timesteps, callback in setup:
            callback.on_training_end()

        return self

    def distill(self, iterations=10):

        rollouts = []

        rollouts = [next(expert.rollout_buffer.get(expert.rollout_buffer.buffer_size)) for expert in self.PPO_agents[1:]]
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
        for agent in self.PPO_agents:
            # state_dict['log_std'] = agent.policy.get_parameter('log_std')
            agent.policy.load_state_dict(state_dict, strict=False)
            agent.policy.optimizer.state = collections.defaultdict(dict)

    def get_interface(self):
        return [MAPPOInterface(self.PPO_agents[i], self.envs[i].observation_space.shape[0], idx=i) for i in
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
            PPO_agents.append(PPO.load(env=env, path=os.path.join(model_dir, "model" + str(i) + ".zip")))

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
        model = PPOTRPOEvalWrapper(bc.reconstruct_policy(model_load_path))
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


class Metrics:
    @staticmethod
    def symmetric_kl(info_vars_1, info_vars_2):
        side1 = th.mean(Metrics.diag_gaussian_kl(info_vars_2, info_vars_1))
        side2 = th.mean(Metrics.diag_gaussian_kl(info_vars_1, info_vars_2))
        return (side1 + side2) / 2

    @staticmethod
    def kl_on_others(dist, dist_info_vars):
        # print(dist)
        # print(type(dist_info_vars))
        # \sum_{j=1} E_{\sim S_j}[D_{kl}(\pi_j || \pi_i)]
        # if len(dist_info_vars) < 2:
        #     return 0

        kl_with_others = 0
        for i in range(len(dist_info_vars)):
            # if i != n:
            kl_with_others += Metrics.symmetric_kl(dist, dist_info_vars[i])

        return kl_with_others / (len(dist_info_vars))  # - 1)

    @staticmethod
    def diag_gaussian_kl(old_dist, new_dist):
        # print(old_dist)
        # print(new_dist)
        old_means = old_dist.means
        old_std = old_dist.stds
        new_means = new_dist.means
        new_std = new_dist.stds
        new_log_stds = th.log(new_std)
        old_log_stds = th.log(old_std)
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices
        """
        # old_std = th.exp(old_log_stds)
        # new_std = th.exp(new_log_stds)
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        numerator = th.square(old_means - new_means) + \
                    th.square(old_std) - th.square(new_std)
        denominator = 2 * th.square(new_std) + 1e-8
        out = numerator / denominator + new_log_stds - old_log_stds
        return out.sum(axis=-1)
