import gym
import numpy as np
from gym import spaces
from abc import ABC, abstractmethod

from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

from deep_sprl.teachers.util import Buffer
# from stable_baselines3.common.vec_env import VecEnv


class AbstractTeacher(ABC):

    @abstractmethod
    def sample(self):
        pass


class BaseWrapper(TaskSettableEnv):

    def __init__(self, env, teacher, discount_factor, context_visible,reward_from_info=False, ctx_norm = None):
        # gym.Env.__init__(self)
        self.stats_buffer = Buffer(3, 1000, True)
        self.ctx_norm = ctx_norm
        self.env = env
        self.teacher = teacher
        self.discount_factor = discount_factor

        if context_visible:
            context = self.teacher.sample()
            low_ext = np.concatenate((self.env.observation_space.low, teacher.bounds[0]))
            high_ext = np.concatenate((self.env.observation_space.high, teacher.bounds[1]))
            self.observation_space = gym.spaces.Box(low=low_ext, high=high_ext)
        else:
            self.observation_space = self.env.observation_space

        if ctx_norm:
            self.observation_space = gym.spaces.Box(low = np.ones_like(self.observation_space.low)*(-np.inf), high = np.ones_like(self.observation_space.low)*(np.inf))
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.context_space = env.context_space
        # self.context_buffer = Buffer(3, 1000, True)
        # self.ctx_hist = Buffer (2, 100, False)
        self.undiscounted_reward = 0.
        self.discounted_reward = 0.
        self.cur_disc = 1.
        self.step_length = 0.

        self.context_visible = context_visible
        self.cur_context = None
        self.cur_initial_state = None

        self.reward_from_info = reward_from_info

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward):
        # self.context_buffer.update_buffer((cur_initial_state, cur_context, discounted_reward))
        return


    def step(self, action):
        step = self.env.step(action)

        if self.context_visible:
            if self.ctx_norm:
                cur_ctx_norm = (self.cur_context- self.ctx_norm[0])/self.ctx_norm[1]
                step = np.concatenate((step[0], cur_ctx_norm)), step[1], step[2], step[3]
            else:
                step = np.concatenate((step[0], self.cur_context)), step[1], step[2], step[3]
            step_internal = np.concatenate((step[0], self.cur_context)), step[1], step[2], step[3]
            self.update(step_internal)
        else:
            self.update(step)
        return step

    def reset(self):
        self.cur_context = self.teacher.sample()
        self.env.set_task(self.cur_context.copy())
        obs = self.env.reset()

        if self.context_visible:
            obs_internal = np.concatenate((obs, self.cur_context))
            if self.ctx_norm:
                cur_ctx_norm = (self.cur_context - self.ctx_norm[0]) / self.ctx_norm[1]
                obs = np.concatenate((obs, cur_ctx_norm))
            else:
                obs = np.concatenate((obs, self.cur_context))

            self.cur_initial_state = obs_internal.copy()
        else:
            self.cur_initial_state = obs.copy()
        # self.ctx_hist.update_buffer((self.cur_initial_state, self.cur_context))
        return obs

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def update(self, step):
        reward = step[3]["reward"] if self.reward_from_info else step[1]
        self.undiscounted_reward += reward
        self.discounted_reward += self.cur_disc * reward
        self.cur_disc *= self.discount_factor
        self.step_length += 1.

        if step[2]:
            self.done_callback(step, self.cur_initial_state.copy(), self.cur_context.copy(), self.discounted_reward)

            self.stats_buffer.update_buffer((self.undiscounted_reward, self.discounted_reward, self.step_length))
            self.undiscounted_reward = 0.
            self.discounted_reward = 0.
            self.cur_disc = 1.
            self.step_length = 0.

            self.cur_context = None
            self.cur_initial_state = None

    def get_statistics(self):
        if len(self.stats_buffer) == 0:
            return 0., 0., 0
        else:
            rewards, disc_rewards, steps = self.stats_buffer.read_buffer()
            mean_reward = np.mean(rewards)
            mean_disc_reward = np.mean(disc_rewards)
            mean_step_length = np.mean(steps)

            return mean_reward, mean_disc_reward, mean_step_length
    def get_context_buffer(self):
        # ins, cons, rewards = self.context_buffer.read_buffer()
        return 0, 0, 0#np.array(ins), np.array(cons), np.array(rewards)

    def get_buffer_size(self):
        return 0

    def get_episodes_statistic(self):
        if len(self.stats_buffer) == 0:
            return 0., 0., 0, -1
        else:
            # print(len(self.stats_buffer) )
            n =len(self.stats_buffer)
            rewards, disc_rewards, steps = self.stats_buffer.read_buffer()
            mean_reward = np.mean(rewards)
            mean_disc_reward = np.mean(rewards)
            mean_step_length = np.mean(steps)

            return rewards, disc_rewards, steps, n
    def get_ctx_hist(self):
        return self.ctx_hist.read_buffer()

    def get_teacher(self):
        return None
    def update_teacher(self, weights):

        return 0

class DNCWrapper(BaseWrapper):

    def __init__(self, env, teacher, discount_factor, max_context_buffer_size=1000,
                 reset_contexts=True):
        BaseWrapper.__init__(self, env, teacher, discount_factor, context_visible = False)
        self.env.teacher = True
    def reset(self):

        self.cur_context = self.teacher.sample()
        self.env.teacher_proposal = self.cur_context
        # self.env.unwrapped.context = self.cur_context.copy()
        obs = self.env.reset()
        self.cur_initial_state= obs.copy()
        return obs

class DummyWrapper(BaseWrapper):
    def __init__(self, env, discount_factor, reward_from_info=False):
        gym.Env.__init__(self)

        self.stats_buffer = Buffer(3, 1000, True)
        self.teacher = None
        self.env = env
        # self.teacher = teacher
        self.discount_factor = discount_factor
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

        self.undiscounted_reward = 0.
        self.discounted_reward = 0.
        self.cur_disc = 1.
        self.step_length = 0.

        self.context_visible = False
        self.cur_context = None
        self.cur_initial_state = None
        self.reward_from_info = reward_from_info


    def reset(self):
        # self.cur_context = self.env.sample()
        # self.env.unwrapped.context = self.cur_context.copy()
        obs = self.env.reset()
        self.cur_context = obs.copy()
        if self.context_visible:
            obs = np.concatenate((obs, self.cur_context))

        self.cur_initial_state = obs.copy()
        return obs