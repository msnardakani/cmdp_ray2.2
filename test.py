# from dm_control import suite
import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
# import copy
from envs.contextual_ball_catching import ContextualBallCatching
from gymnasium.wrappers import TimeLimit
import gym
from envs.contextual_env import  CtxDictWrapper, ctx_visibility, exp_group, GMMCtxEnvWrapper, SquashedGMMCtxEnvWrapper

from brax.envs.half_cheetah import Halfcheetah
from envs.brax.half_cheetah_ctx import HalfcheetahCTX


from envs.epicare import EpiCare

env = EpiCare()

print(env.diseases['Disease_0'])

env2 = EpiCare()

print(env2.diseases['Disease_0'])




# ctx_mode=0
# max_steps=1000

# ctx_lb = np.array([ -np.inf, -np.inf,-np.inf, 0., 0.1])
# ctx_ub = np.array([ -.1, np.inf, np.inf, np.inf, np.inf])

# ctx_dim = 6

# config={'target_mean' : np.array([ -9.8, 0.77459666924, -0.009999999776482582, 20, 9.457333]),
#                             'target_var': np.square([ 0.5, .1, .001, 1., .5]),
#                             'init_mean':np.array([ -9.8, 0.77459666924, -0.009999999776482582, 20, 9.457333]),
#                             'init_var': np.square([ 0.5, .1, .001, 1., .5]),
#                             'prior': None}
# env_creator = lambda config: SquashedGMMCtxEnvWrapper(CtxDictWrapper(
#                 TimeLimit(HalfcheetahCTX(), max_episode_steps=max_steps), key=[ 'gravity', 'friction', 'angular_damping', 'joint_angular_damping', 'torso_mass'],ctx_visible=ctx_mode),
#                 ctx_lb=ctx_lb, ctx_ub=ctx_ub, ctx_mode=ctx_mode, **config)
# env=env_creator(config=config)
# env=HalfcheetahCTX()
env.reset()
for i in range(10000):
    env.step(env.action_space.sample())
    # env.render()
# print(img)
# Load the environment
# random_state = np.random.RandomState(42)
# env = suite.load('cheetah', 'run', task_kwargs={'random': random_state})

# # Simulate episode with random actions
# duration = 4  # Seconds
# frames = []
# ticks = []
# rewards = []
# observations = []

# spec = env.action_spec()
# time_step = env.reset()



# html_video = display_video(frames, framerate=1./env.control_timestep())

# # Show video and plot reward and observations
# num_sensors = len(time_step.observation)

# _, ax = plt.subplots(1 + num_sensors, 1, sharex=True, figsize=(4, 8))
# ax[0].plot(ticks, rewards)
# ax[0].set_ylabel('reward')
# ax[-1].set_xlabel('time')

# for i, key in enumerate(time_step.observation):
#   data = np.asarray([observations[j][key] for j in range(len(observations))])
#   ax[i+1].plot(ticks, data, label=key)
#   ax[i+1].set_ylabel(key)

# html_video.save('tst.gif')
