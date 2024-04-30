# from envs.contextual_env import GMMCtxEnvWrapper
# from envs.contextual_env import make_multi_agent_divide_and_conquer
# from envs.point_mass_2d import PointMassEnv
# from gym.wrappers import TimeLimit
# import numpy as np
#
# max_steps = 128
# ctx_lb = np.array([-4, .5])
# ctx_ub = np.array([4, 4])
# ctx_priors = np.array([1, 1])
# ctx_means = np.array([[3, 1],[-3, 1]])
# ctx_vars = np.array([[.01, .01],[.1, .01]])
# env_creator = lambda config: GMMCtxEnvWrapper(TimeLimit(PointMassEnv(context=np.array([3, .5])), max_episode_steps=max_steps ), ctx_lb=ctx_lb, ctx_ub = ctx_ub, ctx_visible= True,**config )
# config = dict(ctx_means = ctx_means, ctx_vars = ctx_vars, ctx_priors = ctx_priors)
# env = env_creator(config)
#
#
# MADnCEnv = make_multi_agent_divide_and_conquer(env_name_or_creator=env_creator)
#
#
# config2 = dict(num_agents = 3, env_mapping = [0, 2, 1],
#                agent_config = [config,
#                          dict(ctx_means = np.array([[-2, 2],]), ctx_vars = np.array([[.01, .01],])),
#                          dict(ctx_means = np.array([[0, 1],]), ctx_vars = np.array([[.1, .1],]))])
# env_ =MADnCEnv(config2)


# from envs.contextual_ball_catching import ContextualBallCatching
# env = ContextualBallCatching()
# env.reset()
# #%%
# # print('sss')
# print(env.render())

import gymnasium as gym
from gymnasium.envs.registration import register
import mujoco_py
register(
    id='ContextualBallCatching-v1',
    max_episode_steps=200,
    entry_point='envs.contextual_ball_catching:ContextualBallCatching'
)
from envs.contextual_ball_catching import ContextualBallCatching
# env = gym.make('Ant-v3', render_mode = 'human')
# env = gym.make('ContextualBallCatching-v1', render_mode='human')
env = ContextualBallCatching(render_mode= 'human')
obs,_ =env.reset()
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(i, terminated, truncated)
    env.render()
    if terminated or truncated:
        break