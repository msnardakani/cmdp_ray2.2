# import gym
import sys
# import os
import copy
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.utils import seeding
# import itertools
EMPTY = BLACK = 0
WALL = GRAY = 1
TARGET = GREEN = 3
AGENT = RED = 4
SUCCESS = PINK = 6
COLORS = {BLACK: [0.0, 0.0, 0.0], GRAY: [0.5, 0.5, 0.5], GREEN: [0.0, 1.0, 0.0],
          RED: [1.0, 0.0, 0.0], PINK: [1.0, 0.0, 1.0]}

NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4
UP_RIGHT = 5
DOWN_RIGHT = 6
UP_LEFT = 7
DOWN_LEFT = 8

# MAX_T = 15


class GridworldEnv():
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, render_mode = 'rgb_array'):
        size = (11, 11)
        corridor = (1, 3)
        self.render_mode = render_mode
        self.actions = [NOOP, UP, DOWN, LEFT, RIGHT, UP_RIGHT, DOWN_RIGHT, UP_LEFT, DOWN_LEFT]
        self.inv_actions = [0, 2, 1, 4, 3,5,6,7,8]
        self.action_space = spaces.Discrete(9)
        self.action_pos_dict = {NOOP: [0, 0], UP: [-1, 0], DOWN: [1, 0], LEFT: [0, -1], RIGHT: [0, 1],
                                UP_RIGHT : [-1,1], DOWN_RIGHT : [1,1], UP_LEFT:  [-1, -1], DOWN_LEFT : [1, -1], }

        self.img_shape = [256, 256, 3]  # visualize state

        # initialize system state
        # this_file_path = os.path.dirname(os.path.realpath(__file__))
        # self.grid_map_path = os.path.join(this_file_path, 'plan{}.txt'.format(plan))
        self.start_grid_map = self._create_grid_map(size=size, corridor=corridor)  # initial grid map
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
        self.grid_map_shape = self.start_grid_map.shape

        self.observation_space = spaces.Box(low=np.array([-1.0, -1.0, ], dtype=np.float64), \
                                            high=np.array([1.0, 1.0, ], dtype=np.float64), dtype=np.float64)

        #
        # self.observation_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0], dtype=np.float64), \
        #                                     high=np.array([1.0, 1.0, 1.0], dtype=np.float64),dtype=np.float64)

        # agent state: start, target, current state
        self.agent_start_state, self.agent_target_state = self._get_agent_start_target_state()
        self.agent_state = copy.deepcopy(self.agent_start_state)

        # set other parameters
        self.restart_once_done = False  # restart or not once done
        self.time_steps = 0
        # set seed
        # self.seed()

        # consider total episode reward
        self.episode_total_reward = 0.0

        # consider viewer for compatibility with gym
        self.viewer = None

    def seed(self, seed=None):

        # Fix seed for reproducibility

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self, coordinates, action, reward):

        # Return a triple with: current location of the agent in the map
        # given coordinates, the previous action and the previous reward

        ## Normalized for better perform of the NN


        return np.asarray([2. * (self.grid_map_shape[1] * coordinates[0] + coordinates[1]) / (
                    self.grid_map_shape[0] * self.grid_map_shape[1]) - 1., \
                           (action - 4.5) / 9., ])#reward])

    def step(self, action):
        # self.time_steps +=1

        # Return next observation, reward, finished, success

        action = int(action)
        info = {'success': False, }
        done = False

        # Penalties
        penalty_step = 1
        penalty_wall = 0
        # truncated = self.time_steps>=MAX_T
        truncated = False
        reward = -penalty_step
        nxt_agent_state = (self.agent_state[0] + self.action_pos_dict[action][0],
                           self.agent_state[1] + self.action_pos_dict[action][1])

        next_state_out_of_map = (nxt_agent_state[0] < 0 or nxt_agent_state[0] >= self.grid_map_shape[0]) or \
                                (nxt_agent_state[1] < 0 or nxt_agent_state[1] >= self.grid_map_shape[1])
        # successful behavior
        target_position = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]

        if action == NOOP:
            info['success'] = True
            info['distance'] = (self.agent_state[0] - self.agent_target_state[0])**2 + (self.agent_state[1] -  self.agent_target_state[1])**2
            # if truncated:
            #     reward += - info['distance']

            self.episode_total_reward += reward  # Update total reward

            return self.get_state(self.agent_state, action, reward), reward, False, truncated,  info

        # Make a step


        if next_state_out_of_map:
            info['success'] = False
            # info['distance'] = abs(self.agent_state[0] - self.agent_target_state[0]) + abs(self.agent_state[1] -  self.agent_target_state[1])
            info['distance'] = (self.agent_state[0] - self.agent_target_state[0])**2 + (self.agent_state[1] -  self.agent_target_state[1])**2

            # if truncated:
            #     reward+= - info['distance']
                # reward += 2*(.5 - info['distance']/(self.grid_map_shape[1]-1 + self.grid_map_shape[0]-1)) +0.1
            self.episode_total_reward += reward  # Update total reward

            return self.get_state(self.agent_state, action, reward), reward, False, truncated, info



        if target_position == EMPTY:

            self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = AGENT

        elif target_position == WALL:
            # info['distance'] = abs(self.agent_state[0] - self.agent_target_state[0])  + abs(self.agent_state[1] -  self.agent_target_state[1])
            info['distance'] = (self.agent_state[0] - self.agent_target_state[0])**2 + (self.agent_state[1] -  self.agent_target_state[1])**2

            # if truncated:
            #     reward += -info['distance']
                # reward += 2*(.5 - info['distance']/(self.grid_map_shape[1]-1 + self.grid_map_shape[0]-1)) +0.1
            info['success'] = False
            self.episode_total_reward += reward   # Update total reward

            return self.get_state(self.agent_state, action, reward-penalty_wall), (reward - penalty_wall), False, truncated, info

        elif target_position == TARGET:

            self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = SUCCESS

        self.current_grid_map[self.agent_state[0], self.agent_state[1]] = EMPTY
        self.agent_state = copy.deepcopy(nxt_agent_state)

        info['success'] = True

        # info['distance'] = abs(self.agent_state[0] - self.agent_target_state[0])  + abs(
        #             self.agent_state[1] - self.agent_target_state[1])
        info['distance'] = (nxt_agent_state[0] - self.agent_target_state[0]) ** 2 + (
                    nxt_agent_state[1] - self.agent_target_state[1]) ** 2

        if nxt_agent_state[0] == self.agent_target_state[0] and nxt_agent_state[1] == self.agent_target_state[1]:
            done = True
            info['distance'] = 0.0
            # reward +=1
            if self.restart_once_done:
                self.reset()

        # if done or truncated:
        #     reward+= -info['distance']
        self.episode_total_reward += reward  # Update total reward
        return self.get_state(self.agent_state, action, reward), reward, done, truncated, info

    def reset(self, *, seed=None, options=None):

        # Return the initial state of the environment
        self.time_steps=0
        self.agent_state = copy.deepcopy(self.agent_start_state)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.episode_total_reward = 0.0
        return self.get_state(self.agent_state, 0.0, 0.0), {}

    def close(self):
        if self.viewer: self.viewer.close()
        return

    def _create_grid_map(self, size, corridor, start_pt = (1,1), target_pt=(-2, -2)):

        # Return the gridmap imported from a txt plan

        # grid_map = open(grid_map_path, 'r').readlines()
        assert (corridor[0] < size[0] - 3) & (corridor[1] < size[1] - 3)
        grid_map = np.ones(shape=size, dtype=int)

        grid_map[1:-1, 1:-1] = 0
        grid_map[1:-1, (size[1] - corridor[1]) // 2: (size[1] + corridor[1]) // 2 ] = 1
        grid_map[(size[0] - corridor[0]) // 2:(size[0] + corridor[0]) // 2 ,
        (size[1] - corridor[1]) // 2: (size[1] + corridor[1]) // 2 ] = 0
        assert grid_map[start_pt] == 0 and grid_map[target_pt] == 0
        grid_map[start_pt] = AGENT
        grid_map[target_pt] = TARGET

        return grid_map



    def _get_agent_start_target_state(self):
        start_state = np.where(self.start_grid_map == AGENT)
        target_state = np.where(self.start_grid_map == TARGET)

        start_or_target_not_found = not (start_state[0] and target_state[0])
        if start_or_target_not_found:
            sys.exit('Start or target state not specified')
        start_state = (start_state[0][0], start_state[1][0])
        target_state = (target_state[0][0], target_state[1][0])

        return start_state, target_state

    def _gridmap_to_image(self, img_shape=None):

        # Return image from the gridmap

        if img_shape is None:
            img_shape = self.img_shape
        observation = np.random.randn(*img_shape) * 0.0
        gs0 = int(observation.shape[0] / self.current_grid_map.shape[0])
        gs1 = int(observation.shape[1] / self.current_grid_map.shape[1])
        for i in range(self.current_grid_map.shape[0]):
            for j in range(self.current_grid_map.shape[1]):
                for k in range(3):
                    this_value = COLORS[self.current_grid_map[i, j]][k]
                    observation[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1, k] = this_value
        return (255 * observation).astype(np.uint8)
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            # gym.logger.warn(
            #     "You are calling render method without specifying any render mode. "
            #     "You can specify the render_mode at initialization, "
            #     f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            # )
            return
        else:
            return self._render(mode=self.render_mode)

    def _render(self, mode='human', close=False):

        # Returns a visualization of the environment according to specification

        if close:
            plt.close(1)  # Final plot
            return

        img = self._gridmap_to_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            plt.figure()
            plt.imshow(img)
            return
