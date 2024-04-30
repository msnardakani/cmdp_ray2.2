import copy
import random
import torch
import numpy as np
from gymnasium import spaces
from envs.gridworld_env import GridworldEnv, AGENT, TARGET
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env import EnvContext


def correct_region(region, map_size):
    return {k: (v[0] if v[0] > 0 else v[0] + map_size[0],
                v[1] if v[1] > 0 else v[1] + map_size[1]) for k, v in region.items()}


def box2discrete( val, grid_map_shape):
    return tuple(np.round(divmod((val +1)*grid_map_shape[0]*grid_map_shape[1]/2, grid_map_shape[1])).astype(int))


def coordinate_transform(point, grid_map_shape):
    return 2. * (grid_map_shape[1] * point[0] + point[1]) / (
        grid_map_shape[0] * grid_map_shape[1]) - 1.


class TaskSettableGridworld(GridworldEnv, TaskSettableEnv):
    def __init__(self, config: EnvContext):
        super().__init__()
        self.size = config.get("size", (10, 16))
        self.corridor = config.get("corridor", (2, 8))
        self.region = correct_region(config.get('region',{0: (1, 1), 1: (-2, -2)}), self.size)
        self.context_space = spaces.Dict({'corridor': spaces.Box(low=np.array([-1.0, ]), \
                                            high=np.array([1.0, ]),dtype=np.float64),
                                          'region': spaces.Box(low=np.array([-1.0, -1.0, ]), \
                                            high=np.array([1.0, 1.0, ]),dtype=np.float64)})
        self.agent_start_state, self.agent_target_state = self.region[0], self.region[1]
        self.reinit(size = self.size, corridor = self.corridor, start_pt= self.agent_start_state, target_pt= self.agent_target_state)
        # self.corridor = corridor
        # self.agent_start_state, self.agent_target_state = self._get_agent_start_target_state()



    def reinit(self, size, corridor,  start_pt = (1,1), target_pt=(-2, -2)):
        self.corridor = corridor
        self.size = size
        self.start_grid_map = self._create_grid_map(size=size, corridor=corridor,  start_pt = start_pt, target_pt=target_pt)  # initial grid map
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
        self.grid_map_shape = self.start_grid_map.shape
        self.agent_start_state, self.agent_target_state = self._get_agent_start_target_state()
        self.agent_state = copy.deepcopy(self.agent_start_state)

        return

    def sample_tasks(self, n_tasks):
        # Sample Corridor width and length

        tasks = []
        for i in range(n_tasks):
            width =  np.random.randint(low=1, high=self.grid_map_shape[0]-4)
            rows = (self.size[0] - width)//2

            length = np.random.randint(low=1, high=self.grid_map_shape[1]-4)
            cols = (self.size[1] - length) // 2
            row_idx = np.random.randint(low=1, high=rows)
            col_idx = np.random.randint(low=1, high=cols)

            start_pt = (random.choice([row_idx, -1-row_idx]), random.choice([col_idx, -1-col_idx]))
            while True:
                row_idx = np.random.randint(low=1, high=rows)
                col_idx = np.random.randint(low=1, high=cols)

                target_pt = (random.choice([row_idx, -1 - row_idx]), random.choice([col_idx, -1 - col_idx]))
                if target_pt!=start_pt:
                    break
            tasks.append({'corridor':(width, length),
                          'region': {0: start_pt, 1: target_pt}})

        return tasks

    def set_task(self, task):
            # Return the gridmap imported from a txt plan
        if 'region' in task:
            assert task['region'][0] != task['region'][1], "Target region must be different from Start region"

            self.start_grid_map[self.region[0]] = 0
            self.start_grid_map[self.region[1]] = 0
            self.region = {0: box2discrete(task['region'][0], self.grid_map_shape),
                           1: box2discrete(task['region'][1], self.grid_map_shape)}


            # grid_map = open(grid_map_path, 'r').readlines()
        if 'corridor' in task:
            size = self.grid_map_shape
            self.corridor = box2discrete(task['corridor'], size)


            grid_map = np.ones(shape=size, dtype=int)
            grid_map[1:-1, 1:-1] = 0
            grid_map[1:-1, (size[1] - self.corridor[1]) // 2: (size[1] - self.corridor[1]) // 2 + self.corridor[1]] = 1
            grid_map[(size[0] - self.corridor[0]) // 2:(size[0] - self.corridor[0]) // 2 + self.corridor[0],
            (size[1] - self.corridor[1]) // 2: (size[1] - self.corridor[1]) // 2 + self.corridor[1]] = 0

            self.start_grid_map = grid_map


        if 'size' in task:
            assert task['size'] == self.size
            self.start_grid_map

        # if self.start_grid_map[self.region[0]] != 0 or self.start_grid_map[self.region[1]] != 0:
        #     print(self.start_grid_map[self.region[0]], self.start_grid_map[self.region[1]])

        assert self.start_grid_map[self.region[0]] == 0 and self.start_grid_map[self.region[1]] == 0
        self.start_grid_map[self.region[0]] = AGENT
        self.start_grid_map[self.region[1]] = TARGET
        self.grid_map_shape = self.start_grid_map.shape
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.agent_state = copy.deepcopy(self.agent_start_state)
        self.agent_start_state, self.agent_target_state = self._get_agent_start_target_state()
            # self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.episode_total_reward = 0.0

        return



    def reconfig(self, config):
        self.size = config.get("size", (11, 11))
        self.corridor = config.get("corridor", (1, 3))
        self.region = correct_region(config.get('region', {0: (1, 1), 1: (-2, -2)}), self.size)
        self.agent_start_state, self.agent_target_state = self.region[0], self.region[1]
        self.reinit(size = self.size, corridor = self.corridor, start_pt= self.agent_start_state, target_pt= self.agent_target_state)
        return


    def get_task(self):
        return {'corridor': coordinate_transform(self.corridor, self.grid_map_shape),
                'region': np.asarray([coordinate_transform(self.region[0], self.grid_map_shape),
                                      coordinate_transform(self.region[1], self.grid_map_shape)]) }



    def report_task(self):
        return {'corridor': self.corridor, 'region':self.region}