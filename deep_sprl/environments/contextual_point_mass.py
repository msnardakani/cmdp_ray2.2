import numpy as np
import time

from gymnasium import Env, spaces
from ..util.viewer import Viewer
import pygame
from skimage.draw import  rectangle, ellipse, disk
import matplotlib.pyplot as plt
class ContextualPointMass(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, context=np.array([0., 2., 2.])):#, horizon = 100):
        self.action_space = spaces.Box(np.array([-10., -10.], dtype=np.float64), np.array([10., 10.], dtype=np.float64),dtype=np.float64)
        self.observation_space = spaces.Box(np.array([-4., -np.inf, -4., -np.inf], dtype=np.float64),
                                            np.array([4., np.inf, 4., np.inf], dtype=np.float64), dtype=np.float64)

        self._state = None
        self._goal_state = np.array([0., 0., -3., 0.])
        self.context = context
        self._dt = 0.01
        self._viewer = Viewer(8, 8, background=(255, 255, 255))
        # self.steps = 0
        # self.horizon = horizon

        self.screen = self._init_screen()

    def _init_screen(self):
        img= 255*np.ones((800, 800, 3), dtype=np.uint8)
        rr, cc = ellipse(r=700, c=400, r_radius=80, c_radius=8, rotation=45)
        img[rr, cc, :] = (255, 0, 0)


        rr, cc = ellipse(r=700, c=400, r_radius=80, c_radius=8, rotation=-45)
        img[rr, cc, :] = (255, 0, 0)
        return img

    def reset(self, *, seed=None, options=None):
        # self.steps=0
        self._state = np.array([0., 0., 3., 0.], dtype=np.float64)
        return np.copy(self._state), {}

    def _step_internal(self, state, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        state_der = np.zeros(4)
        state_der[0::2] = state[1::2]
        friction_param = self.context[2]
        state_der[1::2] = 1.5 * action - friction_param * state[1::2] + np.random.normal(0, 0.05, (2,))
        new_state = np.clip(state + self._dt * state_der, self.observation_space.low,
                            self.observation_space.high)

        crash = False
        if state[2] >= 0 > new_state[2] or state[2] <= 0 < new_state[2]:
            alpha = (0. - state[2]) / (new_state[2] - state[2])
            x_crit = alpha * new_state[0] + (1 - alpha) * state[0]

            if np.abs(x_crit - self.context[0]) > 0.5 * self.context[1]:
                new_state = np.array([x_crit, 0., 0., 0.], dtype=np.float64)
                crash = True

        return new_state, crash

    def step(self, action):
        if self._state is None:
            raise RuntimeError("State is None! Be sure to reset the environment before using it")

        action = np.clip(action, self.action_space.low, self.action_space.high)

        new_state = self._state
        crash = False
        for i in range(0, 10):
            # self.steps+=1
            new_state, crash = self._step_internal(new_state, action)

            if crash :#or self.steps>self.horizon:
                # crash = True
                break

        self._state = np.copy(new_state)
        distance = np.linalg.norm(self._goal_state[0::2] - new_state[0::2])
        success = distance < 0.25
        info = {"is_success": success}
        reward = np.exp(-0.6 * distance) + 10 if success else np.exp(-0.6 * distance)

        return new_state, reward, crash or success, False,  info

    def render(self, mode='rgb_array'):
        pos = self.context[0] + 4.
        width = self.context[1]
        if mode == 'human':

            self._viewer.line(np.array([0., 4.]), np.array([np.clip(pos - 0.5 * width, 0., 8.), 4.]), color=(0, 0, 0),
                              width=0.2)
            self._viewer.line(np.array([np.clip(pos + 0.5 * width, 0., 8, ), 4.]), np.array([8., 4.]), color=(0, 0, 0),
                              width=0.2)

            self._viewer.line(np.array([3.9, 0.9]), np.array([4.1, 1.1]), color=(255, 0, 0), width=0.1)
            self._viewer.line(np.array([4.1, 0.9]), np.array([3.9, 1.1]), color=(255, 0, 0), width=0.1)

            self._viewer.circle(self._state[0::2] + np.array([4., 4.]), 0.1, color=(0, 0, 0))

            self._viewer.display(self._dt)
            return None
        else:

            frame = self.screen.copy()
            rr, cc = rectangle((380, 0), (420, max(int((pos-width/2)*100),0)))
            frame[rr, cc, :] = (0, 0, 0)
            rr, cc = rectangle((380, min(int((pos+width/2)*100), 799)), (420, 799))
            frame[rr, cc, :] = (0, 0, 0)
            center = (  np.array([4.-self._state[2], 4.+self._state[0]])) *100
            rr, cc = disk(center.astype(int), 20, shape=frame.shape)
            frame[rr,cc, :] = (25,25, 170)
            # frame = (frame*255).astype(np.uint8)

            # print(screen)
            return frame
