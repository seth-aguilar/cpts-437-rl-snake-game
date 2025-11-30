import random
from collections import deque
from enum import Enum

import numpy as py

import pygame

class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

class SnakeEnv:
    '''
    Snake environment with gym api

    Actions:
        0 - keep going straight
        1 - turn left
        2 - turn right

    Methods:
        obs = reset()
        obs, reward, done, info = step(action)
        render()
        close()
    '''

    def __init__(
        self,
        grid_size=(20, 20),
        cell_size=20,
        step_penalty=-0.01,
        food_reward=10.0,
        death_penalty=-10.0,
        render_mode=False,
        max_steps_without_food=200,
    ):
        # Grid & rendering
        self.grid_width, self.grid_height = grid_size
        self.cell_size = cell_size
        self.render_mode = render_mode

        # Rewards
        self.step_penalty = step_penalty
        self.food_reward = food_reward
        self.death_penalty = death_penalty

        self.max_steps_without_food = max_steps_without_food

        # Pygame stuff
        self._pygame_initialized = False
        self.screen = None
        self.clock = None

        # Internal game state
        self.snake = None          # deque of (x, y)
        self.direction = None      # Direction enum
        self.food_pos = None       # (x, y)
        self.done = False
        self.steps_since_last_food = 0
        self.total_steps = 0
        self.score = 0

        # Init game state
        self.reset()