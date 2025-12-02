import random
from collections import deque
from enum import Enum

import numpy as np

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
        reward_shaping=True,  # Enable distance-based reward shaping
    ):
        # Grid & rendering
        self.grid_width, self.grid_height = grid_size
        self.cell_size = cell_size
        self.render_mode = render_mode

        # Rewards
        self.step_penalty = step_penalty
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.reward_shaping = reward_shaping

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
    
    # --------- Public API -----------

    def reset(self):
        '''Reset game to initial state'''
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2

        self.direction = Direction.RIGHT
        self.snake = deque()
        self.snake.appendleft((center_x, center_y))     # reset head
        self.snake.append((center_x - 1, center_y))     # place body segment behind head

        self._place_food()                              # place a piece of food

        # reset game state
        self.done = False
        self.steps_since_last_food = 0
        self.total_steps = 0
        self.score = 0

        return self._get_observation()

    def step(self, action):
        '''Take a step in environment'''
        if self.done:
            raise RuntimeError("call reset() before calling setp again() after game finishes")
        
        # Store previous distance to food for reward shaping
        old_head = self.snake[0]
        old_dist = self._manhattan_distance(old_head, self.food_pos)
        
        # update direction based on action
        self._update_direction(action)

        # compute next head postition
        head_x, head_y = self.snake[0]
        if self.direction == Direction.RIGHT:
            head_x += 1
        elif self.direction == Direction.LEFT:
            head_x -= 1
        elif self.direction == Direction.UP:
            head_y -= 1
        elif self.direction == Direction.DOWN:
            head_y += 1
        
        new_head = (head_x, head_y)

        reward = self.step_penalty

        info = {}

        # check for collisions
        if self._is_collision(new_head):
            self.done = True
            reward += self.death_penalty
            info["reason"] = "collision"
        else:
            # move snake
            self.snake.appendleft(new_head)

            # check if food ate
            if new_head == self.food_pos:
                reward += self.food_reward
                self.score += 1
                self.steps_since_last_food = 0
                self._place_food()
            else:
                # normal move; pop tail
                self.snake.pop()
                self.steps_since_last_food += 1
                
                # Reward shaping: small reward for getting closer to food
                if self.reward_shaping:
                    new_dist = self._manhattan_distance(new_head, self.food_pos)
                    if new_dist < old_dist:
                        reward += 0.1  # Moving closer to food
                    elif new_dist > old_dist:
                        reward -= 0.1  # Moving away from food

                # force termination if stuck too long; half death penalty
                if self.steps_since_last_food > self.max_steps_without_food:
                    self.done = True
                    reward += self.death_penalty / 2
                    info["reason"] = "stuck"
        
        self.total_steps += 1

        obs = self._get_observation()

        return obs, reward, self.done, info
    
    def _manhattan_distance(self, pos1, pos2):
        '''Calculate Manhattan distance between two positions'''
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def render(self, fps=15):
        '''Render game current game state with pygame'''
        if not self.render_mode:
            return
        
        if not self._pygame_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.grid_width * self.cell_size, self.grid_height * self.cell_size)
            )
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()
            self._pygame_initialized = True

        # handle quit so window can be closed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        # draw background
        self.screen.fill((0, 0, 0))

        # draw food
        fx, fy = self.food_pos
        pygame.draw.rect(
            self.screen,
            (200, 0, 0),
            pygame.Rect(
                fx * self.cell_size,
                fy * self.cell_size,
                self.cell_size,
                self.cell_size,
            ),
        )

        # draw snake
        for i, (x, y) in enumerate(self.snake):
            color = (0, 200, 0) if i == 0 else (0, 150, 0)
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                ),
            )

        pygame.display.flip()
        if self.clock is not None:
            self.clock.tick(fps)

    def close(self):
        '''Close game window'''
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False

    # --------- Internal Helpers -----------

    def _place_food(self):
        '''Place food at random empty cell'''
        available_cells = [
            (x, y)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
            if (x, y) not in self.snake
        ]
        if not available_cells:
            # winning game state
            self.food_pos = None
            self.done = True
        else:
            self.food_pos = random.choice(available_cells)

    def _is_collision(self, pos):
        x, y = pos

        # wall collision
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        # self collision
        if pos in self.snake:
            return True
        return False
    
    def _update_direction(self, action):
        '''Update direction based on relative action'''
        # stay same direction
        if action == 0:
            return
        
        curr = self.direction

        # turn left
        if action == 1:  
            if curr == Direction.UP:
                self.direction = Direction.LEFT
            elif curr == Direction.DOWN:
                self.direction = Direction.RIGHT
            elif curr == Direction.LEFT:
                self.direction = Direction.DOWN
            elif curr == Direction.RIGHT:
                self.direction = Direction.UP
        # turn right
        elif action == 2:  
            if curr == Direction.UP:
                self.direction = Direction.RIGHT
            elif curr == Direction.DOWN:
                self.direction = Direction.LEFT
            elif curr == Direction.LEFT:
                self.direction = Direction.UP
            elif curr == Direction.RIGHT:
                self.direction = Direction.DOWN

    def _get_observation(self):
        '''
        Return the current state as a feature vector.

        Enhanced representation with:
        - Immediate danger (1 cell)
        - Look-ahead danger (2 cells)
        - Direction encoding
        - Food direction
        - Body awareness (tail direction, body density)
        '''
        head_x, head_y = self.snake[0]

        # Direction one-hot flags
        dir_up_flag = int(self.direction == Direction.UP)
        dir_down_flag = int(self.direction == Direction.DOWN)
        dir_left_flag = int(self.direction == Direction.LEFT)
        dir_right_flag = int(self.direction == Direction.RIGHT)

        # Helper to check collision at position
        def is_danger(x, y):
            if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
                return 1
            if (x, y) in self.snake:
                return 1
            return 0

        # Get direction vectors
        dir_vectors = {
            Direction.UP: (0, -1),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.RIGHT: (1, 0),
        }

        curr = self.direction

        # Relative directions
        if curr == Direction.UP:
            left_dir, right_dir = Direction.LEFT, Direction.RIGHT
        elif curr == Direction.DOWN:
            left_dir, right_dir = Direction.RIGHT, Direction.LEFT
        elif curr == Direction.LEFT:
            left_dir, right_dir = Direction.DOWN, Direction.UP
        else:
            left_dir, right_dir = Direction.UP, Direction.DOWN

        # Immediate danger (1 cell ahead)
        dx, dy = dir_vectors[curr]
        danger_straight_1 = is_danger(head_x + dx, head_y + dy)
        
        dx, dy = dir_vectors[left_dir]
        danger_left_1 = is_danger(head_x + dx, head_y + dy)
        
        dx, dy = dir_vectors[right_dir]
        danger_right_1 = is_danger(head_x + dx, head_y + dy)

        # Look-ahead danger (2 cells ahead)
        dx, dy = dir_vectors[curr]
        danger_straight_2 = is_danger(head_x + 2*dx, head_y + 2*dy)
        
        dx, dy = dir_vectors[left_dir]
        danger_left_2 = is_danger(head_x + 2*dx, head_y + 2*dy)
        
        dx, dy = dir_vectors[right_dir]
        danger_right_2 = is_danger(head_x + 2*dx, head_y + 2*dy)

        # Diagonal danger (helps detect traps)
        dx_s, dy_s = dir_vectors[curr]
        dx_l, dy_l = dir_vectors[left_dir]
        dx_r, dy_r = dir_vectors[right_dir]
        
        danger_diag_left = is_danger(head_x + dx_s + dx_l, head_y + dy_s + dy_l)
        danger_diag_right = is_danger(head_x + dx_s + dx_r, head_y + dy_s + dy_r)

        # Count open spaces in each direction (path freedom)
        def count_open_spaces(start_x, start_y, dx, dy, max_dist=5):
            count = 0
            for i in range(1, max_dist + 1):
                x, y = start_x + i * dx, start_y + i * dy
                if is_danger(x, y):
                    break
                count += 1
            return count / max_dist  # Normalize

        space_straight = count_open_spaces(head_x, head_y, *dir_vectors[curr])
        space_left = count_open_spaces(head_x, head_y, *dir_vectors[left_dir])
        space_right = count_open_spaces(head_x, head_y, *dir_vectors[right_dir])

        # Food relative position (normalized)
        fx, fy = self.food_pos
        food_dx = (fx - head_x) / max(1, self.grid_width - 1)
        food_dy = (fy - head_y) / max(1, self.grid_height - 1)

        # Food direction relative to snake's heading
        food_ahead = 0
        food_left = 0
        food_right = 0
        
        if curr == Direction.UP:
            food_ahead = 1 if fy < head_y else 0
            food_left = 1 if fx < head_x else 0
            food_right = 1 if fx > head_x else 0
        elif curr == Direction.DOWN:
            food_ahead = 1 if fy > head_y else 0
            food_left = 1 if fx > head_x else 0
            food_right = 1 if fx < head_x else 0
        elif curr == Direction.LEFT:
            food_ahead = 1 if fx < head_x else 0
            food_left = 1 if fy > head_y else 0
            food_right = 1 if fy < head_y else 0
        else:  # RIGHT
            food_ahead = 1 if fx > head_x else 0
            food_left = 1 if fy < head_y else 0
            food_right = 1 if fy > head_y else 0

        # Snake length ratio (how much of grid is filled)
        length_ratio = len(self.snake) / (self.grid_width * self.grid_height)

        state = np.array(
            [
                # Immediate danger
                danger_straight_1,
                danger_left_1,
                danger_right_1,
                # Look-ahead danger
                danger_straight_2,
                danger_left_2,
                danger_right_2,
                # Diagonal danger (trap detection)
                danger_diag_left,
                danger_diag_right,
                # Open space in each direction
                space_straight,
                space_left,
                space_right,
                # Direction
                dir_up_flag,
                dir_down_flag,
                dir_left_flag,
                dir_right_flag,
                # Food info
                food_dx,
                food_dy,
                food_ahead,
                food_left,
                food_right,
                # Snake info
                length_ratio,
            ],
            dtype=np.float32,
        )

        return state