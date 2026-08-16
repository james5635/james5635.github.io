# Train a Neural Network to Play the Snake Game

## Introduction
In this tutorial, we will walk through the process of training a neural network to play the classic Snake game using reinforcement learning. The neural network will learn to make decisions based on the current state of the game, aiming to maximize its score by collecting food while avoiding collisions.

## Prerequisites
Before we begin, ensure you have the following installed:

- Python 3.x
- PyTorch
- Pygame

## Setup
```bash
pip install torch torchvision torchaudio pygame
```
## Code Overview
The code is organized into several files, each responsible for different aspects of the game and the training process:

- `main.py`: The main training loop where the agent interacts with the game environment.
- `agent.py`: Defines the Agent class, which includes methods for decision-making, memory management, and training the neural network.
- `game.py`: Implements the Snake game logic, including the game state, collision detection, and rendering.
- `model.py`: Defines the neural network architecture used by the agent to predict actions based on the game state.

```python {title="main.py"}
from agent import Agent
from game import SnakeGame
import time


def train():
    agent = Agent()
    game = SnakeGame()  # Custom Pygame or UI-less game instance
    game.set_theme("monokai")

    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state parameters
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Game over: reset environment and train long-term memory
            game.reset()
            agent.n_games += 1
            print("memory: " + str(len(agent.memory)))
            agent.train_long_memory()
            print(f"Game {agent.n_games}, Score: {score}")
            # time.sleep(1)


train()

```

```python {title="agent.py"}
import torch.optim as optim
import random
import numpy as np
from collections import deque
from model import LinearQNet
import torch.nn as nn
import torch
from game import SnakeGame


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0  # Exploration rate
        self.gamma = 0.9  # Discount rate for future rewards
        self.memory = deque(maxlen=100_000)  # Memory buffer for experience replay
        self.model = LinearQNet(11, 256, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def get_state(self, game: SnakeGame):
        # Implement logic to extract the 11 boolean variables based on game physics
        # (Returns a numpy array of shape (11,) filled with 0s and 1s)
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        # [11],[3],x,[11],x
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        # Epsilon-Greedy Strategy
        self.epsilon = max(0.05, 1.0 - (self.n_games * 0.01))  # Decrease exploration over time
        final_move = [0, 0, 0]

        # [0,1.0)
        if random.random() < self.epsilon:
            move_idx = random.randint(0, 2)
            final_move[move_idx] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move_idx = torch.argmax(prediction).item()
            final_move[move_idx] = 1

        return final_move

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train on a single step to quickly adapt to immediate feedback
        self.train_step([state], [action], [reward], [next_state], [done])

    def train_long_memory(self):
        # Experience Replay: Train on a random batch of past memories
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def train_step(self, state, action, reward, next_state, done):
        # Convert all training inputs to PyTorch Tensors
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        # 1. Predicted Q values with current state
        pred = self.model(state)

        # 2. Compute Bellman Equation Target: Q_new = reward + gamma * max(next_predicted_Q_value)
        #    Adjust target value by using new Q value
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                )

            # Update target value for the specific action taken
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3. Optimize parameters
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


```


```python {title="game.py}


# using pygame
import random
from collections import namedtuple
from typing import List, Tuple
import numpy as np
import pygame  # Make sure to run: pip install pygame

# A clean way to handle 2D coordinates
Point = namedtuple("Point", ["x", "y"])

THEMES = {
    "monokai": {
        "bg": (46, 46, 46),  # Charcoal (#2E2E2E)
        "food": (255, 97, 136),  # Pink (#FF6188)
        "head": (166, 226, 46),  # Green (#A6E22E)
        "body1": (244, 191, 117),  # Yellow (#F4BF75)
        "body2": (253, 151, 31),  # Orange (#FD971F)
        "text": (248, 248, 242),  # Off-white (#F8F8F2)
    },
    "catppuccin": {
        "bg": (30, 30, 46),  # Mocha Base (#1E1E2E)
        "food": (245, 194, 231),  # Pink (#F5C2E7)
        "head": (166, 227, 161),  # Green (#A6E3A1)
        "body1": (137, 180, 250),  # Blue (#89B4FA)
        "body2": (180, 190, 254),  # Lavender (#B4BEFE)
        "text": (205, 214, 244),  # Text (#CDD6F4)
    },
    "one-dark": {
        "bg": (40, 44, 52),  # Dark Gray (#282C34)
        "food": (224, 108, 117),  # Dark Red (#E06C75)
        "head": (152, 195, 121),  # Light Green (#98C379)
        "body1": (97, 175, 239),  # Blue (#61AFEF)
        "body2": (198, 120, 221),  # Purple (#C678DD)
        "text": (171, 178, 191),  # Light Gray (#ABB2BF)
    },
    "classic": {
        "bg": (0, 0, 0),  # Black
        "food": (200, 0, 0),  # Red
        "head": (0, 0, 255),  # Blue
        "body1": (0, 0, 255),  # Blue
        "body2": (0, 100, 255),  # Light Blue
        "text": (255, 255, 255),  # White
    },
}


class SnakeGame:
    def __init__(
        self, width: int = 640, height: int = 480, block_size: int = 20, fps: int = 40
    ) -> None:
        self.w: int = width
        self.h: int = height
        self.block_size: int = block_size
        self.fps: int = fps  # Controlling training/game view speed

        # 1. Initialize Pygame Video and Text Engines
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("arial", 25)

        # 2. Setup Windows Surfaces
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake AI Training Window")
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self) -> None:
        """Resets the game state completely for a brand new game."""
        self.head: Point = Point(self.w / 2, self.h / 2)

        self.snake: List[Point] = [
            self.head,
            Point(self.head.x - self.block_size, self.head.y),
            Point(self.head.x - (2 * self.block_size), self.head.y),
        ]

        self.score: int = 0
        self.food: Point = Point(0, 0)
        self._place_food()

        self.direction: int = 0
        self.frame_iteration: int = 0

    def _place_food(self) -> None:
        x: int = (
            random.randint(0, (self.w - self.block_size) // self.block_size)
            * self.block_size
        )
        y: int = (
            random.randint(0, (self.h - self.block_size) // self.block_size)
            * self.block_size
        )
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def play_step(self, action: List[int]) -> Tuple[float, bool, int]:
        """Executes one single game step, updates window drawings, and ticks frame-rate clock."""
        self.frame_iteration += 1

        # 1. Capture User Exit Request (Ensures you can close the window cleanly)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Calculate movement shifts
        self._move(action)
        self.snake.insert(0, self.head)

        reward: float = 0.0
        game_over: bool = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10.0
            return reward, game_over, self.score

        # 3. Check food collection mechanics
        if self.head == self.food:
            self.score += 1
            reward = 10.0
            self._place_food()
        else:
            self.snake.pop()

        # 4. Render the current frame visually to the display window
        self._update_ui()
        self.clock.tick(
            self.fps
        )  # Slows execution speed down so a human can track moves

        return reward, game_over, self.score

    def is_collision(self, pt: Point = None) -> bool:
        if pt is None:
            pt = self.head

        if pt.x > self.w - self.block_size or pt.x < 0:
            return True
        if pt.y > self.h - self.block_size or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True

        return False

    def _move(self, action: List[int]) -> None:
        clockwise_directions: List[int] = [0, 1, 2, 3]

        if action[0] == 1:
            pass
        elif action[1] == 1:
            next_idx = (self.direction + 1) % 4
            self.direction = clockwise_directions[next_idx]
        elif action[2] == 1:
            next_idx = (self.direction - 1) % 4
            self.direction = clockwise_directions[next_idx]

        x: float = self.head.x
        y: float = self.head.y

        if self.direction == 0:  # RIGHT
            x += self.block_size
        elif self.direction == 1:  # DOWN
            y += self.block_size
        elif self.direction == 2:  # LEFT
            x -= self.block_size
        elif self.direction == 3:  # UP
            y -= self.block_size

        self.head = Point(x, y)

    def set_theme(self, theme_name: str) -> None:
        """Dynamically switches the active drawing color profile palette."""
        # Fall back to classic mode if a typo is entered
        if theme_name not in THEMES:
            theme_name = "classic"

        self.current_theme = THEMES[theme_name]

    def _update_ui(self) -> None:
        """Draws all game assets onto the canvas using the current selected theme profile."""
        theme = self.current_theme

        # Clean background slate
        self.display.fill(theme["bg"])

        # Draw the Snake
        for idx, pt in enumerate(self.snake):
            # Highlight the head block uniquely
            color1 = theme["head"] if idx == 0 else theme["body1"]
            color2 = theme["body2"]

            pygame.draw.rect(
                self.display,
                color1,
                pygame.Rect(pt.x, pt.y, self.block_size, self.block_size),
            )
            pygame.draw.rect(
                self.display, color2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
            )

        # Draw Food Asset
        pygame.draw.rect(
            self.display,
            theme["food"],
            pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size),
        )

        # Display Current Game Score Text Overlay
        text_surface = self.font.render(
            "Score: " + str(self.score), True, theme["text"]
        )
        self.display.blit(text_surface, [10, 10])

        # Flip the double-buffer surface to make drawings live
        pygame.display.flip()

    def get_state(self) -> np.ndarray:
        point_l = Point(self.head.x - self.block_size, self.head.y)
        point_r = Point(self.head.x + self.block_size, self.head.y)
        point_u = Point(self.head.x, self.head.y - self.block_size)
        point_d = Point(self.head.x, self.head.y + self.block_size)

        dir_l = self.direction == 2
        dir_r = self.direction == 0
        dir_u = self.direction == 3
        dir_d = self.direction == 1

        state = [
            (dir_r and self.is_collision(point_r))
            or (dir_l and self.is_collision(point_l))
            or (dir_u and self.is_collision(point_u))
            or (dir_d and self.is_collision(point_d)),
            (dir_u and self.is_collision(point_r))
            or (dir_d and self.is_collision(point_l))
            or (dir_l and self.is_collision(point_u))
            or (dir_r and self.is_collision(point_d)),
            (dir_d and self.is_collision(point_r))
            or (dir_u and self.is_collision(point_l))
            or (dir_r and self.is_collision(point_u))
            or (dir_l and self.is_collision(point_d)),
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            self.food.x < self.head.x,
            self.food.x > self.head.x,
            self.food.y < self.head.y,
            self.food.y > self.head.y,
        ]

        return np.array(state, dtype=np.int64)

```

```python {title="model.py"}

import torch
import torch.nn as nn

# The Neural Network Architecture
class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)

```

## Demonstration
To see the neural network in action, watch the following demonstration video:

<iframe width="800" height="445" src="https://www.youtube.com/embed/XVrmH0q45kA?si=HoDSXsCBA9jjVbo7" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>