import gym
import gym.spaces as spaces
import numpy as np

import numba
import numba.types as nt


@numba.njit(nt.void(nt.bool_[:, ::1], nt.int_, nt.int_, nt.float_, nt.float_), cache=True)
def maze_generation(maze, width=81, height=51, complexity=.75, density=.75):
    """
    Simple maze generation algorithm I've taken from Wikipedia
    https://en.wikipedia.org/wiki/Maze_generation_algorithm#Python_code_example

    Should do the job, maybe some other algorithm can be fasteer, but this one was free in development time ;)
    """
    # Only odd shapes
    assert width % 2 == 1, "Only odd maze shapes are supported"
    assert height % 2 == 1, "Only odd maze shapes are supported"

    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (width + height))) # number of components
    density    = int(density * ((width // 2) * (height // 2))) # size of components

    # Fill borders
    maze[0, :] = maze[-1, :] = 1
    maze[:, 0] = maze[:, -1] = 1

    # Make aisles
    for i in range(density):
        # pick a random position
        x, y = np.random.randint(0, height // 2 + 1) * 2, np.random.randint(0, width // 2 + 1) * 2

        maze[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:             neighbours.append((y, x - 2))
            if x < height - 2:  neighbours.append((y, x + 2))
            if y > 1:             neighbours.append((y - 2, x))
            if y < width - 2:  neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[np.random.randint(0, len(neighbours))]

                if maze[y_, x_] == 0:
                    maze[y_, x_] = 1
                    maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_


@numba.njit(nt.void(nt.uint8[:, :, ::1], nt.bool_[:, ::1], nt.Tuple([nt.uint8, nt.uint8, nt.uint8])), cache=True)
def fill_observation_from_maze(observation, maze, wall_color):
    """ Fill observation from maze """

    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j]:
                observation[i, j] = wall_color


@numba.njit(nt.void(numba.int32[::1], nt.bool_[:, ::1], numba.int32[::1], numba.int_), cache=True)
def move_agent(state: np.ndarray, maze: np.ndarray, bounds: np.ndarray, action_idx: int) -> None:
    new_state = state.copy()

    if action_idx == 1:
        new_state[0] = max(state[0] - 1, 0)
    elif action_idx == 2:
        new_state[0] = min(state[0] + 1, bounds[0]-1)
    elif action_idx == 3:
        new_state[1] = max(state[1] - 1, 0)
    elif action_idx == 4:
        new_state[1] = min(state[1] + 1, bounds[1]-1)

    if not maze[new_state[0], new_state[1]]:
        state[0] = new_state[0]
        state[1] = new_state[1]


class RandomMazeEnv(gym.Env):
    """
    An environment where the goal is to solve a random maze.
    """

    wall_color = (255, 255, 255)
    agent_color = (0, 69, 173)
    goal_color = (255, 213, 0)

    metadata = {'render.modes': [
        'rgb_array'
    ]}

    action_space = spaces.Discrete(5)

    action_meaning = {
        0: 'NOOP',
        1: 'UP',
        2: 'DOWN',
        3: 'LEFT',
        4: 'RIGHT'
    }

    def __init__(self, maze_width, maze_height, maze_complexity=0.75, maze_density=0.75, start_end_distance=None,
                 win_reward=1.0, move_reward=0.0):

        self.maze_width = maze_width
        self.maze_height = maze_height
        self.maze_complexity = maze_complexity
        self.maze_density = maze_density

        self.win_reward = win_reward
        self.move_reward = move_reward

        self.start_end_distance = start_end_distance

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.maze_width, self.maze_height, 3), dtype=np.uint8)

        self.maze = None
        self.observation = None

        self.agent_location = None
        self.goal_location = None

        self._bounds = np.array([self.maze_width, self.maze_height], dtype=np.int32)
        self._agent_color_array = np.array(self.agent_color, dtype=np.uint8)
        self._goal_color_array = np.array(self.goal_color, dtype=np.uint8)

        self._info = {'start_end_distance': self.start_end_distance}

        self._internal_reset()

    def _internal_reset(self):
        self.maze = np.zeros((self.maze_width, self.maze_height), dtype=bool)
        maze_generation(self.maze, self.maze_width, self.maze_height, self.maze_complexity, self.maze_density)

        self.observation = np.zeros((self.maze_width, self.maze_height, 3), dtype=np.uint8)

        fill_observation_from_maze(self.observation, self.maze, self.wall_color)

        self.goal_location = None
        self.agent_location = None

        # Find suitable goal location
        for i in range(1000):
            i = np.random.randint(self.maze_width)
            j = np.random.randint(self.maze_height)

            if self.maze[i, j]:
                continue
            else:
                self.goal_location = np.array([i, j], dtype=np.int32)

        assert self.goal_location is not None, "Unable to find good goal location"

        # Find suitable agent starting location
        for i in range(1000):
            if self.start_end_distance is None:
                i = np.random.randint(self.maze_width)
                j = np.random.randint(self.maze_height)
            else:
                i = np.random.randint(
                    low=max(0, self.goal_location[0] - self.start_end_distance),
                    high=min(self.maze_width, self.goal_location[0] + self.start_end_distance)
                )

                j = np.random.randint(
                    low=max(0, self.goal_location[1] - self.start_end_distance),
                    high=min(self.maze_height, self.goal_location[1] + self.start_end_distance)
                )

            if self.maze[i, j]:
                continue
            elif i == self.goal_location[0] and j == self.goal_location[1]:
                continue
            else:
                self.agent_location = np.array([i, j], dtype=np.int32)

        assert self.agent_location is not None, "Unable to find good agent location"

        self.observation[self.agent_location[0], self.agent_location[1], :] = self._agent_color_array
        self.observation[self.goal_location[0], self.goal_location[1], :] = self._goal_color_array

    def is_solved(self):
        """ Check if given cube is solved """
        return (self.agent_location == self.goal_location).all()

    def step(self, action: int):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return
                            undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        assert 0 <= action < 5, "Action must be within range"

        self.observation[self.agent_location[0], self.agent_location[1], :] = 0

        move_agent(self.agent_location, self.maze, self._bounds, action)

        self.observation[self.agent_location[0], self.agent_location[1], :] = self._agent_color_array

        if self.is_solved():
            return self.observation, self.win_reward, True, self._info.copy()
        else:
            return self.observation, self.move_reward, False, self._info.copy()

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        self._internal_reset()
        return self.observation

    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        if mode == 'rgb_array':
            return self.observation
        else:
            super().render(mode=mode)

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return

    def get_keys_to_action(self):
        keyword_to_key = {
            'UP':      ord('w'),
            'DOWN':    ord('s'),
            'LEFT':    ord('a'),
            'RIGHT':   ord('d'),
        }

        keys_to_action = {}

        for action_id, action_meaning in self.action_meaning.items():
            keys = []
            for keyword, key in keyword_to_key.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id
        return keys_to_action


if __name__ == '__main__':
    import gym.utils.play as play

    # register()
    # env = RandomMazeEnv(127, 127)
    env = RandomMazeEnv(63, 63, start_end_distance=5)
    play.play(env, zoom=16.0)
