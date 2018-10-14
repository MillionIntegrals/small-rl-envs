import gym
import gym.spaces as spaces
import numpy as np

import numba
import numba.types as nt


@numba.njit(nt.void(numba.double[::1], numba.int_, numba.double), cache=True)
def up_right_play(state: np.ndarray, action_idx: int, stepsize: float) -> None:
    if action_idx == 0:
        state[0] = max(state[0] - stepsize, 0.0)
    elif action_idx == 1:
        state[0] = min(state[0] + stepsize, 1.0)
    elif action_idx == 2:
        state[1] = max(state[1] - stepsize, 0.0)
    elif action_idx == 3:
        state[1] = min(state[1] + stepsize, 1.0)


class UpRightSparseEnv(gym.Env):
    """
    An environment where the goal is to reach an upper right corner by the agent.
    Sounds easy but is actually hard to solve using generic methods.
    """

    wall_color = (255, 255, 255)
    agent_color = (0, 69, 173)
    goal_color = (255, 213, 0)

    metadata = {'render.modes': [
        'rgb_array'
    ]}

    action_space = spaces.Discrete(4)
    observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.double)

    def __init__(self, stepsize=0.01, win_reward=1.0, move_reward=0.0):
        self.reward_range = (min(win_reward, move_reward), max(win_reward, move_reward))

        self.stepsize = stepsize
        self.win_reward = win_reward
        self.move_reward = move_reward

        self._state = np.array([0.0, 0.0], dtype=np.double)
        self._goal_state = np.array([1.0, 1.0], dtype=np.double)
        self._info = {'stepsize': stepsize}

    def is_solved(self):
        """ Check if given cube is solved """
        return (self._state == self._goal_state).all()

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
        assert 0 <= action < 4, "Action must be within range"

        up_right_play(self._state, action, self.stepsize)

        if self.is_solved():
            return self._state, self.win_reward, True, self._info.copy()
        else:
            return self._state, self.move_reward, False, self._info.copy()

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        self._state = np.zeros(shape=(2,), dtype=np.int)
        return self._state

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
            canvas = np.zeros(shape=(5 + 128, 5 + 128, 3), dtype=np.uint8)

            canvas[:2, :] = self.wall_color
            canvas[-2:, :] = self.wall_color
            canvas[:, :2] = self.wall_color
            canvas[:, -2:] = self.wall_color

            canvas[
                int(np.round(127 * self._goal_state[0])) + 2: int(np.round(127 * self._goal_state[0])) + 4,
                int(np.round(127 * self._goal_state[1])) + 2: int(np.round(127 * self._goal_state[1])) + 4
            ] = self.goal_color

            canvas[
                int(np.round(127 * self._state[0])) + 2: int(np.round(127 * self._state[0])) + 4,
                int(np.round(127 * self._state[1])) + 2: int(np.round(127 * self._state[1])) + 4
            ] = self.agent_color

            return canvas
        else:
            super().render(mode=mode)

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return

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

