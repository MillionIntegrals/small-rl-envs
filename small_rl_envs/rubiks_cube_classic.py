import gym
import gym.spaces as spaces
import numpy as np
import numba


@numba.jit(nopython=True)
def rubiks_cube_play(state: np.ndarray, action_idx: int) -> None:
    """ Most important function in this module. How does rotating each face affect the cube? - numba version """
    face_idx = action_idx // 2
    # Rotate direction 0 - clockwise
    # Rotate direction 1 - counterclockwise
    rotate_direction = action_idx % 2

    # Each rotation is first rotating the main face and then four attached faces
    if rotate_direction == 0:
        state[face_idx] = state[face_idx, ::-1, :].T
    else:
        state[face_idx] = state[face_idx, :, ::-1].T

    if face_idx == 0:
        temp = state[1, :, 0].copy()

        if rotate_direction == 0:
            state[1, :, 0] = state[5, ::-1, 2]
            state[5, :, 2] = state[3, ::-1, 0]
            state[3, :, 0] = state[2, :, 0]
            state[2, :, 0] = temp
        else:
            state[1, :, 0] = state[2, :, 0]
            state[2, :, 0] = state[3, :, 0]
            state[3, :, 0] = state[5, ::-1, 2]
            state[5, :, 2] = temp[::-1]
    elif face_idx == 1:
        temp = state[0, 0, :].copy()

        if rotate_direction == 0:
            state[0, 0, :] = state[2, 0, :]
            state[2, 0, :] = state[4, 0, :]
            state[4, 0, :] = state[5, 0, :]
            state[5, 0, :] = temp
        else:
            state[0, 0, :] = state[5, 0, :]
            state[5, 0, :] = state[4, 0, :]
            state[4, 0, :] = state[2, 0, :]
            state[2, 0, :] = temp
    elif face_idx == 2:
        temp = state[0, :, 2].copy()

        if rotate_direction == 0:
            state[0, :, 2] = state[3, 0, :]
            state[3, 0, :] = state[4, ::-1, 0]
            state[4, :, 0] = state[1, 2, :]
            state[1, 2, :] = temp[::-1]
        else:
            state[0, :, 2] = state[1, 2, ::-1]
            state[1, 2, :] = state[4, :, 0]
            state[4, :, 0] = state[3, 0, ::-1]
            state[3, 0, :] = temp
    elif face_idx == 3:
        temp = state[0, 2, :].copy()

        if rotate_direction == 0:
            state[0, 2, :] = state[5, 2, :]
            state[5, 2, :] = state[4, 2, :]
            state[4, 2, :] = state[2, 2, :]
            state[2, 2, :] = temp
        else:
            state[0, 2, :] = state[2, 2, :]
            state[2, 2, :] = state[4, 2, :]
            state[4, 2, :] = state[5, 2, :]
            state[5, 2, :] = temp
    elif face_idx == 4:
        temp = state[1, :, 2].copy()

        if rotate_direction == 0:
            state[1, :, 2] = state[2, :, 2]
            state[2, :, 2] = state[3, :, 2]
            state[3, :, 2] = state[5, ::-1, 0]
            state[5, :, 0] = temp[::-1]
        else:
            state[1, :, 2] = state[5, ::-1, 0]
            state[5, :, 0] = state[3, ::-1, 2]
            state[3, :, 2] = state[2, :, 2]
            state[2, :, 2] = temp
    elif face_idx == 5:
        temp = state[0, :, 0].copy()

        if rotate_direction == 0:
            state[0, :, 0] = state[1, 0, ::-1]
            state[1, 0, :] = state[4, :, 2]
            state[4, :, 2] = state[3, 2, ::-1]
            state[3, 2, :] = temp
        else:
            state[0, :, 0] = state[3, 2, :]
            state[3, 2, :] = state[4, ::-1, 2]
            state[4, :, 2] = state[1, 0, :]
            state[1, 0, :] = temp[::-1]


def rubiks_cube_play_original(state: np.ndarray, action_idx: int):
    """ Most important function in this module. How does rotating each face affect the cube?"""
    face_idx = action_idx // 2
    # Rotate direction 0 - clockwise
    # Rotate direction 1 - counterclockwise
    rotate_direction = action_idx % 2

    # Parameter for np.rot90
    numpy_rot_k = 1 if rotate_direction == 1 else 3

    # Each rotation is first rotating the main face and then four attached faces

    # Main face rotation
    state[face_idx] = np.rot90(state[face_idx], k=numpy_rot_k)

    if face_idx == 0:
        temp = state[1, :, 0].copy()

        if rotate_direction == 0:
            state[1, :, 0] = np.flip(state[5, :, 2])
            state[5, :, 2] = np.flip(state[3, :, 0])
            state[3, :, 0] = state[2, :, 0]
            state[2, :, 0] = temp
        else:
            state[1, :, 0] = state[2, :, 0]
            state[2, :, 0] = state[3, :, 0]
            state[3, :, 0] = np.flip(state[5, :, 2])
            state[5, :, 2] = np.flip(temp)
    elif face_idx == 1:
        temp = state[0, 0, :].copy()

        if rotate_direction == 0:
            state[0, 0, :] = state[2, 0, :]
            state[2, 0, :] = state[4, 0, :]
            state[4, 0, :] = state[5, 0, :]
            state[5, 0, :] = temp
        else:
            state[0, 0, :] = state[5, 0, :]
            state[5, 0, :] = state[4, 0, :]
            state[4, 0, :] = state[2, 0, :]
            state[2, 0, :] = temp
    elif face_idx == 2:
        temp = state[0, :, 2].copy()

        if rotate_direction == 0:
            state[0, :, 2] = state[3, 0, :]
            state[3, 0, :] = np.flip(state[4, :, 0])
            state[4, :, 0] = state[1, 2, :]
            state[1, 2, :] = np.flip(temp)
        else:
            state[0, :, 2] = np.flip(state[1, 2, :])
            state[1, 2, :] = state[4, :, 0]
            state[4, :, 0] = np.flip(state[3, 0, :])
            state[3, 0, :] = temp
    elif face_idx == 3:
        temp = state[0, 2, :].copy()

        if rotate_direction == 0:
            state[0, 2, :] = state[5, 2, :]
            state[5, 2, :] = state[4, 2, :]
            state[4, 2, :] = state[2, 2, :]
            state[2, 2, :] = temp
        else:
            state[0, 2, :] = state[2, 2, :]
            state[2, 2, :] = state[4, 2, :]
            state[4, 2, :] = state[5, 2, :]
            state[5, 2, :] = temp
    elif face_idx == 4:
        temp = state[1, :, 2].copy()

        if rotate_direction == 0:
            state[1, :, 2] = state[2, :, 2]
            state[2, :, 2] = state[3, :, 2]
            state[3, :, 2] = np.flip(state[5, :, 0])
            state[5, :, 0] = np.flip(temp)
        else:
            state[1, :, 2] = np.flip(state[5, :, 0])
            state[5, :, 0] = np.flip(state[3, :, 2])
            state[3, :, 2] = state[2, :, 2]
            state[2, :, 2] = temp
    elif face_idx == 5:
        temp = state[0, :, 0].copy()

        if rotate_direction == 0:
            state[0, :, 0] = np.flip(state[1, 0, :])
            state[1, 0, :] = state[4, :, 2]
            state[4, :, 2] = np.flip(state[3, 2, :])
            state[3, 2, :] = temp
        else:
            state[0, :, 0] = state[3, 2, :]
            state[3, 2, :] = np.flip(state[4, :, 2])
            state[4, :, 2] = state[1, 0, :]
            state[1, 0, :] = np.flip(temp)


class RubiksCubeClassicEnv(gym.Env):
    """
    An environment where the goal is to solve Rubiks cube. Reward is 1.0 only when the cube is solved and 0.0 otherwise.
    This is a fixed 6x3x3 classic cube.

    Cube's faces are ordered in the following way:
      ---
      |1|
    ---------
    |0|2|4|5|
    ---------
      |3|
      ---

    shuffle_count is a parameter specifying how many times should random actions be played on the environment
    to initialize it "randomly"
    """

    # Set this in SOME subclasses
    metadata = {'render.modes': [
        'rgb_array'
    ]}
    reward_range = (0.0, 1.0)
    spec = None

    colors = [
        # Taken from https://www.schemecolor.com/rubik-cube-colors.php
        (185, 0, 0),
        (0, 69, 173),
        (255, 213, 0),
        (0, 155, 72),
        (255, 89, 0),
        (255, 255, 255)
    ]

    # Where in canvas each face should be rendered
    canvas_coords = [
        (0, 33),
        (33, 0),
        (33, 33),
        (33, 66),
        (66, 33),
        (99, 33)
    ]

    # Set these in ALL subclasses
    # 12 actions because there are 6 faces that can be turned clockwise or counterclockwise
    action_space = spaces.Discrete(12)
    observation_space = spaces.Box(low=0, high=5, shape=(6, 3, 3), dtype=np.uint8)

    def __init__(self, shuffle_count=300):
        self._state = np.zeros(shape=(6, 3, 3), dtype=np.uint8)
        self._shuffle_count = shuffle_count

        self._initialize_starting_cube()
        self._shuffle()

    def _initialize_starting_cube(self):
        """ Initialize cube to the initial 'solved' state """
        for i in range(6):
            self._state[i] = i

    def _shuffle(self):
        """ Shuffle the cube """
        actions = np.random.randint(0, self.action_space.n, size=self._shuffle_count)

        for action in actions:
            rubiks_cube_play(self._state, action)

    def is_solved(self):
        """ Check if given cube is solved """
        return False

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
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        assert 0 <= action < 12, "Action must be within range"
        rubiks_cube_play(self._state, action)

        if self.is_solved():
            return self._state, 1.0, True, {}
        else:
            return self._state, 0.0, False, {}

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        self._initialize_starting_cube()
        self._shuffle()
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
            canvas = np.zeros(shape=(100, 133, 3), dtype=np.uint8)

            for face_idx in range(6):
                face_coords = self.canvas_coords[face_idx]

                for i in range(3):
                    for j in range(3):
                        color = self.colors[self._state[face_idx, i, j]]

                        start_x = face_coords[0] + j * 11 + 1
                        end_x = face_coords[0] + j * 11 + 10 + 1

                        start_y = face_coords[1] + i * 11 + 1
                        end_y = face_coords[1] + i * 11 + 10 + 1

                        canvas[start_y:end_y, start_x:end_x] = color

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

