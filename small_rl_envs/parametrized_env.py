import gym


class ParametrizedEnv(gym.Env):
    """
    An environment that is described by a set of parameters, which can change between rollouts.
    It is important that observation and action spaces stay constant or at least somehow compatible.
    Constants cannot be changed once env has been instantiated
    """

    def __init__(self, parameters, constants=None):
        super().__init__()

        self._parameters = parameters
        self._constants = constants
        self._initialize()
        self._initialize_from_current_parameters()

    @property
    def parameters(self):
        return self._parameters

    @property
    def constants(self):
        return self._constants

    def reset_with_new_params(self, new_parameters):
        """ Reset this environment by setting new state"""
        self._parameters = new_parameters
        self._initialize_from_current_parameters()
        return self.reset()

    def _initialize(self):
        """ Optional one-time initialization of the environment """
        pass

    def _initialize_from_current_parameters(self):
        """ Initialize internal state of the environment from a currently set set of parameters """
        raise NotImplementedError


class RandomizedParametrizedEnv(ParametrizedEnv):
    """
    An environment that randomly samples a set of parameters for an underlying env.
    Parametrized itself by a set of randomization "metaparameters"
    """

    def __init__(self, metaparameters, constants=None):
        super().__init__(metaparameters, constants=constants)

        self._underlying_params = self._sample_underlying_params()
        self._underlying_env = self.underlying_env_constructor(self.underlying_parameters, self.constants)

    @property
    def underlying_parameters(self):
        return self._underlying_params

    @property
    def underlying_env_constructor(self):
        """ Create an instance of an underlying env """
        raise NotImplementedError

    def _initialize_from_current_parameters(self):
        """ Initialize internal state of the environment from a currently set set of parameters """
        # There is no real internal initialization here by default
        pass

    def _sample_underlying_params(self):
        """ Draw from a random distribution of underlying environment parameters """
        raise NotImplementedError

    def step(self, action):
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
        return self._underlying_env.step(action)

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        self._underlying_params = self._sample_underlying_params()
        return self._underlying_env.reset_with_new_params(self._underlying_params)

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
        return self._underlying_env.render(mode=mode)

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._underlying_env.close()

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
        self._underlying_env.seed(seed=seed)

    @property
    def unwrapped(self):
        """Completely unwrap this env.
        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self._underlying_env.unwrapped

    def get_keys_to_action(self):
        """ Map keys to action in env 'play' utility """
        return self._underlying_env.get_keys_to_action()

    @property
    def action_space(self):
        return self._underlying_env.action_space

    @property
    def observation_space(self):
        return self._underlying_env.observation_space

    @property
    def metadata(self):
        return self._underlying_env.metadata
