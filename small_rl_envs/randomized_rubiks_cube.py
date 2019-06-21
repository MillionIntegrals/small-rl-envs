import attr
import numpy as np
import os.path

from gym import Env
from gym.envs.registration import EnvSpec

from vel.openai.baselines import logger
from vel.openai.baselines.bench import Monitor
from vel.rl.api import EnvFactory
from vel.rl.env.wrappers.clip_episode_length import ClipEpisodeLengthWrapper
from vel.util.situational import process_environment_settings

from small_rl_envs.parametrized_env import RandomizedParametrizedEnv
from small_rl_envs.rubiks_cube_classic import RubiksCubeConstants, RubiksCubeParameters, RubiksCubeClassicEnv


@attr.s(auto_attribs=True)
class RandomizedRubiksCubeParameters:
    """ Parameters of the rubiks cube environment """
    # How many times should we scramble the cube maximum
    max_shuffle_count: int = 30


class RandomizedRubiksCube(RandomizedParametrizedEnv):
    """ Randomized rubiks cube environment """

    def __init__(self, parameters, constants=None):
        if parameters is None:
            parameters = RandomizedRubiksCubeParameters()
        elif isinstance(parameters, dict):
            parameters = RandomizedRubiksCubeParameters(**parameters)

        if constants is None:
            constants = RubiksCubeConstants()
        elif isinstance(constants, dict):
            constants = RubiksCubeConstants(**constants)

        super().__init__(parameters, constants)

    @property
    def underlying_env_constructor(self):
        return RubiksCubeClassicEnv

    def _sample_underlying_params(self):
        return RubiksCubeParameters(
            shuffle_count=np.random.randint(0, self.parameters.max_shuffle_count+1)
        )


class RandomizedRubiksCubeFactory(EnvFactory):
    """ Factory class for the rubiks cube environment """

    DEFAULT_SETTINGS = {
        'default': {
            'max_episode_frames': 10000,
            'monitor': False,
            'allow_early_resets': False,
            'parameters': {},
            'constants': {}
        }
    }

    def __init__(self, settings=None, presets=None):
        self.settings = process_environment_settings(self.DEFAULT_SETTINGS, settings, presets)

    def specification(self) -> EnvSpec:
        """ Return environment specification """
        raise NotImplementedError

    def instantiate(self, seed=0, serial_id=1, preset='default', extra_args=None) -> Env:
        """ Create a new Env instance """
        settings = self.settings[preset]

        env = RandomizedRubiksCube(
            parameters=settings['parameters'],
            constants=settings['constants']
        )

        env = ClipEpisodeLengthWrapper(env, max_episode_length=settings['max_episode_frames'])

        # Monitoring the env
        if settings['monitor']:
            logdir = logger.get_dir() and os.path.join(logger.get_dir(), str(serial_id))
        else:
            logdir = None

        env = Monitor(env, logdir, allow_early_resets=settings['allow_early_resets'])

        return env


def create(settings=None, presets=None):
    """ Vel factory function """

    return RandomizedRubiksCubeFactory(settings, presets)
