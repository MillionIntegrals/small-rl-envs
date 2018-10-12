from small_rl_envs.rubiks_cube_classic import RubiksCubeClassicEnv

import numpy as np
import numpy.testing as nt


def test_scrambling():
    """ Tested against real Rubiks cube """
    env = RubiksCubeClassicEnv(shuffle_count=0)

    env.step(1)
    env.step(2)
    env.step(5)
    env.step(6)
    env.step(9)
    env.step(10)
    env.step(0)
    env.step(3)
    env.step(4)
    env.step(7)
    env.step(8)
    env.step(11)

    array = np.array([[[3, 0, 4],
                       [4, 0, 5],
                       [1, 1, 5]],
                      [[4, 5, 2],
                       [3, 1, 2],
                       [1, 2, 3]],
                      [[5, 1, 5],
                       [0, 2, 0],
                       [0, 3, 0]],
                      [[3, 2, 1],
                       [5, 3, 4],
                       [2, 2, 0]],
                      [[4, 0, 4],
                       [1, 4, 5],
                       [5, 3, 2]],
                      [[1, 3, 2],
                       [4, 5, 1],
                       [3, 4, 0]]], dtype=np.uint8)

    nt.assert_array_almost_equal(env._state, array)


def test_reversion():
    env = RubiksCubeClassicEnv()

    numbers = np.random.randint(6, size=100)

    for number in numbers:
        env.reset()
        state = env._state.copy()

        env.step(number * 2)
        env.step(number * 2 + 1)

        nt.assert_array_almost_equal(state, env._state)

        env.reset()
        state = env._state.copy()

        env.step(number * 2 + 1)
        env.step(number * 2)

        nt.assert_array_almost_equal(state, env._state)


