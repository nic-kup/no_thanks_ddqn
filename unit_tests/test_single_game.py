import pytest
from game import NoThanks
from single_game import single_game, k_step_game
from model import predict, init_random_params
import jax.random as jr

# We need to first get a `predict` function to be able to run tests.

mygame = NoThanks(4, 11)
mygame.start_game()
INPUT_SIZE = len(mygame.get_things())
print(f"Input size: {INPUT_SIZE}")

SEED = 4
key = jr.PRNGKey(SEED)
key, sbkey = jr.split(key)
params = init_random_params(sbkey, (-1, INPUT_SIZE))[1]
key, sbkey = jr.split(key)


def test_run_single_game():
    single_game(predict, param_list=[params])


def test_run_k_step_game():
    k_step_game(predict, param_list=[params])
