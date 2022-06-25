"""Contains the model"""
from jax.experimental.stax import (
    serial,
    Sigmoid,
    Dense,
    LeakyRelu,
    parallel,
    FanOut,
)
from jax import jit, grad
import jax.numpy as jnp


def Dueling():
    """Combining value and action via dueling architecture."""
    init_fun = lambda rng, input_shape: (input_shape[-1], ())

    @jit
    def apply_fun(params, inputs, **kwargs):
        discounted_action = inputs[1] - jnp.max(inputs[1], axis=-1).reshape((-1, 1))
        return inputs[0] + discounted_action

    return init_fun, apply_fun


# Try using my conv and splitting card / token from player states
# FanOut an parallel as dualistic agent
init_random_params, predict = serial(
    Dense(300),
    LeakyRelu,
    Dense(600),
    LeakyRelu,
    Dense(600),
    LeakyRelu,
    FanOut(2),
    parallel(Dense(300), Dense(300)),
    parallel(LeakyRelu, LeakyRelu),
    parallel(Dense(60), Dense(60)),
    parallel(LeakyRelu, LeakyRelu),
    parallel(Dense(4), Dense(8)),
    parallel(LeakyRelu, LeakyRelu),
    parallel(Dense(1), Dense(2)),
    parallel(Sigmoid, Sigmoid),
    Dueling(),
)

predict = jit(predict)
