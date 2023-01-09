"""Contains the model"""
from jax.example_libraries.stax import (
    serial,
    Sigmoid,
    Dense,
    Relu,
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
    FanOut(2),
    parallel(Dense(300), Dense(300)),
    parallel(LeakyRelu, LeakyRelu),
    parallel(Dense(100), Dense(100)),
    parallel(LeakyRelu, LeakyRelu),
    parallel(Dense(8), Dense(16)),
    parallel(LeakyRelu, LeakyRelu),
    parallel(Dense(1), Dense(2)),
    parallel(Sigmoid, Sigmoid),
    Dueling(),
)

predict = jit(predict)
