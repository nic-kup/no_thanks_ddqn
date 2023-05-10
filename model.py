"""Contains the model"""
from jax.example_libraries.stax import (
    serial,
    Sigmoid,
    Dense,
    Relu,
    parallel,
    FanOut,
)
from jax import jit
import jax.numpy as jnp


def Dueling():
    """Combining value and action via dueling architecture."""

    def init_fun(rng, input_shape):
        return input_shape[-1], ()

    @jit
    def apply_fun(params, inputs, **kwargs):
        discounted_action = inputs[1] - jnp.max(inputs[1], axis=-1).reshape((-1, 1))
        return inputs[0] + discounted_action

    return init_fun, apply_fun


def build_model():
    """Builds the Dueling DDQN model."""
    return serial(
        Dense(500),
        Relu,
        Dense(500),
        Relu,
        FanOut(2),
        parallel(Dense(50), Dense(50)),
        parallel(Relu, Relu),
        parallel(Dense(25), Dense(25)),
        parallel(Relu, Relu),
        parallel(Dense(1), Dense(2)),
        Dueling(),
    )


# Initialize model and prediction function
init_random_params, predict = build_model()

# Compile the predict function
predict = jit(predict)
