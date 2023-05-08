"""Contains the model"""
from jax.example_libraries.stax import (
    serial,
    Sigmoid,
    Dense,
    LeakyRelu,
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
        discounted_action = inputs[1] - jnp.mean(inputs[1], axis=-1).reshape((-1, 1))
        return inputs[0] + discounted_action

    return init_fun, apply_fun


def build_model():
    """Builds the Dueling DDQN model."""
    return serial(
        Dense(512),
        LeakyRelu,
        Dense(256),
        LeakyRelu,
        FanOut(2),
        parallel(Dense(64), Dense(64)),
        parallel(LeakyRelu, LeakyRelu),
        parallel(Dense(1), Dense(2)),
        Dueling(),
    )


def build_model():
    """Builds the Dueling DDQN model."""
    return serial(
        Dense(256),
        LeakyRelu,
        Dense(128),
        LeakyRelu,
        FanOut(2),
        parallel(Dense(32), Dense(64)),
        parallel(LeakyRelu, LeakyRelu),
        parallel(Dense(1), Dense(2)),
        Dueling(),
    )


# Initialize model and prediction function
init_random_params, predict = build_model()

# Compile the predict function
predict = jit(predict)
