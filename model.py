"""Contains the model"""
from jax.example_libraries.stax import (
    serial,
    Sigmoid,
    Dense,
    Relu,
    parallel,
    FanOut,
    FanInSum,
)
from jax import jit
from custom_layers import (
    MultiHeadAttn,
    ExpandDims,
    Flatten,
    LayerNorm,
    Dueling,
    Identity,
)
import jax.numpy as jnp


# Possible model for future?
def build_model():
    """Builds the Dueling DDQN model."""
    return serial(
        Dense(512),
        Relu,
        Dense(512),
        Relu,
        FanOut(4),
        parallel(
            Dense(1),  # Value
            Dense(2),  # Action pair
            Dense(171),  # s_t+1
            Dense(1),  # Immediate reward
        ),
    )


def build_model():
    """Builds the Dueling DDQN model."""
    return serial(
        Dense(512),
        Relu,
        Dense(512),
        Relu,
        Dense(256),
        Relu,
        Dense(64),
        Relu,
        FanOut(2),
        parallel(Dense(1), Dense(2)),
        parallel(Sigmoid, Identity()),
        Dueling(),
    )


# def build_model():
#     """Builds the Dueling DDQN model."""
#     return serial(
#         Dense(200),
#         ExpandDims(),
#         Dense(5),
#         Relu,
#         LayerNorm(),
#         MultiHeadAttn((200, 200), (5, 1), 1),
#         Relu,
#         Flatten(),
#         Dense(50),
#         Relu,
#         FanOut(2),
#         parallel(Dense(25), Dense(25)),
#         parallel(Relu, Relu),
#         parallel(Dense(1), Dense(2)),
#         Dueling(),
#     )


# Initialize model and prediction function
init_random_params, predict = build_model()

# Compile the predict function
predict = jit(predict)
