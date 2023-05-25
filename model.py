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
        FanOut(3),
        parallel(
            FanOut(2),  # Value Action
            Dense(256),  # s_t+1
            Dense(16),  # Immediate reward
        ),
        parallel(
            parallel(
                Sigmoid,  # Value
                Identity(),  # Effect of Action
            ),
            Relu,  # s_t+1
            Relu,  # Immediate reward
        ),
        parallel(
            Dueling(),  # Value
            Dense(171),  # s_{n+1}
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


@jit
def loss(params, batch, old_params, key=None):
    """Loss function for predictions"""
    s, a, r, sn, done = batch

    # Calculate various Q-values
    new_q_values = predict(params, s)
    new_next_q_values = predict(params, sn)
    old_next_q_values = predict(old_params, sn)

    # Apply to action
    q_values = jnp.sum(new_q_values * a, axis=-1)

    # actions via old
    next_actions = jnp.argmax(new_next_q_values, axis=-1)
    old_next_q_values_sel = jnp.take_along_axis(
        old_next_q_values, next_actions[:, None], axis=-1
    ).squeeze()

    # Hardcoded discount
    target = r + 0.98 * done * old_next_q_values_sel
    return jnp.mean(jnp.square(q_values - target))


@jit
def all_loss(params, batch, old_params, key=None):
    s, a, r, sn, done = batch
    new_q_values, hat_sn, hat_r = predict(params, s)
    nnew_q_values, nhat_sn, nhat_r = predict(params, sn)
    old_q_values, ohat_sn, ohat_r = predict(params, s)

    # Apply to action
    q_values = jnp.sum(new_q_values * a, axis=-1)


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
