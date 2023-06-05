"""Contains the model"""
from jax.example_libraries.stax import (
    Dense,
    Identity,
    Relu,
    serial,
    Sigmoid,
    Softmax,
    parallel,
    FanOut,
    FanInSum,
)
from jax import jit
from jax.lax import stop_gradient
from custom_layers import (
    MultiHeadAttn,
    ExpandDims,
    Flatten,
    Linear,
    ResDense,
    LayerNorm,
    Dueling,
)
import jax.numpy as jnp
from game import NoThanks


mygame = NoThanks(4, 11)
mygame.start_game()
GAME_STATE_SIZE = len(mygame.get_things())
del mygame


def build_model():
    """Builds the Dueling DDQN model."""
    return serial(
        Dense(512),
        Relu,
        Dense(512),
        Relu,
        Dense(64),
        Relu,
        FanOut(2),
        parallel(Dense(1), Dense(2)),
        parallel(Sigmoid, Identity),
        Dueling(),
    )


def build_model():
    """Builds the Dueling DDQN model."""
    return serial(
        Linear(512),
        ResDense(1024),
        FanOut(2),
        parallel(
            serial(
                FanOut(2),
                parallel(Dense(1), Dense(2)),
                Dueling(),
            ),
            serial(Linear(GAME_STATE_SIZE), Relu),
        ),
    )


# @jit
def loss(params, batch, old_params, key=None):
    """Loss function for predictions"""
    s, a, r, sn, done = batch

    # Calculate various Q-values
    new_q_values = predict(params, s)
    new_next_q_values = predict(params, sn)
    old_next_q_values = predict(old_params, sn)

    # Apply to action
    q_values = jnp.sum(new_q_values * a, axis=-1)

    # action from params, but value from old_params
    next_actions = jnp.argmax(new_next_q_values, axis=-1)
    old_next_q_values_sel = jnp.take_along_axis(
        old_next_q_values, next_actions[:, None], axis=-1
    ).squeeze()

    # Hardcoded discount
    target = r + 0.98 * done * old_next_q_values_sel
    return jnp.mean(jnp.square(q_values - target))


# Initialize model and prediction function
init_random_params, predict = build_model()

# Compile the predict function
predict = jit(predict)


@jit
def all_loss(params, batch, old_params, key=None):
    s, a, r, sn, done = batch
    new_q_values, hat_sn = predict(params, s)
    nnew_q_values, hat_snn = predict(params, sn)
    old_q_values, ohat_snn = predict(old_params, sn)
    hatn_q_values, hatn_snn = predict(params, hat_sn)

    # Next state prediction + consistency
    loss_sn = jnp.mean(jnp.mean(jnp.square(sn - hat_sn), axis=-1))
    loss_snn = 0.5 * jnp.mean(
        jnp.mean(jnp.square(stop_gradient(hat_snn) - hatn_snn), axis=-1)
    )

    embedd = params[0][0]
    unbedd = params[-1][-1][0][0]

    loss_bedd = 2.0 * jnp.mean(jnp.square(jnp.dot(embedd, unbedd) - jnp.eye(GAME_STATE_SIZE)))

    # Apply to action
    q_values = jnp.sum(new_q_values * a, axis=-1)

    # New v old
    next_actions = jnp.argmax(nnew_q_values, axis=-1)
    old_next_q_values_sel = jnp.take_along_axis(
        old_q_values, next_actions[:, None], axis=-1
    ).squeeze()

    # Hardcoded discount
    target = r + 0.98 * done * old_next_q_values_sel
    # Consistency Loss
    loss_con = jnp.mean(jnp.square(q_values - target))
    return loss_con + loss_sn + loss_snn + loss_bedd
