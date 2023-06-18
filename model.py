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
    PrintShape,
    FakeAttention,
    ExpandDims,
    SoLU,
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


partial_predict = jit(serial(Linear(128), ResDense(16), ResDense(16))[1])



def build_model():
    """Builds the Dueling DDQN model."""
    return serial(
        Linear(100),
        ResDense(200),
        ResDense(200),
        ResDense(200),
        FanOut(2),
        parallel(
            serial(
                Linear(50),
                Relu,
                FanOut(2),
                parallel(Linear(1), Linear(2)),
                Dueling(),
            ),
            Linear(100),
            Identity,
        ),
    )



def build_model():
    """Builds the Dueling DDQN model."""
    return serial(
        Linear(64),
        FakeAttention(),
        Relu,
        Linear(32),
        FakeAttention(),
        Relu,
        FanOut(2),
        parallel(Linear(1), Linear(2)),
        parallel(Sigmoid, Identity),
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

    # action from params, but value from old_params
    next_actions = jnp.argmax(new_next_q_values, axis=-1)
    old_next_q_values_sel = jnp.take_along_axis(
        old_next_q_values, next_actions[:, None], axis=-1
    ).squeeze()

    # Hardcoded discount
    target = r + 0.98 * done * old_next_q_values_sel
    return jnp.mean((4.0 - 3.0 * done) * jnp.square(q_values - target))


# Initialize model and prediction function
init_random_params, predict = build_model()

# Compile the predict function
predict = jit(predict)

# loss: the predicted Q value from the state without prediction should also be the same


@jit
def all_loss(params, batch, old_params, key=None):
    s, a, r, sn, done = batch
    new_q_values, hat_sn = predict(params, s)
    nnew_q_values, hat_snn = predict(params, sn)
    old_q_values, ohat_snn = predict(old_params, sn)

    # Next state prediction + consistency

    embedd = params[0][0]

    loss_sn = jnp.mean(jnp.mean(jnp.square(jnp.dot(sn, embedd) - hat_sn), axis=-1))

    # Apply to action
    q_values = jnp.sum(new_q_values * a, axis=-1)

    # New v old
    next_actions = jnp.argmax(nnew_q_values, axis=-1)
    old_next_q_values_sel = jnp.take_along_axis(
        old_q_values, next_actions[:, None], axis=-1
    ).squeeze()

    # Hardcoded discount
    target = r + 0.98 * done * old_next_q_values_sel
    # Bellman Loss with extra weighting on terminal episodes
    loss_con = jnp.mean((3.0 - 2.0 * done) * jnp.square(q_values - target))
    return loss_con + loss_sn
