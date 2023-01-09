import random
import numpy as np
from jax import jit, grad
import jax.numpy as jnp


@jit
def hard_max(x, axis=-1):
    """returns 1.0 at max and 0.0 else"""
    return 1.0 - jnp.sign(jnp.max(x, axis=axis).reshape((-1, 1)) - x)


def sample_from(experience_list, predict_fun, params, old_params, k=32):
    """Batch from experiences for DDQN"""
    sample = random.sample(experience_list, k=k)

    state = jnp.array([x[0] for x in sample])
    action = jnp.array([(1 - x[1], x[1]) for x in sample])

    reward = jnp.array([x[2] for x in sample])
    state_p = jnp.array(np.array([x[3] for x in sample]))
    final = jnp.array([x[4] for x in sample]).reshape((-1, 1))

    x = predict_fun(params, state_p)
    arg_x = hard_max(x)

    x_target = predict_fun(old_params, state_p)

    target = reward + jnp.sum(x_target * arg_x * final, axis=-1)

    return state, action, target


def sample_all(experiences, predict_fun, params, old_params):
    """Huge sample using all experiences for DDQN"""
    state = jnp.array([x[0] for x in experiences])
    action = jnp.array([(1 - x[1], x[1]) for x in experiences])

    reward = jnp.array([x[2] for x in experiences])
    state_p = jnp.array([x[3] for x in experiences])
    final = jnp.array([x[4] for x in experiences]).reshape((-1, 1))

    x = predict_fun(params, state_p)
    arg_x = hard_max(x)

    x_target = predict_fun(old_params, state_p)

    target = reward + jnp.sum(x_target * arg_x * final, axis=-1)

    return state, action, target
