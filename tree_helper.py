"""Contains functions for modifying trees"""
from jax import jit, tree_map
import jax.numpy as jnp


@jit
def tree_square_sum(x):
    return sum(tree_map(jnp.sum, tree_map(jnp.square, x)))


@jit
def tree_zeros_like(x):
    return tree_map(lambda x: x * 0.0, x)


@jit
def update_step(
    adam_step, step_size, cur_x, grad, momentum, square_weight, beta1=0.9, beta2=0.999
):
    t = adam_step + 1
    """Aam update step"""
    x_out = tree_map(lambda x: x * (1 - step_size * 0.5), cur_x)
    momentum = tree_map(lambda x, y: beta1 * x + (1 - beta1) * y, momentum, grad)
    square_grad = tree_map(jnp.square, grad)
    square_weight = tree_map(
        lambda x, y: beta2 * x + (1 - beta2) * y, square_weight, square_grad
    )

    abs_hat = tree_map(jnp.sqrt, square_weight)

    step_hat = step_size * jnp.sqrt(1 - beta2**t) / (1 - beta1**t)
    x_out = tree_map(
        lambda x, y, z: x - step_hat * y / (z + 1e-09), x_out, momentum, abs_hat
    )

    return x_out, momentum, square_weight
