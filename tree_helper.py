"""Contains functions for modifying trees"""
from jax import jit, tree_map
import jax.numpy as jnp


@jit
def tree_square_sum(weights):
    """Returns the sum of the squares of a jax-tree"""
    return sum(tree_map(jnp.sum, tree_map(jnp.square, weights)))


@jit
def convex_comb(tree_a, tree_b, t):
    """Takes convex combination of two trees"""
    return tree_map(lambda x, y: (1 - t) * x + t * y, tree_a, tree_b)


@jit
def tree_zeros_like(weights):
    """Returns a tree with same structure as given tree, but only 0"""
    return tree_map(lambda x: x * 0.0, weights)


@jit
def lion_step(step_size, cur_x, grad, momentum, beta1=0.9, beta2=0.99, wd=1.1):
    """Applies lion optimzer step to weights given grad and momentum"""
    # Mix with momentum
    update = convex_comb(grad, momentum, beta1)
    # Take the sign
    update = tree_map(jnp.sign, update)

    # Update momentum
    momentum = convex_comb(grad, momentum, beta2)

    # Add weight decay to update
    weight_decay = tree_map(lambda x: x * wd, cur_x)
    update = tree_map(lambda x, y: x + y, update, weight_decay)

    # Step size
    update = tree_map(lambda x: x * step_size, update)
    return tree_map(lambda x, y: x - y, cur_x, update), momentum
