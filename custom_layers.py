"""Architecture for a transforer in jax stax format"""
import jax.numpy as jnp
import jax.random as jr
from math import prod
from jax import jit
from jax.lax import slice_in_dim
from jax.nn import softmax
from jax.nn.initializers import glorot_normal, normal, zeros, ones
from jax.example_libraries.stax import (
    Dense,
    FanOut,
    FanInSum,
    Identity,
    LeakyRelu,
    parallel,
    serial,
)


@jit
def attention(qkv):
    """Unmasked attention function"""
    q, k, v = qkv
    d_k_in = q.shape[-1]
    temp = jnp.einsum("...ij,...jk->...ik", q, k.swapaxes(-1, -2)) / jnp.sqrt(d_k_in)
    temp = softmax(temp)
    return jnp.einsum("...ij,...jk->...ik", temp, v)


def LayerNorm(eps=1e-07, beta_init=zeros, gamma_init=ones):
    """Layer normalization layer"""

    def init_fun(rng, input_shape):
        k1, k2 = jr.split(rng)
        beta, gamma = beta_init(k1, input_shape[1:]), gamma_init(k2, input_shape[1:])
        return input_shape, (beta, gamma)

    @jit
    def apply_fun(params, inputs, **kwargs):
        beta, gamma = params
        out = (inputs - inputs.mean(-1, keepdims=True)) / jnp.sqrt(
            inputs.var(-1, keepdims=True) + eps
        )
        return out * gamma + beta

    return init_fun, apply_fun



def ExpandDims(axis=-1):
    """Adds a dimension on end for shenanigans"""

    def init_fun(rng, input_shape):
        if axis == -1:
            output_shape = input_shape + (1,)
        else:
            output_shape = input_shape[:axis] + (1,) + input_shape[axis:]
        return output_shape, ()

    @jit
    def apply_fun(params, inputs, **kwargs):
        return jnp.expand_dims(inputs, axis)

    return init_fun, apply_fun


def Flatten():
    """Flattens to single dimension"""

    def init_fun(rng, input_shape):
        return (input_shape[0], prod(input_shape[1:])), ()

    @jit
    def apply_fun(params, inputs, **kwargs):
        return jnp.reshape(inputs, (inputs.shape[0], -1))

    return init_fun, apply_fun


def Linear(out_dim, W_init=glorot_normal(), b_init=normal()):
    """Layer constructor function for a linear layer."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = jr.split(rng)
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
        return output_shape, (W, b)

    @jit
    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return jnp.einsum("...ij,...jk->...ik", inputs, W) + b

    return init_fun, apply_fun


def SingleAttention(out_dim, inner_dim):
    """Single Headed attention for single token dim"""
    init_q, apply_q = Linear(inner_dim)
    init_k, apply_k = Linear(inner_dim)
    init_v, apply_v = Linear(out_dim)
    init_out, apply_out = Linear(out_dim)

    def init_fun(rng, input_shape):
        k1, k2, k3, k4 = jr.split(rng, 4)
        _, q_parm = init_q(k1, (input_shape))
        _, k_parm = init_k(k2, (input_shape))
        _, v_parm = init_v(k3, (input_shape))
        _, out_parm = init_out(k4, (input_shape))
        return (input_shape[:-1], out_dim), (
            q_parm,
            k_parm,
            v_parm,
            out_parm,
        )

    @jit
    def apply_fun(params, inputs, **kwargs):
        q_params, k_params, v_params, out_params = params

        Q = (
            apply_q(q_params, q_inputs)
            .reshape(-1, n_cont_out, num_heads, d_k_in)
            .swapaxes(1, 2)
        )
        K = (
            apply_k(k_params, inputs)
            .reshape(-1, n_cont_in, num_heads, d_k_in)
            .swapaxes(1, 2)
        )
        V = (
            apply_v(v_params, inputs)
            .reshape(-1, n_cont_in, num_heads, d_k_out)
            .swapaxes(1, 2)
        )

        attn = attention((Q, K, V)).reshape((-1, n_cont_out, d_model_out))
        return apply_out(out_params, attn)

    return init_fun, apply_fun


def MultiHeadAttn(n_context, d_model, num_heads):
    """Multi Headed Attention a la Perciever AR"""
    n_cont_in, n_cont_out = n_context
    d_model_in, d_model_out = d_model

    d_k_in = d_model_in // num_heads
    d_k_out = d_model_out // num_heads

    init_q, apply_q = Linear(d_model_in)
    init_k, apply_k = Linear(d_model_in)
    init_v, apply_v = Linear(d_model_out)
    init_out, apply_out = Linear(d_model_out)

    def init_fun(rng, input_shape):
        k1, k2, k3, k4 = jr.split(rng, 4)
        _, q_parm = init_q(k1, (input_shape[0], n_cont_out, d_model_in))
        _, k_parm = init_k(k2, (input_shape[0], n_cont_in, d_model_in))
        _, v_parm = init_v(k3, (input_shape[0], n_cont_in, d_model_in))
        _, out_parm = init_out(k4, (input_shape[0], n_cont_out, d_model_out))
        return (input_shape[0], n_cont_out, d_model_out), (
            q_parm,
            k_parm,
            v_parm,
            out_parm,
        )

    @jit
    def apply_fun(params, inputs, **kwargs):
        q_params, k_params, v_params, out_params = params

        q_inputs = slice_in_dim(inputs, n_cont_in - n_cont_out, n_cont_in, axis=1)

        Q = (
            apply_q(q_params, q_inputs)
            .reshape(-1, n_cont_out, num_heads, d_k_in)
            .swapaxes(1, 2)
        )
        K = (
            apply_k(k_params, inputs)
            .reshape(-1, n_cont_in, num_heads, d_k_in)
            .swapaxes(1, 2)
        )
        V = (
            apply_v(v_params, inputs)
            .reshape(-1, n_cont_in, num_heads, d_k_out)
            .swapaxes(1, 2)
        )

        attn = attention((Q, K, V)).reshape((-1, n_cont_out, d_model_out))
        return apply_out(out_params, attn)

    return init_fun, apply_fun
