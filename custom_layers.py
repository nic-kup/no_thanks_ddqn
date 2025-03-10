"""Architecture for a transforer in jax stax format"""
import jax.numpy as jnp
import jax.random as jr
from math import prod
from jax import jit
from jax.lax import slice_in_dim
from jax.nn import softmax, relu
from jax.nn.initializers import glorot_normal, normal, zeros, ones
from jax.example_libraries.stax import (
    Relu,
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


def Dueling():
    """Combining value and action via dueling architecture."""

    def init_fun(rng, input_shape):
        return input_shape[-1], ()

    @jit
    def apply_fun(params, inputs, **kwargs):
        # Advanced advantage calculation with better normalization
        # Center advantages around zero for better stability
        # Subtract mean (per batch element) to ensure advantages sum to zero for each state
        advantage = inputs[1] - jnp.mean(inputs[1], axis=-1, keepdims=True)
        
        # Optional: Apply full standardization (not just mean centering)
        # This adds another level of normalization similar to the advantage calculation in loss()
        # advantage = advantage / (jnp.std(advantage, axis=-1, keepdims=True) + 1e-8)
        
        # Use value function as state value estimate
        state_value = inputs[0]
        
        # Return Q(s,a) = V(s) + A(s,a)
        return state_value + advantage

    return init_fun, apply_fun


@jit
def solu(x, axis=-1):
    return x * softmax(x, axis=axis)


def SoLU(axis=-1):
    """Softmax Linear Unit"""

    def init_fun(rng, input_shape):
        return input_shape, ()

    @jit
    def apply_fun(params, inputs, **kwargs):
        return solu(inputs, axis=axis)

    return init_fun, apply_fun


def Softmax(axis=-1):
    """Softmax"""

    def init_fun(rng, input_shape):
        return input_shape, ()

    @jit
    def apply_fun(params, inputs, **kwargs):
        return softmax(inputs, axis=axis)

    return init_fun, apply_fun


def Reshape(new_shape):
    """Reshape"""

    def init_fun(rng, input_shape):
        return (-1, *new_shape), ()

    def apply_fun(params, inputs, **kwargs):
        return inputs.reshape((-1, *new_shape))

    return init_fun, apply_fun


def PrintShape():
    """Debug Layer"""

    def init_fun(rng, input_shape):
        print(input_shape)
        return input_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return inputs

    return init_fun, apply_fun


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


def ResDense(size, W_init=glorot_normal()):
    """Residual Dense Layer with LayerNorm"""

    def init_fun(rng, input_shape):
        ki, ko = jr.split(rng)
        W_i = W_init(ki, (input_shape[-1], size))
        W_o = W_init(ko, (size, input_shape[-1]))
        return input_shape, (W_i, W_o)

    @jit
    def apply_fun(params, inputs, **kwargs):
        W_i, W_o = params

        x = inputs @ W_i
        x = relu(x)
        x = x @ W_o
        return x + inputs

    return init_fun, apply_fun


def FakeAttention(W_init=glorot_normal()):
    """Transformer for non-time series inputs"""

    def init_fun(rng, input_shape):
        kq, kk, kv = jr.split(rng, 3)
        Wq = W_init(kq, (input_shape[-1], input_shape[-1]))
        Wk = W_init(kk, (input_shape[-1], input_shape[-1]))
        Wv = W_init(kv, (input_shape[-1], input_shape[-1]))
        return input_shape, (Wq, Wk, Wv)

    @jit
    def apply_fun(params, inputs, **kwargs):
        Wq, Wk, Wv = params
        qvec = jnp.dot(inputs, Wq)
        kvec = jnp.dot(inputs, Wk)

        qk = softmax(jnp.einsum("...i, ...j->...ij", qvec, kvec))
        return inputs + jnp.einsum("...i,...ij->...j", inputs, qk @ Wv)

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
        key = jr.split(rng)[0]
        W = W_init(key, (input_shape[-1], out_dim))
        return output_shape, (W,)

    @jit
    def apply_fun(params, inputs, **kwargs):
        (W,) = params
        return jnp.einsum("...ij,...jk->...ik", inputs, W)

    return init_fun, apply_fun


def serial(*layers):
    """Combinator for composing multiple layers
    Args:
    *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

    Returns:
    A new layer, meaning an (init_fun, apply_fun) pair, representing the serial
    composition of the given sequence of layers."""
    nlayers = len(layers)
    init_funs, apply_funs = zip(*layers)

    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = jr.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_fun(params, inputs, **kwargs):
        rng = kwargs.pop("rng", None)
        rngs = jr.split(rng, nlayers) if rng is not None else (None,) * nlayers
        for fun, param, rng in zip(apply_funs, params, rngs):
            inputs = fun(param, inputs, rng=rng, **kwargs)
        return inputs

    return init_fun, apply_fun


# def SingleAttention(out_dim, inner_dim):
#     """Single Headed attention for single token dim"""
#     init_q, apply_q = Dense(inner_dim)
#     init_k, apply_k = Dense(inner_dim)
#     init_v, apply_v = Dense(out_dim)
#     init_out, apply_out = Dense(out_dim)

#     def init_fun(rng, input_shape):
#         k1, k2, k3, k4 = jr.split(rng, 4)
#         _, q_parm = init_q(k1, (input_shape))
#         _, k_parm = init_k(k2, (input_shape))
#         _, v_parm = init_v(k3, (input_shape))
#         _, out_parm = init_out(k4, (input_shape))
#         return (input_shape[:-1], out_dim), (
#             q_parm,
#             k_parm,
#             v_parm,
#             out_parm,
#         )

#     @jit
#     def apply_fun(params, inputs, **kwargs):
#         q_params, k_params, v_params, out_params = params

#         Q = (
#             apply_q(q_params, q_inputs)
#             .reshape(-1, n_cont_out, num_heads, d_k_in)
#             .swapaxes(1, 2)
#         )
#         K = (
#             apply_k(k_params, inputs)
#             .reshape(-1, n_cont_in, num_heads, d_k_in)
#             .swapaxes(1, 2)
#         )
#         V = (
#             apply_v(v_params, inputs)
#             .reshape(-1, n_cont_in, num_heads, d_k_out)
#             .swapaxes(1, 2)
#         )

#         attn = attention((Q, K, V)).reshape((-1, n_cont_out, d_model_out))
#         return apply_out(out_params, attn)

#     return init_fun, apply_fun


def MultiHeadAttn(n_context, d_model, num_heads):
    """Multi Headed Attention a la Perciever AR"""
    n_cont_in, n_cont_out = n_context
    d_model_in, d_model_out = d_model

    d_k_in = d_model_in // num_heads
    d_k_out = d_model_out // num_heads

    init_q, apply_q = Dense(d_model_in)
    init_k, apply_k = Dense(d_model_in)
    init_v, apply_v = Dense(d_model_out)
    init_out, apply_out = Dense(d_model_out)

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
