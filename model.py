"""Contains the model definition and loss functions for the No Thanks DDQN."""

from jax.example_libraries.stax import (
    Dense, 
    Relu, 
    serial, 
    parallel, 
    FanOut
)
from jax import jit
from custom_layers import ResDense, Dueling, Linear
import jax.numpy as jnp
from game import NoThanks
from functools import partial

# Get the game state size once at module level
mygame = NoThanks(4, 11)
mygame.start_game()
GAME_STATE_SIZE = len(mygame.get_things())
del mygame

# Define partial_predict for feature embedding (needed by play.py)
partial_predict = jit(serial(Linear(128), ResDense(16), ResDense(16))[1])


def build_model():
    """Builds the Dueling DDQN model with wider networks for better representation."""
    return serial(
        Dense(256),  # Increased network width for better representation
        Relu,
        Dense(128),  # Increased intermediate layer size
        Relu,
        ResDense(64),  # Added residual connection to improve gradient flow
        Relu,
        Dense(32),  # Additional layer for more complex representations
        Relu,
        FanOut(2),
        parallel(Dense(1), Dense(2)),
        Dueling(),
    )


def compute_advantage(q_values, q_values_taken, clip_value=10.0, epsilon=1e-5):
    """Compute normalized advantage values.
    
    Args:
        q_values: The Q-values for all actions
        q_values_taken: The Q-values for the actions actually taken
        clip_value: Value to clip Q-values to prevent extreme differences
        epsilon: Small value to prevent division by zero
        
    Returns:
        Normalized advantage values
    """
    # Estimate value function as mean of Q-values
    v_s_estimate = jnp.mean(q_values, axis=-1)
    
    # Clip Q-values to prevent extreme values
    q_values_taken_clipped = jnp.clip(q_values_taken, -clip_value, clip_value)
    v_s_estimate_clipped = jnp.clip(v_s_estimate, -clip_value, clip_value)
    
    # Calculate advantage = Q(s,a) - V(s)
    advantage = q_values_taken_clipped - v_s_estimate_clipped
    
    # Normalize advantages for stability
    mean_adv = jnp.mean(advantage)
    std_adv = jnp.std(advantage)
    return (advantage - mean_adv) / (std_adv + epsilon)


def compute_huber_loss(predictions, targets, delta=1.0):
    """Compute Huber loss between predictions and targets.
    
    Args:
        predictions: Predicted values
        targets: Target values
        delta: Threshold for quadratic vs. linear loss
        
    Returns:
        Huber loss values
    """
    abs_error = jnp.abs(predictions - targets)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + delta * linear


@partial(jit, static_argnums=(3,))
def loss(params, batch, old_params, include_state_prediction=False, key=None, kl_weight=0.1):
    """Unified loss function with optional state prediction and KL divergence component.
    
    Args:
        params: Parameters of the current policy network
        batch: Tuple of (states, actions, rewards, next_states, dones)
        old_params: Parameters of the target network
        include_state_prediction: Whether to include state prediction component
        key: Random key for operations that need randomness
        kl_weight: Weight for KL divergence term between current and target policy
        
    Returns:
        Loss value with GRPO and KL divergence components
    """
    s, a, r, sn, done = batch
    batch_size = s.shape[0]
    
    # Calculate Q-values based on whether state prediction is included
    # Use a boolean constant instead of directly checking the parameter
    use_state_pred = bool(include_state_prediction)
    
    if use_state_pred:
        new_q_values, hat_sn = predict(params, s)
        new_next_q_values, _ = predict(params, sn)
        old_next_q_values, _ = predict(old_params, sn)
        old_q_values, _ = predict(old_params, s)
        
        # Next state prediction loss
        embedd = params[0][0]
        loss_sn = jnp.mean(jnp.mean(jnp.square(jnp.dot(sn, embedd) - hat_sn), axis=-1))
    else:
        new_q_values = predict(params, s)
        new_next_q_values = predict(params, sn)
        old_next_q_values = predict(old_params, sn)
        old_q_values = predict(old_params, s)
        loss_sn = 0.0

    # Apply to action - current policy action values
    q_values = jnp.sum(new_q_values * a, axis=-1)
    
    # Old policy action values for the same action
    old_q_values_taken = jnp.sum(old_q_values * a, axis=-1)
    
    # Calculate advantage using helper function
    advantage = compute_advantage(old_q_values, old_q_values_taken)
    
    # Calculate probability ratio for GRPO with safety measures
    q_diff = jnp.clip(q_values - old_q_values_taken, -5.0, 5.0)
    ratio = jnp.exp(q_diff)
    ratio_clipped = jnp.clip(ratio, 0.9, 1.1)
    
    # Action from params, but value from old_params (standard DDQN)
    next_actions = jnp.argmax(new_next_q_values, axis=-1)
    old_next_q_values_sel = jnp.take_along_axis(
        old_next_q_values, next_actions[:, None], axis=-1
    ).squeeze()

    # Discount factor - different based on loss type
    discount_factor = 0.98 if include_state_prediction else 0.99
    target = r + discount_factor * done * old_next_q_values_sel
    
    # Advantage weighting
    advantage_weight = jnp.exp(jnp.clip(advantage, -1.0, 1.0))
    
    # GRPO modified target
    grpo_target = target * jnp.minimum(ratio, ratio_clipped)
    
    # Huber loss for robustness
    huber_loss = compute_huber_loss(q_values, grpo_target)
    
    # Weight the loss by advantage weight (removed importance sampling)
    weighted_loss = huber_loss * advantage_weight
    
    # Calculate KL divergence between current and target policy Q-distributions
    # Convert to probabilities using softmax
    new_q_probs = jnp.exp(new_q_values) / jnp.sum(jnp.exp(new_q_values), axis=-1, keepdims=True)
    old_q_probs = jnp.exp(old_q_values) / jnp.sum(jnp.exp(old_q_values), axis=-1, keepdims=True)
    
    # Avoid zeros in the division for numerical stability
    old_q_probs = jnp.clip(old_q_probs, 1e-5, 1.0)
    
    # KL divergence: sum(p_old * log(p_old / p_new))
    kl_div = jnp.sum(old_q_probs * jnp.log(old_q_probs / jnp.clip(new_q_probs, 1e-5, 1.0)), axis=-1)
    
    # Either return basic loss or combined with state prediction
    if use_state_pred:
        # Extra weighting on terminal episodes
        loss_q = jnp.mean((3.0 - 2.0 * done) * weighted_loss)
        # Add KL divergence term
        loss_with_kl = loss_q + kl_weight * jnp.mean(kl_div)
        # Add state prediction loss
        loss_weight = 0.5  # Weight for state prediction component
        return loss_with_kl + loss_weight * loss_sn
    else:
        # Add KL divergence term to the basic loss
        return jnp.mean(weighted_loss) + kl_weight * jnp.mean(kl_div)


# Initialize model and prediction function
init_random_params, predict = build_model()

# Compile the predict function
predict = jit(predict)

# Convenience function for backward compatibility
def all_loss(params, batch, old_params, key=None, kl_weight=0.1):
    """Loss function with state prediction component and KL divergence."""
    return loss(params, batch, old_params, True, key, kl_weight)

# JIT the all_loss function
all_loss = jit(all_loss)
