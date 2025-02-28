"""Helper functions for simple experience sampling for DDQN."""
import random
import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Any
from jax import Array


def hard_max(x: Array, axis: int = -1) -> Array:
    """Returns 1.0 at max position and 0.0 elsewhere.
    
    Args:
        x: Input array
        axis: Axis along which to find maximum
        
    Returns:
        Binary array with 1.0 at maximum positions
    """
    return 1.0 - jnp.sign(jnp.max(x, axis=axis).reshape((-1, 1)) - x)


def format_batch(sample: List[Any]) -> Tuple[Array, Array, Array, Array, Array]:
    """Convert a list of experience tuples to batch format.
    
    Optimized version that uses pre-allocation and direct array creation.
    
    Args:
        sample: List of experience tuples
        
    Returns:
        Tuple of (states, actions, rewards, next_states, dones)
    """
    # Pre-allocate arrays when possible for better performance
    batch_size = len(sample)
    
    # For numeric data that we know the shape of, we can pre-allocate
    action_array = np.zeros((batch_size, 2), dtype=np.float32)
    reward_array = np.zeros(batch_size, dtype=np.float32)
    final_array = np.zeros((batch_size, 1), dtype=np.float32)
    
    # We need to determine state shape from the first example
    state_shape = sample[0][0].shape
    state_array = np.zeros((batch_size,) + state_shape, dtype=np.float32)
    state_p_array = np.zeros((batch_size,) + state_shape, dtype=np.float32)
    
    # Fill arrays directly - faster than list comprehensions
    for i, x in enumerate(sample):
        state_array[i] = x[0]
        action_array[i] = [1 - x[1], x[1]]
        reward_array[i] = x[2]
        state_p_array[i] = x[3]
        final_array[i, 0] = x[4]
    
    # Convert to JAX arrays once at the end
    return (
        jnp.array(state_array),
        jnp.array(action_array),
        jnp.array(reward_array),
        jnp.array(state_p_array),
        jnp.array(final_array)
    )


def sample_sequentially(experience_list: List, k: int = 32, j: int = 0) -> Tuple:
    """Sample experiences sequentially from the buffer.
    
    Args:
        experience_list: List of experiences to sample from
        k: Batch size
        j: Offset index for sampling
        
    Returns:
        Tuple of (states, actions, rewards, next_states, dones)
    """
    length = len(experience_list)
    sample = experience_list[(j * k) % length : ((j + 1) * k) % length]
    return format_batch(sample)


def sample_from(experience_list: List, k: int = 32) -> Tuple:
    """Randomly sample experiences from the buffer.
    
    Args:
        experience_list: List of experiences to sample from
        k: Number of samples to draw
        
    Returns:
        Tuple of (states, actions, rewards, next_states, dones)
    """
    # Handle case where we have fewer experiences than requested
    if len(experience_list) <= k:
        return format_batch(experience_list)
        
    # Otherwise sample randomly
    sample = random.sample(experience_list, k=k)
    return format_batch(sample)


def sample_all(experiences: List) -> Tuple:
    """Convert all experiences to batch format.
    
    Args:
        experiences: List of all experiences
        
    Returns:
        Tuple of (states, actions, rewards, next_states, dones)
    """
    return format_batch(experiences)


def sample_traj(experience_list: List[List], batch_size: int = 32) -> Tuple:
    """Sample trajectories of experiences.
    
    Args:
        experience_list: List of trajectory lists
        batch_size: Number of trajectories to sample
        
    Returns:
        Tuple of (states, actions, rewards, next_states, dones) with trajectory dimension
    """
    # Ensure we don't try to sample more trajectories than available
    actual_batch_size = min(batch_size, len(experience_list))
    sample = random.sample(experience_list, k=actual_batch_size)
    
    # Get trajectory length from first sample
    traj_len = len(sample[0])
    
    # Extract each component and maintain trajectory structure
    states, actions, rewards, next_states, dones = [], [], [], [], []
    
    for traj in sample:
        traj_states = []
        traj_actions = []
        traj_rewards = []
        traj_next_states = []
        traj_dones = []
        
        for step in traj:
            traj_states.append(step[0])
            traj_actions.append(step[1])
            traj_rewards.append(step[2])
            traj_next_states.append(step[3])
            traj_dones.append(step[4])
            
        states.append(traj_states)
        actions.append(traj_actions)
        rewards.append(traj_rewards)
        next_states.append(traj_next_states)
        dones.append(traj_dones)
    
    # Convert to JAX arrays
    states_array = jnp.array(states)
    actions_array = jnp.array(actions)
    
    # Format actions as one-hot
    actions_one_hot = jnp.stack(
        (jnp.ones_like(actions_array) - actions_array, actions_array), 
        axis=-1
    )
    
    return (
        states_array,               # [batch_size, traj_length, state_dim]
        actions_one_hot,            # [batch_size, traj_length, 2]
        jnp.array(rewards),         # [batch_size, traj_length]
        jnp.array(next_states),     # [batch_size, traj_length, state_dim]
        jnp.array(dones)            # [batch_size, traj_length]
    )


# Removed advantage computation functions that were used for prioritization


# Removed prioritized sampling functions to improve performance
