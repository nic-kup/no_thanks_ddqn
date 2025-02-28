"""Contains code for automatic game playing using DDQN."""
from game import NoThanks
import numpy.random as npr
from jax.nn import sigmoid
import jax.numpy as jnp
from typing import List, Tuple, Callable, Any, Dict
from functools import partial
from jax import jit


def apply_exploration_bonus(q_vals: jnp.ndarray, 
                           cur_player: int,
                           center_tokens: int,
                           action_counts: Dict[int, Dict[str, int]],
                           consecutive_actions: List[int],
                           exploration_bonus: float = 0.1,
                           max_consecutive: int = 3) -> jnp.ndarray:
    """Calculate and apply exploration bonuses to Q-values.
    
    Args:
        q_vals: Original Q-values from the model
        cur_player: Current player index
        center_tokens: Number of tokens in the center
        action_counts: Dictionary tracking action counts for each player
        consecutive_actions: List tracking consecutive same actions per player
        exploration_bonus: Exploration factor strength
        max_consecutive: Threshold before applying consecutive action penalty
        
    Returns:
        Modified Q-values with exploration bonuses applied
    """
    # Initialize bonuses array
    bonuses = jnp.zeros(2)
    
    # Incentivize taking card when there are tokens in center
    center_tokens_bonus = min(center_tokens * 0.05, 0.3)
    bonuses = bonuses.at[0].set(bonuses[0] + center_tokens_bonus)
    
    # Apply bonus for action diversity when needed
    if consecutive_actions[cur_player] >= max_consecutive:
        # Calculate bonus value based on streak length
        bonus_value = exploration_bonus * (consecutive_actions[cur_player] - max_consecutive + 1)
        
        # Encourage opposite action based on history
        if action_counts[cur_player]["take"] > action_counts[cur_player]["no_thanks"]:
            bonuses = bonuses.at[1].set(bonus_value)  # Encourage "no thanks"
        else:
            bonuses = bonuses.at[0].set(bonuses[0] + bonus_value)  # Encourage "take"
    
    # Return Q-values with bonuses applied
    return q_vals + bonuses


def single_game(predict: Callable, 
               param_list: Tuple, 
               reward_factor: float = 1.0, 
               inv_temp: float = 5.0, 
               exploration_bonus: float = 0.1) -> List:
    """Run a complete game with the given predict function and collect experiences.
    
    Args:
        predict: Q-value prediction function
        param_list: Tuple of model parameters for different players
        reward_factor: Scaling factor for rewards
        inv_temp: Inverse temperature for softmax action selection
        exploration_bonus: Magnitude of exploration bonus
        
    Returns:
        List of experiences collected during the game
    """
    new_exp = []
    mygame = NoThanks(4, 11, reward_factor=reward_factor)
    mygame.start_game()
    game_going = 1
    
    # Randomly assign parameters to each player
    player_param = npr.randint(0, len(param_list), size=4)
    
    # Initialize player state storage
    player_store = [
        (mygame.get_things_perspective(player), 1) for player in range(mygame.n_players)
    ]
    
    # Action diversity tracking
    player_action_counts = {player: {"take": 0, "no_thanks": 0} for player in range(mygame.n_players)}
    consecutive_same_action = [0] * mygame.n_players
    max_consecutive_actions = 3
    
    # Main game loop
    while game_going:
        cur_player = mygame.player_turn
        
        # Get current state and predict Q-values
        state = mygame.get_things()
        q_vals = predict(
            param_list[player_param[cur_player]], state.reshape((1, -1))
        ).ravel()
        
        # Apply exploration bonuses
        q_vals = apply_exploration_bonus(
            q_vals,
            cur_player,
            mygame.center_tokens,
            player_action_counts,
            consecutive_same_action,
            exploration_bonus,
            max_consecutive_actions
        )
        
        # Store current experience
        new_exp.append([*player_store[cur_player], state, 1.0])
        
        # Determine action using softmax or greedy selection
        if inv_temp is None:
            # Deterministic (greedy) selection
            take_action = q_vals[0] > q_vals[1]
        else:
            # Probabilistic selection with temperature
            take_prob = sigmoid(inv_temp * (q_vals[0] - q_vals[1]))
            take_action = take_prob > npr.random()
        
        # Take selected action and update state
        if take_action:
            # Take card action
            game_going, reward = mygame.take_card()
            player_store[cur_player] = (state, 0, reward)
            player_action_counts[cur_player]["take"] += 1
            
            # Update consecutive action counter
            consecutive_same_action[cur_player] = 1 if consecutive_same_action[cur_player] < 0 else consecutive_same_action[cur_player] + 1
        else:
            # No thanks action
            game_going, reward = mygame.no_thanks()
            player_store[cur_player] = (state, 1, reward)
            player_action_counts[cur_player]["no_thanks"] += 1
            
            # Update consecutive action counter
            consecutive_same_action[cur_player] = -1 if consecutive_same_action[cur_player] > 0 else consecutive_same_action[cur_player] - 1
    
    # Game over - record final experiences with terminal rewards
    winner = mygame.winning()
    for player in range(mygame.n_players):
        new_exp.append([
            player_store[player][0],  # s_t
            player_store[player][1],  # a_t
            winner[player],          # r_t (final reward)
            mygame.get_things_perspective(player),  # s_{t+1}
            0.0,                     # game_going = False
        ])
    
    # Filter out any incomplete experiences
    return [exp for exp in new_exp if len(exp) == 5]


def k_step_game(predict: Callable, 
               param_list: Tuple, 
               k: int = 1, 
               reward_factor: float = 1.0, 
               inv_temp: float = 5.0) -> List:
    """Run a game and collect k-step trajectories for each player.
    
    Args:
        predict: Q-value prediction function
        param_list: Tuple of model parameters for different players
        k: Number of steps in each trajectory
        reward_factor: Scaling factor for rewards
        inv_temp: Inverse temperature for softmax action selection
        
    Returns:
        List of k-step trajectories collected during the game
    """
    new_exp = []
    mygame = NoThanks(4, 11, reward_factor=reward_factor)
    mygame.start_game()
    game_going = 1
    
    # Randomly assign parameters to each player
    player_param = npr.randint(0, len(param_list), size=4)
    
    # Initialize k-step buffers for each player
    k_step_buffer = [[] for _ in range(mygame.n_players)]
    
    # Initialize player state storage
    player_store = [(mygame.get_things(), 1.0) for player in range(mygame.n_players)]
    
    # Main game loop
    while game_going:
        cur_player = mygame.player_turn
        
        # Get current state and predict Q-values
        state = mygame.get_things()
        q_vals = predict(
            param_list[player_param[cur_player]], state.reshape((1, -1))
        ).ravel()
        
        # Store experience in k-step buffer if valid
        if len(player_store[cur_player]) == 3:
            k_step_buffer[cur_player].append([*player_store[cur_player], state, 1.0])
        
        # Determine action
        if inv_temp is None:
            take_action = q_vals[0] > q_vals[1]
        else:
            take_prob = sigmoid(inv_temp * (q_vals[0] - q_vals[1]))
            take_action = take_prob > npr.random()
        
        # Take selected action
        if take_action:
            game_going, reward = mygame.take_card()
            player_store[cur_player] = (state, 0, reward)
        else:
            game_going, reward = mygame.no_thanks()
            player_store[cur_player] = (state, 1, reward)
        
        # If k-step buffer reaches size k, add to experiences and remove oldest
        if len(k_step_buffer[cur_player]) == k:
            new_exp.append(k_step_buffer[cur_player].copy())
            k_step_buffer[cur_player].pop(0)
    
    # Add final experiences with terminal rewards
    winner = mygame.winning()
    for player in range(mygame.n_players):
        k_step_buffer[player].append([
            player_store[player][0],  # s_t
            player_store[player][1],  # a_t
            winner[player],          # r_t (final reward)
            mygame.get_things_perspective(player),  # s_{t+1}
            0.0,                     # game_going = False
        ])
        
        # If buffer has reached size k, add to experiences
        if len(k_step_buffer[player]) == k:
            new_exp.append(k_step_buffer[cur_player].copy())
    
    return new_exp
