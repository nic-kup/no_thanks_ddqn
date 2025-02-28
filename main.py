"""Training script for Dueling DDQN to play No Thanks card game."""
from random import sample
from numpy import load
import time
import jax.numpy as jnp
import jax.random as jr
from jax.nn import sigmoid
import numpy.random as npr
from jax import jit, grad
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from joblib import Parallel, delayed
from functools import partial
from game import NoThanks
from single_game import single_game
from play import print_cards_from_one_hot
from tree_helper import lion_step, tree_zeros_like, convex_comb
from model import init_random_params, predict, loss
from sample_helpers import sample_from, sample_all
import logging
import os


def setup_logging():
    """Setup logging configuration for tracking training progress."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('DDQN-NT')


@jit
def pred_q_values(params, state):
    """Get Q-values for a given state using the current policy network."""
    return predict(params, state)[0]


def play_games(predict_fn, param_list, num_games, inv_temp, exploration_bonus=0.1):
    """Play multiple games in parallel and collect experiences.
    
    Args:
        predict_fn: Function to predict Q-values from states
        param_list: Tuple of parameter sets for different players
        num_games: Number of games to play
        inv_temp: Inverse temperature for softmax action selection
        exploration_bonus: Magnitude of exploration bonus
        
    Returns:
        List of experiences collected from all games
    """
    return Parallel(n_jobs=-1, backend="threading")(
        delayed(single_game)(predict_fn, param_list, 0.0, inv_temp, exploration_bonus)
        for _ in range(num_games)
    )


def load_model_params(path, params_template):
    """Load model parameters from file.
    
    Args:
        path: Path to the parameter file
        params_template: Template parameters with correct structure
        
    Returns:
        Loaded parameters
    """
    npz_files = load(path)
    leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]
    tree_def = tree_flatten(params_template)[1]
    return tree_unflatten(tree_def, leaves)


def save_model_params(params, filename):
    """Save model parameters to file.
    
    Args:
        params: Model parameters to save
        filename: Output filename
    """
    leaves, _ = tree_flatten(params)
    jnp.savez(filename, *leaves)


def run_sample_game(params, logger):
    """Run a sample game using the trained model and log the results.
    
    Args:
        params: Trained model parameters
        logger: Logger for output
    """
    logger.info("Running example game with trained model")
    
    mygame = NoThanks(4, 11)
    mygame.start_game()
    game_going = 1
    game_states = []
    
    logger.info("|-------|")
    while game_going:
        cur_player = mygame.player_turn
        state = mygame.get_things().reshape((1, -1))
        game_states.append(state)
        q_vals = pred_q_values(params, state).ravel()
        player_tkns = mygame.get_player_tokens_int(cur_player)
        player_cards = mygame.player_cards[cur_player]

        logger.info(f"Player: {mygame.player_turn} | Tokens: {player_tkns}")
        logger.info(f"Cards: {print_cards_from_one_hot(player_cards)}")
        logger.info(f"Center: Card {mygame.center_card}, Tokens {mygame.center_tokens}")
        logger.info(f"Cards left: {len(mygame.cards)}")
        logger.info(f"Q_vals: {q_vals}")

        if q_vals[0] > q_vals[1]:
            game_going, rew = mygame.take_card()
            logger.info(f"Action: take, Reward: {rew}")
        else:
            game_going, rew = mygame.no_thanks()
            logger.info(f"Action: no_thanks, Reward: {rew}")

        logger.info("-----------")

    logger.info(f"Final scores: {mygame.score()}")
    logger.info(f"Winner rewards: {mygame.winning()}")

    for x in range(mygame.n_players):
        logger.info(
            f"Player {x}: Tokens={mygame.get_player_tokens_int(x):<3} | Cards={print_cards_from_one_hot(mygame.player_cards[x])}"
        )


def main():
    """Main training function."""
    # Setup logging
    logger = setup_logging()
    
    # Hyperparameters
    config = {
        # Optimization parameters
        "step_size_initial": 3e-4,  # Initial learning rate
        "step_size_final": 8e-5,    # Final learning rate
        "weight_decay_initial": 0.25, # Initial weight decay
        "weight_decay_final": 0.5,   # Final weight decay
        "polyak_tau": 0.99,          # Target network update rate
        
        # Training schedule
        "epochs": 299,               # Total training epochs
        "reset_target_every": 40,    # How often to save old target networks
        "max_inv_temp": 100,         # Maximum inverse temperature for exploration
        "max_replay_buffer": 50000,  # Maximum experience replay buffer size
        
        # No longer using prioritized replay
        
        # KL divergence parameters
        "kl_weight_initial": 0.05,       # Initial KL divergence weight
        "kl_weight_final": 0.2,          # Final KL divergence weight
        
        # Other settings
        "continue_training": False,  # Whether to continue from a saved model
        "seed": 4                    # Random seed
    }
    
    # Log configuration
    logger.info("Starting training with configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize game environment to get input size
    mygame = NoThanks(4, 11)
    mygame.start_game()
    input_size = len(mygame.get_things())
    logger.info(f"State representation size: {input_size}")
    
    # Initialize random keys
    key = jr.PRNGKey(config["seed"])
    key, subkey = jr.split(key)
    
    # Initialize model parameters
    params = init_random_params(subkey, (-1, input_size))[1]
    key, subkey = jr.split(key)
    
    # Initialize optimizer state
    old_params = tree_zeros_like(params)  # Target network parameters
    oldp_params = init_random_params(subkey, (-1, input_size))[1]  # Previous target network
    key, subkey = jr.split(key)
    momentum = tree_zeros_like(params)  # Optimizer momentum
    
    # Initialize training state
    inv_temp = 1  # Initial exploration (inverse temperature)
    experiences = []  # Experience replay buffer
    
    # Define specialized gradient function for loss with KL divergence
    @jit
    def loss_grad(params, batch, old_params, key, kl_weight=0.1):
        """Compute gradients of the loss function with KL divergence term."""
        return grad(lambda p: loss(p, batch, old_params, False, key, kl_weight))(params)
    
    # Load previous training state if requested
    if config["continue_training"]:
        logger.info("Continuing from previous training run")
        if os.path.exists("params_end.npz"):
            params = load_model_params("params_end.npz", params)
            old_params = params.copy()
            inv_temp = config["max_inv_temp"] * 2  # Use high inverse temp (less exploration)
        else:
            logger.warning("No previous parameters found, starting fresh")
    else:
        logger.info("Starting a new training run")
    
    # Initialize training variables
    step_size = config["step_size_initial"]
    weight_decay = config["weight_decay_initial"]
    game_loss = 0.0
    
    # Main training loop
    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()
        
        # Update learning parameters based on training progress
        if epoch < 2 * config["max_inv_temp"]:
            # Early training phase - higher learning rates, more exploration
            weight_decay = config["weight_decay_initial"]
            step_size = config["step_size_initial"]
            inv_temp = min(epoch, config["max_inv_temp"])
        else:
            # Late training phase - lower learning rates, less exploration
            weight_decay = config["weight_decay_final"]
            step_size = config["step_size_final"]
            inv_temp = 2 * config["max_inv_temp"]

        # Manage target networks
        if epoch % config["reset_target_every"] == 1 and epoch > 2:
            logger.info("Saving previous target network for diversity in self-play")
            oldp_params = old_params.copy()  # Store previous target network

        # Apply soft update to target network using Polyak averaging
        if epoch > 0:  # Skip first epoch to avoid zeros
            old_params = convex_comb(params, old_params, config["polyak_tau"])
        
        # Phase 1: Generate new experiences through self-play
        start_time = time.time()
        
        # Dynamic exploration schedule
        exploration_bonus = max(0.3 * (1.0 - epoch / config["epochs"]), 0.05)
        
        # Dynamic game count schedule
        games_to_play = 5
        if epoch <= 1:
            # More games at the beginning to bootstrap learning
            games_to_play += 200
        if epoch % 10 == 0:
            # Periodically refresh experience buffer
            games_to_play += 10
        
        # Run games in parallel to collect experiences
        game_results = play_games(
            pred_q_values,
            (params, old_params, oldp_params),
            games_to_play,
            inv_temp,
            exploration_bonus
        )
        
        # Flatten results from all games
        new_experiences = [item for sublist in game_results for item in sublist]
        game_time = time.time() - start_time
        
        # Update experience replay buffer
        experiences = new_experiences + sample(
            experiences, 
            k=min(config["max_replay_buffer"] - len(new_experiences), len(experiences))
        )
        
        # Phase 2: Train model on collected experiences
        start_time = time.time()
        
        # Dynamic training schedule - fewer steps with larger batches
        num_gradient_steps = 128
        if epoch < 5:
            # More updates in early epochs
            num_gradient_steps += 64
            
        # Use larger batch sizes for better vectorization and performance
        batch_size = 1024
        if epoch % 5 == 0:
            # Periodically use even larger batches for better gradient estimates
            batch_size = 2048
        
        # No longer using prioritized replay
        
        # Perform gradient updates
        for step in range(num_gradient_steps):
            # Small learning rate jitter for better convergence
            current_step_size = step_size / (jr.uniform(subkey, shape=(1,)) + 0.1)
            
            # Use simple uniform sampling for all batches (much faster)
            batch = sample_from(experiences, k=batch_size)
            
            # Calculate KL weight with proper schedule
            kl_progress = min(1.0, epoch / 100)
            kl_weight = config["kl_weight_initial"] + kl_progress * (config["kl_weight_final"] - config["kl_weight_initial"])
            
            # Calculate gradients with KL divergence
            grads = loss_grad(params, batch, old_params, subkey, kl_weight)
            
            # Update parameters using Lion optimizer
            params, momentum = lion_step(current_step_size, params, grads, momentum, wd=weight_decay)
            key, subkey = jr.split(key)
            
        training_time = time.time() - start_time
        
        # Log progress
        logger.info(
            f"Epoch {epoch:3d}: |new_exp|={len(new_experiences):<5} | "
            f"game_time={game_time:5.2f}s | train_time={training_time:5.2f}s"
        )
        
        # Periodically evaluate and save model
        if epoch % 5 == 0 or epoch == config["epochs"] - 1:
            # Calculate current loss on a large batch
            eval_batch = sample_all(experiences)
            # Use the loss function directly for evaluation, with include_state_prediction=False
            game_loss = jnp.mean(loss(params, eval_batch, old_params, False, subkey))
            key, subkey = jr.split(key)
            
            # Save intermediate parameters
            save_model_params(params, "all_params")
            
            logger.info(
                f"Evaluation at epoch {epoch:3d}: Loss={game_loss:9.4f} | "
                f"Buffer size={len(experiences)}"
            )
    
    # Save final model
    save_model_params(params, "params_end")
    logger.info("Training complete. Saved final parameters as params_end.npz")
    
    # Run a sample game with the trained model
    run_sample_game(params, logger)


if __name__ == "__main__":
    main()
