# CLAUDE.md - Commands and Style Guidelines

## Commands
- Run main training loop: `python main.py`
- Play against trained model: `python play.py`
- Run all tests: `pytest unit_tests/`
- Run specific test: `pytest unit_tests/test_single_game.py::test_run_single_game`

## Code Style Guidelines
- **Imports**: Group standard library, third-party, and local imports with a blank line between groups
- **Formatting**: Use consistent indentation (4 spaces)
- **Naming**: Use snake_case for functions/variables, CamelCase for classes
- **Types**: Use JAX types (jnp.array, etc.) consistently
- **Functions**: Add docstrings for all functions explaining purpose
- **Error Handling**: Use appropriate error handling with descriptive messages
- **Model Architecture**: Use JAX's functional programming style for layer definitions
- **Performance**: Use JIT compilation (@jit) for performance-critical functions
- **Comments**: Add comments for complex algorithms or calculations

## Project Structure
- Model code in `model.py`
- Game logic in `game.py` 
- Training loop in `main.py`
- Custom JAX layers in `custom_layers.py`
- Analysis tools in `analysis/` directory
