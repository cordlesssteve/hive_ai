#Training Script
#training.py

import torch
import torch.optim as optim
import logging
import datetime
import os
from typing import Tuple, List
import numpy as np
from tqdm import tqdm
from game import HiveGame
from model import HiveNetwork, SelfPlayTrainer

class TrainingConfig:
    def __init__(self):
        self.num_games = 1000           # Total number of self-play games
        self.batch_size = 128           # Training batch size
        self.epochs_per_checkpoint = 50  # Save model every N epochs
        self.temperature_threshold = 30  # Number of moves before temperature drops
        self.initial_temp = 1.0         # Initial temperature for move selection
        self.final_temp = 0.1           # Final temperature after threshold
        self.learning_rate = 0.001      # Learning rate for optimizer
        self.checkpoint_dir = "checkpoints"
        self.log_dir = "logs"

class TrainingManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_directories()
        self.setup_logging()
        
        # Initialize network and training components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = HiveNetwork().to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=config.learning_rate
        )
        self.trainer = SelfPlayTrainer(self.network, self.optimizer)
        
        logging.info(f"Using device: {self.device}")
        logging.info(f"Network architecture:\n{self.network}")

    def setup_directories(self):
        """Create necessary directories for checkpoints and logs"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

    def setup_logging(self):
        """Configure logging"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.config.log_dir, f"training_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def train(self):
        """Main training loop"""
        best_loss = float('inf')
        
        for game_idx in range(self.config.num_games):
            logging.info(f"Starting game {game_idx + 1}/{self.config.num_games}")
            
            # Generate self-play game
            game_data = self.generate_game(game_idx)
            
            # Train on game data
            epoch_losses = self.train_on_game(game_data)
            avg_loss = np.mean([p + v for p, v in epoch_losses])
            
            # Log progress
            self.log_training_progress(game_idx, epoch_losses)
            
            # Save checkpoints
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(game_idx, best=True)
            
            if (game_idx + 1) % self.config.epochs_per_checkpoint == 0:
                self.save_checkpoint(game_idx)

    def generate_game(self, game_idx: int) -> List[Tuple]:
        """Generate a single self-play game with progress tracking"""
        logging.info("Generating self-play game...")
        game = HiveGame()
        states, policies, values = [], [], []
        
        with tqdm(total=100, desc="Game Progress") as pbar:
            move_count = 0
            while not game.is_game_over()[0]:
                # Update temperature based on move count
                temp = self.get_temperature(move_count)
                
                # Get MCTS policy
                state = game.get_state()
                mcts_policy = self.trainer.mcts.select_move(game, temperature=temp)
                
                # Store data
                states.append(state)
                policies.append(mcts_policy)
                
                # Make move
                game.make_move(*mcts_policy)
                move_count += 1
                
                # Update progress bar (estimate 50 moves per game)
                pbar.update(2)
        
        # Calculate final values based on game outcome
        outcome = game.is_game_over()[1]
        values = self.calculate_game_values(len(states), outcome)
        
        return list(zip(states, policies, values))

    def train_on_game(self, game_data: List[Tuple]) -> List[Tuple[float, float]]:
        """Train network on a single game's data"""
        losses = []
        num_batches = (len(game_data) + self.config.batch_size - 1) // self.config.batch_size
        
        with tqdm(total=num_batches, desc="Training") as pbar:
            for i in range(0, len(game_data), self.config.batch_size):
                batch = game_data[i:i + self.config.batch_size]
                policy_loss, value_loss = self.trainer.train_step(batch)
                losses.append((policy_loss, value_loss))
                pbar.update(1)
        
        return losses

    def get_temperature(self, move_count: int) -> float:
        """Get temperature for move selection based on move count"""
        if move_count < self.config.temperature_threshold:
            return self.config.initial_temp
        return self.config.final_temp

    def calculate_game_values(self, num_states: int, outcome: int) -> List[float]:
        """Calculate values for all game states based on final outcome"""
        winner_value = 1 if outcome == 0 else -1
        values = []
        current_value = winner_value
        
        for _ in range(num_states):
            values.append(current_value)
            current_value = -current_value
        
        return values

    def save_checkpoint(self, game_idx: int, best: bool = False):
        """Save model checkpoint"""
        prefix = 'best' if best else f'game_{game_idx}'
        path = os.path.join(self.config.checkpoint_dir, f'{prefix}_model.pt')
        
        checkpoint = {
            'game_idx': game_idx,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint to {path}")

    def log_training_progress(self, game_idx: int, losses: List[Tuple[float, float]]):
        """Log training progress"""
        avg_policy_loss = np.mean([p for p, v in losses])
        avg_value_loss = np.mean([v for p, v in losses])
        total_loss = avg_policy_loss + avg_value_loss
        
        logging.info(
            f"Game {game_idx + 1} complete:\n"
            f"  Average Policy Loss: {avg_policy_loss:.4f}\n"
            f"  Average Value Loss: {avg_value_loss:.4f}\n"
            f"  Total Loss: {total_loss:.4f}"
        )

def main():
    # Set up configuration
    config = TrainingConfig()
    
    # Create training manager and start training
    manager = TrainingManager(config)
    
    try:
        manager.train()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
    finally:
        logging.info("Training completed")

if __name__ == "__main__":
    main()