import os
import sys
import argparse
from training import TrainingConfig, TrainingManager

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Hive AI model')
    parser.add_argument('--games', type=int, default=1000,
                      help='Number of self-play games to generate')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Training batch size')
    parser.add_argument('--checkpoint-freq', type=int, default=50,
                      help='Save checkpoint every N games')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                      help='Directory to save logs')
    parser.add_argument('--resume', type=str,
                      help='Path to checkpoint to resume training from')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create configuration
    config = TrainingConfig()
    config.num_games = args.games
    config.batch_size = args.batch_size
    config.epochs_per_checkpoint = args.checkpoint_freq
    config.learning_rate = args.lr
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir
    
    # Create training manager
    manager = TrainingManager(config)
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        manager.network.load_state_dict(checkpoint['model_state_dict'])
        manager.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_game = checkpoint['game_idx'] + 1
        print(f"Resuming from game {start_game}")
    
    # Start training
    try:
        manager.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()