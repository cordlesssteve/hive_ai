import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from game import HiveGame, HexCoord

class HiveConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))

class HiveNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input: 91x91x12 (board state)
        # The board is large enough to accommodate any reasonable game state
        # 12 channels: 2 players * (5 piece types + 1 for stack height)
        self.conv_blocks = nn.Sequential(
            HiveConvBlock(12, 64),
            HiveConvBlock(64, 128),
            HiveConvBlock(128, 256),
            HiveConvBlock(256, 256),
            HiveConvBlock(256, 256),
        )
        
        # Policy head (outputs move probabilities)
        self.policy_conv = nn.Conv2d(256, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(91 * 91 * 32, 91 * 91)  # One probability per possible move
        
        # Value head (outputs win probability)
        self.value_conv = nn.Conv2d(256, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(91 * 91 * 32, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Common layers
        x = self.conv_blocks(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 91 * 91 * 32)  # Flatten
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 91 * 91 * 32)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class MCTS:
    """Monte Carlo Tree Search for move selection"""
    def __init__(self, network: HiveNetwork, num_simulations: int = 800):
        self.network = network
        self.num_simulations = num_simulations

    def select_move(self, game: HiveGame, temperature: float = 1.0) -> Tuple[Optional[HexCoord], HexCoord]:
        """Select a move using Monte Carlo Tree Search"""
        print("\nStarting new MCTS search")
        root = MCTSNode(game)
        
        print(f"\nStarting MCTS with {self.num_simulations} simulations")
        for i in range(self.num_simulations):
            if i % 100 == 0:
                print(f"Simulation {i}/{self.num_simulations}")
            
            node = root
            search_path = [node]
            print(f"\nSimulation {i} - Starting selection phase")
            
            # Selection
            selection_depth = 0
            while node.expanded() and not node.game.is_game_over()[0]:
                try:
                    action, next_node = node.select_child()
                    search_path.append(next_node)
                    node = next_node
                    selection_depth += 1
                    print(f"Selection depth: {selection_depth}")
                except ValueError as e:
                    print(f"Selection stopped: {e}")
                    break
            
            print(f"Selection phase complete. Final depth: {selection_depth}")
            
            # Expansion and evaluation
            print("Starting expansion phase")
            try:
                value = node.expand(self.network)
                print(f"Node expanded with value: {value}")
            except Exception as e:
                print(f"Expansion failed: {e}")
                continue
            
            # Backup
            print("Starting backup phase")
            for node in reversed(search_path):
                node.backup(-value)
                value = -value
            print("Backup complete")
        
        if not root.children:
            raise ValueError("No valid moves available from root position")
        
        print(f"\nMCTS completed. Root node stats:")
        print(f"Number of children: {len(root.children)}")
        print(f"Total visits: {root.visit_count}")
        
        # Select move based on visit counts with safeguards
        if temperature == 0:
            # Select most visited move
            action = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        else:
            counts = np.array([child.visit_count for child in root.children.values()])
            
            # Add small constant to prevent division by zero
            counts = counts + 1e-8
            
            # Apply temperature
            if temperature != 1.0:
                counts = counts ** (1.0 / temperature)
            
            # Normalize probabilities with extra safety checks
            total = np.sum(counts)
            if total > 0:
                probs = counts / total
            else:
                # If all counts are zero (shouldn't happen), use uniform distribution
                probs = np.ones_like(counts) / len(counts)
            
            # Ensure probabilities sum to 1 and are valid
            probs = np.clip(probs, 0, 1)
            probs = probs / np.sum(probs)
            
            print(f"Move probabilities: min={np.min(probs):.6f}, max={np.max(probs):.6f}, sum={np.sum(probs):.6f}")
            
            try:
                action_idx = np.random.choice(len(root.children), p=probs)
                action = list(root.children.keys())[action_idx]
            except ValueError as e:
                print(f"Warning: Problem with probability distribution: {e}")
                # Fallback to most visited move
                action = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        
        print(f"Selected action: {action}")
        return action

class MCTSNode:
    def __init__(self, game, parent=None, prior=0):
        self.game = game
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.state = None
        
    def expanded(self) -> bool:
        return len(self.children) > 0
        
    def select_child(self) -> Tuple[Tuple[Optional[HexCoord], HexCoord], 'MCTSNode']:
        """Select a child node using PUCT algorithm"""
        print(f"Selecting child from {len(self.children)} children")
        
        if not self.children:
            raise ValueError("Cannot select child from node with no children")
            
        c_puct = 1.0
        
        def puct_score(child: 'MCTSNode') -> float:
            prior_score = c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            if child.visit_count > 0:
                value_score = -child.value()
            else:
                value_score = 0
            return value_score + prior_score
        
        best_move, best_child = max(self.children.items(), key=lambda x: puct_score(x[1]))
        print(f"Selected move: {best_move}")
        return best_move, best_child
        
    def expand(self, network: HiveNetwork) -> float:
        """Expand node and return value estimate"""
        print(f"Starting node expansion")
        print(f"Current game state: {len(self.game.board)} pieces on board")
        
        # Get valid moves first to avoid unnecessary computation if there are none
        valid_moves = self.game.get_valid_moves()
        print(f"Valid moves: {len(valid_moves)}")
        
        if not valid_moves:
            raise ValueError("No valid moves available for expansion")
        
        # Get network prediction
        state_tensor = torch.FloatTensor(self.game.get_state()).unsqueeze(0)
        print(f"Created state tensor with shape: {state_tensor.shape}")
        
        with torch.no_grad():
            policy_logits, value = network(state_tensor)
            print(f"Network evaluation complete. Policy shape: {policy_logits.shape}")
        
        # Process policy
        policy = F.softmax(policy_logits.squeeze(), dim=0)
        print(f"Processing {len(valid_moves)} valid moves")
        
        # Create children nodes
        for move in valid_moves:
            print(f"Processing move: {move}")
            try:
                next_game = self.game.clone()
                next_game.make_move(*move)
                move_idx = self._move_to_index(move)
                self.children[move] = MCTSNode(
                    next_game,
                    parent=self,
                    prior=policy[move_idx].item()
                )
            except Exception as e:
                print(f"Failed to process move {move}: {str(e)}")
        
        print(f"Node expansion complete. Created {len(self.children)} children")
        return value.item()

    def backup(self, value: float) -> None:
        """Update node statistics"""
        self.value_sum += value
        self.visit_count += 1
        
    def value(self) -> float:
        """Get mean value of node"""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0
        
    def _move_to_index(self, move: Tuple[Optional[HexCoord], HexCoord]) -> int:
        """Convert move to policy index.
        For placement moves, use the destination position.
        For movement moves, use the source position.
        This ensures consistency with the network's policy output."""
        _, to_pos = move  # Always use the destination position for index
        return (to_pos.r + 45) * 91 + (to_pos.q + 45)

class SelfPlayTrainer:
    def __init__(self, network: HiveNetwork, optimizer: torch.optim.Optimizer):
        self.network = network
        self.optimizer = optimizer
        self.mcts = MCTS(network)
        
    def generate_game(self) -> list:
        """Generate a self-play game and return training data"""
        game = HiveGame()
        states, policies, values = [], [], []
        
        while not game.is_game_over()[0]:
            # Store current state
            state = game.get_state()
            
            # Create policy vector
            policy = np.zeros(91 * 91)  # Initialize empty policy vector
            
            # Get move from MCTS
            valid_moves = game.get_valid_moves()
            selected_move = self.mcts.select_move(game, temperature=1.0)
            
            # Create policy vector from MCTS probabilities
            for move in valid_moves:
                move_idx = self._move_to_index(move)
                # Set 1 for the selected move, 0 for others
                # Could be enhanced to use actual MCTS visit counts
                policy[move_idx] = 1.0 if move == selected_move else 0.0
            
            # Store data
            states.append(state)
            policies.append(policy)
            
            # Make move
            game.make_move(*selected_move)
        
        # Get game outcome
        outcome = game.is_game_over()[1]
        winner_value = 1.0 if outcome == 0 else -1.0
        
        # Populate values based on game outcome
        current_value = winner_value
        for _ in range(len(states)):
            values.append(current_value)
            current_value = -current_value
            
        return list(zip(states, policies, values))
        
    def train_step(self, data: list) -> Tuple[float, float]:
        """Train network on a batch of data"""
        states, policies, values = zip(*data)
        
        # Convert to tensors with correct shapes and types
        state_tensor = torch.FloatTensor(np.array(states))
        policy_tensor = torch.FloatTensor(np.array(policies))  # Changed from LongTensor
        value_tensor = torch.FloatTensor(values)
        
        # Forward pass
        policy_logits, value_pred = self.network(state_tensor)
        
        # Calculate loss
        # Changed from cross_entropy to KL divergence since we're using probability distributions
        policy_loss = -(policy_tensor * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(value_pred.squeeze(), value_tensor)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()

# Training example usage:
def train_network():
    network = HiveNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    trainer = SelfPlayTrainer(network, optimizer)
    
    num_games = 1000
    batch_size = 128
    
    for game_idx in range(num_games):
        # Generate self-play game
        game_data = trainer.generate_game()
        
        # Train on game data
        for i in range(0, len(game_data), batch_size):
            batch = game_data[i:i + batch_size]
            policy_loss, value_loss = trainer.train_step(batch)
            
            print(f"Game {game_idx}, Batch {i//batch_size}")
            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
        
        # Save model periodically
        if game_idx % 100 == 0:
            torch.save(network.state_dict(), f"hive_model_{game_idx}.pt")