import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

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
        
    def select_move(self, game, temperature: float = 1.0) -> Tuple[Optional[HexCoord], HexCoord]:
        root = MCTSNode(game)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.expanded() and not node.game.is_game_over()[0]:
                action, node = node.select_child()
                search_path.append(node)
            
            # Expansion and evaluation
            value = node.expand(self.network)
            
            # Backup
            for node in reversed(search_path):
                value = -value
                node.backup(value)
        
        # Select move based on visit counts
        if temperature == 0:
            action = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        else:
            counts = np.array([child.visit_count for child in root.children.values()])
            probs = counts ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            action = list(root.children.keys())[np.random.choice(len(root.children), p=probs)]
        
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
        c_puct = 1.0
        
        def puct_score(child: 'MCTSNode') -> float:
            prior_score = c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            value_score = -child.value()
            return value_score + prior_score
        
        return max(self.children.items(), key=lambda x: puct_score(x[1]))
        
    def expand(self, network: HiveNetwork) -> float:
        """Expand node and return value estimate"""
        state_tensor = torch.FloatTensor(self.game.get_state()).unsqueeze(0)
        policy_logits, value = network(state_tensor)
        policy = F.softmax(policy_logits.squeeze(), dim=0)
        
        for move in self.game.get_valid_moves():
            next_game = self.game.clone()
            next_game.make_move(*move)
            self.children[move] = MCTSNode(next_game, parent=self, prior=policy[self._move_to_index(move)])
            
        return value.item()
        
    def backup(self, value: float) -> None:
        """Update node statistics"""
        self.value_sum += value
        self.visit_count += 1
        
    def value(self) -> float:
        """Get mean value of node"""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0
        
    def _move_to_index(self, move: Tuple[Optional[HexCoord], HexCoord]) -> int:
        """Convert move to policy index"""
        from_pos, to_pos = move
        if from_pos is None:
            # Placement move
            return (to_pos.r + 45) * 91 + (to_pos.q + 45)
        else:
            # Movement move
            return (from_pos.r + 45) * 91 + (from_pos.q + 45)

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
            # Get MCTS policy
            state = game.get_state()
            mcts_policy = self.mcts.select_move(game, temperature=1.0)
            
            # Store data
            states.append(state)
            policies.append(mcts_policy)
            
            # Make move
            game.make_move(*mcts_policy)
        
        # Get game outcome
        outcome = game.is_game_over()[1]
        winner_value = 1 if outcome == 0 else -1
        
        # Populate values based on game outcome
        current_value = winner_value
        for _ in range(len(states)):
            values.append(current_value)
            current_value = -current_value
            
        return list(zip(states, policies, values))
        
    def train_step(self, data: list) -> Tuple[float, float]:
        """Train network on a batch of data"""
        states, policies, values = zip(*data)
        state_tensor = torch.FloatTensor(np.array(states))
        policy_tensor = torch.LongTensor(policies)
        value_tensor = torch.FloatTensor(values)
        
        # Forward pass
        policy_logits, value_pred = self.network(state_tensor)
        
        # Calculate loss
        policy_loss = F.cross_entropy(policy_logits, policy_tensor)
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