import numpy as np
from enum import Enum
from typing import List, Tuple, Dict, Set

class PieceType(Enum):
    QUEEN_BEE = 1
    SPIDER = 2
    BEETLE = 3
    GRASSHOPPER = 4
    ANT = 5

class Piece:
    def __init__(self, piece_type: PieceType, player: int):
        self.piece_type = piece_type
        self.player = player  # 0 or 1
        self.is_on_top = True  # For beetles climbing

class HexCoord:
    """Axial coordinate system for hexagonal grid"""
    def __init__(self, q: int, r: int):
        self.q = q
        self.r = r
    
    def __eq__(self, other):
        return self.q == other.q and self.r == other.r
    
    def __hash__(self):
        return hash((self.q, self.r))
    
    def get_neighbors(self) -> List['HexCoord']:
        """Returns all adjacent hex coordinates"""
        directions = [
            (1, 0), (1, -1), (0, -1),
            (-1, 0), (-1, 1), (0, 1)
        ]
        return [HexCoord(self.q + q, self.r + r) for q, r in directions]

class HiveGame:
    def __init__(self):
        self.board: Dict[HexCoord, List[Piece]] = {}  # Stack of pieces at each position
        self.player_hands: List[Dict[PieceType, int]] = [
            {
                PieceType.QUEEN_BEE: 1,
                PieceType.SPIDER: 2,
                PieceType.BEETLE: 2,
                PieceType.GRASSHOPPER: 3,
                PieceType.ANT: 3
            },
            {
                PieceType.QUEEN_BEE: 1,
                PieceType.SPIDER: 2,
                PieceType.BEETLE: 2,
                PieceType.GRASSHOPPER: 3,
                PieceType.ANT: 3
            }
        ]
        self.current_player = 0
        self.turn = 0
        self.queen_bee_placed = [False, False]

    def get_valid_moves(self) -> List[Tuple[HexCoord, HexCoord]]:
        """Returns list of valid moves as (from_pos, to_pos) tuples"""
        moves = []
        
        # If it's turn 4 and queen isn't placed, only allow queen placement
        if self.turn >= 6 and not self.queen_bee_placed[self.current_player]:
            return self._get_valid_placements(PieceType.QUEEN_BEE)
            
        # Get valid piece placements
        for piece_type in PieceType:
            if self.player_hands[self.current_player][piece_type] > 0:
                moves.extend(self._get_valid_placements(piece_type))
        
        # Get valid piece movements
        for pos, stack in self.board.items():
            if stack[-1].player == self.current_player:
                moves.extend(self._get_valid_movements(pos))
        
        return moves

    def _get_valid_placements(self, piece_type: PieceType) -> List[Tuple[None, HexCoord]]:
        """Get valid positions to place a new piece"""
        if not self.board:  # First move
            return [(None, HexCoord(0, 0))]
            
        valid_positions = set()
        
        # Must be adjacent to friendly piece
        for pos, stack in self.board.items():
            if stack[-1].player == self.current_player:
                for neighbor in pos.get_neighbors():
                    if neighbor not in self.board:
                        valid_positions.add(neighbor)
        
        # Cannot be adjacent to enemy pieces
        for pos, stack in self.board.items():
            if stack[-1].player != self.current_player:
                for neighbor in pos.get_neighbors():
                    valid_positions.discard(neighbor)
        
        return [(None, pos) for pos in valid_positions]

    def _get_valid_movements(self, from_pos: HexCoord) -> List[Tuple[HexCoord, HexCoord]]:
        """Get valid positions to move an existing piece"""
        moves = []
        piece = self.board[from_pos][-1]
        
        # Check if moving this piece would break the hive
        if not self._can_remove_piece(from_pos):
            return []
            
        if piece.piece_type == PieceType.QUEEN_BEE:
            moves.extend(self._get_queen_moves(from_pos))
        elif piece.piece_type == PieceType.SPIDER:
            moves.extend(self._get_spider_moves(from_pos))
        elif piece.piece_type == PieceType.BEETLE:
            moves.extend(self._get_beetle_moves(from_pos))
        elif piece.piece_type == PieceType.GRASSHOPPER:
            moves.extend(self._get_grasshopper_moves(from_pos))
        elif piece.piece_type == PieceType.ANT:
            moves.extend(self._get_ant_moves(from_pos))
            
        return moves

    def make_move(self, from_pos: HexCoord, to_pos: HexCoord) -> None:
        """Execute a move, either placing a new piece or moving an existing one"""
        if from_pos is None:  # Placing new piece
            piece_type = self._determine_piece_type(to_pos)
            piece = Piece(piece_type, self.current_player)
            if piece_type == PieceType.QUEEN_BEE:
                self.queen_bee_placed[self.current_player] = True
            self.player_hands[self.current_player][piece_type] -= 1
            self.board[to_pos] = [piece]
        else:  # Moving existing piece
            piece = self.board[from_pos].pop()
            if not self.board[from_pos]:  # Remove empty stacks
                del self.board[from_pos]
            if to_pos in self.board:
                self.board[to_pos].append(piece)
            else:
                self.board[to_pos] = [piece]
                
        self.current_player = 1 - self.current_player
        self.turn += 1

    def _can_remove_piece(self, pos: HexCoord) -> bool:
        """Check if removing a piece would break the hive"""
        if len(self.board[pos]) > 1:  # Can always remove from stack
            return True
            
        # Temporarily remove the piece
        temp_board = self.board.copy()
        del temp_board[pos]
        
        if not temp_board:  # No other pieces
            return True
            
        # Use flood fill to check if all pieces are connected
        start_pos = next(iter(temp_board.keys()))
        connected = self._flood_fill(start_pos, temp_board)
        
        return len(connected) == len(temp_board)

    def _flood_fill(self, start: HexCoord, board: Dict[HexCoord, List[Piece]]) -> Set[HexCoord]:
        """Returns set of all positions connected to start position"""
        connected = {start}
        stack = [start]
        
        while stack:
            current = stack.pop()
            for neighbor in current.get_neighbors():
                if neighbor in board and neighbor not in connected:
                    connected.add(neighbor)
                    stack.append(neighbor)
                    
        return connected

    def is_game_over(self) -> Tuple[bool, int]:
        """Returns (is_over, winner). Winner is None for draw, -1 for not over"""
        for player in [0, 1]:
            queen_pos = self._find_queen(player)
            if queen_pos and self._is_surrounded(queen_pos):
                return True, 1 - player
        return False, -1

    def _find_queen(self, player: int) -> HexCoord:
        """Find position of player's queen bee"""
        for pos, stack in self.board.items():
            piece = stack[-1]
            if piece.player == player and piece.piece_type == PieceType.QUEEN_BEE:
                return pos
        return None

    def _is_surrounded(self, pos: HexCoord) -> bool:
        """Check if position is completely surrounded by pieces"""
        return all(neighbor in self.board for neighbor in pos.get_neighbors())

    def get_state(self) -> np.ndarray:
        """Convert current game state to neural network input format"""
        # Create a 91x91 board (large enough for any realistic game)
        # Each position has 12 channels:
        # 2 players * (5 piece types + 1 for stack height)
        state = np.zeros((91, 91, 12))
        
        center = 45, 45  # Center of the board
        
        for pos, stack in self.board.items():
            x = center[0] + pos.q
            y = center[1] + pos.r
            
            # Set piece type and stack height for top piece
            piece = stack[-1]
            channel = piece.player * 6 + piece.piece_type.value - 1
            state[x, y, channel] = 1
            
            # Set stack height
            height_channel = piece.player * 6 + 5
            state[x, y, height_channel] = len(stack)
            
        return state
    