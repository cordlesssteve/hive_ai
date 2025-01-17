import numpy as np
from enum import Enum
from typing import List, Tuple, Dict, Set, Optional


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

    def _determine_piece_type(self, to_pos: HexCoord) -> PieceType:
        """Determine which piece type to place based on game state and rules"""
        # If queen bee must be placed, return queen bee
        if self.turn >= 6 and not self.queen_bee_placed[self.current_player]:
            return PieceType.QUEEN_BEE
            
        # Get available pieces for the current player
        available_pieces = [
            piece_type
            for piece_type, count in self.player_hands[self.current_player].items()
            if count > 0
        ]
        
        if not available_pieces:
            raise ValueError("No pieces available to place")
            
        # For now, just return the first available piece type
        # This can be enhanced to implement more sophisticated piece selection
        return available_pieces[0]

    def get_valid_moves(self) -> List[Tuple[Optional[HexCoord], HexCoord]]:
        """Returns list of valid moves as (from_pos, to_pos) tuples. 
        from_pos is None for placement moves."""
        print(f"\nCalculating valid moves:")
        print(f"Current player: {self.current_player}")
        print(f"Turn: {self.turn}")
        print(f"Queen bee placed: {self.queen_bee_placed}")
        
        moves = []
        
        # If it's turn 4 and queen isn't placed, only allow queen placement
        if self.turn >= 6 and not self.queen_bee_placed[self.current_player]:
            print("Forcing queen bee placement")
            queen_moves = self._get_valid_placements(PieceType.QUEEN_BEE)
            return queen_moves if queen_moves else []
            
        # Get valid piece placements
        print("\nChecking piece placements:")
        for piece_type in PieceType:
            pieces_left = self.player_hands[self.current_player][piece_type]
            print(f"{piece_type}: {pieces_left} pieces left")
            if pieces_left > 0:
                placement_moves = self._get_valid_placements(piece_type)
                if placement_moves:
                    print(f"Found {len(placement_moves)} valid placements for {piece_type}")
                    moves.extend(placement_moves)
        
        # Get valid piece movements
        print("\nChecking piece movements:")
        for pos, stack in self.board.items():
            if stack[-1].player == self.current_player:
                print(f"Checking movements for piece at ({pos.q}, {pos.r})")
                movement_moves = self._get_valid_movements(pos)
                if movement_moves:
                    print(f"Found {len(movement_moves)} valid movements")
                    moves.extend(movement_moves)
        
        print(f"\nTotal valid moves found: {len(moves)}")
        return moves

    def _get_valid_placements(self, piece_type: PieceType) -> List[Tuple[None, HexCoord]]:
        """Get valid positions to place a new piece"""
        print(f"\nCalculating valid placements:")
        print(f"Current player: {self.current_player}")
        print(f"Board state: {len(self.board)} pieces")
        print(f"Piece type: {piece_type}")
        
        if not self.board:  # First move
            print("First move of the game - placing at origin")
            return [(None, HexCoord(0, 0))]
            
        valid_positions = set()
        has_friendly_pieces = False
        
        # Check if player has any pieces on board
        for pos, stack in self.board.items():
            if stack[-1].player == self.current_player:
                has_friendly_pieces = True
                print(f"Found friendly piece at ({pos.q}, {pos.r})")
                break
        
        print(f"Has friendly pieces: {has_friendly_pieces}")
        
        if not has_friendly_pieces:
            # First placement for this player - must be adjacent to enemy
            print("First placement for current player - checking adjacent to enemy")
            for pos, stack in self.board.items():
                if stack[-1].player != self.current_player:
                    print(f"Found enemy piece at ({pos.q}, {pos.r})")
                    for neighbor in pos.get_neighbors():
                        if neighbor not in self.board:
                            print(f"Adding valid position at ({neighbor.q}, {neighbor.r})")
                            valid_positions.add(neighbor)
        else:
            # Normal placement - must be adjacent to friendly piece
            print("Normal placement - checking adjacent to friendly")
            for pos, stack in self.board.items():
                if stack[-1].player == self.current_player:
                    print(f"Found friendly piece at ({pos.q}, {pos.r})")
                    for neighbor in pos.get_neighbors():
                        if neighbor not in self.board:
                            print(f"Adding valid position at ({neighbor.q}, {neighbor.r})")
                            valid_positions.add(neighbor)
            
            # Cannot be adjacent to enemy pieces (except for first placement)
            for pos, stack in self.board.items():
                if stack[-1].player != self.current_player:
                    print(f"Found enemy piece at ({pos.q}, {pos.r})")
                    for neighbor in pos.get_neighbors():
                        if neighbor in valid_positions:
                            print(f"Removing position at ({neighbor.q}, {neighbor.r})")
                            valid_positions.discard(neighbor)
        
        result = [(None, pos) for pos in valid_positions]
        print(f"Final valid placements: {len(result)}")
        return result

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
    
    def _get_queen_moves(self, pos: HexCoord) -> List[Tuple[HexCoord, HexCoord]]:
        """Get valid moves for Queen Bee: can move one space while maintaining connectivity"""
        moves = []
        for neighbor in pos.get_neighbors():
            if neighbor not in self.board and self._is_slideable(pos, neighbor):
                moves.append((pos, neighbor))
        return moves

    def _get_beetle_moves(self, pos: HexCoord) -> List[Tuple[HexCoord, HexCoord]]:
        """Get valid moves for Beetle: can move one space and climb on top of other pieces"""
        moves = []
        for neighbor in pos.get_neighbors():
            # Beetle can move to empty spaces or on top of other pieces
            if self._is_slideable(pos, neighbor) or neighbor in self.board:
                moves.append((pos, neighbor))
        return moves

    def _get_grasshopper_moves(self, pos: HexCoord) -> List[Tuple[HexCoord, HexCoord]]:
        """Get valid moves for Grasshopper: must jump in straight line over pieces"""
        moves = []
        directions = [
            (1, 0), (1, -1), (0, -1),
            (-1, 0), (-1, 1), (0, 1)
        ]
        
        for dq, dr in directions:
            current = HexCoord(pos.q + dq, pos.r + dr)
            # Must jump over at least one piece
            if current not in self.board:
                continue
            
            # Keep moving in same direction until empty space found
            while current in self.board:
                current = HexCoord(current.q + dq, current.r + dr)
            
            # Add landing spot if we jumped over at least one piece
            if current != HexCoord(pos.q + dq, pos.r + dr):
                moves.append((pos, current))
                
        return moves

    def _get_spider_moves(self, pos: HexCoord) -> List[Tuple[HexCoord, HexCoord]]:
        """Get valid moves for Spider: must move exactly three spaces around the hive"""
        moves = set()
        current_positions = {pos}
        
        # For each of three steps
        for step in range(3):
            next_positions = set()
            # For each position reachable in previous steps
            for current in current_positions:
                # Check all neighbors
                for neighbor in current.get_neighbors():
                    if (neighbor not in self.board and 
                        self._is_slideable(current, neighbor) and
                        neighbor != pos):  # Can't return to start
                        if step == 2:  # Only add final positions
                            moves.add((pos, neighbor))
                        else:
                            next_positions.add(neighbor)
            current_positions = next_positions
            
        return list(moves)

    def _get_ant_moves(self, pos: HexCoord) -> List[Tuple[HexCoord, HexCoord]]:
        """Get valid moves for Ant: can move to any position around the hive"""
        moves = set()
        visited = {pos}
        to_visit = {pos}
        
        while to_visit:
            current = to_visit.pop()
            for neighbor in current.get_neighbors():
                if (neighbor not in self.board and 
                    self._is_slideable(current, neighbor) and
                    neighbor not in visited):
                    visited.add(neighbor)
                    to_visit.add(neighbor)
                    if neighbor != pos:  # Don't include starting position
                        moves.add((pos, neighbor))
                        
        return list(moves)

    def _is_slideable(self, from_pos: HexCoord, to_pos: HexCoord) -> bool:
        """Check if a piece can slide from one position to another.
        Pieces can't slide if it would break connectivity or if the gap is too narrow."""
        # Get the two positions that would be touched while sliding
        dq = to_pos.q - from_pos.q
        dr = to_pos.r - from_pos.r
        
        touch_points = []
        if dq == 1 and dr == -1:  # Northeast
            touch_points = [HexCoord(from_pos.q + 1, from_pos.r), 
                          HexCoord(from_pos.q, from_pos.r - 1)]
        elif dq == 1 and dr == 0:  # East
            touch_points = [HexCoord(from_pos.q + 1, from_pos.r - 1),
                          HexCoord(from_pos.q + 1, from_pos.r + 1)]
        elif dq == 0 and dr == 1:  # Southeast
            touch_points = [HexCoord(from_pos.q + 1, from_pos.r),
                          HexCoord(from_pos.q - 1, from_pos.r + 1)]
        elif dq == -1 and dr == 1:  # Southwest
            touch_points = [HexCoord(from_pos.q, from_pos.r + 1),
                          HexCoord(from_pos.q - 1, from_pos.r)]
        elif dq == -1 and dr == 0:  # West
            touch_points = [HexCoord(from_pos.q - 1, from_pos.r + 1),
                          HexCoord(from_pos.q - 1, from_pos.r - 1)]
        elif dq == 0 and dr == -1:  # Northwest
            touch_points = [HexCoord(from_pos.q + 1, from_pos.r - 1),
                          HexCoord(from_pos.q - 1, from_pos.r)]
            
        # If both touch points are occupied, can't slide through
        return not all(pos in self.board for pos in touch_points)

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
    
    def clone(self) -> 'HiveGame':
        """Create a deep copy of the game state"""
        new_game = HiveGame()
        
        # Deep copy the board state
        new_game.board = {
            pos: [Piece(p.piece_type, p.player) for p in stack]
            for pos, stack in self.board.items()
        }
        
        # Copy player hands
        new_game.player_hands = [hand.copy() for hand in self.player_hands]
        
        # Copy simple attributes
        new_game.current_player = self.current_player
        new_game.turn = self.turn
        new_game.queen_bee_placed = self.queen_bee_placed.copy()
        
        return new_game
    
    def get_state(self) -> np.ndarray:
        """Convert current game state to neural network input format.
        Returns a tensor of shape (channels, height, width) = (12, 91, 91)
        for PyTorch's Conv2d layers."""
        state = np.zeros((12, 91, 91))  # (channels, height, width)
        center = 45, 45  # Center of the board
        
        for pos, stack in self.board.items():
            x = center[0] + pos.q
            y = center[1] + pos.r
            
            piece = stack[-1]
            # Calculate channel index: 6 channels per player (5 piece types + stack height)
            channel = piece.player * 6 + piece.piece_type.value - 1
            state[channel, y, x] = 1  # Note: channel comes first
            
            # Set stack height in the last channel for each player
            height_channel = piece.player * 6 + 5
            state[height_channel, y, x] = len(stack)
        return state