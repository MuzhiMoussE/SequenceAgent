# Analysis of the problem

### 1. Observable

All players can see the entire game state, including:
- The board state (chip placements)
- The positions of completed sequences
- The public discard/trade pile

Partial observability exists only in hidden cards (private hand):
- Each agent can only observe its own hand, not the opponent's.

### 2. Deterministic and Stochastic

If the game state is considering the board state:
Given a state, an action, and a rule, the next state is uniquely determined (no dice, no chance). No random transition in environment or outcome.

If the game state is considering the borad state and draft area, etc.,:
Given a state, an sction, and a rule, the next state is Stochastic as the draft area is randomly generated.


### 3. State Space Complexity

The game state combines a 10×10 board, each player’s 5-card hand, and the 5-card draft, yielding around 10²⁵ possible configurations. In practice, however, the board’s fixed geometry (corners, center), repeating line and block patterns, and a limited card set (ranks, suits, Jokers) mean that only a small fraction of those states matter for decision-making.

### 4. Time Constraint

The agent has only 1 second to determine it's move, so limiting the time complexity is needed.

### 5. Zero Sum

The agent and opponent can demonstrate the following ending: agent win, opponent lose; agent lose, opponent win; tie; So this is a zero-sum game.
