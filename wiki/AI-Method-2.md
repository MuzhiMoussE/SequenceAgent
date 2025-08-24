# AI Method 2 - Goal recognition Techniques


# Table of Contents
  * [Motivation](#motivation)
  * [Solved challenges](#solved-challenges)
  * [Detail ](#detail)
  * [Trade-offs](#trade-offs)
     - [Advantages](#advantages)
     - [Disadvantages](#disadvantages)
  * [Future improvements](#future-improvements)
 
### Motivation
In order to better predict the opponent's behavior, I implemented a card counter. With the help of card counter, the agent determine whether it can block the opponent's potential reward sequence.

[Back to top](#table-of-contents)

### Solved Challenges
Only taking actions that are beneficial to forming self's sequence is not enough for this game as the opponent may form the sequence faster and eventually win. So, it is important to stop the opponent when the agent can do it. 

[Back to top](#table-of-contents)


### Detail

#### Obtain the information of the opponent's hand cards -- Card counter

It can achieve incremental synchronization. At the beginning of each round, increment the record of the opponent's known hand cards and undisclosed cards based on the current Game state, to simulate the information that cannot be directly obtained from the Game state, that is, opponent's hand.
The logic for initializing the card_counter is:
1. Traverse the opponent's action trace and obtain the opponent's known hand cards by adding draft_card and removing play_card
2. Using the equation 
   > potential_cards = all cards - all hands - all draft - all discards
   - Derive the unknown cards on the left using the known information on the right side
3. Update logic of the card counter:
The logic for updating the opponent's known hand cards is the same as the initialization
4. Update potential_cards by:
   >  known potential cards during this round = enemy played card from unknown hand + all drawn cards + all current draft - previous draft 
   >  updated potential_cards = previous potential_cards - known potential cards during this round

** Note, please see [Future improvements](#future-improvements) for detail about the card counter.

#### Determine whether it can block the opponent's potential scoring sequence
I defined the opponent case as any sequence in which the opponent has at least 3 tokens, and the sequence is unblocked by my own pieces. This typically indicates that the opponent is only 1–2 moves away from winning, making it a high-priority threat.

Defensive Module Logic:
The module begins by analyzing the current game state to identify all potentially threatening sequences. Each sequence is split into two categories:

Occupied cells (with opponent tokens)

Empty cells (available for blocking)

For occupied cells, the blocking strategy is to use one-eyed Jacks to remove opponent tokens.

For empty cells, the agent attempts to place its own tokens to directly interrupt the sequence.

To maximize defensive impact, the module prioritizes cells that are part of multiple threatening sequences, and also takes into account any cases where the opponent's scoring is clearly foreseeable (i.e., "Opponent's reward is predictable" conditions).


[Back to top](#table-of-contents)

### Trade-offs  
The tread-off is between stopping opponent and forming own sequence continuously. The strategy is listed in [Design Motivation](AI-Agent-1.md#design-motivation)

#### *Advantages*  
Stop the opponent intention to form a line and prevent it from winning.

#### *Disadvantages*
Always stop the opponent possibly means the agent needs to stop taking actions beneficial to forming own line.

[Back to top](#table-of-contents)

### Future improvements
The potential_cards is designed for use in MCTS to simulate the probabilistic cards which can be got later, and this part of logic is not used in this agent. So, in the future, the card counter can be used in other agents and provide additional information for these agents.

[Back to top](#table-of-contents)

