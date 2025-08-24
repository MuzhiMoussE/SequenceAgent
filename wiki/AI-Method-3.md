# AI Method 3 - Monte Carlo Tree Search

We adopted **Monte Carlo Tree Search (MCTS)** as the primary decision-making algorithm for this agent, augmented with a domain-specific heuristic to bias exploration and improve efficiency. The tree policy balances exploration and exploitation using the UCB1 formula. When expanding nodes in MCTS, actions with higher heuristic scores are prioritized, guiding MCTS to explore promising strategy regions more efficiently. 

- **Tree Policy (Selection and Expansion):**
    - We implemented the standard four-stage MCTS pipeline: selection, expansion, simulation, and backpropagation.
    - For **selection**, we use an extended UCB1 formula:
        
        $\frac{v}{n + \epsilon} + c \sqrt{\frac{2 \log N}{n + \epsilon}} + \beta \cdot \mathrm{heuristic}(a)$
        
        where v is the node’s value, n is its visit count, N is the parent’s total visits, and $\beta$ is a `bias_weight` parameter (default 0.3) guiding selection toward high-value heuristics.
        
    - **Expansion** is limited to the top-K scoring actions (K=5), using the heuristic score to prioritize promising children and reduce branching factor.
- **Simulation Policy (Rollout):**
    - We use a **biased simulation policy**. With 50% probability, the rollout chooses among the top-3 actions based on heuristic scores. Otherwise, it samples from the top few actions randomly, preserving diversity and avoiding overfitting to the heuristic.
- **Heuristic Score Function:**
    - The `_heuristic_action_score()` computes a rich evaluation of an action, incorporating:
        - Sequence formation potential (including HOTBSEQ priority),
        - Control of the central 2x2 squares and overall central area,
        - Alignment with existing chips and sequence directionality,
        - Opponent threat blocking based on chain detection,
        - Judicious use of Jacks (wild or one-eyed),
        - Penalty for edge placements based on Manhattan distance.
    - This function is applied during both node expansion and simulation.
- **Reward Function:**
    - We use a simple, interpretable terminal state reward:
        
        We define our reward function based on a heuristic evaluation of the final game state after simulation. The function returns:
        
        - A base reward of `10 * number of completed sequences`,
        - Plus a bonus/malus of `+1/-1` for each empty tile controlled by our agent or the opponent.
        
        This function approximates how “favorable” a final state is without requiring the game to end in a win/loss, thus stabilizing reward signals for backpropagation.
        
        $\text{reward} = 10 \times (\text{agent's completed sequences}) + \text{chip control differential}$
        
        where chip control differential measures advantage in ownership over empty tiles.
        
- **Opponent Modeling:**
    - Opponent influence is considered through:
        - Explicit detection of potential threats (chains of length ≥ 3),
        - Heuristic penalties for ignoring dangerous tiles,
        - High reward for using one-eyed Jacks to remove key opponent chips.
- **Iterations per Move & Depth Control:**
    
    To ensure our agent responds within the required time limit, we apply **strict time-based iteration control** rather than fixed simulation depth or iteration count.
    
    - As soon as the elapsed time exceeds 0.9 seconds, the search loop terminates immediately. This ensures that no matter the hardware or game state complexity, the agent never exceeds the per-move time limit.
    - The actual number of iterations varies per move depending on:
        - The complexity of the legal action space,
        - The branching factor after expansion,
        - The speed of successor generation and simulation rollout.
    - On average, the algorithm completes **78 iterations** per move on average within the time budget.

### Challenges and Solutions

- **Balancing Heuristic Bias vs. Exploration:**
    - Over-reliance on heuristics may reduce exploration diversity, leading to local optima.
    - To address this, we used a **`bias_weight` parameter** (default 0.3) and randomized action selection in rollouts to maintain stochasticity.
- **Fine-Tuning Heuristic Weights:**
    - Defining priorities—e.g., whether to block threats or promote sequence formation—proved complex due to diverse in-game situations.
    - We adopted **iterative tuning** by testing agent performance against various opponents and tweaking weights empirically.

### Strengths

- **Time-Bounded Search:**
    
    Ensures stable performance under strict per-move time limits using a dynamic time loop.
    
- **Heuristic-Guided Exploration:**
    
    Domain-specific heuristics guide expansion and rollout, improving efficiency and action quality.
    

### Limitations

- **Heuristic Hand-Crafted:**
    
    Scoring weights and logic are manually tuned and may not generalize to all game scenarios.
    
- **Simulation Quality Variability:**
    
    Rollout policy remains partly stochastic; action quality in simulations may vary.
    
- **May not generalize to all game scenarios**: 

    May overfit to heuristic; stochastic variance in rollouts

### Future Improvements

If more time were available, we would:
- Add Lookahead or Simulation:
    Introduce one-step or shallow rollout to evaluate not just the current move, but also possible opponent reactions.
- Learnable Heuristic Weights:
    Use supervised learning or Q-learning to automatically tune action scores based on game outcomes and self-play data.
- Board Feature Abstraction:
    Extract and generalize more robust board features (e.g., line threats, sequence chains) to improve evaluation consistency across various game stages.
- Integrate Opponent Behavior History:
    Track and learn patterns in the opponent’s past moves to better anticipate likely responses.
