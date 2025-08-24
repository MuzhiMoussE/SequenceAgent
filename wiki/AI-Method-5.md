# AI Method 5 **Greedy Best-first Search**

We adopted a **greedy best-first search approach** in a deprecated agent, relying on handcrafted heuristics to evaluate and select the most promising action each turn. 

- **Search Strategy – Greedy Best-First Search:**
    - The agent evaluates each action independently using a heuristic score.
    - No search tree or history is maintained — it always picks the action with the **best immediate heuristic value**.
    - This enables fast decision-making with minimal overhead.
- **Heuristic Scoring:**
    - Actions are scored based on multiple game-specific features:
        - **Sequence Formation**: Simulates placing a chip to check if it completes a sequence (`HOTBSEQ` gets priority).
        - **Central Control**: Adds bonus for targeting center tiles (rows and cols 3–6).
        - **Chip Alignment**: Rewards placing near friendly chips; minor bonus for placing near opponents.
        - **Opponent Threat Detection**: For `remove` actions, detects clusters of 3+ opponent chips and prioritizes breaking them using one-eyed Jacks.
        - **Special Cards**: Adds fixed bonuses for using two-eyed or one-eyed Jacks.
- **Action Type Weighting:**
    - Base scores are assigned to each action type (`place`, `remove`, `trade`) to reflect their strategic value.

### Limitations

- **No Forward Planning:**
    - Without simulation or tree search, the agent may overlook deeper strategies or traps.
    - To compensate, we designed rich heuristics to capture sequence potential and positional strategy as much as possible.
- **Heuristic Balance Tuning:**
    - It was difficult to manually balance the weights for different conditions (e.g., forming vs. blocking sequences).
    - We addressed this through repeated tuning and empirical testing, observing performance and adjusting scores accordingly.
- **Simplified Opponent Modeling:**
    - Detecting opponent threats without simulations relies on **pattern-based local checks**, which may miss advanced strategies.
    - We implemented a rule to detect clusters of 3+ opponent chips as likely threat zones.