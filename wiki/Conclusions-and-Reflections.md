## Conclusions and Learnings


Through this project, we gained deep hands-on experience in designing competitive game-playing agents using different AI methods. Below are the key takeaways from the development and experimentation process from different methods we implemented:

### 1. BFS:
- Simple method is also helpful: Simple method can have excellent performance when you utilize it in correct circumstances.

- Prune: Limit state space is important in game design, especially when time is limited.

### 2. Goal Recognition:

- Learn from opponent: Close observation of opponent's movement and analyze it, smartness of the opponent helps with changing own logic.

- Infer opponent's state: Use helper function such as Card Counter to infer opponent's intention precisely.  

### 3. MCTS:

- Heuristic Design: Designing a good heuristic was not easy. We had to balance offense and defense, adjust weights through testing, and ensure the agent didn’t overfit to shallow patterns.

- Opponent Modeling: Even without full observability, We could detect opponent threats by analyzing chip patterns, which made my agent more defensive and smarter.

### 4. Q-Learning:

- State Abstraction is Critical : Designing an effective 15-dimensional feature vector was essential for capturing key game elements while remaining generalizable across different board states.

- Reward Shaping Drives Behavior  : Careful reward design (+0.5 for blocking, +0.2 for center control) was crucial for teaching defensive and offensive strategies during offline training.

- Key Insight  : Q-Learning excels at pattern recognition and rapid response to familiar situations, but works best when combined with other approaches that can handle novel scenarios, highlighting the value of hybrid architectures.

## Comparison of the agents

|Agents|Pros|Cons|Performance|Oppenent Blocking|Sequence Formation|
|---|---|---|---|---|---|
|Heuristic-based MCTS|Bias exploration and improve efficiency| May not generalize to all game scenarios. |18/0/22|Poor|Fair|
|Q-Learning with MCTS|Enhanced threat detection and rapid decision making|Training data dependency and static learning limitation|24/0/16|Good|Good|
|Fast BFS|Adopt different actions based on state|Logic is complex and hard to understand|38/0/2|High|Great|


