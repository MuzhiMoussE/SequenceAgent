# AI Method 4 - Q-Learning

We adopted **tabular Q-Learning** as a complementary strategy to our MCTS agent. The agent can move quickly and purely greedily in familiar positions while playing by discretizing game states into a compact feature representation and learning state–action values offline.

- **State Abstraction:**

  - Extract a compact feature vector rather than the full 10×10 grid:

    - Completed sequences for both players (`completed_me`, `completed_op`).

    - Open-threat counts (any line of length ≥3 with enough empty slots) for both players (`threat_me`, `threat_op`).

    - Hand size (`hand_count`).

    - Center control counts (`center_me`, `center_op`).

    - Suit distribution in hand (counts of H, D, C, S).

    - Current draft snapshot (first 3 draft cards).

    - Score difference (`score_diff = my.score − opponent.score`).

  - This reduces millions of raw board states to a few thousand abstract keys.

- **Action Encoding:**

  - Represent each move as a tuple `(coords, play_card, draft_card)`.
  - For the same board location, playing different cards or drafting different replacements become distinct Q entries.

- **Offline Training Loop:**

  - **Limitation:** 

    Run for a long period offline (600 s training + 60 s replay), and set a large fixed‐step truncation (`MAX_STEPS = 100`) until terminal state.

  - **ε-Greedy:** 

    During offline training, choose a random legal move with probability ε (starting at 0.5 decayed to 0.01), otherwise pick the action with highest Q(s,a).

  - **Reward Design:**

    - +Δscore for points gained.
    - +0.5 for blocking opponent’s imminent win.
    - +0.2 for occupying central squares.

  - **Q-Update:**

    After each simulated step, update

    $Q(s,a) ← Q(s,a) + α[r+γmaxQ(s',a')-Q(s,a)]$

    with α=0.1, γ=0.9

- **Experience Replay:**

  - Store every tuple `(s,a,r,s')` in a list.
  - After the main loop, sample uniformly for a fixed time (60 s) to reinforce rare transitions and stabilize learning.

- **Compression & Loading:**

  - Run `train_q.py` for an extended period (600 s training + 60 s replay) to build a rich Q-table, then compress it via gzip to `q_table.pkl.gz` for fast loading before any game .
  - In our agent, only load this file and use ε=0 greedy lookups for runtime decisions.

### Challenges and Solutions

- **Time Constraint vs. Table Richness:**

  - The initial plan was to train and generate a Q-table in real-time during the 15 second warm-up time before the game started, but due to the short training time, the Q-table entries were sparse and could not cover the mid to late game situation.

  - Our solution is to move all training offline and perform deep exploration over a 10 minute budget to produce a comprehensive Q-table.

- **State–Action Explosion vs. Compression:**

  - Full-board serialization combined with only drop-point actions results in a vast state space and insufficient training coverage.
  - We abstract the state into a 15 dimensional feature vector, encode actions as (`coords, play_card`, `draft_card`) and apply gzip compression to keep the file size within 30–40 MB.

- **Runtime Latency:**

  - Online training plus fallback heuristics risked timeout.
  - We achieve nearly zero decision latency by limiting training and search to runtime table lookups.

### Strengths

- **Ultra-low decision latency:**

  Once the Q-table is loaded, each move is a single table lookup and argmax (ε=0 greedy), taking microseconds—far faster than any online search or simulation.

- **High coverage of familiar states:**

  Extended offline budgets and richer abstraction yield thousands of learned state–action pairs. The agent responds optimally in most mid-game and endgame scenarios.

- **Seamless integration:**

  Loading a single compressed file requires no change to the core decision loop; Q-Learning sits alongside rather than replacing existing MCTS, enabling hybrid fallback.

- **Predictable performance:**

  No runtime variability—memory lookups are consistent, eliminating the risk of unexpected slowdowns or timeouts under varying board conditions.

### Limitations

- **Memory Footprint:**

  Even compressed tables can reach tens of megabytes, which may strain limited environments.

- **Static Policy:**

  Once loaded, the agent does not learn further unless retrained offline—cannot adapt mid-tournament.

- **Feature Bias:**

  Quality heavily depends on chosen abstraction; poor features lead to suboptimal Q estimates.

### Evolution and Experiments

- **Parameter Sweeps:**

  Tested α∈{0.05,0.1,0.2}, γ∈{0.8,0.9,0.99}, ε-decay rates to maximize coverage vs. convergence speed.

- **Ablation Studies:**

  Compared full vs. minimal abstraction (7D vs. 15D) and observed a 10% win-rate drop when omitting draft or suit features.

- **Hybrid Policies:**

  Experimented with combining Q-lookup for early game and MCTS for endgame, yielding the best of both worlds.

### Conclusions and Reflections

Tabular Q-Learning provides a **lightweight**, **interpretable** policy that excels in familiar positions with negligible decision latency. Its offline training demands and reliance on hand-crafted features are offset by its ease of integration and complementarity with search-based methods. For future work, migrating to **function approximation** could overcome memory limits and enable continual online learning.