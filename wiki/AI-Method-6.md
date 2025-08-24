# AI Method 6 **Goal-Recognition Guided Greedy Search**



We adopted a **goal-recognition guided greedy search approach** in a deprecated agent that combined opponent modeling with greedy action selection, using goal recognition to identify opponent strategies and respond defensively.

- Search Strategy – Goal-Recognition Guided Greedy:
  - The agent uses a **minimum-step calculation** to identify the opponent's most likely target sequence based on current board state.
  - Actions are evaluated through a **two-phase scoring system**: defensive priority followed by offensive optimization.
  - No search tree is maintained — decisions are made greedily based on immediate threat assessment and heuristic evaluation.
- Goal Recognition Algorithm:
  - Threat Detection: Analyzes all possible winning sequences to find the opponent's closest completion target using `minStep()` function.
  - Priority Thresholds: If opponent can win in ≤2 steps, the system triggers **emergency defense mode**.
  - Multi-Sequence Analysis: Evaluates both regular sequences and special center control (`CENTER` squares) as potential targets.
- Heuristic Scoring Framework:
  - **Defensive Scoring:**
    - Blocking Bonus: +0.5 reward for placing chips on opponent's identified target sequence.
    - Threat Neutralization: Higher priority for `remove` actions targeting opponent clusters of 3+ chips.
  - **Offensive Scoring:**
    - Sequence Completion: Direct score increase based on `state.agents[agent_id].score` differential.
    - Center Control: +0.2 bonus for occupying strategic `CENTER` positions.
    - Alignment Building: Rewards based on `_count_alignment()` and `_directional_alignment()` functions.
- Action Selection Pipeline:
  1. Threat Detection: Analyzes all possible winning sequences to find the opponent's closest completion target using `minStep()` function.
  2. Defensive Filtering: If immediate threat detected, prioritize blocking actions over offensive moves.
  3. Heuristic Evaluation: Score all legal actions using combined defensive and offensive metrics.
  4. Greedy Selection: Choose action with highest composite score, with tie-breaking via randomization.

### Limitations

- Shallow Planning Horizon:
  - Without multi-step lookahead, the agent cannot anticipate opponent counter-moves or complex tactical sequences.
  - Goal recognition only identifies current threats, missing potential future setups.
- Oversimplified Opponent Modeling:
  - The `minStep()` function assumes opponents always pursue the closest completion target.
  - Cannot model deceptive strategies, feints, or multi-objective play.
- Reactive Rather Than Proactive:
  - Heavily biased toward defensive responses; may miss offensive opportunities.
  - Lacks strategic initiative in neutral game states without clear threats.

- Heuristic Brittleness:
  - Fixed weight system (+0.5 for blocking, +0.2 for center) may not adapt to different game phases.

### Deprecation Reasons

- Performance Ceiling:

  The purely reactive approach showed limited improvement against stronger opponents who employed multi-step strategies.

- Predictability:

  Deterministic goal recognition made the agent's responses predictable and exploitable.

- Integration Complexity:

  Difficult to balance goal recognition priorities with other strategic considerations in a unified framework.