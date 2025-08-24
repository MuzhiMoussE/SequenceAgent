# Experiments and Analysis

## Experiments


<details>
  <summary>Experiment 1 </summary>
  
  - Experiment Aim: Test the performance and count of iterations of MCTS codes with the agents to find an optimized approach to cut-off the branches.
  
  - Experiment Approach: 
  
  1. Push the codes of MCTS with tag `test-submission` and analyze the result and replay files.

  2. Add debug information for the iteration in the simulation process in local plays.

  ### Result

  1. The opponent quickly occupied the center positions.


  2. At the same time, we find that the count of iterations in the simulation remains 1 in the output log, which is wrong:


  We noticed that the simulation remained with the depth of 1, causing the whole MCTS acted more likely to a greedy heuristic search in the final best-child function.

</details>

<details>
  <summary>Experiment 2</summary>

  - Experiment Aim: Test the fixed code against wrong simulation depth and performance of `Top-K` scoring function.
  
  - Experiment Approach: 

  1. Update the new codes that changed the logic of simulation and fixed the errors in state-updating, and test.

  2. Remain the debug information output like experiment 1, and check the correctness of simulation and evaluate the performance of `Top-K` scoring. 

  ### Result

  The performance of the MCTS agent is better.

  The count of iterations is recorded as:   

  This means the process of simulation is correct. And the average count of iterations was approximately 78.
</details>
<details>
  <summary>Experiment 3</summary>

  - Experiment Aim: Test the hybrid agent combining Q-Learning with MCTS to evaluate the performance improvement from integrating fast Q-table lookups with strategic tree search.

  - Experiment Approach: 

  1. Integrate the pre-trained Q-table (`q_table.pkl.gz`) with the MCTS agent to create a hybrid decision-making system.

  2. Implement priority-based action selection: Q-Learning for familiar states (when `s_key in Q` and `Q[s_key]` exists), MCTS fallback for novel positions.

  3. Maintain the same debugging output and iteration counting from previous experiments to monitor both Q-table hits and MCTS search performance.

  4. Test the hybrid agent against the same baseline opponents to measure performance gains from the combined approach.

  ### Result

The performance of the hybrid Q-Learning + MCTS agent shows significant improvement compared to pure MCTS approach.


</details>

<details>
  <summary>Experiment 4</summary>

  Implement the Fast-BFS method in the agent. Most of logic in this-version Fast-BFS is the same as the final one. 
  - Experiment Aim: Test the performance against the teching-team agent, and observe shortcomes.
  
  - Experiment Approach: 
  
  1. Push the codes of Fast-BFS with tag `test-submission` and analyze the result and replay files.

  2. Summarize lose cases and found out the most important bug or improvement needed.


  ### Result

</details>

<details>
  <summary>Experiment 5</summary>

  The detection logic is improved. A helper - card counter is introduced.

  ### Result


</details>


---
## Analysis

<details>
  <summary>My First Agent - MCTS with heuristic</summary>

  The heuristic function is unable to effectively counter the opponent's threat, such as the central four-square rule. In Sequence, random rollouts often miss strategic opportunities (e.g., sequence alignment, special card usage).
  
  In the first experiment we find that the process of simulation went wrong, which means the algorithm actually acted as a heuristic-based greedy search with UCB1 to find a best child, without correct simulation and backpropagation in MCTS. After fixing the errors, the correct MCTS got better score, still performed poor with the opponent's threats.


  ### Demo

  #### Group results: Game - 18/0/22 
  #### Tournament results: Position - We did't submit the tag of tourname on this version.

  #### Strategy summary

  | Pros | Cons |
  |-----------------|:-------------|
  | Balance Heuristic Bias vs. Exploration | The consideration of opponent's threat is not enough. |
  | Fine-Tuning Heuristic Weights    | May not generalize to all game scenarios.       |
</details>
<details>
  <summary>My Optimized First Agent - MCTS & Q-Learning </summary>

  The hybrid Q-Learning and MCTS agent addresses critical weaknesses in pure MCTS, particularly opponent threat recognition and defensive positioning. The Q-Learning component uses `recognize_opponent_target()` to identify threats (≤2 steps to win) and provides +0.5 reward bonuses for blocking actions, compensating for MCTS rollouts that miss defensive opportunities.

  ### Demo


  Our experiments revealed that the original MCTS suffered from simulation errors that reduced it to greedy heuristic search. After implementing the hybrid system with 600 seconds of offline Q-Learning training, the agent shows substantial improvement in threat recognition and strategic positioning while maintaining exploration capabilities.

  #### Group results: Game - 24/0/16 
  #### Tournament results: Position - We did't submit the tag of tourname on this version.



  #### Strategy summary

  | Pros | Cons |
  |-----------------|:-------------|
  | Enhanced Threat Detection | Training Data Dependency  |
  | Rapid Decision Making     | Static Learning Limitation  |
  | Complementary AI Methods     | Memory Overhead  |

</details>

<details>
  <summary>My Second Agent - Fast BFS</summary>

  BFS is limited to a small state space and is only run when necessary. At most of times, defence action (goal detection) and evaluation determined action are taken.

  ### Demo

  
  When the agent can not gain reward, the agent does not waste time in impossible reach goal search. So it can in spend time on calculate evaluation and take reasonable action.


  #### Group results: Game - 31/0/9 
  #### Tournament results: Position - We did't submit the tag of tourname on this version.



  #### Strategy summary

  | Pros | Cons |
  |-----------------|:-------------|
  | Take different strategy based on the situation | Logic is complex  |
  | Stop the opponent early     | Opponent may not able to win when we take defense action, as we do not know opponent's hand          |

</details>

<details>
  <summary>My Optimized Second Agent - Fast BFS</summary>

  A card counter is introduced to record opponent's hand, so the prediction of opponent's intention results in high sccuracy. When there is a card that is important to opponent in the draft area, the agent will always take it unless the next situation happens. Change priority to ensure two-eye Jack is assigned with a large number in evaluation, so it is always taken.

  ### Demo



  Defence is even stronger and maintain the advantage of last version.

  #### Group results: Game - 38/0/2 
  #### Tournament results: Position - 6th



  #### Strategy summary

  | Pros | Cons |
  |-----------------|:-------------|
  | Take different strategy based on the situation | Logic is complex  |
  | Stop the opponent early and precisely    | Further technique like MIN-MAX can be introduced   |

</details>
