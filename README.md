This project focuses on developing intelligent agents for the Sequence game. The core challenge lies in designing agents that operate under strict computational constraints while navigating the game's complexities: partial observability (hidden opponent hands), adversarial dynamics, and combinatorial action spaces.

Our approaches evolved through experimentation with two agents and four main AI methods:

- **Fast BFS** - Blind Search: Combines breadth-first search, and heuristic evaluation to decompose the game into tractable sub-problems.

- **Fast BFS** - Goal Recognition: Uses card counter to predict opponent threats and prioritize blocking.

- **Q-learning with MCTS** - Monte Carlo Tree Search (MCTS): Enhanced with domain-specific heuristic scores for biased rollouts.

- **Q-learning with MCTS** - Q-Learning: Trained offline for microsecond decision latency via state abstraction.

And we also introduced some deprecated methods that required some improvements. 

## Contents

### 1. [Home and Introduction](README.md)
### 2. [Problem Analysis](wiki/Problem-Analysis.md)

2.1. [Agent One - FastBFS](wiki/AI-Agent-1.md)

- 2.1.1. [Blind Search Algorithm](wiki/AI-Method-1.md)

- 2.1.2. [Goal Recognition Techniques](wiki/AI-Method-2.md)

- 2.1.3. [Heuristic Evaluation](wiki/Heuristic-Evaluation.md)

2.2. [Agent Two - Q-Learning with MCTS](wiki/AI-Agent-2.md)

- 2.2.1. [Monte-Carlo-Tree-Search](wiki/AI-Method-3.md)

- 2.2.2. [Q-Learning](wiki/AI-Method-4.md)

2.3. Other Agents

- 2.3.1. [Greedy Best First Search - Deprecated ](wiki/AI-Method-5.md)

- 2.3.2. [Goal-Recognition - Deprecated ](wiki/AI-Method-6.md)

### 3. [Evolution and Experiments](wiki/Evolution.md)
### 4. [Conclusions and Reflections](wiki/Conclusions-and-Reflections.md)

---


Agents were iteratively refined through 40+ experimental matches, culminating in FastBFS—the tournament-ready solution achieving 90% win rates (38/40 wins) and 6th place in class-wide competitions. This wiki documents our technical journey, from theoretical foundations to empirical breakthroughs.


Due to the university's policy, most of the project codes and our video presentation are not available to GitHub repositories. If you want to get access to it, please contact us via email(zhangxinmu_mousse@163.com).