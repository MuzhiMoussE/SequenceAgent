# AI Method 1 - Blind Search Algorithm: BFS Technique

# Table of Contents
  * [Motivation](#motivation)
  * [Solved challenges](#solved-challenges)
  * [Detail](#detail)
  * [Trade-offs](#trade-offs)     
     - [Advantages](#advantages)
     - [Disadvantages](#disadvantages)
  * [Future improvements](#future-improvements)
 
### Motivation  

Finding the optimal solution for the current game state is an NP problem, but determining whether a scoring opportunity exists based on known information is a P problem. Therefore, I chose to treat this specific subproblem—detecting scoring opportunities—with a specialized method to strengthen the agent's decision-making.

This is motivated by a common issue in tree search methods: important states are often missed due to limited search depth or breadth.

[Back to top](#table-of-contents)

### Solved Challenges

By limiting the circumstances that BFS is triggered and simplifying the search space, fastBFS makes efficient use of the one-second decision time. If no reward opportunity is found, the agent can quickly shift to alternative strategies without wasting time on unproductive BFS search.

[Back to top](#table-of-contents)

### Detail  

#### Determine whether it is possible to get reward based on the currently known information
To detect scoring possibilities, I consider all cards in the agent's hand plus one card from the draft area (i.e., any one that the agent could definitely take). These cards are treated as the agent's available resources. Then simulate all sequences of playing these cards, representing each as a decision tree, and check every path to a leaf node to find the first path that results in a completed sequence.

The reason for only including one draft card is that the agent can guarantee pick only one draft card per turn—others may be taken by the opponent in the next round.

The decision to stop at the first successful path is a simplification intended to reduce computational cost and limit the complexity of processing search results.

Lastly, to handle the combinatorial explosion caused by two-eyed Jacks (wildcards), I avoid full expansion of all possible plays. Instead, I calculate the minimum number of tokens needed on the board to complete a sequence, and compare that with the number of two-eyed Jacks in hand to determine whether scoring is feasible.

#### Find the shortest path to complete the sequence through BFS

Although the classic BFS algorithm is capable of finding a valid solution, I implemented strict limitations on the BFS search space to ensure it operates efficiently within limited computational resources. These constraints guarantee that if a "scoring opportunity is 100 percent sure", the agent will be able to find a valid solution within the allowed time.

Since BFS is only triggered under the assumption that a scoring path is already detectable, I simplify the task of "finding the shortest path to a completed sequence" into "finding the shortest path to complete a specific sequence". This specific target sequence is obtained from the preliminary step, where the agent identifies sequences that are close to completion during the "scoring opportunity check".

To handle a special case—where the agent's hand contains no useful cards, and the critical card is only available in the draft pool—I added a safeguard. When such a situation is detected, the agent automatically falls back to a score-based action selection strategy instead of attempting an nouse BFS. This ensures robustness even when immediate scoring is impossible due to temporary resource constraints.

[Back to top](#table-of-contents)

### Trade-offs  
#### *Advantages*  
Save search time when there is no goal state.

#### *Disadvantages*
When there are multiple potential way to win, as we use only one card from draft, it is possible that with the draft we choose, 2 steps are needed to win. And with another draft card, only 1 step is needed to win.

[Back to top](#table-of-contents)

### Future improvements  
BFS is a blind search algorithm and the agent tries to minimize effort needed. MIN-MAX may be a suitable alternative method to deal with similar logic.

[Back to top](#table-of-contents)
