import random
import time
from template import Agent
from agents.MyAgents.MCTS.deterministic_rule import DeterministicGameRule as DeterministicGameRule
from agents.MyAgents.qLearning.train_q import abstract_state, choose_action, Q

class MCTSNode:
    def __init__(self, state, game_rule, agent_id, parent=None, action=None):
        self.state = state
        self.game_rule = game_rule
        self.agent_id = agent_id
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        legal_actions = self.game_rule.getLegalActions(self.state, self.agent_id)
        return len(self.children) == len(legal_actions)

    def best_child(self, exploration_weight=1.0, bias_weight=0.3):
        return max(
            self.children,
            key=lambda child: child.value / (child.visits + 1e-6) +
                              exploration_weight * (2 * (self.visits + 1) ** 0.5 / (child.visits + 1e-6))
                              + bias_weight * self._heuristic_action_score(child.action)
        )

    def expand(self):
        legal_actions = self.game_rule.getLegalActions(self.state, self.agent_id)
        #print(f"[DEBUG] Legal actions before filtering: {legal_actions}")

        untried_actions = [
            action for action in legal_actions
            if action not in [child.action for child in self.children]
        ]
        #print(f"[DEBUG] Children actions: {[child.action for child in self.children]}")

        if not untried_actions:
            #print("[DEBUG] No untried actions available")
            return None
        #print("[DEBUG] Count of untried actions")
        #只选择得分前 K 的动作
        K = 5
        top_actions = sorted(untried_actions, key=self._heuristic_action_score, reverse=True)[:K]
        action = random.choice(top_actions)

        next_state = self.game_rule.generateSuccessor(self.state, action, self.agent_id)
        if next_state is None:
            #print(f"[DEBUG] generateSuccessor returned None for action: {action}")
            return None
        child_node = MCTSNode(next_state, self.game_rule, self.agent_id, parent=self, action=action)
        self.children.append(child_node)
        if child_node is None:
            #print(f"[DEBUG] Child node is None for action: {action}")
            return None
        return child_node

    def update(self, reward):
        self.visits += 1
        self.value += reward

    def _identify_opponent_threats(self, min_chain_len=3):
        board = self.state.board.chips
        opp_color = self.state.agents[self.agent_id].opp_colour
        threats = set()
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for r in range(10):
            for c in range(10):
                if board[r][c] != opp_color:
                    continue
                for dr, dc in directions:
                    chain = []
                    for i in range(-4, 5):
                        nr, nc = r + i * dr, c + i * dc
                        if 0 <= nr < 10 and 0 <= nc < 10:
                            val = board[nr][nc]
                            if val == opp_color or val is None:
                                chain.append((nr, nc))
                            else:
                                if len(chain) >= min_chain_len:
                                    empties = [p for p in chain if board[p[0]][p[1]] is None]
                                    threats.update(empties)
                                chain = []
                    if len(chain) >= min_chain_len:
                        empties = [p for p in chain if board[p[0]][p[1]] is None]
                        threats.update(empties)
        return threats

    def _directional_alignment(self, r, c, color):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        best_dir = None
        best_count = 0
        for dr, dc in directions:
            count = 1
            for d in [1, -1]:
                for i in range(1, 5):
                    nr, nc = r + d * i * dr, c + d * i * dc
                    if 0 <= nr < 10 and 0 <= nc < 10 and self.state.board.chips[nr][nc] == color:
                        count += 1
                    else:
                        break
            if count > best_count:
                best_count = count
                best_dir = (dr, dc)
        return best_count

    def _count_alignment(self, r, c, color):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        max_count = 0
        for dr, dc in directions:
            count = 1
            for d in [1, -1]:
                for i in range(1, 5):
                    nr, nc = r + d * i * dr, c + d * i * dc
                    if 0 <= nr < 10 and 0 <= nc < 10 and self.state.board.chips[nr][nc] == color:
                        count += 1
                    else:
                        break
            max_count = max(max_count, count)
        return max_count

    def _heuristic_action_score(self, action):
        if action['type'] != 'place':
            return 0

        r, c = action['coords']
        chips = self.state.board.chips
        plr_state = self.state.agents[self.agent_id]
        opp_color = plr_state.opp_colour

        #Temporarily place a chip to evaluate potential sequences
        orig_value = chips[r][c]
        chips[r][c] = plr_state.colour
        seq_result, seq_type = self.game_rule.checkSeq(chips, plr_state, (r, c))
        chips[r][c] = orig_value

        #If a sequence is formed, assign a high score
        if seq_result:
            base = 800
            if seq_type == 'HOTBSEQ':
                base = 1000  # Highest priority
            return base + 100 * (seq_result['num_seq'] - 1)

        score = 0
        #Prioritize controlling the center squares (2, 3, 4, and 5 of hearts)
        center_squares = [(2, 2), (2, 3), (3, 2), (3, 3)]
        if (r, c) in center_squares:
            score += 500  # High priority for center squares

        #Control of central positions (higher priority)
        if 3 <= r <= 6 and 3 <= c <= 6:
            score += 50

        #Bonus: Directional alignment consistency
        aligned = self._count_alignment(r, c, plr_state.colour)
        if aligned >= 3:
            score += 50 * (aligned - 1)  # Align 2 adds 50, align 3 adds 100...
        aligned = self._directional_alignment(r, c, plr_state.colour)
        score += aligned ** 3  # Strengthen alignment progression

        #Defend key positions (prevent opponent chains)
        threat_coords = self._identify_opponent_threats(min_chain_len=3)
        if (r, c) in threat_coords:
            score += 200

        #Bonus: Block opponent's long chain (length >= 3)
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count = 0
            for d in [1, -1]:
                for i in range(1, 5):
                    nr, nc = r + d * i * dr, c + d * i * dc
                    if 0 <= nr < 10 and 0 <= nc < 10:
                        if chips[nr][nc] == opp_color:
                            count += 1
                        elif chips[nr][nc] is None:
                            continue
                        else:
                            break
            if count >= 3:
                score += 50  # Strongly prioritize blocking 3+ opponent chain
                if count ==4:
                    score += 100

        #Check if it is a J card
        is_wild_jack = is_remove_jack = False
        if 'card' in action and isinstance(action['card'], str):
            card = action['card'].lower()
            if card in ['jd', 'jc']:
                is_wild_jack = True
            elif card in ['jh', 'js']:
                is_remove_jack = True

        #One-eyed Jack - Remove opponent's chip
        if action['type'] == 'remove' and is_remove_jack:
            r, c = action['coords']
            opp_threats = self._identify_opponent_threats(min_chain_len=3)
            if (r, c) in opp_threats:
                return 900
            elif self._max_sequence_potential(r, c, opp_color) >= 4:
                return 700
            else:
                return 200

        #Two-eyed Jack - Place in any empty position
        if is_wild_jack:
            if seq_result:
                score += 1200 if seq_type == 'HOTBSEQ' else 1000
            else:
                score += 100
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 10 and 0 <= nc < 10:
                        if chips[nr][nc] == plr_state.colour:
                            score += 30
                        elif chips[nr][nc] == opp_color:
                            score += 10
        #Penalize edge placement
        manhattan = abs(r - 4.5) + abs(c - 4.5)
        score -= int(manhattan * 5)  # The farther from the center, the lower the score

        return score

    def _max_sequence_potential(self, r, c, color):
        max_len = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1  # Current position
            for d in [1, -1]:
                for i in range(1, 5):
                    nr, nc = r + d * i * dr, c + d * i * dc
                    if 0 <= nr < 10 and 0 <= nc < 10:
                        if self.state.board.chips[nr][nc] == color:
                            count += 1
                        elif self.state.board.chips[nr][nc] is None:
                            continue
                        else:
                            break
            max_len = max(max_len, count)
        return max_len


class MCTS:
    def __init__(self, game_rule, time_limit=0.9):
        self.game_rule = game_rule
        self.time_limit = time_limit

    def search(self, initial_state, agent_id):
        root = MCTSNode(initial_state, self.game_rule, agent_id)
        start_time = time.time()
        count = 0
        while time.time() - start_time < self.time_limit:
            node = self._select(root)
            if node is None:
                #print("[Debug] Node is None, skipping iteration")
                continue
            reward = self._simulate(node.state, agent_id)
            self._backpropagate(node, reward)
            count += 1
        print("[Debug]Iterations:", count)

        exploration_weight = max(0.5, 1.0 - (time.time() - start_time) / self.time_limit)
        return root.best_child(exploration_weight=exploration_weight).action

    def _select(self, node):
        while not self._is_terminal(node.state) and node.is_fully_expanded():
            #print("[Debug] Node is fully expanded, selecting best child")
            node = node.best_child()
        if not self._is_terminal(node.state):
            #print("[Debug] Node is not terminal, expanding")
            return node.expand()
        return node

    def simple_policy_select(self, state, agent_id, game_rule):
        actions = game_rule.getLegalActions(state, agent_id)
        if random.random() < 0.6:
            return random.choice(actions)
        actions = game_rule.getLegalActions(state, agent_id)
        best_score = -float('inf')
        best_actions = []
        chips = state.board.chips
        plr = state.agents[agent_id]
        opp = plr.opp_colour

        for action in actions:
            if action['type'] != 'place':
                continue
            r, c = action['coords']

            #simulate
            orig = chips[r][c]
            chips[r][c] = plr.colour
            seq, seq_type = game_rule.checkSeq(chips, plr, (r, c))
            chips[r][c] = orig

            score = 0
            if seq:
                score += 1000 if seq_type == 'HOTBSEQ' else 800
            else:
                for dr, dc in [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 10 and 0 <= nc < 10 and chips[nr][nc] == plr.colour:
                        score += 30
                    elif 0 <= nr < 10 and 0 <= nc < 10 and chips[nr][nc] == opp:
                        score += 10

                #center reward
                if 3 <= r <= 6 and 3 <= c <= 6:
                    score += 20

            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return random.choice(best_actions) if best_actions else random.choice(actions)

    def _simulate(self, state, agent_id):
        current_state = state
        while not self._is_terminal(current_state):
            actions = self.game_rule.getLegalActions(current_state, agent_id)
            if not actions:
                return self._heuristic_state_score(current_state, agent_id)
            #Randomly select an action
            choice = random.random()
            if choice < 0.5:
                #heuristic action selection
                node = MCTSNode(current_state, self.game_rule, agent_id)  # 临时节点用于调用启发式函数
                scored_actions = sorted(
                    actions,
                    key=lambda a: node._heuristic_action_score(a),
                    reverse=True
                )
                #select top 3 actions
                k = min(3, len(scored_actions))
                action = random.choice(scored_actions[:k])
            else:
                action = random.choice(actions[:3])
            #else:
                #simple policy
                #action = self.simple_policy_select(current_state, agent_id, self.game_rule)

            current_state = self.game_rule.generateSuccessor(current_state, action, agent_id)
        return self._heuristic_state_score(current_state, agent_id)

    def _backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

    def _is_terminal(self, state):
        return self.game_rule.gameEnds()

    def _heuristic_state_score(self, state, agent_id):
        agent_state = state.agents[agent_id]
        score = agent_state.completed_seqs * 10
        for r, c in state.board.empty_coords:
            if (r, c) in state.board.plr_coords[agent_state.colour]:
                score += 1
            elif (r, c) in state.board.plr_coords[agent_state.opp_colour]:
                score -= 1
        return score

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.mcts = None

    def SelectAction(self, actions, game_state):
        # Q-learning first
        s_key = abstract_state(game_state, self.id)
        if s_key in Q and Q[s_key]:
            q_act = choose_action(s_key, actions, epsilon=0.0)
            if q_act is not None:
                return q_act

        if self.mcts is None:
            self.mcts = MCTS(DeterministicGameRule(len(game_state.agents)))
        if not hasattr(game_state.deck, 'cards'):
            game_state.deck.cards = game_state.deck.new_deck()
        best_action = self.mcts.search(game_state, self.id)
        return best_action