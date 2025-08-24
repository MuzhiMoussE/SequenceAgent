import heapq
import math
import os
import pickle
import time, random
from itertools import count

# Ignore card draw entirely vs keeping track of which cards are probably left in the deck
# from Sequence.sequence_model import SequenceGameRule as GameRule
from agents.MyAgents.MCTS.deterministic_rule import DeterministicGameRule as GameRule

from Sequence.sequence_utils import EMPTY
from copy import deepcopy

from agents.MyAgents.qLearning.train_q import train_q
from template import Agent
from collections import defaultdict

THINKTIME   = 0.9
NUM_PLAYERS = 2
THRESH_LOW  = -3.0
THRESH_HIGH =  3.0
# the center of the board (2, 3, 4, and 5 of hearts)
CENTER = [(4,4), (4,5), (5,4), (5,5)]

# List all winning sequences
WINNING_SEQUENCES = []
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
for row in range(10):
    for column in range(10):
        for x,y in DIRECTIONS:
            seq = [(row + x*n, column + y*n) for n in range(5)]
            # Filter out-of-bound sequences
            if all(0 <= i < 10 and 0 <= j < 10 for i,j in seq):
                WINNING_SEQUENCES.append(seq)

THREAT_WEIGHTS = {
    'open_four': 6.0,
    'open_three': 2.5,
}
MAX_DEPTH = 5


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.game_rule = GameRule(NUM_PLAYERS)
        self.turn_count = 0

        # killer moves & history 用于更优走子排序
        self.killer_moves = defaultdict(list)
        self.history_table = {}

        # auto–train / load Q-table
        qpath = os.path.join(os.path.dirname(__file__), 'qtable.pkl')
        if not os.path.isfile(qpath):
            # execute in startup period
            train_q()
        # load the trained table
        try:
            with open(qpath, 'rb') as f:
                self.Q = pickle.load(f)
        except FileNotFoundError:
            self.Q = {}

    # Generates actions from this state.
    def GetActions(self, state):
        return self.game_rule.getLegalActions(state, self.id)

    # Carry out a given action on this state and return True if goal is reached received.
    def DoAction(self, state, action):
        score = state.agents[self.id].score
        next_state = self.game_rule.generateSuccessor(state, action, self.id)

        # Check whether it reached goal or not
        goal_reached = False
        if next_state.agents[self.id].score > score:
            goal_reached = True

        return next_state, goal_reached


    # Return the first action that leads to goal, if any was found.
    def SelectAction(self, actions, rootstate):
        start_time = time.time()
        # Q-Learning
        s_key = self.serialize(rootstate)
        if s_key in self.Q:
            a_key, _ = max(self.Q[s_key].items(), key=lambda kv: kv[1])
            for a in actions:
                if self.serialize_action(a) == a_key:
                    return a

        # Block
        target = self.recognize_opponent_target(rootstate)
        if target:
            tset = set(target)
            for action in actions:
                if action.get('type') == 'place' and action.get('coords') in tset \
                        and rootstate.board.chips[action['coords'][0]][action['coords'][1]] == EMPTY:
                    return action

        # Candidates
        cands = [a for a in actions
                 if a.get('type') in ('place', 'remove', 'trade')]

        # Use heuristic for `place` actions; otherwise, assign a fixed priority
        def cand_score(state, action):
            # Q
            s_key = self.serialize(state)
            ak = self.serialize_action(action)
            qv = self.Q.get(s_key, {}).get(ak, 0.0)
            # Heuristic
            succ = self.game_rule.generateSuccessor(state, action, self.id)
            hv = self.heuristic(succ)
            return -1.0 * qv + 1.0 * hv

        # Heuristic
        # Dynamic K + depth
        chips = rootstate.board.chips
        all_seqs = WINNING_SEQUENCES + [CENTER]
        op = rootstate.agents[1 - self.id]
        opp_cols = (op.colour, op.seq_colour)
        op_missing, _ = self.minStep(chips, all_seqs, opp_cols)
        me = rootstate.agents[self.id]
        my_cols = (me.colour, me.seq_colour)
        my_missing, _ = self.minStep(chips, all_seqs, my_cols)

        # Decider
        if op_missing <= 1 or my_missing <= 1:
            K = 6
            max_depth = 5
        else:
            K = 12
            max_depth = 4

        candidates = sorted(cands, key=lambda a: cand_score(rootstate, a))[:K]

        # Cache
        cache = {} # (state_key, depth, maximizing) -> (value, action)
        succ_cache = {} # (state_key, action_key) -> next_state
        # Move ordering helpers
        killer_moves = {d: [] for d in range(max_depth+1)}
        history_table = {}

        def order_moves(mvs, depth, state):
            # instant win/lose first
            inst = [a for a in mvs if self.DoAction(state, a)[1]]
            rest = [a for a in mvs if a not in inst]
            # killer
            km = killer_moves.get(depth, [])
            # sort rest by history then heuristic
            rest.sort(key=lambda a: (
                -int(a in km),
                -history_table.get((depth, self.serialize_action(a)), 0),
                cand_score(state, a)
            ))
            return inst + rest

        def get_moves(st, pid):
            try:
                return self.game_rule.getLegalActions(st, pid)
            except AttributeError:
                return []

        def get_succ(state, action, pid):
            sk = self.serialize(state)
            ak = self.serialize_action(action)
            key = (sk, ak, pid)
            if key not in succ_cache:
                succ_cache[key] = self.game_rule.generateSuccessor(state, action, pid)
            return succ_cache[key]

        # Minimax-αβ
        node_count = 0
        MAX_NODES = 500
        def minimax_ab(state, depth, alpha, beta, moves, maximizing):
            nonlocal start_time
            if time.time() - start_time > THINKTIME:
                return self.evaluate(state), None

            nonlocal node_count
            if node_count > MAX_NODES:
                return self.evaluate(state)
            node_count += 1

            # Null‐move pruning
            if depth > 1 and not (op_missing <= 1 or my_missing <= 1):
                R = 2 if depth > 4 else 1
                r, _ = minimax_ab(state, depth - 1 - R, -beta, -beta + 1,
                                  get_moves(state, 1 - self.id), False)
                if r >= beta:
                    return beta, None

            sk = self.serialize(state)
            key = (sk, depth, maximizing)
            if key in cache:
                return cache[key]

            if depth == 0:
                val = self.quiesce(state,alpha,beta)
                cache[key] = (val, None)
                return val, None

            best_act = None
            if maximizing:
                best_val = -math.inf
                pid = self.id
                for a in order_moves(moves, depth, state):
                    # Simulate my actions
                    next_state = get_succ(deepcopy(state), a, pid)
                    # I win, highest score
                    if next_state.agents[pid].score > state.agents[pid].score:
                        cache[key] = (math.inf, a)
                        return math.inf, a
                    # opponent's next action
                    child = get_moves(next_state, 1 - pid)
                    val, _ = minimax_ab(next_state, depth - 1, alpha, beta, child, False)
                    if val > best_val:
                        best_val, best_act = val, a
                    if best_val < THRESH_LOW:
                        break
                    alpha = max(alpha, best_val)
                    # record killer/history
                    if alpha >= beta:
                        if depth not in killer_moves:
                            killer_moves[depth] = []
                        killer_moves[depth].append(a)
                        history_table[(depth, self.serialize_action(a))] = \
                            history_table.get((depth, self.serialize_action(a)), 0) + 1
                        break

                cache[key] = (best_val, best_act)
                return best_val, best_act

            else:
                best_val = math.inf
                pid = 1 - self.id
                for a in order_moves(moves, depth, state):
                    # Simulate opponent actions
                    next_state = get_succ(deepcopy(state), a, pid)
                    # Opponent win, lowest score
                    if next_state.agents[pid].score > state.agents[pid].score:
                        cache[key] = (-math.inf, a)
                        return -math.inf, a
                    # My next action
                    child = get_moves(next_state, self.id)
                    val, _ = minimax_ab(next_state, depth - 1, alpha, beta, child, True)
                    if val < best_val:
                        best_val, best_act = val, a
                    if best_val > THRESH_HIGH:
                        break
                    beta = min(beta, best_val)
                    if beta <= alpha:
                        if depth not in killer_moves:
                            killer_moves[depth] = []
                        killer_moves[depth].append(a)
                        history_table[(depth, self.serialize_action(a))] = \
                            history_table.get((depth, self.serialize_action(a)), 0) + 1
                        break
                cache[key] = (best_val, best_act)
                return best_val, best_act
        # Iterative Deepening
        best_move = None
        prev_val = 0.0
        for depth in range(1, MAX_DEPTH + 1):
            if time.time() - start_time > THINKTIME - 0.08:
                break
            if depth > 1:
                low, high = prev_val - 1.0, prev_val + 1.0
                res = minimax_ab(rootstate, depth, low, high, candidates, True)
                if res is None or res[0] <= low or res[0] >= high:
                    res = minimax_ab(rootstate, depth, -math.inf, math.inf, candidates, True)
            else:
                res = minimax_ab(rootstate, depth, -math.inf, math.inf, candidates, True)
            if res is None:
                continue
            prev_val, move = res
            if move:
                best_move = move
        if best_move:
            return best_move

        # Greedy fallback
        return max(actions, key=lambda a: self.evaluate(self.game_rule.generateSuccessor(rootstate, a, self.id)))

    # Record the accessed status to prevent duplicate expansion
    def serialize(self, state):
        flat_board = tuple(sum(state.board.chips, []))
        flat_hand = tuple(state.agents[self.id].hand)
        flat_draft = tuple(state.board.draft)
        return flat_board, flat_hand, flat_draft

    # Serialize action
    def serialize_action(self, action):
        coords = tuple(action.get('coords') or ())
        card = action.get('play_card')
        draft = action.get('draft_card')
        return coords, card, draft

    # Attack: 5-line + Center
    def heuristic(self, state):
        chips = state.board.chips
        me = state.agents[self.id]
        my_colours = (me.colour, me.seq_colour)

        my_seq_step ,_ = self.minStep(chips, WINNING_SEQUENCES, my_colours)
        my_ctr_step ,_ = self.minStep(chips, [CENTER], my_colours)

        # Weight
        attack_seq = 1.0
        attack_ctr = 2.0

        return attack_seq * my_seq_step + attack_ctr * my_ctr_step

    # Calculate the minimum number of steps required for any sequence
    def minStep(self, chips, seqs, colours):
        min = float('inf')
        best_seq = None
        for seq in seqs:
            count = 0
            # Count the occupied number
            for (x, y) in seq:
                if chips[x][y] in colours:
                    count += 1
            # Calculate the remaining steps
            step = len(seq) - count
            # Find the minimum number
            if step < min:
                min = step
                best_seq = seq
        return min, best_seq

    # Goal Recognition
    def recognize_opponent_target(self, state):
        chips = state.board.chips
        op = state.agents[1 - self.id]
        op_colours = (op.colour, op.seq_colour)

        min_step, best_seq = self.minStep(chips, WINNING_SEQUENCES + [CENTER], op_colours)

        # If the opponent is about to win(in 2 steps), prioritize defense
        return best_seq if min_step <= 2 else None

    # Composite evaluation: Q-Learning + Heuristic + Center Threat
    def evaluate(self, state):
        s_key = self.serialize(state)
        q_dict = self.Q.get(s_key, {})
        q_val = max(q_dict.values()) if q_dict else 0.0
        # Assume game phase = fraction of non‐empty on board
        filled = sum(1 for row in state.board.chips for c in row if c != EMPTY)
        phase = filled / (10 * 10)
        h_val = -self.heuristic(state)
        # Normalization
        norm_h = (h_val + 12) / 12
        norm_q = min(q_val / 5.0, 1.0)

        chips = state.board.chips
        me = state.agents[self.id]
        op = state.agents[1 - self.id]
        my_cols = (me.colour, me.seq_colour)
        op_cols = (op.colour, op.seq_colour)
        my_ctr = sum(1 for x, y in CENTER if chips[x][y] in my_cols)
        norm_c = my_ctr / 4

        base = (4.0 * norm_q) * (1-phase) + (1.5 * norm_h) * phase + 1.0 * norm_c

        my_of4 = self.count_open_patterns(chips, my_cols, 4)
        my_of3 = self.count_open_patterns(chips, my_cols, 3)
        op_of4 = self.count_open_patterns(chips, op_cols, 4)
        op_of3 = self.count_open_patterns(chips, op_cols, 3)

        threat = (THREAT_WEIGHTS['open_four']  * (my_of4 - op_of4)
                  + THREAT_WEIGHTS['open_three']  * (my_of3 - op_of3))

        return base + threat

    # Count to evaluate threat
    def count_open_patterns(self, chips, pid_cols, length):
        cnt = 0
        for seq in WINNING_SEQUENCES + [CENTER]:
            n = len(seq)
            if n < length:
                continue
            # slide a window of size=length within seq
            for i in range(n - length + 1):
                segment = seq[i:i + length]
                ends = []
                if i > 0:
                    ends.append(seq[i - 1])
                if i + length < n:
                    ends.append(seq[i + length])
                if all(chips[r][c] in pid_cols for r, c in segment) \
                        and all(0 <= rr < 10 and 0 <= cc < 10 and chips[rr][cc] == EMPTY
                                for rr, cc in ends):
                    cnt += 1
        return cnt

    # Quiescence Search
    def quiesce(self, state, alpha, beta):
        stand_pat = self.evaluate(state)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        # Only extend “threatening” moves: those creating open‐four
        moves = [a for a in self.GetActions(state) if self.creates_open_four(state, a)]
        for a in moves:
            nxt = self.game_rule.generateSuccessor(state, a, self.id)
            score = -self.quiesce(nxt, -beta, -alpha)
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    # Weather the action falls on the state and causes our side to display the "Open Four" mode.
    def creates_open_four(self, state, action):
        next_state, _ = self.DoAction(deepcopy(state), action)
        chips_before = state.board.chips
        chips_after = next_state.board.chips
        cols = (next_state.agents[self.id].colour,
                next_state.agents[self.id].seq_colour)

        cnt_before = self.count_open_patterns(chips_before, cols, 4)
        cnt_after = self.count_open_patterns(chips_after, cols, 4)

        return cnt_after > cnt_before