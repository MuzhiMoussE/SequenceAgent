import random
import pickle
import gzip
import os
import time
from collections import defaultdict

from agents.MyAgents.MCTS.deterministic_rule import DeterministicGameRule as GameRule
from Sequence.sequence_utils import EMPTY

"""
    Q-Learning training script for Sequence Agent
"""

MAX_STEPS    = 100
ALPHA        = 0.1
GAMMA        = 0.9
EPSILON_START = 0.5
EPSILON_END   = 0.01
DECAY_RATE    = 0.9999  # ε *= DECAY_RATE
TRAIN_TIME     = 600.0
REPLAY_TIME    = 60.0

Q = defaultdict(lambda: defaultdict(float))

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

# Abstract state
def abstract_state(state, agent_id):
    board = state.board.chips
    me = state.agents[agent_id]
    op = state.agents[1-agent_id]

    # Completed sequence count
    completed_me = sum(all(board[x][y]==me.colour for x,y in seq)
                       for seq in WINNING_SEQUENCES)
    completed_op = sum(all(board[x][y]==op.colour for x,y in seq)
                       for seq in WINNING_SEQUENCES)

    # Threat count
    def count_threat(col):
        cnt = 0
        for seq in WINNING_SEQUENCES:
            vals = [board[x][y] for x,y in seq]
            if vals.count(col) >= 3 and vals.count(EMPTY) >= (5-vals.count(col)):
                cnt += 1
        return cnt
    threat_me = count_threat(me.colour)
    threat_op = count_threat(op.colour)

    hand_count = len(me.hand)

    # Center count
    center_me = sum(1 for x,y in CENTER if board[x][y] == me.colour)
    center_op = sum(1 for x,y in CENTER if board[x][y] == op.colour)

    # Suits
    suits = ['H', 'D', 'C', 'S']
    hand_suit_counts = tuple(
        sum(1 for card in me.hand if card.upper().endswith(s))
        for s in suits
    )

    draft = tuple(state.board.draft[:3])
    score_diff = me.score - op.score

    return (
        completed_me, completed_op,
        threat_me, threat_op,
        hand_count, center_me, center_op,
        *hand_suit_counts,
        draft, score_diff
    )

# Serialize action
def serialize_action(action):
    coords = tuple(action.get('coords') or ())
    play_card = action.get('play_card')
    draft_card = action.get('draft_card')
    return (coords, play_card, draft_card)

#  ε-greedy
def choose_action(state_key, actions, epsilon):
    if not actions:
        return None

    # Explore
    if random.random() < epsilon:
        return random.choice(actions)

    # Exploit
    ak_vals = []
    for a in actions:
        ak = serialize_action(a)
        ak_vals.append((ak, Q[state_key].get(ak, 0.0)))
    best_val = max(v for _, v in ak_vals)
    # Tie-break
    best_aks = [ak for ak, v in ak_vals if v == best_val]
    chosen_ak = random.choice(best_aks)
    for a in actions:
        if serialize_action(a) == chosen_ak:
            return a

    return random.choice(actions)

# Training loop
def train_q(time_budget = TRAIN_TIME):
    start = time.time()
    env = GameRule(2)
    experience_pool = []
    epsilon = EPSILON_START

    ep = 0
    while time.time() - start < time_budget:
        state = env.initialGameState()
        agent_id = 0
        step = 0

        while step < MAX_STEPS:
            s_key = abstract_state(state, agent_id)
            actions = env.getLegalActions(state, agent_id)
            if not actions:
                break
            action = choose_action(s_key, actions, epsilon)

            # Simulate
            next_state = env.generateSuccessor(state, action, agent_id)

            # Reward: Defend > Attack
            reward = next_state.agents[agent_id].score - state.agents[agent_id].score
            op_id = 1 - agent_id
            seq = recognize_opponent_target(state, op_id)
            # Defend, +0.5
            if seq and action['coords'] in seq:
                reward += 0.5
            # Center, +0.2
            if action.get('coords') in CENTER:
                reward += 0.2

            next_key  = abstract_state(next_state, agent_id)
            ak = serialize_action(action)
            experience_pool.append((s_key, ak, reward, next_key))

            # Best Q
            next_actions = env.getLegalActions(next_state, agent_id)
            best_next = 0.0
            if next_actions:
                best_next = max(Q[next_key ][serialize_action(a)] for a in next_actions)

            # qLearning update
            Q[s_key][ak] += ALPHA * (reward + GAMMA * best_next - Q[s_key][ak])

            state = next_state
            agent_id = 1 - agent_id
            step += 1

        epsilon = max(EPSILON_END, epsilon * DECAY_RATE)
        ep += 1

    # Experience replay for remaining REPLAY_TIME
    replay_start = time.time()
    while time.time() - replay_start < REPLAY_TIME and experience_pool:
        s_key, ak, reward, next_key = random.choice(experience_pool)
        best_next = 0.0
        if Q.get(next_key):
            best_next = max(Q[next_key].values())
        Q[s_key][ak] += ALPHA * (reward + GAMMA * best_next - Q[s_key][ak])

    # Save Q-table
    num_states = len(Q)
    num_entries = sum(len(d) for d in Q.values())

    gz_file = os.path.join(os.path.dirname(__file__), 'q_table.pkl.gz')
    with gzip.open(gz_file, 'wb') as f:
        pickle.dump(dict(Q), f)
    print(f"train_q: ran {ep} episodes in {time.time()-start:.2f}s, saved Q-table")

# Automatically load pre-training Q table
gz_path=os.path.join(os.path.dirname(__file__),'q_table.pkl.gz')
if os.path.exists(gz_path):
    with gzip.open(gz_path,'rb') as f:
        data=pickle.load(f)
        Q=defaultdict(lambda: defaultdict(float), data)

# Calculate the minimum number of steps required for any sequence
def minStep(chips, seqs, colours):
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
def recognize_opponent_target(state, op_id):
    chips = state.board.chips
    op = state.agents[op_id]
    op_colours = (op.colour, op.seq_colour)

    min_step, best_seq = minStep(chips, WINNING_SEQUENCES + [CENTER], op_colours)

    # If the opponent is about to win(in 2 steps), prioritize defense
    return best_seq if min_step <= 2 else None


if __name__ == '__main__':
    train_q()