import random
from itertools import product

from agents.MyAgents.Fast_BFS.Constant_info import *
from agents.MyAgents.Fast_BFS.Constant_info import SLIDE_DIRECTION_CENTER


# Generates actions from this state.
def GetActions(game_state, agent_id):
    actions = []
    agent_state = game_state.agents[agent_id]

    # First, give the agent the option to trade a dead card, if they haven't just done so.
    if not agent_state.trade:
        for card in agent_state.hand:
            if card[0] != 'j':
                free_spaces = 0
                for r, c in COORDS[card]:
                    if game_state.board.chips[r][c] == EMPTY:
                        free_spaces += 1
                if not free_spaces:  # No option to place, so card is considered dead and can be traded.
                    if game_state.board.draft:
                        for draft in game_state.board.draft:
                            actions.append({'play_card': card, 'draft_card': draft, 'type': 'trade', 'coords': None})
                    else:
                        actions.append({'play_card': card, 'draft_card': None, 'type': 'trade', 'coords': None})

        if len(actions):  # If trade actions available, return those, along with the option to forego the trade.
            actions.append({'play_card': None, 'draft_card': None, 'type': 'trade', 'coords': None})
            return actions

    # If trade is prohibited, or no trades available, add action/s for each card in player's hand.
    # For each action, add copies corresponding to the various draft cards that could be selected at end of turn.
    for card in agent_state.hand:
        if card in ['jd', 'jc']:  # two-eyed jacks
            for r in range(10):
                for c in range(10):
                    if game_state.board.chips[r][c] == EMPTY:
                        if game_state.board.draft:
                            for draft in game_state.board.draft:
                                actions.append(
                                    {'play_card': card, 'draft_card': draft, 'type': 'place', 'coords': (r, c)})
                        else:
                            actions.append({'play_card': card, 'draft_card': None, 'type': 'place', 'coords': (r, c)})

        elif card in ['jh', 'js']:  # one-eyed jacks
            for r in range(10):
                for c in range(10):
                    if game_state.board.chips[r][c] == agent_state.opp_colour:
                        if game_state.board.draft:
                            for draft in game_state.board.draft:
                                actions.append(
                                    {'play_card': card, 'draft_card': draft, 'type': 'remove', 'coords': (r, c)})
                        else:
                            actions.append(
                                {'play_card': card, 'draft_card': None, 'type': 'remove', 'coords': (r, c)})

        else:  # regular cards
            for r, c in COORDS[card]:
                if game_state.board.chips[r][c] == EMPTY:
                    if game_state.board.draft:
                        for draft in game_state.board.draft:
                            actions.append({'play_card': card, 'draft_card': draft, 'type': 'place', 'coords': (r, c)})
                    else:
                        actions.append({'play_card': card, 'draft_card': None, 'type': 'place', 'coords': (r, c)})

    return actions

    # Carry out a given action on this state and return True if goal is reached received.


def DoAction(agent_id, state, action):
    score = state.agents[agent_id].score
    possible_state = generateSuccessor(state, action, agent_id)
    new_score = possible_state.agents[agent_id].score
    if new_score > score:
        # print(f"score {score} new score {new_score} action {action}")
        goal_reached = True
    else:
        goal_reached = False
    return possible_state, goal_reached


def generateSuccessor(state, action, agent_id):
    state.board.new_seq = False
    plr_state = state.agents[agent_id]
    plr_state.last_action = action  # Record last action such that other agents can make use of this information.
    reward = 0

    # Update agent state. Take the card in play from the agent, discard, draw the selected draft, deal a new draft.
    # If agent was allowed to trade but chose not to, there is no card played, and hand remains the same.
    card = action['play_card']
    draft = action['draft_card']
    if card:
        plr_state.hand.remove(card)  # Remove card from hand.
        plr_state.discard = card  # Add card to discard pile.
        state.deck.discards.append(
            card)  # Add card to global list of discards (some agents might find tracking this helpful).
        if draft:
            state.board.draft.remove(draft)  # Remove draft from draft selection.
            plr_state.hand.append(draft)  # Add draft to player hand.
        # state.board.draft.extend(state.deck.deal())  # Replenish draft selection.

    # If action was to trade in a dead card, action is complete, and agent gets to play another card.
    if action['type'] == 'trade':
        plr_state.trade = True  # Switch trade flag to prohibit agent performing a second trade this turn.
        plr_state.agent_trace.action_reward.append((action, reward))  # Log this turn's action and score (zero).
        return state

    # Update Sequence board. If action was to place/remove a marker, add/subtract it from the board.
    r, c = action['coords']
    if action['type'] == 'place':
        state.board.chips[r][c] = plr_state.colour
        state.board.empty_coords.remove(action['coords'])
        state.board.plr_coords[plr_state.colour].append(action['coords'])
    elif action['type'] == 'remove':
        state.board.chips[r][c] = EMPTY
        state.board.empty_coords.append(action['coords'])
        state.board.plr_coords[plr_state.opp_colour].remove(action['coords'])
    else:
        print("Action unrecognised.")

    # Check if a sequence has just been completed. If so, upgrade chips to special sequence chips.
    if action['type'] == 'place':
        seq, seq_type = checkSeq(state.board.chips, plr_state, (r, c))
        if seq:
            reward += seq['num_seq']
            state.board.new_seq = seq_type
            for sequence in seq['coords']:
                for r, c in sequence:
                    if state.board.chips[r][c] != JOKER:  # Joker spaces stay jokers.
                        state.board.chips[r][c] = plr_state.seq_colour
                        try:
                            state.board.plr_coords[plr_state.colour].remove(action['coords'])
                        except:  # Chip coords were already removed with the first sequence.
                            pass
            plr_state.completed_seqs += seq['num_seq']
            plr_state.seq_orientations.extend(seq['orientation'])

    plr_state.trade = False  # Reset trade flag if agent has completed a full turn.
    plr_state.agent_trace.action_reward.append((action, reward))  # Log this turn's action and any resultant score.
    plr_state.score += reward
    return state


# Returns a list of sequence coordinates if a sequence has just been formed. Else, returns None.
def checkSeq(chips, plr_state, last_coords):
    clr, sclr = plr_state.colour, plr_state.seq_colour
    oc, os = plr_state.opp_colour, plr_state.opp_seq_colour
    seq_type = TRADSEQ
    seq_coords = []
    seq_found = {'vr': 0, 'hz': 0, 'd1': 0, 'd2': 0, 'hb': 0}
    found = False
    nine_chip = lambda x, clr: len(x) == 9 and len(set(x)) == 1 and clr in x
    lr, lc = last_coords

    # All joker spaces become player chips for the purposes of sequence checking.
    for r, c in COORDS['jk']:
        chips[r][c] = clr

    # First, check "heart of the board" (2h, 3h, 4h, 5h). If possessed by one team, the game is over.
    coord_list = [(4, 4), (4, 5), (5, 4), (5, 5)]
    heart_chips = [chips[y][x] for x, y in coord_list]
    if EMPTY not in heart_chips and (clr in heart_chips or sclr in heart_chips) and not (
            oc in heart_chips or os in heart_chips):
        seq_type = HOTBSEQ
        seq_found['hb'] += 2
        seq_coords.append(coord_list)

    # Search vertical, horizontal, and both diagonals.
    vr = [(-4, 0), (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    hz = [(0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    d1 = [(-4, -4), (-3, -3), (-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    d2 = [(-4, 4), (-3, 3), (-2, 2), (-1, 1), (0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]
    for seq, seq_name in [(vr, 'vr'), (hz, 'hz'), (d1, 'd1'), (d2, 'd2')]:
        coord_list = [(r + lr, c + lc) for r, c in seq]
        coord_list = [i for i in coord_list if 0 <= min(i) and 9 >= max(i)]  # Sequences must stay on the board.
        chip_str = ''.join([chips[r][c] for r, c in coord_list])
        # Check if there exists 4 player chips either side of new chip (counts as forming 2 sequences).
        if nine_chip(chip_str, clr):
            seq_found[seq_name] += 2
            seq_coords.append(coord_list)
        # If this potential sequence doesn't overlap an established sequence, do fast check.
        if sclr not in chip_str:
            sequence_len = 0
            start_idx = 0
            for i in range(len(chip_str)):
                if chip_str[i] == clr:
                    sequence_len += 1
                else:
                    start_idx = i + 1
                    sequence_len = 0
                if sequence_len >= 5:
                    seq_found[seq_name] += 1
                    seq_coords.append(coord_list[start_idx:start_idx + 5])
                    break
        else:  # Check for sequences of 5 player chips, with a max. 1 chip from an existing sequence.
            for pattern in [clr * 5, clr * 4 + sclr, clr * 3 + sclr + clr, clr * 2 + sclr + clr * 2,
                            clr + sclr + clr * 3, sclr + clr * 4]:
                for start_idx in range(5):
                    if chip_str[start_idx:start_idx + 5] == pattern:
                        seq_found[seq_name] += 1
                        seq_coords.append(coord_list[start_idx:start_idx + 5])
                        found = True
                        break
                if found:
                    break

    for r, c in COORDS['jk']:
        chips[r][c] = JOKER  # Joker spaces reset after sequence checking.

    num_seq = sum(seq_found.values())
    if num_seq > 1 and seq_type != HOTBSEQ:
        seq_type = MULTSEQ
    return ({'num_seq': num_seq, 'orientation': [k for k, v in seq_found.items() if v], 'coords': seq_coords},
            seq_type) if num_seq else (None, None)


def generate_combinations(hand, empty_coords):
    empty_set = set(empty_coords)
    candidates = []

    for card in hand:
        candidate = []
        for r, c in COORDS[card]:
            if (r, c) in empty_set:
                candidate.append((r, c))
        if candidate:
            candidates.append(candidate)

    return [list(p) for p in product(*candidates)] if candidates else []


def first_action_selection(actions):
    return actions[0]  # If no goal was found in the time limit, return first action.


def random_action_selection(actions):
    return random.choice(actions)  # If no goal was found in the time limit, return first action.


def unique_actions(actions):
    return [dict(s) for s in {frozenset(d.items()) for d in actions}]


def find_critical_locations(chips, target_colour, target_colour_seq):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    empty_pos_all = []
    taken_pos_all = []

    for x in range(SIZE):
        for y in range(SIZE):
            for dx, dy in directions:
                count = 0
                count_seq = 0
                empty_pos = []
                taken_pos = []
                reverse_target_color_contains = False

                for i in range(5):
                    nx, ny = x + i * dx, y + i * dy
                    if not (0 <= nx < SIZE and 0 <= ny < SIZE):
                        break
                    check_location = chips[nx][ny]
                    if check_location == target_colour or check_location == JOKER:
                        count += 1
                        if not check_location == JOKER:
                            taken_pos = taken_pos + [(nx, ny)]
                    elif check_location == EMPTY:
                        empty_pos = empty_pos + [(nx, ny)]
                    elif check_location == target_colour_seq:
                        count_seq += 1
                    else:
                        reverse_target_color_contains = True
                        break

                if not reverse_target_color_contains and len(taken_pos) >= 3 and not len(
                        empty_pos) == 0 and count_seq < 2:
                    empty_pos_all.extend(empty_pos)
                    taken_pos_all.extend(taken_pos)
    if not len(empty_pos_all) == 0 and not len(taken_pos_all) == 0:
        return empty_pos_all, taken_pos_all

    return None, None


def find_critical_location_matrix(chips, target_colour, target_colour_seq):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    empty_pos_all = []
    taken_pos_all = []

    for x in range(SIZE):
        for y in range(SIZE):
            for dx, dy in directions:
                count = 0
                count_seq = 0
                empty_pos = []
                taken_pos = []
                reverse_target_color_contains = False

                for i in range(5):
                    nx, ny = x + i * dx, y + i * dy
                    if not (0 <= nx < SIZE and 0 <= ny < SIZE):
                        break
                    check_location = chips[nx][ny]
                    count += 1
                    if check_location == target_colour or check_location == JOKER:
                        if not check_location == JOKER:
                            taken_pos = taken_pos + [(nx, ny)]
                    elif check_location == EMPTY:
                        empty_pos = empty_pos + [(nx, ny)]
                    elif check_location == target_colour_seq:
                        count_seq += 1
                    else:
                        reverse_target_color_contains = True
                        break

                if not reverse_target_color_contains and count == 5 and len(
                        taken_pos) >= 3 and not len(empty_pos) == 0 and count_seq < 2:
                    empty_pos_all.append(empty_pos)
                    taken_pos_all.append(taken_pos)

    # center check
    count_seq = 0
    empty_pos = []
    taken_pos = []
    reverse_target_color_contains = False
    for x, y in SLIDE_DIRECTION_CENTER:
        check_location = chips[x][y]
        if check_location == target_colour or check_location == JOKER:
            if not check_location == JOKER:
                taken_pos = taken_pos + [(x, y)]
        elif check_location == EMPTY:
            empty_pos = empty_pos + [(x, y)]
        elif check_location == target_colour_seq:
            count_seq += 1
        else:
            reverse_target_color_contains = True
            break

    if not reverse_target_color_contains and len(taken_pos) >= 2 and not len(empty_pos) == 0 and count_seq < 2:
        empty_pos_all.append(empty_pos)
        taken_pos_all.append(taken_pos)

    return empty_pos_all, taken_pos_all
