from copy import deepcopy
from agents.MyAgents.helper.Constant_info import *

initial_heuristic = [[float('inf'), 7, 7, 7, 7, 7, 7, 7, 7, float('inf')],
                    [7, 7, 8, 8, 8, 8, 8, 8, 7, 7],
                    [7, 8, 7, 8, 8, 8, 8, 7, 8, 7],
                    [7, 8, 8, 7, 8, 8, 7, 8, 8, 7],
                    [7, 8, 8, 8, 3, 3, 8, 8, 8, 7],
                    [7, 8, 8, 8, 3, 3, 8, 8, 8, 7],
                    [7, 8, 8, 7, 8, 8, 7, 8, 8, 7],
                    [7, 8, 7, 8, 8, 8, 8, 7, 8, 7],
                    [7, 7, 8, 8, 8, 8, 8, 8, 7, 7],
                    [float('inf'), 7, 7, 7, 7, 7, 7, 7, 7, float('inf')]]

class HeuristicGameRule:
    def __init__(self):
        self.heuristic_board = initial_heuristic

    def generateSuccessor(self, state, action, agent_id, heuristic_in_cal):
        state.board.new_seq = False
        plr_state = state.agents[agent_id]
        plr_state.last_action = action  # Record last action such that other agents can make use of this information.
        reward = 0
        heuristic_min = min(min(row) for row in heuristic_in_cal)
        heuristic_new = heuristic_in_cal

        # Update agent state. Take the card in play from the agent, discard, draw the selected draft, deal a new draft.
        # If agent was allowed to trade but chose not to, there is no card played, and hand remains the same.
        card = action['play_card']
        draft = action['draft_card']
        if card:
            plr_state.hand.remove(card)  # Remove card from hand.
            plr_state.discard = card  # Add card to discard pile.
            state.deck.discards.append(
                card)  # Add card to global list of discards (some agents might find tracking this helpful).
            state.board.draft.remove(draft)  # Remove draft from draft selection.
            plr_state.hand.append(draft)  # Add draft to player hand.
            #state.board.draft.extend(state.deck.deal())  # Replenish draft selection.

        # If action was to trade in a dead card, action is complete, and agent gets to play another card.
        if action['type'] == 'trade':
            plr_state.trade = True  # Switch trade flag to prohibit agent performing a second trade this turn.
            plr_state.agent_trace.action_reward.append((action, reward))  # Log this turn's action and score (zero).
            return state, heuristic_min, heuristic_new

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
            seq, seq_type, heuristic_min, heuristic_new = self.checkHeuristic(state.board.chips, plr_state, (r, c), heuristic_in_cal)
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
        return state, heuristic_min, heuristic_new

        # Returns a list of sequence coordinates if a sequence has just been formed. Else, returns None.

    def checkHeuristic(self, chips, plr_state, last_coords, heuristic_in_cal):
        heuristic_new = deepcopy(heuristic_in_cal)
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
        number_own = 0
        number_opp = 0
        number_fail = 0
        # heart_chips = [chips[x][y] for x, y in coord_list]
        if (lr,lc) in coord_list:
            for check_r, check_c in coord_list:
                if os in chips[check_r][check_c]:
                    number_fail += 1
                if clr in chips[check_r][check_c] or sclr in chips[check_r][check_c]:
                    number_own += 1
                if oc in chips[check_r][check_c]:
                    number_opp += 1
            if number_fail != 0:
                heuristic_new[lr][lc] = float('inf')
            else:
                for check_r, check_c in coord_list:
                    heuristic_new[check_r][check_c] = min(heuristic_new[check_r][check_c],len(coord_list)-number_own+number_opp)
                if number_own == len(coord_list):
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
                for r, c in coord_list:
                    heuristic_new[r][c] = 0
            # If this potential sequence doesn't overlap an established sequence, do fast check.
            else:
                for i,j in coord_list:
                    # min_step = [float('inf')] * 4
                    min_step = float('inf')
                    direction_index = 0
                    # for seq_sub, seq_names_sub in [(vr, 'vr'), (hz, 'hz'), (d1, 'd1'), (d2, 'd2')]:
                    for seq_sub, seq_names_sub in [(seq, seq_name)]:
                        coord_list_sub = [(r + i, c + j) for r, c in seq_sub]
                        coord_list_sub = [p for p in coord_list_sub if 0 <= min(p) and 9 >= max(p)]
                        chip_str_sub = ''.join([chips[r][c] for r, c in coord_list_sub])
                        if nine_chip(chip_str_sub, clr):
                            heuristic_new[i][j] = 0
                            # min_step[direction_index] = 0
                            min_step = 0
                        else:
                            for index_window in range(len(chip_str_sub) - 4):
                                window = chip_str_sub[index_window:index_window + 5]
                                # color_count = window.count(clr) +  window.count(sclr)
                                color_count = window.count(clr)
                                opp_color_count = window.count(oc)
                                opp_sequence_color_count = window.count(os)
                                tmp_cal = float('inf') if opp_sequence_color_count > 0 else max(0,5-color_count+opp_color_count)
                                # min_step[direction_index] = min(min_step[direction_index], tmp_cal)
                                min_step = min(min_step, tmp_cal)
                            # direction_index += 1
                    # heuristic_new[i][j] = min(heuristic_new[i][j], sum(sorted(min_step)[:2]))
                    heuristic_new[i][j] = min(heuristic_new[i][j],min_step)
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
            heuristic_new[r][c]=float('inf')

        num_seq = sum(seq_found.values())
        if num_seq > 1 and seq_type != HOTBSEQ:
            seq_type = MULTSEQ

        heuristic_min = min(min(row) for row in heuristic_new)

        return ({'num_seq': num_seq, 'orientation': [k for k, v in seq_found.items() if v], 'coords': seq_coords},
                seq_type, heuristic_min, heuristic_new) if num_seq else (None, None, heuristic_min, heuristic_new)