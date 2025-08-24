# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#

import time
import traceback
from collections import deque, Counter
from copy import deepcopy
from agents.MyAgents.Fast_BFS.Parameter_info import DISCOUNT_FACTOR, NUM_PLAYER, SEQUENCE_WEIGHT, \
    SEQUENCE_EXPONENT, LINE_WEIGHT, LINE_EXPONENT, BOARD_WEIGHT, BOARD_EXPONENT
from agents.MyAgents.Fast_BFS.action_utils import DoAction, GetActions, checkSeq, \
    generate_combinations, random_action_selection, first_action_selection, unique_actions, \
    find_critical_location_matrix
from agents.MyAgents.Fast_BFS.card_counter import CardCounter
from agents.MyAgents.Fast_BFS.Evaluator import Evaluator
from agents.MyAgents.Fast_BFS.Constant_info import *
from template import Agent

FULL_TIME = 0.95
NUM_PLAYERS = 2


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.enemy_id = NUM_PLAYER - self.id - 1
        self.card_counter = None
        self.start_time = None

    def SelectAction(self, actions, game_state):
        if not self.card_counter:
            self.card_counter = CardCounter(self.id, self.enemy_id, game_state)
            self.start_time = time.time()
        else:
            self.start_time = time.time()
            self.card_counter.update(game_state)

        action = self._selectAction(actions, game_state)
        return action

    def _selectAction(self, actions, game_state):
        evaluator = Evaluator(self.id, game_state)
        draft_value_map = self.cal_draft_score(game_state, evaluator)
        best_draft = max(draft_value_map, key=draft_value_map.get)

        lose_score, defence_draft, defence_coordinates = self.is_goal_reachable(self.card_counter.known_enemy_hand,
                                                                                     self.enemy_id, game_state)
        if defence_draft:
            best_draft = defence_draft

        # trade with the highest score draft if has
        trade_action = next((a for a in actions if a['type'] == 'trade' and a['draft_card'] == best_draft), None)
        if trade_action:
            return trade_action

        #  Should the optimal draw be used when successfully defending?
        if lose_score:
            best_acton = self.get_defence_action(actions, game_state, best_draft, defence_coordinates)
            if best_acton:
                return best_acton

            goal_reached, draft, goal_coordinates = self.is_goal_reachable(game_state.agents[self.id].hand,
                                                                                self.id, game_state)
            if goal_reached:
                if draft:
                    best_draft = draft
                game_state.board.draft = [best_draft]
                return self.fast_bfs(game_state, goal_coordinates, evaluator, self.start_time)
            if defence_draft:
                game_state.board.draft = [defence_draft]
        else:
            goal_reached, draft, goal_coordinates = self.is_goal_reachable(game_state.agents[self.id].hand,
                                                                                self.id, game_state)
            if goal_reached:
                if draft:
                    best_draft = draft
                game_state.board.draft = [best_draft]
                return self.fast_bfs(game_state, goal_coordinates, evaluator, self.start_time)

            best_acton = self.get_defence_action(actions, game_state, best_draft, defence_coordinates)
            if best_acton:
                return best_acton

        if best_draft in TWO_EYED_JACKS:
            game_state.board.draft = [best_draft]

        return self.evaluate_by_reward_func(actions, game_state, evaluator, self.start_time)

    def get_defence_action(self, actions, game_state, best_draft, defence_coordinates):
        empty_pos_counter, taken_pos_counter = self.find_defense_locations(game_state)
        action_map = {a['coords']: a for a in actions if a['type'] == 'remove' and a['draft_card'] == best_draft}
        if action_map and len(taken_pos_counter) > 0:
            # Try to filter from defence_coordinates
            filtered_items = [
                item for item in action_map.items() if item[0] in defence_coordinates
            ] if defence_coordinates else action_map.items()
            #  When removing, prioritize the removal of elements from longer queues? Not implement yet
            best_action = max(
                (item for item in filtered_items if item[0] in taken_pos_counter),
                key=lambda item: taken_pos_counter.get(item[0]),
                default=(None, None)
            )[1]
            if best_action:
                return best_action

        if len(empty_pos_counter) > 0:
            best_action = max(
                (a for a in actions if a['draft_card'] == best_draft and a['coords'] in empty_pos_counter),
                key=lambda a: empty_pos_counter.get(a['coords']),
                default=None
            )
            if best_action:
                return best_action

    def find_defense_locations(self, game_state):
        target_colour = game_state.agents[self.id].opp_colour
        target_colour_seq = game_state.agents[self.id].opp_seq_colour
        empty_pos_all, taken_pos_all = find_critical_location_matrix(game_state.board.chips, target_colour,
                                                                     target_colour_seq)
        flat_empty_pos = [pos for group in empty_pos_all for pos in group]
        empty_pos_counter = Counter(flat_empty_pos)
        flat_taken_pos = [pos for group in taken_pos_all for pos in group]
        taken_pos_counter = Counter(flat_taken_pos)
        return empty_pos_counter, taken_pos_counter

    def evaluate_by_reward_func(self, actions, game_state, evaluator, start_time):
        # changed function structure, currently is correct and can save more space
        # play_card -> evaluator
        evaluator_map = dict()
        max_score = 0
        best_action = None
        node_count = 0

        # try:
        ua = unique_actions(actions)
        # queue = deque()
        timer_record = 0
        timer_start = time.time()
        for action in ua:
            if time.time() - start_time < (FULL_TIME - timer_record):
                timer_start = time.time()
                node_count += 1
                play_card = action['play_card']
                play_coords = action['coords']
                if evaluator_map.get(play_coords):
                    if best_action['coords'] == play_coords and play_card not in TWO_EYED_JACKS:
                        best_action['play_card'] = play_card
                else:
                    node_evaluator = evaluator.get_deepcopy()
                    line_score, board_score, sequence_score = node_evaluator.update_by_action(action)
                    score = self.cal_reward_func(line_score, board_score, sequence_score)
                    node_state = deepcopy(game_state)
                    node_state.board.chips[play_coords[0]][play_coords[1]] = node_state.agents[self.id].colour
                    draft_card_reward = self.cal_draft_score(node_state, node_evaluator)
                    max_draft, max_reward = max(draft_card_reward.items(), key=lambda item: item[1])
                    score = score + max_reward * DISCOUNT_FACTOR * DISCOUNT_FACTOR
                    action['draft_card'] = max_draft
                    if max_score < score:
                        max_score = score
                        best_action = action
                    evaluator_map[play_coords] = score
            timer_record = max(time.time() - timer_start, timer_record)

        if best_action:
            return best_action
        '''
        except Exception as e:
            # print(f"exception! {e}")
            traceback.print_exc()'''
        return first_action_selection(actions)

    def fast_bfs(self, game_state, goal_important_loc, evaluator, start_time):
        actions = GetActions(game_state, self.id)
        ua = unique_actions(actions)
        queue = deque()
        default_acton = actions[0]
        for action in ua:
            if goal_important_loc and action['coords'] not in goal_important_loc:
                continue
            queue.append((game_state, [action]))
            # default_acton = action

        if len(queue) == 0:
            # fallback to evaluation
            return self.evaluate_by_reward_func(actions, game_state, evaluator, start_time)

        if queue:
            _, action_list = queue[0]
            default_acton = action_list[0]

        timer_record = 0
        max_depth = 0
        node_count = 0

        while len(queue) and time.time() - start_time < (FULL_TIME - timer_record):
            node_count += 1
            timer_start = time.time()
            state, path = queue.popleft()  # Pop the next node (state, path) in the queue.
            action = path[-1]

            if len(path) > max_depth:
                max_depth = len(path)

            # Carry out this action on the state, and check for goal
            next_state, goal = DoAction(self.id, deepcopy(state), action)

            if goal:
                return path[0]  # If the current action reached the goal, return the initial action that led there.

            for a in unique_actions(GetActions(next_state, self.id)):
                if goal_important_loc and a['coords'] not in goal_important_loc:
                    continue
                next_path = path + [a]
                queue.append((next_state, next_path))

            timer_record = max(time.time() - timer_start, timer_record)

        if default_acton:
            return default_acton

        # fast_bfs fallback!!!
        return random_action_selection(actions)  # If no goal was found in the time limit, return first action.

    def is_goal_reachable(self, agent_hand, agent_id, state):
        # check goal could be reached with which draft
        hand = agent_hand
        # use all non j cards
        combinations = generate_combinations(hand, state.board.empty_coords)
        color = state.agents[agent_id].colour
        color_seq = state.agents[agent_id].seq_colour
        # check hand cards
        hand_j_count = hand.count('jc') + hand.count('jd')
        chips = state.board.chips
        for coordinates in combinations:
            chips_copy = [row[:] for row in chips]
            taken_coordinates = []
            for coordinate in coordinates:
                r, c = coordinate
                if chips_copy[r][c] != EMPTY:
                    continue
                taken_coordinates += [coordinate]
                chips_copy[r][c] = color
                seq, seq_type = checkSeq(chips_copy, state.agents[agent_id], coordinate)
                if seq_type:
                    return True, None, set([item for row in seq['coords'] for item in row])

            # find and use two-eyed jacks by find_defense_move
            if hand_j_count > 0:
                empty_pos_all, taken_pos_all = find_critical_location_matrix(chips_copy, color, color_seq)
                min_length = min((len(group) for group in empty_pos_all), default=5)
                if min_length <= hand_j_count:
                    # Find out that for all indexes i, the corresponding empty_pos_all[i] is the shortest
                    shortest_indices = [i for i, group in enumerate(empty_pos_all) if len(group) == min_length]
                    # Merge the elements corresponding to these indexes in "empty" and "taken"
                    merged_shortest = [pos for i in shortest_indices for pos in (empty_pos_all[i] + taken_pos_all[i])]
                    # When the judgment condition is met, return the merged result
                    return True, None, set(taken_coordinates + merged_shortest)

            # check one draft cards
            for draft in state.board.draft:
                for r, c in COORDS[draft]:
                    if chips_copy[r][c] != EMPTY:
                        continue
                    chips_copy_2 = [row[:] for row in chips_copy]
                    chips_copy_2[r][c] = color
                    seq, seq_type = checkSeq(chips_copy_2, state.agents[agent_id], (r, c))
                    if seq_type:
                        return True, draft, set([item for row in seq['coords'] for item in row])

        return False, None, None

    def cal_draft_score(self, state, evaluator):
        root_line_score = evaluator.get_line_score()
        root_board_score = evaluator.get_board_score()
        root_sequence_score = evaluator.sequence_score

        draft_value_map = dict()
        for card in state.board.draft:
            if card in TWO_EYED_JACKS:  # two-eyed jacks
                draft_value_map[card] = self.cal_two_eyed_jack_score(root_line_score, root_board_score,
                                                                     root_sequence_score)
                continue
            elif card in ONE_EYED_JACKS:  # one-eyed jacks
                draft_value_map[card] = self.cal_one_eyed_jack_score(root_line_score, root_board_score,
                                                                     root_sequence_score)
                continue

            max_reward = 0
            for r, c in COORDS[card]:
                if state.board.chips[r][c] != EMPTY:
                    continue
                action = {'type': 'place', 'coords': (r, c)}
                line_score, board_score, sequence_score = evaluator.get_deepcopy().update_by_action(action)
                reward = self.cal_reward_func(line_score, board_score, sequence_score)
                max_reward = max(max_reward, reward)
            draft_value_map[card] = max_reward

        valid_scores = [reward for card, reward in draft_value_map.items() if card not in JACKS]
        max_non_jack_score = max(valid_scores, default=0)

        for card in state.board.draft:
            if card in TWO_EYED_JACKS:
                draft_value_map[card] = max(draft_value_map.get(card, 0), max_non_jack_score + 1)

        return draft_value_map

    def cal_reward_func(self, line_score, board_score, sequence_score):
        return SEQUENCE_WEIGHT * sequence_score ** SEQUENCE_EXPONENT + LINE_WEIGHT * line_score ** LINE_EXPONENT + BOARD_WEIGHT * board_score ** BOARD_EXPONENT

    def cal_two_eyed_jack_score(self, line_score, board_score, sequence_score):
        return self.cal_reward_func(line_score + 1, board_score + 30, sequence_score)

    def cal_one_eyed_jack_score(self, line_score, board_score, sequence_score):
        return self.cal_reward_func(line_score, board_score + 16, sequence_score)

# END FILE -----------------------------------------------------------------------------------------------------------#
