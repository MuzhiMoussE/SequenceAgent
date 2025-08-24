# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#

import time
import traceback
from collections import deque
from copy import deepcopy

from agents.MyAgents.Fast_BFS.Parameter_info import K1, K2, K3, DISCOUNT_FACTOR
from agents.MyAgents.Fast_BFS.action_utils import DoAction, GetActions, checkSeq, \
    generate_combinations, random_action_selection, first_action_selection, unique_actions, find_critical_locations
from agents.MyAgents.MCTS_online.Evaluator_MCTS import Evaluator
from agents.MyAgents.helper.Constant_info import *
from template import Agent

FULL_TIME = 0.95
NUM_PLAYERS = 2


# Defines this agent.
class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        # More advanced agents might find it useful to not be bound by the functions in GameRule, instead executing
        # their own custom functions under GetActions and DoAction.

    # Take a list of actions and an initial state, and perform breadth-first search within a time limit.
    # Return the first action that leads to goal, if any was found.
    def SelectAction(self, actions, game_state):
        start_time = time.time()

        evaluator = Evaluator(self.id, game_state)
        draft_value_map = self.cal_draft_score(game_state, evaluator)
        max_draft = max(draft_value_map, key=draft_value_map.get)

        important_loc, removable_locations = self.find_defense_move(game_state)
        if important_loc:
            next_action = next((a for a in actions if a['play_card'] in important_loc and a['draft_card'] == max_draft), None)
            if next_action:
                return next_action
        if removable_locations:
            action_map = {a['coords']: a for a in actions if a['type'] == 'remove' and a['draft_card'] == max_draft}
            for loc in removable_locations:
                if loc in action_map:
                    return action_map[loc]

        is_goal_reached, draft, goal_important_loc = self.is_goal_reachable_dfs2(game_state)

        if is_goal_reached:
            if not draft:
                draft = max_draft
            game_state.board.draft = [draft]
            return self.fast_bfs(game_state, goal_important_loc, start_time)

        return self.evaluate_by_reward_func(actions, game_state, evaluator, start_time)

    def find_defense_move(self, game_state):
        target_colour = game_state.agents[self.id].opp_colour
        target_colour_seq = game_state.agents[self.id].opp_seq_colour
        return find_critical_locations(game_state.board.chips, target_colour, target_colour_seq)

    def evaluate_by_reward_func(self, actions, game_state, evaluator, start_time):
        # changed function structure, currently is correct and can save more space
        # play_card -> evaluator
        evaluator_map = dict()
        max_score = 0
        best_action = None

        try:
            ua = unique_actions(actions)
            # queue = deque()
            timer_record = 0
            timer_start = time.time()
            for action in ua:
                if time.time() - start_time < (FULL_TIME - timer_record):
                    timer_start = time.time()
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
                        # max_draft = max(draft_card_reward, key=draft_card_reward.get)
                        # max_reward = draft_card_reward.get(max_draft)
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
        except Exception as e:
            traceback.print_exc()
        return first_action_selection(actions)

        # queue = deque()
        # for action in ua:
        #     queue.append((game_state, [action]))
        #
        # timer_record = 0
        # node_count = 0
        #
        # while len(queue) and time.time() - start_time < (FULL_TIME - timer_record):
        #     node_count += 1
        #     timer_start = time.time()
        #     state, path = queue.popleft()  # Pop the next node (state, path) in the queue.
        #     action = path[-1]
        #
        #     play_card = action['play_card']
        #     evaluator_tuple = evaluator_map.get(play_card, None)
        #     if not evaluator_tuple:
        #         node_evaluator = evaluator.get_deepcopy()
        #         line_score, board_score, sequence_score = node_evaluator.update_by_action(action)
        #         score = self.cal_reward_func(line_score, board_score, sequence_score)
        #         evaluator_tuple = (node_evaluator, score)
        #         evaluator_map[play_card] = evaluator_tuple
        #
        #     draft_card = action['draft_card']
        #     node_evaluator = evaluator_tuple[0]
        #     root_line_score = node_evaluator.get_line_score()
        #     root_board_score = node_evaluator.get_board_score()
        #     root_sequence_score = node_evaluator.sequence_score
        #     if draft_card in TWO_EYED_JACKS:  # two-eyed jacks
        #         max_reward = self.cal_two_eyed_jack_score(root_line_score, root_board_score, root_sequence_score)
        #     elif draft_card in ONE_EYED_JACKS:  # one-eyed jacks
        #         max_reward = self.cal_one_eyed_jack_score(root_line_score, root_board_score, root_sequence_score)
        #     else:
        #         max_notj_reward = 0
        #         for r, c in COORDS[draft_card]:
        #             if state.board.chips[r][c] != EMPTY:
        #                 continue
        #             mock_action = {'type': 'place', 'coords': (r, c)}
        #             line_score, board_score, sequence_score = node_evaluator.get_deepcopy().update_by_action(
        #                 mock_action)
        #             reward = self.cal_reward_func(line_score, board_score, sequence_score)
        #             max_notj_reward = max(max_notj_reward, reward)
        #         max_reward = max_notj_reward
        #
        #     score = evaluator_tuple[1] + max_reward * DISCOUNT_FACTOR * DISCOUNT_FACTOR
        #
        #     if max_score < score:
        #         max_score = score
        #         best_action = action
        #
        #     timer_record = max(time.time() - timer_start, timer_record)
        #
        # # print(f"Fast BFS 3 id {self.id} evaluate node count = {node_count}, time consume = {time.time() - start_time:.3f} left node = {len(queue)}")
        # if best_action:
        #     return best_action
        #
        # # print("evaluate_by_reward_func fallback!!!")
        # return first_action_selection(actions)

    def fast_bfs(self, game_state, goal_important_loc, start_time):
        actions = GetActions(game_state, self.id)
        ua = unique_actions(actions)
        queue = deque()
        default_acton = actions[0]
        for action in ua:
            if goal_important_loc and action['coords'] not in goal_important_loc:
                continue
            queue.append((game_state, [action]))
            # default_acton = action

        if queue:
            last_state, action_list = queue[0]
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

    def is_goal_reachable_dfs2(self, state):
        # check goal could be reached with which draft
        hand = state.agents[self.id].hand
        # use all non j cards
        combinations = generate_combinations(hand, state.board.empty_coords)
        color = state.agents[self.id].colour
        color_seq = state.agents[self.id].seq_colour
        # check hand cards
        j_count = hand.count('jc') + hand.count('jd')
        for coordinates in combinations:
            chips_copy = [row[:] for row in state.board.chips]
            for coordinate in coordinates:
                r, c = coordinate
                chips_copy[r][c] = color
                seq, seq_type = checkSeq(chips_copy, state.agents[self.id], coordinate)
                if seq_type:
                    return True, None, set([item for row in seq['coords'] for item in row])

            # find and use two-eyed jacks by find_defense_move
            if j_count > 0:
                important_loc, _ = find_critical_locations(chips_copy, color, color_seq)
                if important_loc and len(important_loc) <= j_count:
                    return True, None, set(coordinates + important_loc)

            # check one draft cards
            for draft in state.board.draft:
                for r, c in COORDS[draft]:
                    chips_copy = [row[:] for row in chips_copy]
                    chips_copy[r][c] = color
                    seq, seq_type = checkSeq(chips_copy, state.agents[self.id], (r, c))
                    if seq_type:
                        return True, draft, set([item for row in seq['coords'] for item in row] + [(r, c)])

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
        return draft_value_map

    def cal_reward_func(self, line_score, board_score, sequence_score):
        return K1 * sequence_score + K2 * line_score * line_score + K3 * board_score

    def cal_two_eyed_jack_score(self, line_score, board_score, sequence_score):
        return self.cal_reward_func(line_score + 1, board_score + 20, sequence_score)

    def cal_one_eyed_jack_score(self, line_score, board_score, sequence_score):
        return self.cal_reward_func(line_score, board_score + 16, sequence_score)

# END FILE -----------------------------------------------------------------------------------------------------------#
