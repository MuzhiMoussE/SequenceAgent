# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#

import time
from collections import deque
from copy import deepcopy

from agents.MyAgents.Fast_BFS.action_utils import DoAction, GetActions, checkSeq, \
    generate_combinations, random_action_selection, first_action_selection, find_critical_locations
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

        # try to stop the other agent
        important_loc, removable_locations = self.find_defense_move(game_state)
        if important_loc:
            # next_action = next((a for a in actions if a['play_card'] == important_loc), None)
            for action_item in actions:
                if action_item["action"] == "place" and action_item["coords"] in important_loc:
                    return action_item
        if removable_locations:
            action_map = {a['coords']: a for a in actions if a['type'] == 'remove'}
            for loc in removable_locations:
                if loc in action_map:
                    return action_map[loc]

        selected_action = self.fast_bfs(actions, game_state, start_time)

        return selected_action

    def find_defense_move(self, game_state):
        target_colour = game_state.agents[self.id].opp_colour
        target_colour_seq = game_state.agents[self.id].opp_seq_colour
        return find_critical_locations(game_state.board.chips, target_colour, target_colour_seq)


    def fast_bfs(self, actions, game_state, start_time):
        timer_record = 0
        queue = deque()
        last_action = None

        # timeout selection, try to get j when it is present
        for card in game_state.board.draft:
            if card == 'jc' or card == 'jd' :
                for action_item in actions:
                    if action_item['draft_card'] == card :
                        last_action = action_item
            elif card == 'jh' or card == 'js' :
                for action_item in actions:
                    if action_item['draft_card'] == card :
                        last_action = action_item

        for action in actions:
            draw = 1
            if ['type'] == 'trade':
                draw += 1
            queue.append((game_state, draw, [action]))

        while len(queue) and time.time() - start_time < (FULL_TIME - timer_record):
            timer_start = time.time()
            state, draw, path = queue.popleft()  # Pop the next node (state, path) in the queue.
            action = path[-1]

            # Carry out this action on the state, and check for goal
            draw -= 1
            next_state, goal = DoAction(self.id, deepcopy(state), action)

            if goal:
                return path[0]  # If the current action reached the goal, return the initial action that led there.
            elif draw > 0:
                for a in GetActions(next_state, self.id):
                    # Else, simply add this state and its path to the queue.
                    next_path = path + [a]
                    queue.append((next_state, draw, next_path))
            else:
                next_state.board.draft = []
                # check if this can reach goal by DFS
                # if goal cannot reach, continue loop without append queue
                if not self.is_goal_reachable_dfs(next_state):
                    continue
                # here means the path with the draw can win,
                # so restore the previous state and do BFS with the draw
                state.board.draft = [action['draft_card']]
                path.pop()
                queue = deque()
                for a in GetActions(state, self.id):
                    next_path = path + [a]
                    queue.append((state, 100, next_path))

            timer_record = max(time.time() - timer_start, timer_record)

        if last_action:
            return last_action
        else:
            if len(queue) == 0:
                return first_action_selection(actions)  # If no goal was found in the time limit, return first action.
            else:
                return random_action_selection(actions)  # If no goal was found in the time limit, return random action.

    def is_goal_reachable_dfs(self, state):
        # don't consider remove
        # filtered_list = [x for x in state.agents[self.id].hand if x not in {'jh', 'js'}]
        # skip if has ['jh', 'js'] two two-eyed jacks because it's too complex to check
        j_count = state.agents[self.id].hand.count('jc') + state.agents[self.id].hand.count('jd')
        # j_count = filtered_list.count('jc') + filtered_list.count('jd')
        # if j_count > 1:
        #     return False
        # use all non j cards
        combinations = generate_combinations(state.agents[self.id].hand, state.board.empty_coords)
        target_colour = state.agents[self.id].colour
        target_colour_seq = state.agents[self.id].seq_colour
        for coordinates in combinations:
            chips_copy = [row[:] for row in state.board.chips]
            for coordinate in coordinates:
                r, c = coordinate
                chips_copy[r][c] = target_colour
                _, seq_type = checkSeq(chips_copy, state.agents[self.id], coordinate)
                if seq_type:
                    return True

            # find and use two-eyed jacks by find_defense_move
            if j_count > 0:
                important_loc, _ = find_critical_locations(chips_copy, target_colour, target_colour_seq)
                if important_loc:
                    return True

        return False
# END FILE -----------------------------------------------------------------------------------------------------------#
