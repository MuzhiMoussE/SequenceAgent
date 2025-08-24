# add detection to normal BFS

# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#

import time
from copy import deepcopy
from collections import deque
from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule
from agents.MyAgents.helper.Search_rule import PlanningGameRule
from agents.MyAgents.helper.Util import find_defense_move

FULL_TIME = 1
NUM_PLAYERS = 2

# FUNCTIONS ----------------------------------------------------------------------------------------------------------#

# Defines this agent.
class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id  # Agent needs to remember its own id.
        self.game_rule = GameRule(NUM_PLAYERS)  # Agent stores an instance of GameRule, from which to obtain functions.
        # More advanced agents might find it useful to not be bound by the functions in GameRule, instead executing
        # their own custom functions under GetActions and DoAction.

    # Generates actions from this state.
    def GetActions(self, state):
        return self.game_rule.getLegalActions(state, self.id)

    # Carry out a given action on this state and return True if goal is reached received.
    def DoAction(self, receive_state, action):
        score = receive_state.agents[self.id].score
        planner = PlanningGameRule()
        possible_state = planner.generateSuccessor(receive_state, action, self.id)
        new_score = possible_state.agents[self.id].score
        if new_score > score:
            goal_reached = True
        else:
            goal_reached = False
        return goal_reached

    # Take a list of actions and an initial state, and perform breadth-first search within a time limit.
    # Return the first action that leads to goal, if any was found.
    def SelectAction(self, actions, rootstate):
        start_time = time.time()
        queue = deque([(deepcopy(rootstate), [])])  # Initialise queue. First node = root state and an empty path.
        timer_record = 0

        important_loc = find_defense_move(rootstate, self.id)  # try to stop the other player
        if important_loc:
            next_action = next((a for a in actions if a['play_card'] == important_loc), None)
            if next_action:
                return next_action

        # Conduct BFS starting from root state.
        while len(queue) and time.time() - start_time < (FULL_TIME - timer_record):
            timer_start = time.time()
            state, path = queue.popleft()  # Pop the next node (state, path) in the queue.
            new_actions = self.GetActions(state)  # Obtain new actions available to the agent in this state.

            for a in new_actions:  # Then, for each of these actions...
                next_state = deepcopy(state)  # Copy the state.
                next_path = path + [a]  # Add this action to the path.

                goal_reachable = self.DoAction(next_state, a)  # Carry out this action on the state, and check for goal

                if goal_reachable:
                    return next_path[
                        0]  # If the current action reached the goal, return the initial action that led there.
                else:
                    queue.append((next_state, next_path))  # Else, simply add this state and its path to the queue.

            timer_record = max(time.time() - timer_start,timer_record)
        return actions[0]  # If no goal was found in the time limit, return the first move

# END FILE -----------------------------------------------------------------------------------------------------------#