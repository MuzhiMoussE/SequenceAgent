# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Purpose: Implements an weighted ASTAR search agent for the COMP90054 competitive game environment.
# change timeout choice, try the action with best cost+weight*weight when not found

# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#

import time, random
import heapq
from itertools import count
from copy import deepcopy
from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule
from agents.MyAgents.helper.Heuristic_rule import HeuristicGameRule

FULL_TIME = 1
NUM_PLAYERS = 2
WEIGHT = 1
STEP_COST = 1

# FUNCTIONS ----------------------------------------------------------------------------------------------------------#

# Defines this agent.
class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id  # Agent needs to remember its own id.
        self.game_rule = GameRule(NUM_PLAYERS)  # Agent stores an instance of GameRule, from which to obtain functions.
        self.planner = HeuristicGameRule()

        # More advanced agents might find it useful to not be bound by the functions in GameRule, instead executing
        # their own custom functions under GetActions and DoAction.

    # Generates actions from this state.
    def GetActions(self, state):
        return self.game_rule.getLegalActions(state, self.id)

    # Carry out a given action on this state and return True if goal is reached received.
    def DoAction(self, state, action,heuristic_matrix):
        possible_state, heuristic, heuristic_matrix = self.planner.generateSuccessor(state, action, self.id, heuristic_matrix)
        return heuristic, heuristic_matrix

    # Take a list of actions and an initial state, and perform Astar search within a time limit.
    def SelectAction(self, actions, rootstate):
        start_time = time.time()
        counter = count()
        my_queue = []
        heapq.heappush(my_queue, (0, next(counter), (deepcopy(rootstate), [], float('inf'), self.planner.heuristic_board)))
        best_g = dict()
        timer_record = 0
        first_indicate = True  # choose an action when timeout
        timeout_action = None

        while len(my_queue) and time.time() - start_time < (FULL_TIME - timer_record):  # to avoid timeout
            timer_start = time.time()
            priority, count_id, queue_value= heapq.heappop(my_queue)    # Pop the next node (cost+weighted heuristic, counter, (state, path, cost, heuristic matrix)) in the queue.
            state, path, existing_cost, heuristic_matrix = queue_value  # heuristic matrix is saved to save time, although it cost more on memory
            if (not state in best_g) or (priority < best_g[state]):
                best_g[state] = priority
                new_actions = self.GetActions(state)  # Obtain new actions available to the agent in this state.
                for a in new_actions:
                    next_state = deepcopy(state)  # Copy the state.
                    next_path = path + [a]  # Add this action to the path.
                    heuristic_next, heuristic_matrix_next = self.DoAction(next_state, a, heuristic_matrix)  # Carry out this action on the state, and return heuristic
                    if heuristic_next==0:
                        # save the heuristic board that is only one step taken, for use in next SelectAction()
                        _, self.planner.heuristic_board = self.DoAction(deepcopy(rootstate), next_path[0], self.planner.heuristic_board)

                        return next_path[
                            0]  # If the current action reached the goal, return the initial action that led there.
                    else:
                        if first_indicate:
                            # choose the best h node when timeout, assuming this is the best action
                            timeout_action = next_path[0]
                            first_indicate = False
                        heapq.heappush(my_queue, (existing_cost+STEP_COST+WEIGHT*heuristic_next, next(counter), (next_state, next_path, existing_cost+STEP_COST, heuristic_matrix_next )))  # Else, simply add this state and its path to the queue.

            timer_record = max(time.time() - timer_start,timer_record)
        if timeout_action:
            timeout_action = actions[0]
        _, self.planner.heuristic_board = self.DoAction(deepcopy(rootstate), timeout_action, self.planner.heuristic_board)
        return timeout_action  # If no goal was found in the time limit, return a timeout action.

# END FILE -----------------------------------------------------------------------------------------------------------#