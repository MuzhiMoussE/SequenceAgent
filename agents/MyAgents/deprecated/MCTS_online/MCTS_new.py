import time
import random
from agents.MyAgents.MCTS_online.Node_MCTS import SingleAgentNode
from agents.MyAgents.MCTS_online.Constant_info_MCTS import K1, K2, K3, DISCOUNT_FACTOR


class MCTS:
    def __init__(self, mdp, qfunction, bandit):
        self.mdp = mdp
        self.qfunction = qfunction
        self.bandit = bandit

    """Execute the MCTS algorithm from the initial state given, with timeout in seconds"""
    def mcts(self, timeout=1, root_node=None):
        if root_node is None:
            root_node = self.create_root_node()

        start_time = time.time()
        current_time = time.time()
        while current_time < start_time + timeout:

            # Find a state node to expand
            selected_node = root_node.select()
            if not selected_node:
                break
                # selected_node = root_node
            if not self.mdp.isEnds(selected_node.state):

                child = selected_node.expand()
                if not child:
                    continue
                reward = self.simulate(child)
                selected_node.back_propagate(reward, child)

            current_time = time.time()

        return root_node

    """ Create a root node representing an initial state """
    def create_root_node(self): pass

    """ Choose a random action. Heustics can be used here to improve simulations. """
    def choose(self, state):
        return random.choice(self.mdp.getLegalActions(state, self.mdp.ownid))

    """ Simulate until a terminal state """
    def simulate(self, node):
        state = node.state
        cumulative_reward = 0.0
        depth = 0

        # change depth here, if needed. Initial there is no depth limit
        while not self.mdp.isEnds(state) and depth <=2:
            # Choose an action to execute
            action = self.choose(state)

            # Execute the action
            (next_state, reward_game, done) = self.mdp.execute(state, action)
            reward_line = self.qfunction.linereward(next_state, self.mdp.ownid)
            reward_board = self.qfunction.boardreward(next_state, self.mdp.ownid)
            reward = K1*reward_game + K2*reward_line + K3*reward_board

            # Discount the reward
            cumulative_reward += pow(DISCOUNT_FACTOR, depth) * reward
            depth += 1

            state = next_state

        return cumulative_reward


class SingleAgentMCTS(MCTS):
    def create_root_node(self):
        return_root_node = SingleAgentNode(
            self.mdp, None, self.mdp.initialGameState(), self.qfunction, self.bandit
        )
        # return_root_node.expand()
        return return_root_node
