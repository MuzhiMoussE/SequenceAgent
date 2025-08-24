import traceback

from template import Agent
from agents.MyAgents.MCTS_online.Bandit_MCTS import UpperConfidenceBounds
from agents.MyAgents.MCTS_online.QFunction_MCTS import QFunction
from agents.MyAgents.MCTS_online.Sequence_rule_MCTS import SequenceGameRule
from agents.MyAgents.MCTS_online.MCTS_new import SingleAgentMCTS

FULL_TIME = 0.95
NUM_PLAYERS = 2

class myAgent(Agent):

    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id  # Agent needs to remember its own id.

    # Initialize needed variable
        self.mdp = SequenceGameRule(NUM_PLAYERS, _id)
        self.qfunction = QFunction()
        self.bandit = UpperConfidenceBounds()
        self.timeout = FULL_TIME

     # build SingleAgentMCTS, start mcts
        self.mcts_algorithm = SingleAgentMCTS(self.mdp, self.qfunction, self.bandit)

    def SelectAction(self, actions, game_state):
        try:
            """Use MCTS to choose one action"""
            root_node = self.mcts_algorithm.create_root_node()
            root_node.state = game_state

            # run MCTS
            root_node = self.mcts_algorithm.mcts(timeout=self.timeout, root_node=root_node)

            # Select best action（Based on Q）
            # best_action = max(root_node.children.keys(), key=lambda action: root_node.get_visits())
            best_action1 = max(root_node.children.items(), key=lambda item: self.qfunction.qtable[1])[0]
            best_action2 = max(
                root_node.children.items(),
                key=lambda item: self.qfunction.qtable[item[0]]
            )[0]
        except Exception as e:
            traceback.print_exc()

        return dict(best_action1)
