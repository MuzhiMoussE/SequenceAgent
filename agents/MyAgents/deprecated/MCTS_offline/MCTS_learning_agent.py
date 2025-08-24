from template import Agent
from agents.MyAgents.MCTS_offline.MCTS_learning import SingleAgentMCTS
from agents.MyAgents.MCTS_offline.Bandit_learning import UpperConfidenceBounds
from agents.MyAgents.MCTS_offline.Qtable_learning import QTable
from agents.MyAgents.MCTS_offline.Sequence_rule import SequenceGameRule

# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#

FULL_TIME = 1
NUM_PLAYERS = 2
FILENAME = "LearnQ"

# FUNCTIONS ----------------------------------------------------------------------------------------------------------#
class myAgent(Agent):

    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id  # Agent needs to remember its own id.

        # Initialize needed variable
        self.mdp = SequenceGameRule(NUM_PLAYERS, _id)
        self.qfunction = QTable()
        self.qfunction.load(FILENAME)    # load file, save file to other places if needed
        self.bandit = UpperConfidenceBounds()
        self.timeout = FULL_TIME

        # build SingleAgentMCTS, start mcts
        self.mcts_algorithm = SingleAgentMCTS(self.mdp, self.qfunction, self.bandit)

    def SelectAction(self, actions, game_state):
        # Use MCTS to choose one action
        root_node = self.mcts_algorithm.create_root_node()
        root_node.state = game_state

        # run MCTS
        root_node = self.mcts_algorithm.mcts(timeout=self.timeout, root_node=root_node)

        # Select best action（Based on Q）
        # best_action = max(root_node.children.keys(), key=lambda action: root_node.get_visits())
        best_action = max(root_node.children.items(), key=lambda item: self.qfunction.qtable[1])[0]

        return dict(best_action)

     # Currently it is not used, change agent file to have this function, and invoke it from game main process
    def SaveResult(self):
        self.qfunction.save(FILENAME)
