from collections import defaultdict
import random
from copy import deepcopy
from agents.MyAgents.helper.Util import removeActions

class Node:

    # Record a unique node id to distinguish duplicated states
    next_node_id = 0

    # Records the number of times states have been visited
    visits = defaultdict(lambda: 0)

    def __init__(self, mdp, parent, state, qfunction, bandit, reward=0.0, action=None):
        self.mdp = mdp
        self.parent = parent
        self.state = state
        self.id = Node.next_node_id
        Node.next_node_id += 1

        # The Q function used to store state-action values
        self.qfunction = qfunction

        # A multi-armed bandit for this node
        self.bandit = bandit

        # The immediate reward received for reaching this state, used for backpropagation
        self.reward = reward

        # The action that generated this node
        self.action = action

    """ Select a node that is not fully expanded """
    def select(self): pass

    """ Expand a node if it is not a terminal node """
    def expand(self): pass

    """ Backpropogate the reward back to the parent node """
    def back_propagate(self, reward, child): pass

    """ Return the value of this node """
    def get_value(self):
        max_q_value = self.qfunction.get_max_q(
            self.state, self.mdp.get_actions(self.state)
        )
        return max_q_value

    """ Get the number of visits to this state """
    def get_visits(self):
        return Node.visits[self.state]


class SingleAgentNode(Node):
    def __init__(
            self,
            mdp,
            parent,
            state,
            qfunction,
            bandit,
            reward=0.0,
            action=None,
    ):
        super().__init__(mdp, parent, state, qfunction, bandit, reward, action)

        # A dictionary from actions to a set of node-probability pairs
        self.children = {}

    """ Return true if and only if all child actions have been expanded """
    def is_fully_expanded(self):
        valid_actions = self.mdp.getLegalActions(self.state, self.mdp.ownid)
        if len(valid_actions) == len(self.children):
            return True
        else:
            return False

    """ Select a node that is not fully expanded """
    def select(self):
        if not self.is_fully_expanded() or self.mdp.isEnds(self.state):
            return self
        else:
            actions = list(self.children.keys())
            action = self.bandit.select(self.state, actions, self.qfunction, self.mdp.ownid)
            return self.get_outcome_child(dict(action)).select()

    """ Expand a node if it is not a terminal node """
    def expand(self):
        if not self.mdp.isEnds(self.state):
            # Randomly select an unexpanded action to expand
            actions = self.mdp.getLegalActions(self.state, self.mdp.ownid)
            target = self.children.keys()
            # actions = set(self.mdp.getLegalActions(self.state, self.mdp.ownid)) - set(self.children.keys())
            unexpanded_actions = removeActions(actions, target)
            if not unexpanded_actions:
                unexpanded_actions = actions
            action = random.choice(unexpanded_actions)
            self.children[frozenset(action.items())] = []

            return self.get_outcome_child(action)
        return self

    """ Backpropogate the reward back to the parent node """
    def back_propagate(self, reward, child):
        action = child.action

        Node.visits[self.state] = Node.visits[self.state] + 1
        #  Node.visits[(self.state, action)] = Node.visits[(self.state, action)] + 1
        Node.visits[(self.state, frozenset(action))] =  Node.visits[(self.state, frozenset(action))] + 1

        # q_value = self.qfunction.get_q_value(self.state, action)
        #if self.parent:
        #    if self.parent.id==1:
        # delta = (1 / (Node.visits[(self.state, frozenset(action))])) * (
        #         reward - self.qfunction.get_q_value(child.state.board.chips)
        # )
        delta = (1 / (Node.visits[(self.state, frozenset(action))])) * (
                reward - self.qfunction.get_q_value(self.state, action, self.mdp.ownid)
        )
        # self.qfunction.update(child.state.board.chips, delta)
        self.qfunction.update(self.state, action, delta, self.mdp.ownid)

        if self.parent:
            self.parent.back_propagate(self.reward + reward, self)

    """ Simulate the outcome of an action, and return the child node """
    def get_outcome_child(self, action):
        # Choose one outcome based on transition probabilities
        current_state = deepcopy(self.state)
        (next_state, reward, done) = self.mdp.execute(current_state, action)

        # Find the corresponding state and return if this already exists
        # for (child, _) in self.children[frozenset(action.items())]:
        #     if next_state == child.state:
        #         return child
        for (child, _) in self.children[frozenset(action.items())]:
            if next_state == child.state:
                return child

        # This outcome has not occurred from this state-action pair previously
        new_child = SingleAgentNode(
            self.mdp, self, next_state, self.qfunction, self.bandit, reward, action
        )

        # Find the probability of this outcome (only possible for model-based) for visualising tree
        # probability = 0.0
        # for (outcome, probability) in self.mdp.get_transitions(self.state, action):
        #     if outcome == next_state:
        #         self.children[frozenset(action.items())] += [(new_child, probability)]

        self.children[frozenset(action.items())] += [(new_child, 1)]
        return new_child