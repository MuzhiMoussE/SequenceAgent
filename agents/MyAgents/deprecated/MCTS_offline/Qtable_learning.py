from collections import defaultdict
import json
from agents.MyAgents.MCTS_offline.Board_former import BoardGeneralize


class QTable:
    def __init__(self, alpha=0.1, default_q_value=0.0):
        self.qtable = defaultdict(lambda: default_q_value)
        # self.load(FILENAME)
        self.alpha = alpha
        self.boardformer = BoardGeneralize()

    # def update(self, state, action, delta):
    #     self.qtable[(tuple(tuple(row) for row in state.board.chips), frozenset(action.items()))] = self.qtable[(tuple(tuple(row) for row in state.board.chips), frozenset(action.items()))] + self.alpha * delta

    def update(self, state, action, delta, agentid):
        key_beforetransform = self.twoColorBoard(state, action, agentid)
        key_aftertransform=self.boardformer.get_canonical_form(key_beforetransform)
        self.qtable[key_aftertransform] = self.qtable[key_aftertransform] + self.alpha * delta

    def batch_update(self, states, actions, deltas):
        for state, action, delta in zip(states, actions, deltas):
            self.update(state, action, delta)

    # def get_q_value(self, state, action):
    #     return self.qtable[(tuple(tuple(row) for row in state.board.chips),frozenset(action.items()))]

    def get_q_value(self, state, action, agentid):
        key_beforetransform = self.twoColorBoard(state, action, agentid)
        key_aftertransform = self.boardformer.get_canonical_form(key_beforetransform)
        return self.qtable[key_aftertransform]

    def get_q_values(self, states, actions):
        return [self.get_q_value(state, action) for state, action in zip(states, actions)]

    def save(self, filename):
        with open(filename, "w") as file:
            serialised = {str(key): value for key, value in self.qtable.items()}
            json.dump(serialised, file)

    def load(self, filename, default=0.0):
        try:
            with open(filename, "r") as file:
                serialised = json.load(file)
                self.qtable = defaultdict(
                    lambda: default,
                    serialised
                )
        except FileNotFoundError:
            # print(f"File {filename} not found. Creating an empty qtable.")
            self.qtable = defaultdict(lambda: default)

    def twoColorBoard(self, state, action, agentid):
        old_board = state.board.chips
        self_color = state.agents[agentid].colour
        self_color_seq = state.agents[agentid].seq_colour
        opp_color = state.agents[agentid].opp_colour
        opp_color_seq = state.agents[agentid].opp_seq_colour

        new_board = [
            ['A' if cell == self_color or cell == self_color_seq else 'B' if cell == opp_color or cell == opp_color_seq else cell for cell in row]
            for row in old_board
        ]

        if action.get('type') == 'place':
            new_board[action.get('coords')[0]][action.get('coords')[1]] = 'A'
        elif action.get('type') =='remove':
            new_board[action.get('coords')[0]][action.get('coords')[1]] = '-'

        return new_board

