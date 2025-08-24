from collections import defaultdict
import json
from agents.MyAgents.MCTS_online.Constant_info_MCTS import B1, B2, L0, L1, L2, EL1, EL2, ALPHA


class QFunction:
    def __init__(self, alpha=ALPHA, default_q_value=0.0):
        self.qtable = defaultdict(lambda: default_q_value)
        self.alpha = alpha

    def update(self, state, action, delta, agentid):
        self.qtable[(tuple(tuple(row) for row in state.board.chips), frozenset(action.items()))] = self.qtable[(tuple(tuple(row) for row in state.board.chips), frozenset(action.items()))] + self.alpha * delta

    # def update(self, state, action, delta, agentid):
    #     key_beforetransform = self.twoColorBoard(state, action, agentid)
    #     key_aftertransform=self.boardformer.get_canonical_form(key_beforetransform)
    #     self.qtable[key_aftertransform] = self.qtable[key_aftertransform] + self.alpha * delta

    def batch_update(self, states, actions, deltas):
        for state, action, delta in zip(states, actions, deltas):
            self.update(state, action, delta)

    def get_q_value(self, state, action, agentid):
        return self.qtable[(tuple(tuple(row) for row in state.board.chips),frozenset(action.items()))]

    # def get_q_value(self, state, action, agentid):
    #     key_beforetransform = self.twoColorBoard(state, action, agentid)
    #     key_aftertransform = self.boardformer.get_canonical_form(key_beforetransform)
    #     return self.qtable[key_aftertransform]

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

    def boardreward(self, nextstate, agentid):
        action_focus = nextstate.agents[agentid].last_action
        type_focus = action_focus.get('type')
        coords_focus = action_focus.get('coords')

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        my_count = 0
        opp_count = 0

        if type_focus == 'place' or type_focus == 'remove':
            for dx, dy in directions:
                for direction in [-1, 1]:
                    coords_x, coords_y = coords_focus[0], coords_focus[1]
                    for n in range(5):
                        x = coords_x + dx * direction * n
                        y = coords_y + dy * direction * n
                        if 0 <= x < 10 and 0 <= y < 10:
                            if nextstate.board.chips[x][y]  == nextstate.agents[agentid].colour or nextstate.board.chips[x][y] == nextstate.agents[agentid].seq_colour:
                                my_count += 1
                            elif nextstate.board.chips[x][y]  == nextstate.agents[agentid].opp_colour or nextstate.board.chips[x][y] == nextstate.agents[agentid].opp_seq_colour:
                                opp_count += 1
        else:
            my_count = 0

        return B1*my_count+B2*opp_count
        # return 0

    def linereward(self, nextstate, agentid):
        if nextstate.agents[agentid].score >= 2:
            return L0
        center_score = 0
        for w in [4,5]:
            for s in [4,5]:
                if nextstate.board.chips[w][s] == nextstate.agents[agentid].colour or nextstate.board.chips[w][s] == nextstate.agents[agentid].seq_colour:
                    center_score += 1
                if nextstate.board.chips[w][s] == nextstate.agents[agentid].opp_colour:
                    center_score -= 1
                if nextstate.board.chips[w][s] == nextstate.agents[agentid].opp_seq_colour:
                    center_score = 0
                    break
        center_score = max(0,center_score)

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 横、纵、正斜、反斜
        size = 10
        max_score = float('-inf')

        for x in range(size):
            for y in range(size):
                for dx, dy in directions:
                    score = self.eachlinereward(nextstate, agentid, x, y, dx, dy)
                    max_score = max(max_score, score)

        return L1*max_score+center_score*L2


    def eachlinereward(self, nextstate, agentid, x, y, dx, dy):
        count_me = 0
        count_opp = 0
        for i in range(5):
            nx, ny = x + i * dx, y + i * dy
            if not (0 <= nx < 10 and 0 <= ny < 10):
                return float('-inf')
            val = nextstate.board.chips[nx][ny]
            if val == nextstate.agents[agentid].colour or val =='#':
                count_me += 1
            elif val == nextstate.agents[agentid].opp_colour:
                count_opp += 1
            elif val == nextstate.agents[agentid].opp_seq_colour:
                return float('-inf')

        return EL1*count_me + EL2 * count_opp

