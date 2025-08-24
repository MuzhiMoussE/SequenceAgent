from agents.MyAgents.MCTS_online.Constant_info_MCTS import *

class Evaluator:
    def __init__(self, agent_id, state):
        self.agent_id = agent_id
        plr_state = state.agents[agent_id]
        self.clr, self.sclr = plr_state.colour, plr_state.seq_colour
        self.oc, self.os = plr_state.opp_colour, plr_state.opp_seq_colour
        self.chips = [row[:] for row in state.board.chips]
        self.board_scorer = [[10 for _ in range(10)] for _ in range(10)]
        self.sequence_score = 0
        self.board_count = 456
        self.line_steps = 10
        self.action_line_steps = 10
        self.init_reward()

    def init_reward(self):
        for r, c in JOKER_COORDINATES:
            self.chips[r][c] = self.clr

        for r in range(10):
            self.process_line(r, 0, SLIDE_DIRECTION_HORIZONTAL)
        for c in range(10):
            self.process_line(0, c, SLIDE_DIRECTION_VERTICAL)
        for r in range(6):
            self.process_line(r, 0, SLIDE_DIRECTION_DIAGONAL)
        for c in range(1, 6):
            self.process_line(0, c, SLIDE_DIRECTION_DIAGONAL)
        for r in range(6):
            self.process_line(r, 9, SLIDE_DIRECTION_ANTI_DIAGONAL)
        for c in range(4, 9):
            self.process_line(0, c, SLIDE_DIRECTION_ANTI_DIAGONAL)

        self.process_line(0, 0, SLIDE_DIRECTION_CENTER, center=True)

        self.sequence_score = self.init_sequence_score()
        self.cal_board_count()

    def process_line(self, r0, c0, direction, center=False):
        if center:
            window = [(r0 + dr, c0 + dc) for dr, dc in direction]
            self.evaluate_window(window, is_center=True)
            return

        d_len = len(direction)
        for i in range(d_len - 4):
            window = []
            valid = True
            for j in range(5):
                dr, dc = direction[i + j]
                r, c = r0 + dr, c0 + dc
                if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                    valid = False
                    break
                window.append((r, c))
            if valid:
                self.evaluate_window(window)

    def evaluate_window(self, window, is_center=False):
        score = 0
        has_clr = False
        blocked = False
        sclr_count = 0

        for r, c in window:
            val = self.chips[r][c]
            if val == EMPTY:
                score += 1
            elif val == self.clr:
                has_clr = True
            elif val == self.sclr:
                sclr_count += 1
            elif val == self.oc:
                score += 2
            elif val == self.os:
                blocked = True
                break

        if blocked or sclr_count > 1:
            for r, c in window:
                self.board_scorer[r][c] = 10
            self.line_steps = min(self.line_steps, 10)
        else:
            if is_center and not has_clr:
                score += 1
            for r, c in window:
                cur = self.board_scorer[r][c]
                if self.chips[r][c] == self.clr:
                    self.board_scorer[r][c] = min(cur, score + 1)
                else:
                    self.board_scorer[r][c] = min(cur, score)
            self.line_steps = min(self.line_steps, score)
            self.action_line_steps = min(self.action_line_steps, score)
            if score == 0:
                # A state that can be initialized cannot have uncounted sequences
                # Therefore, it can be directly calculated to quickly update the action
                self.sequence_score += 1

    def init_sequence_score(self):
        # A state that can be initialized cannot have a sequence that can end the game
        # Therefore, it is only necessary to check whether there is a corresponding sequence color existing
        # Note that this method is not applicable to initializing the terminal state!
        return int(any(self.chips[r][c] == self.sclr for r in range(10) for c in range(10)))

    def cal_board_count(self):
        self.board_count = sum(sum(row) for row in self.board_scorer)

    def get_board_score(self):
        return MAX_BOARD_SCORE - self.board_count

    def get_line_score(self):
        return MAX_LINE_SCORE - self.action_line_steps

    def get_deepcopy(self):
        new_evaluator = object.__new__(Evaluator)  # no __init__ , empty object

        new_evaluator.agent_id = self.agent_id
        new_evaluator.clr = self.clr
        new_evaluator.sclr = self.sclr
        new_evaluator.oc = self.oc
        new_evaluator.os = self.os

        new_evaluator.chips = [row[:] for row in self.chips]
        new_evaluator.board_scorer = [row[:] for row in self.board_scorer]

        new_evaluator.board_count = self.board_count
        new_evaluator.line_steps = self.line_steps
        new_evaluator.sequence_score = self.sequence_score
        new_evaluator.action_line_steps = self.action_line_steps

        return new_evaluator

    def update_by_action(self, action):
        action_type = action['type']
        if action_type != PLACE and  action_type != REMOVE:
            return MAX_LINE_SCORE - self.line_steps, self.get_board_score(), self.sequence_score

        coord = action['coords']
        r, c = coord
        if action_type == PLACE:
            self.chips[r][c] = self.clr
        elif action_type == REMOVE:
            self.chips[r][c] = EMPTY
            self.board_scorer[r][c] -= 1

        # re-calculate
        self.action_line_steps = 10
        if 4 <= r <= 5 and 4 <= c <= 5:
            self.process_line(0, 0, SLIDE_DIRECTION_CENTER, center=True)
        self.evaluate_local_windows(r, c)

        return self.get_line_score(), self.get_board_score() + self.distance_to_edge(r,c), self.sequence_score

    def distance_to_edge(self, r, c):
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return 0

        top = r
        bottom = BOARD_SIZE - 1 - r
        left = c
        right = BOARD_SIZE - 1 - c

        return min(top, bottom, left, right)

    def evaluate_local_windows(self, r, c):
        for direction in SLIDE_DIRECTION_UPDATE_POINT:
            for i in range(len(direction) - 4):
                window = []
                valid = True
                for j in range(5):
                    dr, dc = direction[i + j]
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE):
                        valid = False
                        break
                    window.append((rr, cc))
                if valid:
                    self.evaluate_window(window)

