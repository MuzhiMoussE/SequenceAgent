K1 = 100
K2 = 1
K3 = 1
# Q_function alpha
ALPHA = 0.1
# DISCOUNT_FACTOR
DISCOUNT_FACTOR = 0.9

# limitation of runtime
TIME_LIMIT = 0.95
# number of players
NUM_PLAYER = 2

# discount factor of each step of actions
DISCOUNT_FACTOR = 0.9

# depth of simulation
DEPTH = 3
# weight And exponent
# parameters of reward function:
# reward = SEQUENCE_WEIGHT * sequence_score ** SEQUENCE_EXPONENT + LINE_WEIGHT * line_score ** LINE_EXPONENT + BOARD_WEIGHT * board_score ** BOARD_EXPONENT

# SEQUENCE_WEIGHT = 115
SEQUENCE_WEIGHT = 100
SEQUENCE_EXPONENT = 2

# LINE_WEIGHT = 0.9
LINE_WEIGHT = 1
LINE_EXPONENT = 2

# BOARD_WEIGHT = 1.15
BOARD_WEIGHT = 1
BOARD_EXPONENT = 1