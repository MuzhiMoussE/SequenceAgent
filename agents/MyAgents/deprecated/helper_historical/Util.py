class Util:
    def __init__(self):
        pass

def removeActions(actions, targets):
    target_set = set(targets)
    remaining = [
        action for action in actions
        if frozenset(action.items()) not in target_set
    ]
    return remaining

def find_defense_move(rootstate, ownid):
    # A better find_defense_move in changed BFS2, here no more optimization
    # This is used in BFS
    # here is not correct with the else logic, as there are more than three option in board
    directions = [ (0, 1), (1, 0), (1, 1), (1, -1) ]
    size = 10

    for x in range(size):
        for y in range(size):
            for dx, dy in directions:
                count = 0
                empty_pos = None

                for i in range(5):
                    nx, ny = x + i*dx, y + i*dy
                    if not (0 <= nx < size and 0 <= ny < size):
                        break
                    check_location = rootstate.board.chips[nx][ny]
                    if check_location == rootstate.agents[ownid].opp_colour:
                        count += 1
                    elif rootstate.board.chips[nx][ny]=='_':
                        if empty_pos is None:
                            empty_pos = (nx, ny)
                        else:
                            break
                    else:
                        break
                if count == 4 and empty_pos:
                    return empty_pos

    return None
