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
