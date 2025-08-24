import copy
from Sequence.sequence_model import SequenceGameRule
from Sequence.sequence_utils import EMPTY, JOKER

"""
    Ignore card draw entirely vs keeping track of which cards are probably left in the deck
"""
class DeterministicGameRule(SequenceGameRule):

    def generateSuccessor(self, state, action, agent_id):
        new_state = copy.deepcopy(state)
        new_state.board.new_seq = False
        # print(f"agent id {agent_id}")
        plr_state = new_state.agents[agent_id]
        plr_state.last_action = action  # Record last action such that other agents can make use of this information.
        reward = 0

        # Update agent state. Take the card in play from the agent, discard, draw the selected draft, deal a new draft.
        # If agent was allowed to trade but chose not to, there is no card played, and hand remains the same.
        card = action['play_card']
        draft = action['draft_card']
        if card:
            plr_state.hand.remove(card)  # Remove card from hand.
            plr_state.discard = card  # Add card to discard pile.
            new_state.deck.discards.append(
                card)  # Add card to global list of discards (some agents might find tracking this helpful).
            new_state.board.draft.remove(draft)  # Remove draft from draft selection.
            plr_state.hand.append(draft)  # Add draft to player hand.
            # Ignore card draw entirely
            # state.board.draft.extend(state.deck.deal())  # Replenish draft selection.

        # If action was to trade in a dead card, action is complete, and agent gets to play another card.
        if action['type'] == 'trade':
            plr_state.trade = True  # Switch trade flag to prohibit agent performing a second trade this turn.
            plr_state.agent_trace.action_reward.append((action, reward))  # Log this turn's action and score (zero).
            return new_state

        # Update Sequence board. If action was to place/remove a marker, add/subtract it from the board.
        r, c = action['coords']
        if action['type'] == 'place':
            new_state.board.chips[r][c] = plr_state.colour
            new_state.board.empty_coords.remove(action['coords'])
            new_state.board.plr_coords[plr_state.colour].append(action['coords'])
        elif action['type'] == 'remove':
            new_state.board.chips[r][c] = EMPTY
            new_state.board.empty_coords.append(action['coords'])
            new_state.board.plr_coords[plr_state.opp_colour].remove(action['coords'])
        else:
            print("Action unrecognised.")

        # Check if a sequence has just been completed. If so, upgrade chips to special sequence chips.
        if action['type'] == 'place':
            seq, seq_type = self.checkSeq(new_state.board.chips, plr_state, (r, c))
            if seq:
                reward += seq['num_seq']
                new_state.board.new_seq = seq_type
                for sequence in seq['coords']:
                    for r, c in sequence:
                        if new_state.board.chips[r][c] != JOKER:  # Joker spaces stay jokers.
                            new_state.board.chips[r][c] = plr_state.seq_colour
                            try:
                                new_state.board.plr_coords[plr_state.colour].remove(action['coords'])
                            except:  # Chip coords were already removed with the first sequence.
                                pass
                plr_state.completed_seqs += seq['num_seq']
                plr_state.seq_orientations.extend(seq['orientation'])

        plr_state.trade = False  # Reset trade flag if agent has completed a full turn.
        plr_state.agent_trace.action_reward.append((action, reward))  # Log this turn's action and any resultant score.
        plr_state.score += reward
        return new_state