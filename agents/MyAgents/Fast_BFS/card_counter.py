from collections import Counter

class CardCounter:
    def __init__(self, my_id, enemy_id, game_state):
        self.my_id = my_id
        self.enemy_id = enemy_id
        self.my_action_count = 0
        self.enemy_action_count = 0
        self.draft = game_state.board.draft

        # init hand
        self.known_enemy_hand = []
        enemy_trace = game_state.agents[self.enemy_id].agent_trace.action_reward
        for action in enemy_trace[self.enemy_action_count:]:
            draft_card = action[0].get('draft_card')
            play_card = action[0].get('play_card')
            if draft_card:
                self.known_enemy_hand.append(draft_card)
            if play_card in self.known_enemy_hand:
                self.known_enemy_hand.remove(play_card)
            self.enemy_action_count += 1

        # init potential cards (deck + unknown hand)
        all_cards = [(r + s) for r in ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a'] for s in
                     ['d', 'c', 'h', 's']]
        # potential_cards = all cards - all hands - all draft - all discards
        # self.potential_cards = all_cards * 2
        # for card in self.known_enemy_hand:                  # 0 cards
        #     self.potential_cards.remove(card)
        # for card in game_state.agents[self.my_id].hand:     # 6 cards
        #     self.potential_cards.remove(card)
        # for card in game_state.board.draft:                 # 5 cards
        #     self.potential_cards.remove(card)
        # for card in game_state.deck.discards:               # 0 cards
        #     self.potential_cards.remove(card)
        # counter should faster
        known_card_counter = Counter(
            game_state.deck.discards +
            game_state.board.draft +
            game_state.agents[self.my_id].hand +
            self.known_enemy_hand
        )
        # convert potential_cards to counter
        potential_counter = Counter(all_cards * 2)
        # remove known cards
        potential_counter.subtract(known_card_counter)
        # construct potential_cards with card count > 0
        self.potential_cards = [
            card
            for card, count in potential_counter.items()
            for _ in range(max(0, count))
        ]


    def update(self, game_state):
        # known potential cards during this round = enemy played card from unknown hand + all drawn cards + all current draft - last draft
        delta_potential_cards = []

        # process enemy actions
        enemy_start_index = self.enemy_action_count
        enemy_trace = game_state.agents[self.enemy_id].agent_trace.action_reward
        for action in enemy_trace[enemy_start_index:]:
            # ({'coords': None, 'draft_card': None, 'play_card': None, 'type': 'trade'}, 0) Be careful! This is a legal operation!!
            draft_card = action[0].get('draft_card')
            play_card = action[0].get('play_card')
            if draft_card:
                self.known_enemy_hand.append(draft_card)
                delta_potential_cards.append(draft_card)
            if play_card in self.known_enemy_hand:
                self.known_enemy_hand.remove(play_card)
            elif play_card:
                delta_potential_cards.append(play_card)
            self.enemy_action_count += 1

        # process our actions
        my_start_index = self.my_action_count
        my_trace = game_state.agents[self.my_id].agent_trace.action_reward
        for action in my_trace[my_start_index:]:
            draft_card = action[0].get('draft_card')
            if draft_card:
                delta_potential_cards.append(draft_card)
            self.my_action_count += 1

        # add all current draft
        for card in game_state.board.draft:
            delta_potential_cards.append(card)
        # remove all previous draft
        for card in self.draft:
            delta_potential_cards.remove(card)

        # update potential_cards
        for card in delta_potential_cards:
            if card:
                self.potential_cards.remove(card)

        self.draft = game_state.board.draft
