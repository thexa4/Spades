import torch

from gamestate import GameState

def run_game(batch, models, should_print=False):
    base_card_values = torch.tensor([[
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, # spades 2-A
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,          # hearts 2-A
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,          # clubs: 2-A
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,          # diamonds: 2-A
    ]]).repeat(batch, 1)

    spades_mask = torch.tensor([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # spades 2-A
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # hearts 2-A
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # clubs 2-A
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # diamonds 2-A
    ])
    hearts_mask = torch.tensor([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # spades 2-A
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # hearts 2-A
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # clubs 2-A
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # diamonds 2-A
    ])
    clubs_mask = torch.tensor([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # spades 2-A
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # hearts 2-A
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # clubs 2-A
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # diamonds 2-A
    ])
    diamonds_mask = torch.tensor([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # spades 2-A
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # hearts 2-A
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # clubs 2-A
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # diamonds 2-A
    ])
    pending_samples = []

    deck = torch.stack([torch.randperm(52) for _ in range(batch)])

    decks = [
        deck[:, 0:13],
        deck[:, 13:26],
        deck[:, 26:39],
        deck[:, 39:52]
    ]

    hands = [
        torch.zeros(batch, 52, dtype=torch.uint8),
        torch.zeros(batch, 52, dtype=torch.uint8),
        torch.zeros(batch, 52, dtype=torch.uint8),
        torch.zeros(batch, 52, dtype=torch.uint8)
    ]
    
    for i in range(4):
        hands[i][torch.arange(batch).unsqueeze(1), decks[i]] = 1
    
    team_scores = [
        torch.stack([torch.randint(-20, 50, (batch,)), torch.randint(0, 10, (batch,))], dim=1),
        torch.stack([torch.randint(-20, 50, (batch,)), torch.randint(0, 10, (batch,))], dim=1)
    ]
    hands_won = [
        torch.zeros(batch, 13, dtype=torch.uint8),
        torch.zeros(batch, 13, dtype=torch.uint8),
        torch.zeros(batch, 13, dtype=torch.uint8),
        torch.zeros(batch, 13, dtype=torch.uint8)
    ]

    spades_played = torch.zeros(batch, dtype=torch.uint8)

    cards_played = [
        torch.zeros(batch, 13, 52, dtype=torch.uint8),
        torch.zeros(batch, 13, 52, dtype=torch.uint8),
        torch.zeros(batch, 13, 52, dtype=torch.uint8),
        torch.zeros(batch, 13, 52, dtype=torch.uint8)
    ]

    cards_seen = [
        hands[0],
        hands[1],
        hands[2],
        hands[3]
    ]

    player_bids = [
        torch.zeros(batch, dtype=torch.uint8),
        torch.zeros(batch, dtype=torch.uint8),
        torch.zeros(batch, dtype=torch.uint8),
        torch.zeros(batch, dtype=torch.uint8),
    ]

    player_wins = [
        torch.zeros(batch, dtype=torch.uint8),
        torch.zeros(batch, dtype=torch.uint8),
        torch.zeros(batch, dtype=torch.uint8),
        torch.zeros(batch, dtype=torch.uint8)
    ]

    initiative_player = torch.randint(0, 4, (batch,))
    if should_print:
        print(f"player initiatives: {initiative_player}")

    combined_bid_state = GameState(mask=torch.zeros((batch,)))
    combined_bids = torch.zeros((batch,)).to(torch.uint8)
    for initiative_offset in range(4):
        for player in range(4):
            player_should_play = torch.eq(initiative_player, (player + initiative_offset) % 4).to(torch.uint8)

            wrapped_player = player % 4
            left_player = (wrapped_player - 1 + 4) % 4
            right_player = (wrapped_player + 1) % 4
            teammate_player = (wrapped_player + 2) % 4
            team_id = wrapped_player % 2

            state = GameState(
                my_hand = hands[wrapped_player],
                spades_played = spades_played,
                team_bid = player_bids[teammate_player],
                other_team_bid = torch.clamp(player_bids[left_player] + player_bids[right_player], 0, 13),
                bid_mine = player_bids[wrapped_player],
                bid_teammate = player_bids[teammate_player],
                bid_left = player_bids[left_player],
                bid_right = player_bids[right_player],
                cards_played_me = cards_played[wrapped_player],
                cards_played_teammate = cards_played[teammate_player],
                cards_played_left = cards_played[left_player],
                cards_played_right = cards_played[right_player],
                team_score_mine = team_scores[team_id],
                team_score_other = team_scores[1 - team_id],
                hands_won_mine = hands_won[wrapped_player],
                hands_won_teammate = hands_won[teammate_player],
                hands_won_left = hands_won[left_player],
                hands_won_right = hands_won[right_player],
                players_left = torch.full((batch,), 3 - player),
                hand_number = torch.full((batch,), 0),
                trick_wins_me = player_wins[wrapped_player],
                trick_wins_teammate = player_wins[teammate_player],
                trick_wins_left = player_wins[left_player],
                trick_wins_right = player_wins[right_player],
                cards_seen = cards_seen[wrapped_player],
            )

            result = models[wrapped_player].bid(state)

            bids = torch.clamp(result["bids"], 0, 13)
            player_bids[wrapped_player] = player_bids[wrapped_player] + bids * player_should_play
            
            if wrapped_player == 0:
                # capture samples
                combined_bid_state = combined_bid_state.combine(state, player_should_play)
                combined_bids = combined_bids + bids * player_should_play

            
            #print(f"Player {player} bid: {bids}")
    pending_samples.append(('bids', combined_bid_state, None, combined_bids.detach()))
    combined_bid_state = None
    combined_bids = None
    if should_print:
        print(f"Player bids: {player_bids}")

    for round in range(13):
        if should_print:
            print(f"Round {round + 1}")
        suit_mask = torch.full((batch, 52), 1)
        score_mask = torch.full((batch, 52), 1)

        round_played_cards = [
            torch.zeros((batch,)),
            torch.zeros((batch,)),
            torch.zeros((batch,)),
            torch.zeros((batch,)),
        ]
        round_value = [
            torch.zeros((batch,)),
            torch.zeros((batch,)),
            torch.zeros((batch,)),
            torch.zeros((batch,)),
        ]

        combined_card_state = GameState(mask=torch.zeros((batch,)))
        combined_allowed_cards = torch.zeros((batch, 52)).to(torch.uint8)
        combined_cards = torch.zeros((batch,)).to(torch.uint8)
        sum_played = torch.zeros((batch,))
        for initiative_offset in range(4):
            for player in range(4):
                player_should_play = torch.eq(initiative_player, (player + initiative_offset) % 4).to(torch.uint8)
                sum_played = sum_played + player_should_play

                wrapped_player = player % 4
                left_player = (wrapped_player - 1 + 4) % 4
                right_player = (wrapped_player + 1) % 4
                teammate_player = (wrapped_player + 2) % 4
                team_id = wrapped_player % 2
            
                state = GameState(
                    my_hand = hands[wrapped_player],
                    spades_played = spades_played,
                    team_bid = torch.clamp(player_bids[wrapped_player] + player_bids[teammate_player], 0, 13),
                    other_team_bid = torch.clamp(player_bids[left_player] + player_bids[right_player], 0, 13),
                    bid_mine = player_bids[wrapped_player],
                    bid_teammate = player_bids[teammate_player],
                    bid_left = player_bids[left_player],
                    bid_right = player_bids[right_player],
                    cards_played_me = cards_played[wrapped_player],
                    cards_played_teammate = cards_played[teammate_player],
                    cards_played_left = cards_played[left_player],
                    cards_played_right = cards_played[right_player],
                    team_score_mine = team_scores[team_id],
                    team_score_other = team_scores[1 - team_id],
                    hands_won_mine = hands_won[wrapped_player],
                    hands_won_teammate = hands_won[teammate_player],
                    hands_won_left = hands_won[left_player],
                    hands_won_right = hands_won[right_player],
                    players_left = torch.full((batch,), 3 - player),
                    hand_number = torch.full((batch,), round),
                    trick_wins_me = player_wins[wrapped_player],
                    trick_wins_teammate = player_wins[teammate_player],
                    trick_wins_left = player_wins[left_player],
                    trick_wins_right = player_wins[right_player],
                    cards_seen = cards_seen[wrapped_player],
                )
                
                player_hand = hands[wrapped_player]
                no_cards_left = 1 - torch.sum(player_hand, 1)
                allow_all_play = (1 - player_should_play) * no_cards_left

                allowed_cards = hands[wrapped_player] * suit_mask
                any_playable_cards = torch.max(allowed_cards, 1)[0]
                limitless_override = torch.outer(1 - any_playable_cards, torch.full((52,), 1)) * hands[wrapped_player]
                allowed_cards = torch.maximum(allowed_cards, limitless_override)
                allowed_cards = torch.maximum(allowed_cards, torch.outer(allow_all_play, torch.full((52,), 1)))

                result = models[wrapped_player].play(state, allowed_cards)

                card_played = torch.nn.functional.one_hot(result["cards"], 52)

                round_played_cards[wrapped_player] = round_played_cards[wrapped_player] + result["cards"] * player_should_play
                num_cards_played = torch.sum(card_played * allowed_cards)
                if num_cards_played.item() != batch:
                    print(f"Player {wrapped_player} played invalid cards! Only {num_cards_played.item()} cards played out of {batch}.")
                    crash()
                
                card_is_spades = torch.lt(result["cards"], 13).to(torch.uint8)
                
                should_update_mask = torch.outer(player_should_play, torch.ones((52,)))
                
                if player == 0:
                    card_is_hearts = torch.logical_and(torch.ge(result["cards"], 13), torch.lt(result["cards"], 26)).to(torch.uint8)
                    card_is_clubs = torch.logical_and(torch.ge(result["cards"], 26), torch.lt(result["cards"], 39)).to(torch.uint8)
                    card_is_diamonds = torch.ge(result["cards"], 39).to(torch.uint8)

                    should_spades_mask = torch.outer(card_is_spades, spades_mask) + torch.outer(1 - card_is_spades, torch.full((52,), 1))
                    should_hearts_mask = torch.outer(card_is_hearts, hearts_mask) + torch.outer(1 - card_is_hearts, torch.full((52,), 1))
                    should_clubs_mask = torch.outer(card_is_clubs, clubs_mask) + torch.outer(1 - card_is_clubs, torch.full((52,), 1))
                    should_diamonds_mask = torch.outer(card_is_diamonds, diamonds_mask) + torch.outer(1 - card_is_diamonds, torch.full((52,), 1))

                    new_suit_mask = should_spades_mask * should_hearts_mask * should_clubs_mask * should_diamonds_mask
                    suit_mask = should_update_mask * new_suit_mask + (1 - should_update_mask) * suit_mask
                    new_score_mask = torch.maximum(suit_mask, spades_mask).detach()
                    score_mask = should_update_mask * new_score_mask + (1 - should_update_mask) * score_mask

                round_value[wrapped_player] = round_value[wrapped_player] + torch.sum(card_played * score_mask * base_card_values, 1) * player_should_play

                spades_played = spades_played + torch.maximum(spades_played, card_is_spades) * player_should_play
                rparts = []
                for r in range(13):
                    part = cards_played[wrapped_player][:, r, :]
                    if r == round:
                        rparts.append(torch.maximum(part, card_played * should_update_mask))
                    else:
                        rparts.append(part)
                cards_played[wrapped_player] = torch.stack(rparts, 1)

                for i in range(4):
                    cards_seen[i] = torch.maximum(cards_seen[i], card_played * should_update_mask)
                
                hands[wrapped_player] = hands[wrapped_player] * (1 - (card_played * should_update_mask))

                #cards_still_in_hand = torch.sum(hands[wrapped_player] * card_played * should_update_mask)
                #if cards_still_in_hand.item() > 0:
                #    print(f"Player {wrapped_player} still has cards in hand that have been played!")
                #    crash()

                if wrapped_player == 0:
                    combined_card_state = combined_card_state.combine(state, player_should_play)
                    combined_allowed_cards = combined_allowed_cards + allowed_cards * torch.outer(player_should_play, torch.zeros((52,)))
                    combined_cards = combined_cards + result["cards"] * player_should_play

                
                #print(f"Player {wrapped_player} played {result['cards']} with value {round_value[wrapped_player]}")

        # capture samples
        pending_samples.append(('cards', combined_card_state, combined_allowed_cards.detach(), combined_cards.detach()))

        if should_print:
            print(f"Played cards: {round_played_cards}")
        winners = torch.argmax(torch.stack(round_value, 1), 1)

        round_hot = torch.nn.functional.one_hot(torch.tensor(round), 13)
        for player in range(4):
            is_winner = torch.eq(winners, player).to(torch.uint8)

            hands_won[player] = hands_won[player] + torch.outer(is_winner, round_hot)
            player_wins[player] = player_wins[player] + is_winner

        if should_print:
            print(f"Round end, winners: {winners}")
            print(f"Scores: {player_wins}")
        initiative_player = winners
    
    if should_print:
        print("Game end")
    team_0_bid = torch.clamp(player_bids[0] + player_bids[2], 0, 13)
    team_1_bid = torch.clamp(player_bids[1] + player_bids[3], 0, 13)
    team_0_wins = player_wins[0] + player_wins[2]
    team_1_wins = player_wins[1] + player_wins[3]
    team_0_achieved = torch.ge(team_0_wins, team_0_bid).int()
    team_1_achieved = torch.ge(team_1_wins, team_1_bid).int()
    nil_scores = [
        None,
        None,
        None,
        None,
    ]

    for p in range(4):
        did_bid_nil = torch.eq(player_bids[p], 0).to(torch.uint8)
        achieved_nil = torch.eq(player_wins[p], 0).int()

        nil_scores[p] = did_bid_nil * (achieved_nil * 2 - 1) * 10        
    
    team_0_bag_delta = (team_0_wins - team_0_bid) * team_0_achieved
    team_1_bag_delta = (team_1_wins - team_1_bid) * team_1_achieved

    team_0_delta = (team_0_achieved * 2 - 1) * team_0_bid + nil_scores[0] + nil_scores[2]
    team_1_delta = (team_1_achieved * 2 - 1) * team_1_bid + nil_scores[1] + nil_scores[3]

    team_0_bag_overflow = torch.div(team_scores[0][:, 1].int() + team_0_bag_delta, 10, rounding_mode='floor')
    team_1_bag_overflow = torch.div(team_scores[1][:, 1].int() + team_1_bag_delta, 10, rounding_mode='floor')

    team_0_delta = team_0_delta - 10 * team_0_bag_overflow
    team_1_delta = team_1_delta - 10 * team_1_bag_overflow

    team_0_bag_delta = team_0_bag_delta - 10 * team_0_bag_overflow
    team_1_bag_delta = team_1_bag_delta - 10 * team_1_bag_overflow
    
    team_0_score_delta = torch.stack([team_0_delta, team_0_bag_delta], 1)
    team_1_score_delta = torch.stack([team_1_delta, team_1_bag_delta], 1)

    if should_print:
        print(f"Bids: team 1 won {team_0_wins} out of {team_0_bid}, team 2 got {team_1_wins} out of {team_1_bid}")
        print(f"Scores: team 1 got {team_0_delta * 10 + team_0_bag_delta} points, team 2 got {team_1_delta * 10 + team_1_bag_delta} points")

    return [sample + (team_0_score_delta, team_1_score_delta) for sample in pending_samples]
