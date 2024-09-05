import torch

from models.braindead import BrainDead
from gamestate import GameState

def run_game(batch, models):
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

    memory_vector_shapes = [
        models[0].memory_size(),
        models[1].memory_size(),
        models[2].memory_size(),
        models[3].memory_size(),
    ]
    memories = [
        torch.zeros((batch, memory_vector_shapes[0])),
        torch.zeros((batch, memory_vector_shapes[1])),
        torch.zeros((batch, memory_vector_shapes[2])),
        torch.zeros((batch, memory_vector_shapes[3])),
    ]

    initiative_player = torch.randint(0, 4, (batch,))
    print(f"player initiatives: {initiative_player}")

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
                other_team_bid = player_bids[left_player] + player_bids[right_player],
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
                players_left = torch.full((batch,), 4 - player),
                hand_number = torch.full((batch,), 0),
                trick_wins_me = player_wins[wrapped_player],
                trick_wins_teammate = player_wins[teammate_player],
                trick_wins_left = player_wins[left_player],
                trick_wins_right = player_wins[right_player],
                cards_seen = cards_seen[wrapped_player],
            )

            result = models[wrapped_player].bid(state)

            memory_mask = torch.outer(player_should_play, torch.ones(memory_vector_shapes[player]))
            memories[wrapped_player] = memories[wrapped_player] + result["memory"] * memory_mask

            bids = torch.clamp(result["bids"], 0, 13)
            player_bids[wrapped_player] = player_bids[wrapped_player] + bids * player_should_play
            
            if wrapped_player == 0:
                # capture samples
                pending_samples.append(('bids', state, player_should_play.detach(), bids.detach()))
            
            #print(f"Player {player} bid: {bids}")
    print(f"Player bids: {player_bids}")

    for round in range(13):
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
        old_memories = memories
        memories = [
            torch.zeros((batch, memory_vector_shapes[0])),
            torch.zeros((batch, memory_vector_shapes[1])),
            torch.zeros((batch, memory_vector_shapes[2])),
            torch.zeros((batch, memory_vector_shapes[3])),
        ]

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
                    team_bid = player_bids[teammate_player],
                    other_team_bid = player_bids[left_player] + player_bids[right_player],
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
                    players_left = torch.full((batch,), 4 - player),
                    hand_number = torch.full((batch,), round),
                    trick_wins_me = player_wins[wrapped_player],
                    trick_wins_teammate = player_wins[teammate_player],
                    trick_wins_left = player_wins[left_player],
                    trick_wins_right = player_wins[right_player],
                    cards_seen = cards_seen[wrapped_player],
                )
                
                allowed_cards = hands[wrapped_player] * suit_mask
                any_playable_cards = torch.max(allowed_cards, 1)[0]
                limitless_override = torch.outer(1 - any_playable_cards, torch.full((52,), 1))
                allowed_cards = torch.maximum(allowed_cards, limitless_override)

                result = models[wrapped_player].play(state, old_memories[wrapped_player], allowed_cards)

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
                
                memory_mask = torch.outer(player_should_play, torch.ones(memory_vector_shapes[player]))
                memories[wrapped_player] = memories[wrapped_player] + result["memory"] * memory_mask
                hands[wrapped_player] = hands[wrapped_player] * (1 - (card_played * should_update_mask))

                cards_still_in_hand = torch.sum(hands[wrapped_player] * card_played * should_update_mask)
                if cards_still_in_hand.item() > 0:
                    print(f"Player {wrapped_player} still has cards in han that have been played!")
                    crash()

                if wrapped_player == 0:
                    # capture samples
                    pending_samples.append(('cards', state, player_should_play.detach(), result["cards"].detach()))
                
                #print(f"Player {wrapped_player} played {result['cards']} with value {round_value[wrapped_player]}")

        print(f"Played cards: {round_played_cards}")
        print(f"Playcounts: {(sum_played)}")
        winners = torch.argmax(torch.stack(round_value, 1), 1)

        round_hot = torch.nn.functional.one_hot(torch.tensor(round), 13)
        for player in range(4):
            is_winner = torch.eq(winners, player).to(torch.uint8)

            hands_won[player] = hands_won[player] + torch.outer(is_winner, round_hot)
            player_wins[player] = player_wins[player] + is_winner

        print(f"Round end, winners: {winners}")
        print(f"Hand: {torch.sum(torch.stack(hands, 2), 1)}")
        print(f"Seen: {torch.sum(torch.stack(cards_seen, 2), 1)}")
        print(f"Scores: {player_wins}")
        initiative_player = winners
    print(torch.sum(torch.stack(cards_seen, 2), 1))
    print("Game end")


models = [
    BrainDead(),
    BrainDead(),
    BrainDead(),
    BrainDead(),
]
run_game(2, models)