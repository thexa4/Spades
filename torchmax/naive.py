import torch

from models.braindead import BrainDead
from gamestate import GameState

samples = []
training_model = BrainDead()

batch = 2

models = [
    BrainDead(),
    BrainDead(),
    BrainDead(),
    BrainDead(),
]


base_card_values = torch.tensor([[
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, # spades 2-A
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,          # hearts 2-A
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,          # clubs: 2-A
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,          # diamonds: 2-A
]]).repeat(batch, 1)

spades_mask = torch.tensor([[
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # spades 2-A
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # hearts 2-A
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # clubs 2-A
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # diamonds 2-A
]]).repeat(batch, 1)
hearts_mask = torch.tensor([[
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # spades 2-A
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # hearts 2-A
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # clubs 2-A
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # diamonds 2-A
]]).repeat(batch, 1)
clubs_mask = torch.tensor([[
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # spades 2-A
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # hearts 2-A
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # clubs 2-A
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # diamonds 2-A
]]).repeat(batch, 1)
diamonds_mask = torch.tensor([[
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # spades 2-A
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # hearts 2-A
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # clubs 2-A
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # diamonds 2-A
]]).repeat(batch, 1)

for startplayer in range(4):
    pending_samples = []

    print("Startplayer " + str(startplayer))
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
        torch.zeros(batch, dtype=torch.uint8),
        torch.zeros(batch, dtype=torch.uint8),
        torch.zeros(batch, dtype=torch.uint8),
        torch.zeros(batch, dtype=torch.uint8)
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
        torch.zeros(batch, 13, dtype=torch.uint8),
        torch.zeros(batch, 13, dtype=torch.uint8),
        torch.zeros(batch, 13, dtype=torch.uint8),
        torch.zeros(batch, 13, dtype=torch.uint8)
    ]

    memories = [
        None,
        None,
        None,
        None
    ]
    training_memory = None


    for player in range(4):
        wrapped_player = (player + startplayer) % 4
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

        memories[wrapped_player] = result["memory"]
        bids = torch.clamp(result["bids"], 0, 13)
        player_bids[wrapped_player] = bids
        
        if wrapped_player == 0:
            # capture samples
            pending_samples.append((state, bids.detach()))
        
        print(f"Player {player} bid: {bids}")
        

    for round in range(13):
        print(f"Round {round + 1}")
        suit_mask = torch.full((batch, 52), 1)
        score_mask = torch.full((batch, 52), 1)

        round_played_cards = [
            None,
            None,
            None,
            None
        ]

        for player in range(4):
            wrapped_player = (player + startplayer) % 4
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

            result = models[wrapped_player].play(state, memories[wrapped_player], allowed_cards)

            round_played_cards[wrapped_player] = result["cards"]
            card_played = torch.nn.functional.one_hot(result["cards"], 52)
            num_cards_played = torch.sum(card_played * allowed_cards)
            if num_cards_played.item() != batch:
                print(f"Player {wrapped_player} played invalid cards! Only {num_cards_played.item()} cards played out of {batch}.")
                crash()
            
            card_is_spades = torch.lt(result["cards"], 13).to(torch.uint8)
            
            if player == 0:
                card_is_hearts = torch.logical_and(torch.ge(result["cards"], 13), torch.lt(result["cards"], 26)).to(torch.uint8)
                card_is_clubs = torch.logical_and(torch.ge(result["cards"], 26), torch.lt(result["cards"], 39)).to(torch.uint8)
                card_is_diamonds = torch.gt(torch.ge(result["cards"], 39)).to(torch.uint8)



            spades_played = torch.maximum(spades_played, card_is_spades)
            cards_played[wrapped_player][:, round, :] = torch.maximum(cards_played[wrapped_player][:, round, :], card_played)

            for i in range(4):
                cards_seen[i] = torch.maximum(cards_seen[i], card_played)
            
            memories[wrapped_player] = result["memory"]
            hands[wrapped_player] = hands[wrapped_player] * (1 - card_played)

            if wrapped_player == 0:
                # capture samples
                pending_samples.append((state, result["cards"].detach()))
            
            print(f"Player {wrapped_player} played {result['cards']}")
        print("Round end")

        print(round_played_cards)

        crash()



