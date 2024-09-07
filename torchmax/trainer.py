import torch
import random
import sys
import json
import os.path

from models.braindead import BrainDead
from models.bidder import Bidder
from models.fullplayer import FullPlayer
from simulator import run_game

def train(model, optimizer, samples):
    optimizer.zero_grad()

    bid_losses = []
    bid_counts = []
    card_losses = []
    card_counts = []
    for type, state, allowed_options, chosen_action, team_0_score_delta, team_1_score_delta in samples:
        if type == 'cards':
            result = model.play(state, allowed_options, chosen_action)

            predictions = torch.concat([result["own_score_prediction"], result["other_score_prediction"]], 1)
            outcomes = torch.concat([team_0_score_delta, team_1_score_delta], 1)

            predictions = predictions * torch.outer(should_play, torch.ones((4,)))
            outcomes = outcomes * torch.outer(should_play, torch.ones((4,)))

            playcount = torch.sum(should_play)
            error = torch.nn.functional.mse_loss(predictions, outcomes) * should_play.size(0)

            card_losses.append(error)
            card_counts.append(playcount)
            continue
        
        if type == 'bids':
            result = model.bid(state, chosen_action)

            predictions = torch.concat([result["own_score_prediction"], result["other_score_prediction"]], 1)
            outcomes = torch.concat([team_0_score_delta, team_1_score_delta], 1)

            predictions = predictions * torch.outer(should_play, torch.ones((4,)))
            outcomes = outcomes * torch.outer(should_play, torch.ones((4,)))

            playcount = torch.sum(should_play)
            error = torch.nn.functional.mse_loss(predictions, outcomes) * should_play.size(0)

            bid_losses.append(error)
            bid_counts.append(playcount)
            continue
        print(f"Bad type: '{type}'")
        crash()
    
    bid_sample_count = torch.sum(torch.stack(bid_counts))
    bid_total_loss = torch.sum(torch.stack(bid_losses)) / bid_sample_count
    card_sample_count = torch.sum(torch.stack(card_counts))
    card_total_loss = torch.sum(torch.stack(card_losses)) / card_sample_count

    total_loss = bid_total_loss + card_total_loss
    total_loss.backward()
    optimizer.step()

    return {
        "bids": total_loss.item(),
        "cards": card_total_loss.item(),
        "sample_count": bid_sample_count.item() + card_sample_count.item()
    }

def do_tournament(steering_model, available_models, game_batch_size, num_seatings, model_prob):
    samples = []

    def choose_model(models, p):
        pos = len(models) - 1

        while pos > 0:
            if random.random() < p:
                return models[pos]
            pos = pos - 1
        
        return models[0]

    for s in range(num_seatings):
        sys.stdout.write('.')
        sys.stdout.flush()
        models = [
            steering_model,
            choose_model(available_models, model_prob),
            choose_model(available_models, model_prob),
            choose_model(available_models, model_prob),
        ]

        samples += run_game(game_batch_size, models)
    print()
    return samples

def do_q_generation(q1_models, q2_models, temperature=1000, game_batch_size=128, num_seatings=64, steps=100, model_prob=0.2, lr=None):
    q1_training_model = FullPlayer(0)
    q1_optimizer = torch.optim.Adam(q1_training_model.parameters(), lr=lr)

    q1_sample_count = 0
    for s in range(steps):
        q1_samples = do_tournament(q2_models[-1].with_temperature(temperature), q1_models, game_batch_size, num_seatings, model_prob)
        losses = train(q1_training_model, q1_optimizer, q1_samples)
        q1_sample_count = q1_sample_count + losses["sample_count"]
        print(f"Q1 step {s}, bid {round(losses['bids'], ndigits=1)}, cards: {round(losses['cards'], ndigits=1)}, samples: {q1_sample_count}")
        q1_samples = None
    
    q2_training_model = FullPlayer(0)
    q2_optimizer = torch.optim.Adam(q2_training_model.parameters())

    q2_sample_count = 0
    for s in range(steps):
        q2_samples = do_tournament(q1_models[-1].with_temperature(temperature), q2_models, game_batch_size, num_seatings, model_prob)
        losses = train(q2_training_model, q2_optimizer, q2_samples)
        q2_sample_count = q2_sample_count + losses["sample_count"]
        print(f"Q2 step {s}, bid {round(losses['bids'], ndigits=1)}, cards: {round(losses['cards'], ndigits=1)}, samples: {q2_sample_count}")
        q2_samples = None
    
    q1_models.append(q1_training_model)
    q2_models.append(q2_training_model)


def start_training():
    config = {
        "generation": 0,
        "lr": 0.00005,
        "game_batch_size": 512,
        "temperature": 1000,
        "num_seatings": 64,
        "steps": 20,
        "model_prob": 0.3,
    }
    serialized = json.dumps(config, indent=4)
    with open("results/state.json", "w") as f:
        f.write(serialized)
        f.write("\n")

def continue_training():
    config = None
    with open("results/state.json") as f:
        config = json.load(f)

    available_models_q1 = [
        FullPlayer(1000),
    ]
    available_models_q2 = [
        FullPlayer(1000)
    ]

    for i in range(config["generation"]):
        available_models_q1.append(FullPlayer.load(f"results/q1-{str(i).zfill(4)}.pt", 0))
        available_models_q2.append(FullPlayer.load(f"results/q2-{str(i).zfill(4)}.pt", 0))

    while True:
        with open("results/state.json") as f:
            config = json.load(f)
        print(config)

        current_generation = config["generation"]
        do_q_generation(available_models_q1, available_models_q2,
                        temperature=config["temperature"],
                        game_batch_size=config["game_batch_size"],
                        num_seatings=config["num_seatings"],
                        steps=config["steps"],
                        lr=config["lr"],
                        model_prob=config["model_prob"]
                       )
        new_q1 = available_models_q1[-1]
        new_q2 = available_models_q2[-1]

        new_q1.save(f"results/q1-{str(current_generation).zfill(4)}.pt")
        new_q2.save(f"results/q2-{str(current_generation).zfill(4)}.pt")
        
        with open("results/state.json") as f:
            config = json.load(f)
        config["generation"] = current_generation + 1
        with open("results/state.json", "w") as f:
            f.write(json.dumps(config, indent=4))
            f.write("\n")

if not os.path.isfile("results/state.json"):
    start_training()

#torch.set_default_device('cuda')

continue_training()