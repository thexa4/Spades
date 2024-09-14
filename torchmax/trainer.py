import torch
import random
import sys
import json
import os.path
import collections
import threading
import statistics

from models.braindead import BrainDead
from models.bidder import Bidder
from models.fullplayer import FullPlayer
from simulator import run_game
from training_sample import TrainingSample
from multiprocessing.pool import ThreadPool

def train(model, optimizer, samples, devices=[None]):
    device_models = model.replicate(devices)
    device_samples = []
    total_loss = torch.tensor(0, device=model.device)
    
    for i in range(len(devices)):
        device_samples.append(sum(samples[i::len(devices)], TrainingSample()).with_device(devices[i]))
    
    optimizer.zero_grad()
    
    def do_device(device_id):
        bid_losses = None
        bid_counts = None
        card_losses = []
        card_counts = []

        game = device_samples[device_id]
        bid_result = device_models[device_id].bid(game.bid_state, game.bid_result)

        predictions = torch.concat([bid_result["own_score_prediction"], bid_result["other_score_prediction"]], 1)
        outcomes = torch.concat([game.own_outcome.float(), game.other_outcome.float()], 1).detach()

        playcount = predictions.size(0)

        error = torch.nn.functional.mse_loss(predictions, outcomes.to(device=devices[device_id])) * torch.tensor(predictions.size(0), device=devices[device_id]).float()
        bid_losses = error
        bid_counts = playcount

        for round in range(13):
            card_result = device_models[device_id].play(game.hand_states[round], game.hand_allowed[round], game.hand_results[round])

            predictions = torch.concat([card_result["own_score_prediction"], card_result["other_score_prediction"]], 1)
            outcomes = torch.concat([game.own_outcome.float(), game.other_outcome.float()], 1).detach()

            playcount = predictions.size(0)
            error = torch.nn.functional.mse_loss(predictions, outcomes.to(device=devices[device_id])) * torch.tensor(predictions.size(0), device=devices[device_id]).float()

            card_losses.append(error)
            card_counts.append(playcount)
                
        bid_sample_count = bid_counts
        bid_total_loss = bid_losses / torch.tensor(bid_sample_count).float()
        card_sample_count = sum(card_counts)
        card_total_loss = torch.sum(torch.stack(card_losses)) / torch.tensor(card_sample_count).float()

        return {
            "total": (bid_total_loss + card_total_loss).to(model.device),
            "bids": bid_total_loss.item(),
            "cards": card_total_loss.item(),
            "count": bid_sample_count + card_sample_count
        }
    
    pool = ThreadPool(len(devices))
    results = pool.map(do_device, range(len(devices)))

    total_loss = sum([x['total'] for x in results], total_loss)
    card_loss = statistics.mean([x['cards'] for x in results])
    bid_loss = statistics.mean([x['bids'] for x in results])

    total_loss.backward()
    optimizer.step()
    sys.stdout.write('_')
    sys.stdout.flush()

    return {
        "sample_count": sum([x['count'] for x in results]),
        "bids": bid_loss,
        "cards": card_loss,
    }

def do_tournament(steering_model, available_models, game_batch_size, num_seatings, model_prob, devices=[None]):
    seatings_left = [num_seatings]
    seat_lock = threading.Lock()

    def choose_model(models, p):
        pos = len(models) - 1

        while pos > 0:
            if random.random() < p:
                return models[pos]
            pos = pos - 1
        
        return models[0]

    def do_seating(args):
        device, s_left, s_lock = args

        seat_samples = []
        while True:
            with s_lock:
                if s_left[0] <= 0:
                    return seat_samples
                s_left[0] = s_left[0] - 1
            models = [
                steering_model.with_device(device),
                choose_model(available_models, model_prob).with_device(device),
                choose_model(available_models, model_prob).with_device(device),
                choose_model(available_models, model_prob).with_device(device),
            ]

            seat_samples.append(run_game(game_batch_size, models, device=device).with_device('cpu'))
            sys.stdout.write('.')
            sys.stdout.flush()

    pool = ThreadPool(len(devices))
    return sum(pool.map(do_seating, [(d, seatings_left, seat_lock) for d in devices]), [])

def do_q_generation(q1_models, q2_models, temperature=1000, game_batch_size=128, num_seatings=64, steps=100, model_prob=0.2, lr=None, reuse_factor=10, seat_batch=2):
        
    meta = {
        'mc': len(q1_models) + len(q2_models),
        't': temperature,
        'gbatch': game_batch_size,
        'nseat': num_seatings,
        'model_prob': model_prob,
        'lr': lr,
        'reuse': reuse_factor,
        'sbatch': seat_batch,
    }

    q1_training_model = FullPlayer(0, device='cuda')
    curstep = 0
    if os.path.isfile('results/q1_ckpt.pt'):
        test_training_model, metadata = FullPlayer.load('results/q1_ckpt.pt', 0, device='cuda')
        if metadata['state'] == meta:
            curstep = metadata['curstep'] + 1
            q1_training_model = test_training_model
        else:
            print('Checkpoint for q1 not applicable')
    
    q1_optimizer = torch.optim.Adam(q1_training_model.parameters(), lr=lr)

    resevoir = collections.deque()
    
    q1_sample_count = 0
    for s in range(curstep, steps):
        while len(resevoir) < reuse_factor * num_seatings:
            tournament_samples = do_tournament(q2_models[-1].with_temperature(temperature), q1_models, game_batch_size, reuse_factor * num_seatings - len(resevoir), model_prob, devices=['cuda:0', 'cuda:1'])
            for samp in tournament_samples:
                resevoir.append(samp)

        todo_list = list(resevoir)
        random.shuffle(todo_list)
        todo_list = [todo_list[x:x+seat_batch] for x in range(0, len(todo_list), seat_batch)]

        card_losses = []
        bid_losses = []
        for samples in todo_list:            
            losses = train(q1_training_model, q1_optimizer, samples, devices=['cuda:0', 'cuda:1'])

            card_losses.append(losses['cards'])
            bid_losses.append(losses['bids'])

        q1_sample_count = q1_sample_count + losses["sample_count"]
        print()
        q1_training_model.save('results/q1_ckpt.pt', {'state': meta, 'curstep': s})

        print(f"Q1 step {s}, bid {round(statistics.mean(bid_losses), ndigits=1)}, cards: {round(statistics.mean(card_losses), ndigits=1)}, samples: {q1_sample_count}")
        for _ in range(num_seatings):
            resevoir.popleft()
        samples = None
    q1_optimizer.zero_grad()
    q1_optimizer = None
        
    
    q2_training_model = FullPlayer(0, device='cuda')
    curstep = 0
    if os.path.isfile('results/q2_ckpt.pt'):
        test_training_model, metadata = FullPlayer.load('results/q2_ckpt.pt', 0, device='cuda')
        if metadata['state'] == meta:
            curstep = metadata['curstep'] + 1
            q2_training_model = test_training_model
        else:
            print('Checkpoint for q2 not applicable')

    q2_optimizer = torch.optim.Adam(q2_training_model.parameters())
    
    resevoir = collections.deque()
    q2_sample_count = 0
    for s in range(curstep, steps):
        while len(resevoir) < reuse_factor * num_seatings:
            tournament_samples = do_tournament(q1_models[-1].with_temperature(temperature), q1_models, game_batch_size, reuse_factor * num_seatings - len(resevoir), model_prob, devices=['cuda:0', 'cuda:1'])
            for samp in tournament_samples:
                resevoir.append(samp)

        todo_list = list(resevoir)
        random.shuffle(todo_list)
        todo_list = [todo_list[x:x+seat_batch] for x in range(0, len(todo_list), seat_batch)]

        card_losses = []
        bid_losses = []
        for samples in todo_list:            
            losses = train(q2_training_model, q2_optimizer, samples, devices=['cuda:0', 'cuda:1'])

            card_losses.append(losses['cards'])
            bid_losses.append(losses['bids'])

        q2_sample_count = q2_sample_count + losses["sample_count"]
        print()
        q1_training_model.save('results/q2_ckpt.pt', {'state': meta, 'curstep': s})

        print(f"Q2 step {s}, bid {round(statistics.mean(bid_losses), ndigits=1)}, cards: {round(statistics.mean(card_losses), ndigits=1)}, samples: {q2_sample_count}")
        for _ in range(num_seatings):
            resevoir.popleft()
        samples = None

    q1_models.append(q1_training_model)
    q2_models.append(q2_training_model)


def start_training():
    config = {
        "generation": 0,
        "lr": 0.0005,
        "game_batch_size": 512,
        "num_seatings": 64,
        "steps": 20,
        "model_prob": 0.3,
        "reuse_factor": 10,
        "seat_batch": 2,
        "temp_schedule": [
            [
                0,
                100
            ],
            [
                10,
                10
            ],
            [
                20,
                1
            ]
        ]
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
        FullPlayer(1000, device='cuda:0'),
    ]
    available_models_q2 = [
        FullPlayer(1000, device='cuda:0')
    ]

    for i in range(config["generation"]):
        available_models_q1.append(FullPlayer.load(f"results/{str(i).zfill(4)}-q1.pt", 0, device='cuda:0')[0])
        available_models_q2.append(FullPlayer.load(f"results/{str(i).zfill(4)}-q2.pt", 0, device='cuda:0')[0])

    while True:
        with open("results/state.json") as f:
            config = json.load(f)
        print(config)

        current_generation = config["generation"]

        temp_steps = sorted(config['temp_schedule'], key = lambda x: x[0])
        temp = 1
        if current_generation <= temp_steps[0][0]:
            temp = temp_steps[0][1]
        elif current_generation >= temp_steps[-1][0]:
            temp = temp_steps[-1][0]
        else:
            for i in range(1, len(temp_steps)):
                if temp_steps[i][0] >= current_generation:
                    prev_step = temp_steps[i - 1]
                    next_step = temp_steps[i]
                    percent = (current_generation - prev_step[0]) / (next_step[0] - prev_step[0])
                    temp = next_step * percent + prev_step * (1 - percent)
        print(f"Temperature: {temp}")
        
        do_q_generation(available_models_q1, available_models_q2,
                        temperature=temp,
                        game_batch_size=config["game_batch_size"],
                        num_seatings=config["num_seatings"],
                        steps=config["steps"],
                        lr=config["lr"],
                        model_prob=config["model_prob"],
                        reuse_factor=config["reuse_factor"],
                        seat_batch=config["seat_batch"],
                       )
        new_q1 = available_models_q1[-1]
        new_q2 = available_models_q2[-1]

        new_q1.save(f"results/{str(current_generation).zfill(4)}-q1.pt")
        new_q2.save(f"results/{str(current_generation).zfill(4)}-q2.pt")
        
        with open("results/state.json") as f:
            config = json.load(f)
        config["generation"] = current_generation + 1
        with open("results/state.json", "w") as f:
            f.write(json.dumps(config, indent=4))
            f.write("\n")

if not os.path.isfile("results/state.json"):
    start_training()

#with torch.autograd.set_detect_anomaly(True):
continue_training()
