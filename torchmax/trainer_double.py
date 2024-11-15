import torch
import random
import sys
import json
import os.path
import collections
import threading
import statistics

from models.fullplayer import FullPlayer
from simulator import run_double_game
from models.random_player import RandomPlayer
from models.fullplayer2 import FullPlayer2
from training_sample import TrainingSample
from multiprocessing.pool import ThreadPool

def train(model, optimizer, samples, testset=None, devices=[None]):
    device_models = model.replicate(devices)
    device_samples = []
    total_loss = torch.tensor(0, device=model.device)
    
    for i in range(len(devices)):
        device_samples.append(TrainingSample.concat(samples[i::len(devices)]).with_device(devices[i]))
    
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
        round_losses = []

        for round in range(13):
            card_result = device_models[device_id].play(game.hand_states[round], game.hand_allowed[round], game.hand_results[round])

            predictions = torch.concat([card_result["own_score_prediction"], card_result["other_score_prediction"]], 1)
            outcomes = torch.concat([game.own_outcome.float(), game.other_outcome.float()], 1).detach()

            playcount = predictions.size(0)
            error = torch.nn.functional.mse_loss(predictions, outcomes.to(device=devices[device_id])) * torch.tensor(predictions.size(0), device=devices[device_id]).float()

            card_losses.append(error)
            round_losses.append(error)
            card_counts.append(playcount)
                
        bid_sample_count = bid_counts
        bid_total_loss = bid_losses / torch.tensor(bid_sample_count).float()
        card_sample_count = sum(card_counts)
        card_total_loss = torch.sum(torch.stack(card_losses)) / torch.tensor(card_sample_count).float()

        return {
            "total": (13 * bid_total_loss + card_total_loss).to(model.device),
            "bids": bid_total_loss.item(),
            "cards": card_total_loss.item(),
            "rounds": [r.item() / bid_counts for r in round_losses],
            "count": bid_sample_count + card_sample_count
        }
    
    pool = ThreadPool(len(devices))
    results = pool.map(do_device, range(len(devices)))

    total_loss = sum([x['total'] for x in results], total_loss)
    card_loss = statistics.mean([x['cards'] for x in results])
    bid_loss = statistics.mean([x['bids'] for x in results])

    round_matrix = [x['rounds'] for x in results]
    rounds = []
    for i in range(13):
        rounds.append(round(statistics.mean([r[i] for r in round_matrix]), ndigits=1))

    test_bid_loss = -1
    test_card_losses = [-1 for _ in range (13)]
    if testset != None:
        game = testset
        bid_result = device_models[0].bid(game.bid_state, game.bid_result)

        predictions = torch.concat([bid_result["own_score_prediction"], bid_result["other_score_prediction"]], 1)
        outcomes = torch.concat([game.own_outcome.float(), game.other_outcome.float()], 1).detach()

        error = torch.nn.functional.mse_loss(predictions, outcomes.to(device=devices[0])) * torch.tensor(predictions.size(0), device=devices[0]).float()
        test_bid_loss = error.item()
        test_card_losses = []

        for r in range(13):
            card_result = device_models[0].play(game.hand_states[r], game.hand_allowed[r], game.hand_results[r])

            predictions = torch.concat([card_result["own_score_prediction"], card_result["other_score_prediction"]], 1)
            outcomes = torch.concat([game.own_outcome.float(), game.other_outcome.float()], 1).detach()

            error = torch.nn.functional.mse_loss(predictions, outcomes.to(device=devices[0])) * torch.tensor(predictions.size(0), device=devices[0]).float()

            test_card_losses.append((error / torch.tensor(game.own_outcome.size(0)).float()).item())
        test_bid_loss = test_bid_loss / torch.tensor(game.own_outcome.size(0)).float().item()

    total_loss.backward()
    optimizer.step()
    sys.stdout.write('_')
    sys.stdout.flush()

    return {
        "sample_count": sum([x['count'] for x in results]),
        "bids": bid_loss,
        "cards": card_loss,
        "rounds": rounds,
        "test_bid": test_bid_loss,
        "test_round_loss": test_card_losses,
    }

def do_tournament(steering_model, available_models, game_batch_size, num_seatings, model_prob, devices=[None], store_device=None):
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
        device, s_left, s_lock, store_device = args

        seat_samples = []
        while True:
            with s_lock:
                if s_left[0] <= 0:
                    return seat_samples
                s_left[0] = s_left[0] - 1
            models = [
                steering_model.with_device(device),
                choose_model(available_models, model_prob).with_device(device),
            ]

            sample = run_double_game(game_batch_size, models + models, device=device).with_device(devices[0])
            if store_device != None:
                sample = sample.with_device(store_device)
            seat_samples.append(sample)
            sys.stdout.write('.')
            sys.stdout.flush()

    pool = ThreadPool(len(devices))
    return sum(pool.map(do_seating, [(d, seatings_left, seat_lock, store_device) for d in devices]), [])

def do_q_generation(q1_models, q2_models, temperature=1000, game_batch_size=128, num_seatings=64, steps=100, model_prob=0.2, lr=None, reuse_factor=10, seating_offset=0, seat_batch=2, folder='results', hand_layout=None, bid_layout=None, card_layout=None, embedding_path=None, train_devices=['cuda:0'], store_device='cpu', model_retention=0, model_version="fullplayer/v1"):
        
    meta = {
        'mc': len(q1_models) + len(q2_models),
        't': temperature,
        'gbatch': game_batch_size,
        'nseat': num_seatings,
        'model_prob': model_prob,
        'lr': lr,
        'reuse': reuse_factor,
        'seat_offset': seating_offset,
        'sbatch': seat_batch,
        'bid_layers': bid_layout,
        'card_layers': card_layout,
        'hand_layers': hand_layout,
        'model_version': model_version,
    }
    
    print('Generating testset')
    testset = []
    testset_models = q1_models[-10:] + q2_models[-10:]
    for i in range(10):
        random.shuffle(testset_models)
        pairs = zip(testset_models[::2], testset_models[1::2])
        for a,b in pairs:
            sys.stdout.write('.')
            sys.stdout.flush()
            testset.append(run_double_game(4, [a, b, a, b], device=train_devices[0]).with_device(train_devices[0]))
    testset = sum(testset[1:], testset[0])
    print('')


    if model_version == "fullplayer/v1":
        q1_training_model = FullPlayer(0, bid_layout, card_layout, device=train_devices[0])
    if model_version == "fullplayer/v2":
        if hand_layout == None:
            q1_training_model = FullPlayer2(0, None, bid_layout, card_layout, embedding_path=embedding_path, device=train_devices[0])
        else:
            q1_training_model = FullPlayer2(0, hand_layout, bid_layout, card_layout, device=train_devices[0])
    q1_optimizer = torch.optim.Adam(q1_training_model.parameters(), lr=lr)
    if model_retention > 0 and len(q1_models) > 1:
        q1_training_model = q1_models[-1].mix(q1_training_model, 1 - model_retention)
        q1_optimizer = torch.optim.Adam(q1_training_model.parameters(), lr=lr)
        if q1_models[-1].optim_state != None:
            q1_optimizer.load_state_dict(q1_models[-1].optim_state)
        else:
            print("Previous q1 does not have optimizer state")

    curstep = 0
    if os.path.isfile(f"{folder}/q1_ckpt.pt") or os.path.isfile(f"{folder}/q1_ckpt.pt2"):
        if model_version == "fullplayer/v1":
            test_training_model, metadata = FullPlayer.load(f"{folder}/q1_ckpt.pt", 0, device=train_devices[0])
        if model_version == "fullplayer/v2":
            test_training_model, metadata = FullPlayer2.load(f"{folder}/q1_ckpt.pt2", 0, device=train_devices[0])
        if metadata['state'] == meta:
            curstep = metadata['curstep'] + 1
            q1_training_model = test_training_model
            q1_optimizer = torch.optim.Adam(q1_training_model.parameters(), lr=lr)
            if q1_training_model.optim_state != None:
                q1_optimizer.load_state_dict(q1_training_model.optim_state)
            else:
                print("Q1 checkpoint does not have optimizer state")
        else:
            print('Checkpoint for q1 not applicable')
    

    resevoir = collections.deque()
    
    q1_sample_count = 0
    for s in range(curstep, steps):
        while len(resevoir) < reuse_factor * num_seatings + seating_offset:
            tournament_samples = do_tournament(q2_models[-1].with_temperature(temperature), q1_models, game_batch_size, reuse_factor * num_seatings + seating_offset - len(resevoir), model_prob, devices=train_devices, store_device=store_device)
            for samp in tournament_samples:
                resevoir.append(samp)

        todo_list = list(resevoir)
        random.shuffle(todo_list)
        todo_list = [todo_list[x:x+seat_batch] for x in range(0, len(todo_list), seat_batch)]

        card_losses = []
        bid_losses = []
        round_losses = [list() for _ in range(13)]
        test_bid_losses = []
        test_card_losses = [list() for _ in range(13)]
        for samples in todo_list:
            losses = train(q1_training_model, q1_optimizer, samples, testset=testset, devices=train_devices)

            card_losses.append(losses['cards'])
            bid_losses.append(losses['bids'])
            test_bid_losses.append(losses['test_bid'])
            for i in range(13):
                round_losses[i].append(losses['rounds'][i])
                test_card_losses[i].append(losses['test_round_loss'][i])

        q1_sample_count = q1_sample_count + losses["sample_count"]
        print()
        q1_training_model.optim_state = q1_optimizer.state_dict()
        if model_version == "fullplayer/v1":
            q1_training_model.save(f"{folder}/q1_ckpt.pt", {'state': meta, 'curstep': s})
        if model_version == "fullplayer/v2":
            q1_training_model.save(f"{folder}/q1_ckpt.pt2", {'state': meta, 'curstep': s})

        print(f"Q1 step {s}, bid {round(statistics.mean(bid_losses), ndigits=1)}, cards: {round(statistics.mean(card_losses), ndigits=1)}, samples: {q1_sample_count}")
        print([round(statistics.mean(round_losses[i]), ndigits=1) for i in range(13)])
        print(f"Test: bid {round(statistics.mean(test_bid_losses), ndigits=1)}, cards: {[round(statistics.mean(test_card_losses[i]), ndigits=1) for i in range(13)]}")
        for _ in range(num_seatings):
            resevoir.popleft()
        samples = None
    q1_optimizer.zero_grad()
    q1_optimizer = None
        
    
    if model_version == "fullplayer/v1":
        q2_training_model = FullPlayer(0, bid_layout, card_layout, device=train_devices[0])
    if model_version == "fullplayer/v2":
        if hand_layout == None:
            q2_training_model = FullPlayer2(0, None, bid_layout, card_layout, embedding_path=embedding_path, device=train_devices[0])
        else:
            q2_training_model = FullPlayer2(0, hand_layout, bid_layout, card_layout, device=train_devices[0])
    q2_optimizer = torch.optim.Adam(q2_training_model.parameters())
    if model_retention > 0 and len(q2_models) > 1:
        q2_training_model = q2_models[-1].mix(q2_training_model, 1 - model_retention)
        q2_optimizer = torch.optim.Adam(q2_training_model.parameters())
        if q2_models[-1].optim_state != None:
            q2_optimizer.load_state_dict(q1_models[-1].optim_state)
        else:
            print("Previous q2 does not have optimizer state")
    curstep = 0
    if os.path.isfile(f"{folder}/q2_ckpt.pt") or os.path.isfile(f"{folder}/q2_ckpt.pt2"):
        if model_version == "fullplayer/v1":
            test_training_model, metadata = FullPlayer.load(f"{folder}/q2_ckpt.pt", 0, device=train_devices[0])
        if model_version == "fullplayer/v2":
            test_training_model, metadata = FullPlayer2.load(f"{folder}/q2_ckpt.pt2", 0, device=train_devices[0])
        if metadata['state'] == meta:
            curstep = metadata['curstep'] + 1
            q2_training_model = test_training_model
            q2_optimizer = torch.optim.Adam(q2_training_model.parameters(), lr=lr)
            if q2_training_model.optim_state != None:
                q2_optimizer.load_state_dict(q2_training_model.optim_state)
            else:
                print("Q2 checkpoint does not have optimizer state")
        else:
            print('Checkpoint for q2 not applicable')

    
    resevoir = collections.deque()
    q2_sample_count = 0
    for s in range(curstep, steps):
        while len(resevoir) < reuse_factor * num_seatings + seating_offset:
            tournament_samples = do_tournament(q1_models[-1].with_temperature(temperature), q1_models, game_batch_size, reuse_factor * num_seatings + seating_offset - len(resevoir), model_prob, devices=train_devices, store_device='cpu')
            for samp in tournament_samples:
                resevoir.append(samp)

        todo_list = list(resevoir)
        random.shuffle(todo_list)
        todo_list = [todo_list[x:x+seat_batch] for x in range(0, len(todo_list), seat_batch)]

        card_losses = []
        bid_losses = []
        round_losses = [list() for _ in range(13)]
        test_bid_losses = []
        test_card_losses = [list() for _ in range(13)]
        for samples in todo_list:            
            losses = train(q2_training_model, q2_optimizer, samples, testset=testset, devices=train_devices)

            card_losses.append(losses['cards'])
            bid_losses.append(losses['bids'])
            test_bid_losses.append(losses['test_bid'])
            for i in range(13):
                round_losses[i].append(losses['rounds'][i])
                test_card_losses[i].append(losses['test_round_loss'][i])

        q2_sample_count = q2_sample_count + losses["sample_count"]
        print()
        q2_training_model.optim_state = q2_optimizer.state_dict()
        if model_version == "fullplayer/v1":
            q2_training_model.save(f"{folder}/q2_ckpt.pt", {'state': meta, 'curstep': s})
        if model_version == "fullplayer/v2":
            q2_training_model.save(f"{folder}/q2_ckpt.pt2", {'state': meta, 'curstep': s})

        print(f"Q2 step {s}, bid {round(statistics.mean(bid_losses), ndigits=1)}, cards: {round(statistics.mean(card_losses), ndigits=1)}, samples: {q2_sample_count}")
        print([round(statistics.mean(round_losses[i]), ndigits=1) for i in range(13)])
        print(f"Test: bid {round(statistics.mean(test_bid_losses), ndigits=1)}, cards: {[round(statistics.mean(test_card_losses[i]), ndigits=1) for i in range(13)]}")
        for _ in range(num_seatings):
            resevoir.popleft()
        samples = None

    q1_models.append(q1_training_model)
    q2_models.append(q2_training_model)


def start_training(folder):
    config = {
        "generation": 0,
        "lr": 0.0005,
        "game_batch_size": 512,
        "num_seatings": 64,
        "steps": 20,
        "model_prob": 0.3,
        "reuse_factor": 10,
        "seating_offset": 0,
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
        ],
        "bidnet": [
            512,
            512
        ],
        "cardnet": [
            2560,
            1024,
            512,
            512,
            512
        ],
        "store_device": "cpu",
        "model_retention": 0,
        "model_version": "fullplayer/v2",
    }
    serialized = json.dumps(config, indent=4)
    with open(f"{folder}/state.json", "w") as f:
        f.write(serialized)
        f.write("\n")

def continue_training(folder='results'):
    config = None
    with open(f"{folder}/state.json") as f:
        config = json.load(f)

    available_models_q1 = [
        RandomPlayer(device=config["train_devices"][0])
    ]
    available_models_q2 = [
        RandomPlayer(device=config["train_devices"][0])
    ]

    model_version = config.get("model_version", "fullplayer/v1")
    if model_version == "fullplayer/v1":
        for i in range(config["generation"]):
            q1_model = FullPlayer.load(f"{folder}/{str(i).zfill(4)}-q1.pt", 0, device=config["train_devices"][0])[0]
            available_models_q1.append(q1_model)

            q2_model = FullPlayer.load(f"{folder}/{str(i).zfill(4)}-q2.pt", 0, device=config["train_devices"][0])[0]
            available_models_q2.append(q2_model)
    if model_version == "fullplayer/v2":
        for i in range(config["generation"]):
            q1_model = FullPlayer2.load(f"{folder}/{str(i).zfill(4)}-q1.pt2", 0, device=config["train_devices"][0])[0]
            available_models_q1.append(q1_model)

            q2_model = FullPlayer2.load(f"{folder}/{str(i).zfill(4)}-q2.pt2", 0, device=config["train_devices"][0])[0]
            available_models_q2.append(q2_model)
        


    while True:
        with open(f"{folder}/state.json") as f:
            config = json.load(f)
        print(config)

        current_generation = config["generation"]

        temp_steps = sorted(config['temp_schedule'], key = lambda x: x[0])
        temp = 1
        if current_generation <= temp_steps[0][0]:
            temp = temp_steps[0][1]
        elif current_generation >= temp_steps[-1][0]:
            temp = temp_steps[-1][1]
        else:
            for i in range(1, len(temp_steps)):
                if temp_steps[i][0] >= current_generation:
                    prev_step = temp_steps[i - 1]
                    next_step = temp_steps[i]
                    percent = (current_generation - prev_step[0]) / (next_step[0] - prev_step[0])
                    temp = next_step[1] * percent + prev_step[1] * (1 - percent)
                    break
        print(f"Temperature: {temp}")
        
        do_q_generation(available_models_q1, available_models_q2,
                        temperature=temp,
                        game_batch_size=config["game_batch_size"],
                        num_seatings=config["num_seatings"],
                        steps=config["steps"],
                        lr=config["lr"],
                        model_prob=config["model_prob"],
                        reuse_factor=config["reuse_factor"],
                        seating_offset=config.get('seating_offset', 0),
                        seat_batch=config["seat_batch"],
                        folder=folder,
                        hand_layout=config.get("handnet", None),
                        bid_layout=config["bidnet"],
                        card_layout=config["cardnet"],
                        embedding_path=config["hand_embedding"],
                        train_devices=config["train_devices"],
                        store_device=config["store_device"],
                        model_retention=config["model_retention"],
                        model_version=model_version,
                       )
        new_q1 = available_models_q1[-1]
        new_q2 = available_models_q2[-1]

        if model_version == "fullplayer/v1":
            new_q1.save(f"{folder}/{str(current_generation).zfill(4)}-q1.pt")
            new_q2.save(f"{folder}/{str(current_generation).zfill(4)}-q2.pt")
        if model_version == "fullplayer/v2":
            new_q1.save(f"{folder}/{str(current_generation).zfill(4)}-q1.pt2")
            new_q2.save(f"{folder}/{str(current_generation).zfill(4)}-q2.pt2")
        
        with open(f"{folder}/state.json") as f:
            config = json.load(f)
        config["generation"] = current_generation + 1
        with open(f"{folder}/state.json", "w") as f:
            f.write(json.dumps(config, indent=4))
            f.write("\n")

if not os.path.isfile(f"{sys.argv[1]}/state.json"):
    start_training(sys.argv[1])

continue_training(sys.argv[1])
