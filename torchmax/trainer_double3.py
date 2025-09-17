import torch
import random
import sys
import json
import os.path
import collections
import threading
import statistics
import math

from simulator import run_double_game
from models.random_player import RandomPlayer
from models.fullplayer3 import FullPlayer3
from training_sample import TrainingSample
from multiprocessing.pool import ThreadPool

def train(model, optimizer, lr_scheduler, samples, testset=None, devices=[None]):
    device_models = model.replicate(devices)
    device_samples = []
    total_loss = torch.tensor(0, device=model.device)
    
    for i in range(len(devices)):
        device_samples.append(TrainingSample.concat(samples[i::len(devices)]).with_device(devices[i]))
    
    optimizer.zero_grad()
    
    def compute_error(game, model, testset=False):
        bid_losses = None
        card_losses = []

        bid_result = model.bid(game.bid_state, game.bid_result)

        own_mean = bid_result["own_score_mean"]
        own_stddev2 = bid_result["own_score_stddev"] * bid_result["own_score_stddev"]
        own_log_likelyhood = torch.mean(0.5 * torch.log(2 * math.pi * own_stddev2) + torch.square(game.own_outcome[:, 0].float() - own_mean) / (2 * own_stddev2))

        other_mean = bid_result["other_score_mean"]
        other_stddev2 = bid_result["other_score_stddev"] * bid_result["other_score_stddev"]
        other_log_likelyhood = torch.mean(0.5 * torch.log(2 * math.pi * other_stddev2) + torch.square(game.other_outcome[:, 0].float() - other_mean) / (2 * other_stddev2))

        bid_stddev = torch.mean(torch.concat([bid_result["own_score_stddev"], bid_result["other_score_stddev"]]))

        error = own_log_likelyhood + other_log_likelyhood
        bid_losses = error
        round_losses = []
        round_stddevs = []

        for round in range(13):
            card_result = model.play(game.hand_states[round], game.hand_allowed[round], game.hand_results[round])

            own_mean = card_result["own_score_mean"]
            own_stddev2 = card_result["own_score_stddev"] * bid_result["own_score_stddev"]
            own_log_likelyhood = torch.mean(0.5 * torch.log(2 * math.pi * own_stddev2) + torch.square(game.own_outcome[:, 0].float() - own_mean) / (2 * own_stddev2))
            
            other_mean = card_result["other_score_mean"]
            other_stddev2 = card_result["other_score_stddev"] * bid_result["other_score_stddev"]
            other_log_likelyhood = torch.mean(0.5 * torch.log(2 * math.pi * other_stddev2) + torch.square(game.other_outcome[:, 0].float() - other_mean) / (2 * other_stddev2))
            
            #own_bag_error = torch.nn.functional.mse_loss(card_result["own_score_bags"], game.own_outcome[:, 1].float())
            #other_bag_error = torch.nn.functional.mse_loss(card_result["other_score_bags"], game.own_outcome[:, 1].float())

            #error = 0.1 * (own_bag_error + other_bag_error) + own_log_likelyhood + other_log_likelyhood
            error = own_log_likelyhood + other_log_likelyhood

            card_losses.append(error)
            round_losses.append(error)
        
            round_stddevs.append(torch.mean(torch.concat([card_result["own_score_stddev"], card_result["other_score_stddev"]])))

        bid_total_loss = bid_losses
        card_total_loss = torch.mean(torch.stack(card_losses))

        return {
            "total": (bid_total_loss * 26.0 + card_total_loss).to(model.device),
            "bids": bid_total_loss.item(),
            "bid_stddev": bid_stddev.item(),
            "cards": card_total_loss.item(),
            "rounds": [r.item() for r in round_losses],
            "round_stddevs": [r.item() for r in round_stddevs],
        }


    def do_device(device_id):
        game = device_samples[device_id]
        return compute_error(game, device_models[device_id])
    
    pool = ThreadPool(len(devices))
    results = pool.map(do_device, range(len(devices)))

    total_loss = sum([x['total'].to(device=model.device) for x in results], total_loss)
    card_loss = statistics.mean([x['cards'] for x in results])
    bid_loss = statistics.mean([x['bids'] for x in results])

    bid_stddevs = statistics.mean([x['bid_stddev'] for x in results])

    round_matrix = [x['rounds'] for x in results]
    rounds = []
    round_stddev_matrix = [x['round_stddevs'] for x in results]
    round_stddevs = []
    for i in range(13):
        rounds.append(round(statistics.mean([r[i] for r in round_matrix]), ndigits=1))
        round_stddevs.append(round(statistics.mean([r[i] for r in round_stddev_matrix]), ndigits=1))

    test_bid_loss = -1
    test_card_losses = [-1 for _ in range (13)]
    if testset != None:
        game = testset

        outcome = compute_error(game, device_models[0], testset=True)
        test_bid_loss = outcome["bids"]
        test_card_losses = outcome["rounds"]

    total_loss.backward()
    optimizer.step()
    lr_scheduler.step()
    sys.stdout.write('_')
    sys.stdout.flush()

    return {
        "bids": bid_loss,
        "cards": card_loss,
        "rounds": rounds,
        "test_bid": test_bid_loss,
        "test_round_loss": test_card_losses,
        "bid_stddev": bid_stddevs,
        "round_stddevs": round_stddevs,
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

def do_q_generation(q1_models, q2_models, temperature=1000, caution=0.1, game_batch_size=128, num_seatings=64,
                    steps=100, model_prob=0.2, lr=None, reuse_factor=10, seating_offset=0, seat_batch=2, folder='results',
                    bid_layout=None, card_layout=None, hand_embedding_path=None, played_embedding_path=None,
                    train_devices=['cuda:0'], store_device='cpu', model_retention=0, hand_detach=False,
                    played_detach=False):
        
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
    }
    

    def do_generation(q, models, guiding_model):
        training_model = FullPlayer3(0, caution, bid_layout, card_layout, hand_embedding_path=hand_embedding_path, played_embedding_path=played_embedding_path, device=train_devices[0])
        if hand_detach:
            training_model.detach_handnet()
        if played_detach:
            training_model.detach_playednet()

        optimizer = torch.optim.AdamW(training_model.parameters(), lr=lr)

        if model_retention > 0 and len(models) > 1:
            training_model = models[-1].mix(training_model, 1 - model_retention)
            optimizer = torch.optim.AdamW(training_model.parameters(), lr=lr)
            if models[-1].optim_state != None:
                optimizer.load_state_dict(models[-1].optim_state)
            else:
                print(f"Previous q{q} does not have optimizer state")

        curstep = 0
        if os.path.isfile(f"{folder}/q{q}_ckpt.pt3"):
            test_training_model, metadata = FullPlayer3.load(f"{folder}/q{q}_ckpt.pt3", 0, device=train_devices[0])
            if metadata['state'] == meta:
                curstep = metadata['curstep'] + 1
                training_model = test_training_model
                optimizer = torch.optim.AdamW(training_model.parameters(), lr=lr)
                if training_model.optim_state != None:
                    optimizer.load_state_dict(training_model.optim_state)
                else:
                    print(f"Q{q} checkpoint does not have optimizer state")
            else:
                print(f"Checkpoint for q{q} not applicable")
    
        if curstep >= steps:
            return training_model

        batch_count = math.ceil((reuse_factor * num_seatings + seating_offset) / seat_batch + 0.5)
        last_batch_step = -1
        if curstep > 0:
            last_batch_step = curstep * batch_count
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=batch_count, epochs=steps, last_epoch=last_batch_step)
        
        print('Generating testset')
        testset = []
        testset_models = models[-10:]
        if len(testset_models) < 10:
            testset_models = [RandomPlayer().with_device(train_devices[0]) for _ in range(10 - len(testset_models))] + testset_models
        for i in range(10):
            random.shuffle(testset_models)
            pairs = [(guiding_model, x) for x in testset_models]
            for a,b in pairs:
                sys.stdout.write('.')
                sys.stdout.flush()
                testset.append(run_double_game(4, [a, b, a, b], device=train_devices[0]).with_device(train_devices[0]))
        testset = sum(testset[1:], testset[0])
        print('')
        
        best_error = 10000
        best_model = training_model.with_temperature(0)
        best_model_round = -1

        resevoir = collections.deque()
    
        for s in range(curstep, steps):
            while len(resevoir) < reuse_factor * num_seatings + seating_offset:
                tournament_samples = do_tournament(guiding_model.with_temperature(temperature), models, game_batch_size, reuse_factor * num_seatings + seating_offset - len(resevoir), model_prob, devices=train_devices, store_device=store_device)
                for samp in tournament_samples:
                    resevoir.append(samp)

            todo_list = list(resevoir)
            random.shuffle(todo_list)
            todo_list = [todo_list[x:x+seat_batch] for x in range(0, len(todo_list), seat_batch)]

            card_losses = []
            bid_losses = []
            bid_stddevs = []
            round_stddevs = []
            round_losses = [list() for _ in range(13)]
            test_bid_loss = None
            test_card_losses = None

            sample_counter = 0
            for samples in todo_list:
                sample_counter += 1
                
                if sample_counter < len(todo_list):
                    losses = train(training_model, optimizer, lr_scheduler, samples, testset=None, devices=train_devices)
                else:
                    losses = train(training_model, optimizer, lr_scheduler, samples, testset=testset, devices=train_devices)
                    test_bid_loss = losses['test_bid']
                    test_card_losses = losses['test_round_loss']

                card_losses.append(losses['cards'])
                bid_losses.append(losses['bids'])
                bid_stddevs.append(losses['bid_stddev'])
                round_stddevs.append(statistics.mean(losses['round_stddevs']))
                for i in range(13):
                    round_losses[i].append(losses['rounds'][i])

            print()
            
            total_test_loss = test_bid_loss + sum(test_card_losses)
            training_model.optim_state = optimizer.state_dict()
            training_model.save(f"{folder}/q{q}_ckpt.pt3", {'state': meta, 'curstep': s})

            if total_test_loss < best_error:
                best_error = total_test_loss
                best_model = training_model.with_temperature(0)
                best_model_round = s

            print(f"Q{q} step {s}, bid {round(statistics.mean(bid_losses), ndigits=1)} ~{round(statistics.mean(bid_stddevs), ndigits=1)}, cards: {round(statistics.mean(card_losses), ndigits=1)} ~{round(statistics.mean(round_stddevs), ndigits=1)}, lr: {round(lr_scheduler.get_last_lr()[0], ndigits=6)}")
            print([round(statistics.mean(round_losses[i]), ndigits=1) for i in range(13)])
            print(f"Test: bid {round(test_bid_loss, ndigits=1)}, cards: {[round(test_card_losses[i], ndigits=1) for i in range(13)]}; best: {round(best_error, ndigits=1)} @{best_model_round}")
            for _ in range(num_seatings):
                resevoir.popleft()
            samples = None
        optimizer.zero_grad()
        optimizer = None

        return training_model.with_temperature(0)
    
    q1_training_model = do_generation(1, q1_models, q2_models[-1])
    q2_training_model = do_generation(2, q2_models, q1_models[-1])
    
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
        "hand_detach": False,
        "played_detach": False,
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

    for i in range(config["generation"]):
        q1_model = FullPlayer3.load(f"{folder}/{str(i).zfill(4)}-q1.pt3", 0, device=config["train_devices"][0])[0]
        available_models_q1.append(q1_model)

        q2_model = FullPlayer3.load(f"{folder}/{str(i).zfill(4)}-q2.pt3", 0, device=config["train_devices"][0])[0]
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
                        caution=config["caution"],
                        lr=config["lr"],
                        model_prob=config["model_prob"],
                        reuse_factor=config["reuse_factor"],
                        seating_offset=config.get('seating_offset', 0),
                        seat_batch=config["seat_batch"],
                        folder=folder,
                        bid_layout=config["bidnet"],
                        card_layout=config["cardnet"],
                        hand_embedding_path=config["hand_embedding"],
                        played_embedding_path=config["played_embedding"],
                        train_devices=config["train_devices"],
                        store_device=config["store_device"],
                        model_retention=config["model_retention"],
                        hand_detach=config["hand_detach"],
                        played_detach=config["played_detach"],
                       )
        new_q1 = available_models_q1[-1]
        new_q2 = available_models_q2[-1]

        new_q1.save(f"{folder}/{str(current_generation).zfill(4)}-q1.pt3")
        new_q2.save(f"{folder}/{str(current_generation).zfill(4)}-q2.pt3")
        
        with open(f"{folder}/state.json") as f:
            config = json.load(f)
        config["generation"] = current_generation + 1
        with open(f"{folder}/state.json", "w") as f:
            f.write(json.dumps(config, indent=4))
            f.write("\n")

if not os.path.isfile(f"{sys.argv[1]}/state.json"):
    start_training(sys.argv[1])

continue_training(sys.argv[1])
