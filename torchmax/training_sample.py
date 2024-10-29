import torch

from gamestate import GameState


class TrainingSample:
    def __init__(self):
        self.bid_state = None
        self.hand_states = [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
        self.bid_result = None
        self.hand_results = [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
        self.hand_allowed = [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
        self.own_outcome = None
        self.other_outcome = None

    def __add__(self, other):
        if self.bid_state == None:
            return other
        if other.bid_state == None:
            return self
        
        result = TrainingSample()
        result.bid_state = self.bid_state + other.bid_state
        result.hand_states = [self.hand_states[i] + other.hand_states[i] for i in range(len(self.hand_states))]
        result.bid_result = torch.concat([self.bid_result, other.bid_result], 0)
        result.hand_results = [torch.concat([self.hand_results[i], other.hand_results[i]], 0) for i in range(len(self.hand_results))]
        result.hand_allowed = [torch.concat([self.hand_allowed[i], other.hand_allowed[i]], 0) for i in range(len(self.hand_allowed))]
        result.own_outcome = torch.concat([self.own_outcome, other.own_outcome], 0)
        result.other_outcome = torch.concat([self.other_outcome, other.other_outcome], 0)
        return result
    
    @staticmethod
    def concat(parts):
        active_parts = [x for x in parts if x.bid_state != None]

        result = TrainingSample()
        if len(active_parts) == 0:
            return result
        result.bid_state = GameState.concat([x.bid_state for x in active_parts])
        for i in range(len(active_parts[0].hand_states)):
            result.hand_states[i] = GameState.concat([x.hand_states[i] for x in active_parts])
            result.hand_results[i] = torch.concat([x.hand_results[i] for x in active_parts])
            result.hand_allowed[i] = torch.concat([x.hand_allowed[i] for x in active_parts])

        result.bid_result = torch.concat([x.bid_result for x in active_parts], 0)
        result.own_outcome = torch.concat([x.own_outcome for x in active_parts], 0)
        result.other_outcome = torch.concat([x.other_outcome for x in active_parts], 0)
        return result

    def with_device(self, device):
        result = TrainingSample()
        result.bid_state = self.bid_state.with_device(device)
        result.hand_states = [x.with_device(device) for x in self.hand_states]
        result.bid_result = self.bid_result.to(device=device)
        result.hand_results = [x.to(device=device) for x in self.hand_results]
        result.hand_allowed = [x.to(device=device) for x in self.hand_allowed]
        result.own_outcome = self.own_outcome.to(device=device)
        result.other_outcome = self.other_outcome.to(device=device)
        return result
        
