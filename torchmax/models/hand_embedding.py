import torch
from torch import nn

class HandEmbeddingNet(torch.nn.Module):
    def __init__(self, output_size, layer_sizes):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.output_size = output_size
        
        layers = []
        current_size = 52
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_size, layer_size))
            current_size = layer_size
            layers.append(nn.ReLU())

        layers.append(nn.Linear(current_size, output_size))
        layers.append(nn.Tanh());
        self.linear_relu_stack = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.linear_relu_stack(x)
    
    def saveData(self, meta):
        return {
            'layer_sizes': self.layer_sizes,
            'output_size': self.output_size,
            'weights': self.state_dict(),
            'meta': meta,
        }
    
    def save(self, path, meta):
        state_dict = self.saveData(meta)
        torch.save(state_dict, path)


    @staticmethod
    def load(path, device=None):
        state_dict = torch.load(path, weights_only=False, map_location=device)
        return HandEmbeddingNet.loadState(state_dict, device)

    @staticmethod
    def loadState(state_dict, device=None):
        weights = state_dict['weights']

        layer_sizes = state_dict['layer_sizes']
        output_size = state_dict['output_size']
        
        result = HandEmbeddingNet(output_size, layer_sizes).to(device=device)
        result.load_state_dict(weights)
        return (result, state_dict.get('meta', {}))