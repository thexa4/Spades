import torch
from torch import nn
from models.hand_embedding import HandEmbeddingNet
import sys

margin = 0.3
output_size = 10 
layers = [128]
batch = 10240
device = 'cuda:0'

embedding = HandEmbeddingNet(output_size, layers).to(device)

optimizer = torch.optim.Adam(embedding.parameters(), lr=0.001)
curloss = 10
while curloss > 0.0001:

    optimizer.zero_grad()
    total_loss = torch.tensor(0, device=device).to(torch.float)

    deck1 = torch.stack([torch.randperm(52, device=device)[0:13] for _ in range(batch)])
    deck2 = torch.stack([torch.randperm(52, device=device)[0:13] for _ in range(batch)])

    prev_hand1 = torch.zeros(batch, 52, dtype=torch.uint8, device=device)
    prev_hand2 = torch.zeros(batch, 52, dtype=torch.uint8, device=device)

    prev_hand1[torch.arange(batch).unsqueeze(1), deck1] = 1
    prev_hand2[torch.arange(batch).unsqueeze(1), deck2] = 1

    prev_embedding1 = embedding.forward(prev_hand1.to(torch.float))
    prev_embedding2 = embedding.forward(prev_hand2.to(torch.float))

    for i in range(12):

        mask = torch.tensor([x > i for x in range(13)], device=device).expand((batch, -1))
        masked1 = deck1[mask].reshape((batch, -1))
        masked2 = deck2[mask].reshape((batch, -1))
        
        hand1 = torch.zeros(batch, 52, dtype=torch.uint8, device=device)
        hand2 = torch.zeros(batch, 52, dtype=torch.uint8, device=device)
        
        hand1[torch.arange(batch).unsqueeze(1), masked1] = 1
        hand2[torch.arange(batch).unsqueeze(1), masked2] = 1

        new_embedding1 = embedding.forward(hand1.to(torch.float))
        new_embedding2 = embedding.forward(hand2.to(torch.float))

        error1 = torch.nn.functional.mse_loss(prev_embedding1, new_embedding1)
        error2 = torch.nn.functional.mse_loss(prev_embedding1, prev_embedding2)

        loss = torch.clamp(error1 - error2 + margin, 0, None)

        total_loss += loss

        prev_embedding1 = new_embedding1
        prev_embedding2 = new_embedding2

    
    curloss = total_loss.item()
    print(curloss)
    total_loss.backward()
    optimizer.step()


embedding.save(sys.argv[1], {})
