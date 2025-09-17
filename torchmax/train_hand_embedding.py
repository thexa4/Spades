import torch
from torch import nn
from models.hand_embedding import HandEmbeddingNet
import sys
import random
import pytorch_metric_learning
import pytorch_metric_learning.losses
import pytorch_metric_learning.regularizers

margin = 0.5
#output_size = 1024
#layers = [1024, 1024, 1024]
output_size = 128
layers = [128, 128, 128]
batch = 1024 * 1
summation_steps = 1
device = 'cuda:0'
epochs = 300

embedding = HandEmbeddingNet(output_size, layers).to(device)

optimizer = torch.optim.Adam(embedding.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, steps_per_epoch=1, epochs=epochs)

lossfunc = pytorch_metric_learning.losses.SelfSupervisedLoss(
    pytorch_metric_learning.losses.TripletMarginLoss(
        margin=margin,
        distance=pytorch_metric_learning.distances.CosineSimilarity(),
        triplets_per_anchor=128,
        embedding_regularizer=pytorch_metric_learning.regularizers.ZeroMeanRegularizer()
    ),
    symmetric = True
)
curloss = 10
for i in range(epochs):

    optimizer.zero_grad()

    for step in range(summation_steps):
        total_loss = torch.tensor(0, device=device).to(torch.float)

        prevs = []
        nexts = []
        for target_hand in range(12):
            deck1 = torch.stack([torch.randperm(52, device=device)[0:13] for _ in range(batch)]).requires_grad_(False)

            prev_mask = torch.tensor([x >= target_hand for x in range(13)], device=device).expand((batch, -1))
            cur_mask = torch.tensor([x > target_hand for x in range(13)], device=device).expand((batch, -1))
            
            prev_masked = deck1[prev_mask].reshape((batch, -1)).requires_grad_(False)
            cur_masked = deck1[cur_mask].reshape((batch, -1)).requires_grad_(False)
                
            prev_hand = torch.zeros(batch, 52, dtype=torch.uint8, device=device).requires_grad_(False)
            cur_hand = torch.zeros(batch, 52, dtype=torch.uint8, device=device).requires_grad_(False)
                
            prev_hand[torch.arange(batch).unsqueeze(1), prev_masked] = 1
            cur_hand[torch.arange(batch).unsqueeze(1), cur_masked] = 1

            prevs.append(prev_hand.to(torch.float))
            nexts.append(cur_hand.to(torch.float))

        prev_embedding = embedding.forward(torch.cat(prevs).requires_grad_(False))
        cur_embedding = embedding.forward(torch.cat(nexts).requires_grad_(False))

        loss = lossfunc(prev_embedding, cur_embedding)

        total_loss += loss

    curloss = total_loss.item()
    print(f"lr {lr_scheduler.get_last_lr()}: {curloss}")
    total_loss.backward()
    optimizer.step()
    lr_scheduler.step()


embedding.save(sys.argv[1], {})
