import torch
from torch import nn
from models.played_embedding import PlayedEmbeddingNet
import sys
import random
import pytorch_metric_learning
import pytorch_metric_learning.losses
import pytorch_metric_learning.regularizers

margin = 0.5
output_size = 128
layers = [128, 128, 128]
batch = 1024 * 1
summation_steps = 1
device = 'cuda:1'
epochs = 200

embedding = PlayedEmbeddingNet(output_size, layers).to(device)

optimizer = torch.optim.Adam(embedding.parameters(), lr=0.005)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=1, epochs=epochs)
curloss = 10

lossfunc = pytorch_metric_learning.losses.SelfSupervisedLoss(
    pytorch_metric_learning.losses.TripletMarginLoss(
        margin=margin,
        distance=pytorch_metric_learning.distances.CosineSimilarity(),
        triplets_per_anchor=128,
        embedding_regularizer=pytorch_metric_learning.regularizers.ZeroMeanRegularizer()
    ),
    symmetric = True
)

batch_indices = torch.arange(batch).unsqueeze(1)
for i in range(epochs):

    total_loss = torch.tensor(0, device=device).to(torch.float)

    optimizer.zero_grad()

    for step in range(summation_steps):
        prevs = []
        nexts = []

        for target_round in range(13):
            deck = torch.stack([torch.randperm(52, device=device)[0:13] for _ in range(batch)]).requires_grad_(False)

            played = torch.zeros(batch, 13, 52, dtype=torch.uint8, device=device).requires_grad_(False)

            for i in range(target_round):
                played[batch_indices, i, deck[batch_indices, i]] = 1
            
            prevs.append(embedding.forward(played.to(torch.float)))
            played[batch_indices, target_round, deck[batch_indices, target_round]] = 1
            nexts.append(embedding.forward(played.to(torch.float)))

        loss = lossfunc(torch.cat(prevs), torch.cat(nexts))

        total_loss += loss
    
    curloss = total_loss.item()
    print(f"lr {lr_scheduler.get_last_lr()}: {curloss}")
    total_loss.backward()
    optimizer.step()
    lr_scheduler.step()


embedding.save(sys.argv[1], {})
