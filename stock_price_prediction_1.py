from StockAI import StockAI
from Config import StockAIConfig
from StockDataset import StockPriceDataset, normalize_by_last_unknown_price

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import RMSprop

# Model config
config = StockAIConfig().config

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Init of the Dataset
dataset = StockPriceDataset(config["dataset"]["start_date"], 
                            config["dataset"]["end_date"],
                            config["dataset"]["interval_date"], 
                            config["dataset"]["nb_samples"],
                            transform=normalize_by_last_unknown_price)

# Init dataloader
dataloader = DataLoader(dataset, config["dataset"]["batch_size"], config["dataset"]["shuffle"], drop_last=True)

# Init of the model
model = StockAI(config["model"]["input_size"],
                config["model"]["lstm_size"],
                config["model"]["num_layers"],
                config["model"]["keep_prob"])

# Learning rate to use along the epochs
learning_rates = [config["learning"]["init_lr"] * (config["learning"]["lr_decay"] ** max(float(i + 1 - config["learning"]["init_epoch"]), 0.0)) for i in range(config["learning"]["max_epoch"])]

# Loss
loss_fn = MSELoss()
optimizer = RMSprop(model.parameters(), lr=learning_rates[0], eps=1e-08)

# Learning
for epoch_step in range(config["learning"]["max_epoch"]):
    lr = learning_rates[epoch_step]
    print(f"Running for epoch {epoch_step}...")
    for i_batch, batch in enumerate(dataloader):
        x, y = batch
        x = torch.unsqueeze(x, -1).float()
        y = y.float()
        x, y = x.to(device), y.to(device)
        y_pred = model.forward(x)
        loss = torch.autograd.Variable(loss_fn(y_pred, y), requires_grad=True)

        if i_batch%10==0:
            print(f"step: {i_batch}, loss = {loss}")

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        # loss.backward()
        optimizer.step()
