from StockAI import StockAI
from Config import StockAIConfig
from StockDataset import StockPriceDataset, normalize_by_last_unknown_price
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader, Subset
from torch.nn import MSELoss
from torch.optim import RMSprop

TEST = False

def train_test_dataset(dataset, val_split=0.2):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=val_split, shuffle=False)
    return Subset(dataset, train_idx), Subset(dataset, test_idx)

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

dataset_train, dataset_test = train_test_dataset(dataset, val_split=0.1)
print(f"dataset_train = {dataset_train.dataset}")
print(f"dataset_train = {dataset_test.dataset}")

# Init dataloader
dataloader_train = DataLoader(dataset_train, config["dataset"]["batch_size"], config["dataset"]["shuffle"], drop_last=True)
dataloader_test = DataLoader(dataset_test, config["dataset"]["batch_size"], config["dataset"]["shuffle"], drop_last=True)

# Init of the model
model = StockAI(config["model"]["input_size"],
                config["model"]["lstm_size"],
                config["model"]["num_layers"],
                config["model"]["keep_prob"])
model.to(device)

# Learning rate to use along the epochs
learning_rates = [config["learning"]["init_lr"] * (config["learning"]["lr_decay"] ** max(float(i + 1 - config["learning"]["init_epoch"]), 0.0)) for i in range(config["learning"]["max_epoch"])]

# Loss
loss_fn = MSELoss()
optimizer = RMSprop(model.parameters(), lr=learning_rates[0], eps=1e-08)

# Learning
for epoch_step in range(config["learning"]["max_epoch"]):
    lr = learning_rates[epoch_step]
    print(f"Running for epoch {epoch_step}...")
    for i_batch, batch in enumerate(dataloader_train):
        x, y = batch
        x = torch.unsqueeze(x, -1).float()
        y = y.float()
        x, y = x.to(device), y.to(device)
        y_pred = torch.squeeze(model.forward(x))
        loss = loss_fn(y_pred, y)

        if i_batch%100==0:
            print(f"step: {i_batch}, loss = {loss}")

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#test

if TEST:
    runnning_mape = 0
    for i_batch, batch in enumerate(dataloader_test):
            x, y = batch
            x = torch.unsqueeze(x, -1).float()
            y = y.float()
            x, y = x.to(device), y.to(device)
            y_pred = model.forward(x)
            error = torch.mean(torch.abs((y - y_pred) / y))
            runnning_mape += error

    mape = runnning_mape / len(dataloader_test)
    print("",mape)

# Display predicted data
nb_test = len(dataset_test)
print("longueur du dataset test = ", nb_test)
y_truth = []
y_hat = []
for i in range(nb_test):
  x, y = dataset_test.dataset.__getitem__(i)
  norm_value = dataset_test.dataset.get_normalization_value(i)
  x = torch.unsqueeze(torch.unsqueeze(x, 0), -1).float()
  x = x.to(device)
  y_pred = model.forward(x)
  y_truth.append(torch.Tensor.cpu(y).detach().numpy()*norm_value)
  y_hat.append(float(torch.Tensor.cpu(y_pred).detach().numpy())*norm_value)

plt.plot(y_truth, label="real value")
plt.plot(y_hat, label="predicted_value")
plt.grid()
plt.show()