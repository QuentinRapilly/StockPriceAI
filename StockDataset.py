from torch.utils.data import Dataset, DataLoader
from typing import Union
import torch
import yfinance as yf
import pandas as pd
import numpy as np
import os

class StockPriceDataset(Dataset):
    def __init__(self, start_date: str="jj-mm-aaaa", end_date: str="jj-mm-aaaa", 
                 interval: int=1, nb_samples: int=20, transform=None,
                 file_dir: str="data/", csv_file: str=None):

        # If a local data file must be loaded:
        if csv_file is not None:
            self.root_dir = file_dir
            self.filename = csv_file
            with open(os.path.join(file_dir,csv_file), 'r') as file:
                data = pd.read_csv(file, sep=',', header='infer')

        else: # Data must be loaded on an online database:
            dataset = yf.download('^GSPC', start=start_date, end=end_date, interval=interval)

        self.data = dataset
        self.nb_samples = nb_samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int, overlapping: bool=True) -> Union[torch.Tensor, float]:
        # Load one sample more than nb_samples for normalizing, transform
        if overlapping:
            sample = self.data['Close'][index:index+self.nb_samples+2]
        else:
            sample = self.data['Close'][index*self.nb_samples:(index+1)*self.nb_samples+2]
        sample = torch.tensor(sample)
        if self.transform:
            sample = self.transform(sample)[1:]
        else:
            sample = sample[1:]
        label = sample[-1] # label is the last elem of sample
        sample = sample[:-1] # removes label from sample
        return sample, label

    def get_normalization_value(self, index):
        # Retrieve the normalization value from the last sample
        return self.data['Close'][index]

def normalize_by_last_unknown_price(sample: torch.Tensor) -> torch.Tensor:
    """Divides the whole stock price sample by the last unknown price w_{p*t-1}"""
    last_price = sample[0] # w_{pt-1}
    return sample/last_price


if __name__ == "__main__":

    # download S&P data from yahoo finance
    START_DATE = '1950-01-03'
    END_DATE = '2021-11-16'
    INTERVAL = '1d'
    nb_samples = 20

    dataset = StockPriceDataset(START_DATE, END_DATE, INTERVAL, nb_samples,
                                transform=normalize_by_last_unknown_price)

    dataloader = DataLoader(dataset, batch_size=64)
    for i_batch, batch in enumerate(dataloader):
        print("i_batch = {}, batch = {}".format(i_batch, batch))
