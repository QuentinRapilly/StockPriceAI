from typing import Tuple
import numpy as np
import os

from torch.utils.data import Dataset

class ViTDataset(Dataset):
    def __init__(self, start_date: str="jj-mm-aaaa", end_date: str="jj-mm-aaaa", 
                 interval: int=1, input_size: int=150, output_size : int=10, normalize: bool=True,
                 file_dir: str="data/", csv_file: str=None):
      
        self.input_size = input_size
        self.output_size = output_size
        self.tot_size = input_size + output_size
        self.normalize = normalize

        if csv_file is not None: # If a local data file must be loaded:
            self.root_dir = file_dir
            self.filename = csv_file
            with open(os.path.join(file_dir,csv_file), 'r') as file:
                data = pd.read_csv(file, sep=',', header='infer')
        else: # Data must be loaded on an online database:
            data = yf.download('^GSPC', start=start_date, end=end_date, interval=interval)
        
        self.data = data
        seq_init = data['Close'].tolist()
        if self.normalize :
            seq = [seq_init[i]/seq_init[i-1] -1 for i in range(1, len(seq_init))]

        # split into items of size nb_samples
        X = [np.array(seq[i:i+self.input_size]) for i in range(len(seq)-self.tot_size)]
        X = np.array(X)
        Y = [np.array(seq[i+self.input_size:i+self.tot_size]) for i in range(len(seq)-self.tot_size)]
        Y = np.array(Y)

        self.X = X
        self.Y = Y
        self.norm_val = np.array(seq_init[1:])


    def __len__(self) -> int:
        return self.X.shape[0]


    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load one sample more than nb_samples for normalizing, transform
        input = torch.unsqueeze(torch.tensor(self.X[index]),0) # unsqueeze is used to simulate the channel dimension (in case we have the time to predict multiple stocks)
        output = torch.unsqueeze(torch.tensor(self.Y[index]),0)
        return input, output

    def get_input_output(self):
      return self.X, self.Y
      

    def get_normalization_value(self, index):
        # Retrieve the normalization value from the last sample
        return self.norm_val[index]