from torch.nn import LSTM, Module, Dropout, ModuleList

class StockAI(Module):
     
    def __init__(self, input_size, lstm_size, num_layers, keep_prob) -> None:
        super().__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.lstm = LSTM(self.input_size, hidden_size=self.lstm_size, num_layers=self.num_layers, 
                         dropout=1-keep_prob, batch_first=True, proj_size=1)

    
    def forward(self,x):
        a, b = self.lstm(x)
        return b[0]
    