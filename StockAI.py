from torch.nn import LSTM, Module, Dropout, ModuleList

class StockAI(Module):
     
    def __init__(self, input_size, lstm_size, num_layers, keep_prob) -> None:
        super().__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.lstm_list = ModuleList([LSTM(self.input_size, self.lstm_size) for _ in range(self.num_layers)])
        self.dropout_list = ModuleList([Dropout() for _ in range(self.num_layers)])

    
    def forward(self,x):
        y = x
        for i in range(self.num_layers):
            y = self.lstm_list[i](x)
            y = self.dropout_list[i](x)
        return y
    