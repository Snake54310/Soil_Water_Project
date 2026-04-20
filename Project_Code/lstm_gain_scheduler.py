import torch
import torch.nn as nn

class LSTM_GainScheduler(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        # NEW scaling as requested
        multipliers = torch.sigmoid(self.fc(last_hidden)) * 2.0
        return multipliers