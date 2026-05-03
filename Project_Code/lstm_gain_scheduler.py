import torch
import torch.nn as nn

class LSTM_GainScheduler(nn.Module):
    def __init__(self, input_size=13):
        super().__init__()
        self.linear = nn.Linear(input_size, 3)

    def forward(self, x):
        last = x[:, -1, :]
        raw = torch.sigmoid(self.linear(last)) * 2.0   # [0, 2] range
        return raw