import torch
import torch.nn as nn

class LSTM_GainScheduler(nn.Module):
    def __init__(self, input_size=13, lstm_hidden=8):
        super().__init__()
        self.lstm   = nn.LSTM(input_size, lstm_hidden, batch_first=True)
        self.linear = nn.Linear(input_size + lstm_hidden, 3)

    def forward(self, x):
        _, (h, _) = self.lstm(x)              # h: (1, batch, lstm_hidden)
        context   = h.squeeze(0)              # (batch, lstm_hidden)
        last      = x[:, -1, :]              # (batch, input_size)
        combined  = torch.cat([last, context], dim=1)
        return torch.sigmoid(self.linear(combined)) * 2.0   # [0, 2] range