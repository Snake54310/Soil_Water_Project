import torch
import torch.nn as nn

class LSTM_Threshold(nn.Module):
    def __init__(self, input_size=13, lstm_hidden=8):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, lstm_hidden, batch_first=True)
        self.linear  = nn.Linear(input_size + lstm_hidden, 1)
        self.sigmoid = nn.Sigmoid()
        # Bias is intentionally NOT set here.
        # initialize_lstm_models.py is the single source of truth for bias
        # initialization. Calling initialize_models() or loading a .pth is
        # always required before training or inference.

    def forward(self, x):
        _, (h, _) = self.lstm(x)              # h: (1, batch, lstm_hidden)
        context   = h.squeeze(0)              # (batch, lstm_hidden)
        last      = x[:, -1, :]              # (batch, input_size) — current timestep
        combined  = torch.cat([last, context], dim=1)
        return self.sigmoid(self.linear(combined))