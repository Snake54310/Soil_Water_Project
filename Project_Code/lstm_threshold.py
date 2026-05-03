import torch.nn as nn

class LSTM_Threshold(nn.Module):
    def __init__(self, input_size=13):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        # Bias is intentionally NOT set here.
        # initialize_lstm_models.py is the single source of truth for bias
        # initialization (threshold=1.3 → ~0.79 prob, gain=0.0 → mult 1.0).
        # Calling initialize_models() or loading a .pth is always required
        # before training or inference.

    def forward(self, x):
        last = x[:, -1, :]                 # only last timestep
        return self.sigmoid(self.linear(last))