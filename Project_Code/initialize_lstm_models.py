import torch
import torch.nn as nn
from lstm_threshold import LSTM_Threshold
from lstm_gain_scheduler import LSTM_GainScheduler

def initialize_models(input_size=13, lstm_hidden=8, save=True):
    print(f"Creating LSTM+Linear models ({input_size} inputs, hidden={lstm_hidden})...")

    # Threshold model
    model1 = LSTM_Threshold(input_size=input_size, lstm_hidden=lstm_hidden)
    # LSTM weights: orthogonal init is standard for recurrent layers
    for name, param in model1.lstm.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            param.data.zero_()
    # Linear layer: small Xavier + bias nudged to ~0.79 starting probability
    nn.init.xavier_uniform_(model1.linear.weight, gain=0.5)
    model1.linear.bias.data.fill_(1.3)

    # Gain scheduler model
    model2 = LSTM_GainScheduler(input_size=input_size, lstm_hidden=lstm_hidden)
    for name, param in model2.lstm.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            param.data.zero_()
    # Linear layer: small Xavier + bias=0 → sigmoid(0)*2=1.0, neutral multiplier
    nn.init.xavier_uniform_(model2.linear.weight, gain=0.5)
    model2.linear.bias.data.zero_()

    if save:
        torch.save(model1.state_dict(), "lstm_threshold.pth")
        torch.save(model2.state_dict(), "lstm_gain_scheduler.pth")
        print("Done. Models created and saved.")
        print("   • Threshold starts at probability ~0.7858")
        print("   • Gain scheduler starts at multiplier exactly 1.0")
    else:
        print("Models created in memory (not saved).")

    return model1, model2

if __name__ == "__main__":
    initialize_models(input_size=13, lstm_hidden=8)