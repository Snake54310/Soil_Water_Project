import torch
from torch import nn

from lstm_threshold import LSTM_Threshold
from lstm_gain_scheduler import LSTM_GainScheduler


def initialize_models(input_size=2, save=True):
    """
    Creates fresh LSTM models.
    Default = 13 inputs (12 original + current_error).
    Call with input_size=12 if you ever want to go back.
    """
    print(f"Creating initial LSTM models with {input_size} input features...")

    # LSTM 1 - Threshold (soft tuner)
    model1 = LSTM_Threshold(input_size=input_size)
    nn.init.xavier_uniform_(model1.fc.weight, gain=0.5)  # small but non-zero
    model1.fc.bias.data.fill_(1.335) # sigmoid(1.335) ≈ 0.7917

    # LSTM 2 - Gain Scheduler
    model2 = LSTM_GainScheduler(input_size=input_size)
    nn.init.xavier_uniform_(model2.fc.weight, gain=0.5)
    model2.fc.bias.data.fill_(0.0)  # neutral → final_mult = 1.0

    if save:
        torch.save(model1.state_dict(), "lstm_threshold.pth")
        torch.save(model2.state_dict(), "lstm_gain_scheduler.pth")
        print(f"Done. Initial {input_size}-input models created and saved.")
        print("   • Threshold model starts at ~0.7917 probability")
        print("   • Gain scheduler starts at exactly 1.0 multiplier")
    else:
        print("Models created in memory (not saved).")

    return model1, model2


if __name__ == "__main__":
    # Default: create 13-input models (recommended now that you're adding current_error)
    initialize_models(input_size=2)

    # Uncomment below if you want 12-input models instead:
    # initialize_models(input_size=12)