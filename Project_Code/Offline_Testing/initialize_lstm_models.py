import torch
from lstm_threshold import LSTM_Threshold
from lstm_gain_scheduler import LSTM_GainScheduler

def initialize_models(input_size=13, save=True):
    print(f"Creating fresh SIMPLE LINEAR models with {input_size} input features...")

    # Threshold model - starts neutral (~0.5 probability)
    model1 = LSTM_Threshold(input_size=input_size)
    model1.linear.bias.data.fill_(1.3)

    # Gain scheduler - starts neutral (final mult = 1.0)
    model2 = LSTM_GainScheduler(input_size=input_size)
    model2.linear.bias.data.fill_(0.0)   # neutral

    if save:
        torch.save(model1.state_dict(), "lstm_threshold.pth")
        torch.save(model2.state_dict(), "lstm_gain_scheduler.pth")
        print("Done. Simple linear models created and saved.")
        print("   • Threshold starts at probability ~0.5")
        print("   • Gain scheduler starts at multiplier exactly 1.0")
    else:
        print("Models created in memory (not saved).")

    return model1, model2

if __name__ == "__main__":
    initialize_models(input_size=13)