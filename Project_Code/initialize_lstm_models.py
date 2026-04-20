import torch
from lstm_threshold import LSTM_Threshold
from lstm_gain_scheduler import LSTM_GainScheduler

print("Creating initial LSTM models with special output initialization...")

# LSTM 1 - Threshold (unchanged)
model1 = LSTM_Threshold()
model1.fc.weight.data.zero_()
model1.fc.bias.data.fill_(1.0986)   # sigmoid(1.0986) ≈ 0.75

# LSTM 2 - Gain Scheduler (NEW: produces final_mult = 1.0 after *2 + damping 0.5)
model2 = LSTM_GainScheduler()
model2.fc.weight.data.zero_()
model2.fc.bias.data.fill_(0.0)      # sigmoid(0) = 0.5 → 0.5*2 = 1.0 → final_mult = 1.0

torch.save(model1.state_dict(), "lstm_threshold.pth")
torch.save(model2.state_dict(), "lstm_gain_scheduler.pth")

print("Done. Initial models created (Gain final_mult starts at exactly 1.0).")