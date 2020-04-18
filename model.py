import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Creates an lstm model with Input (Dimention D, Hidden Dimention H and # of LSTM Layers)

Inputs
in = shape (Sequence Length, Batch, D) This is your X
h0 = shape (num_layers, batch, H)
c0 = shape (num_layers, batch, H)

c is cell state (long term memory)
h is hidden state (short term memory)

Outputs
out = shape (Sequence Length, Batch, H)
next_h = shape (num_layers, batch, H)
next_c = shape (num_layers, batch, H)

out, (next_h, next_c) = lstm(in, (h0, c0))
"""

class TweetDecider(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, drop_prob = 0.5):
        super(TweetDecider, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
