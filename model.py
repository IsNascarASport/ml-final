import torch
import torch.nn as nn
import torch.nn.functional as F

import csv
import random

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

X_train, y_train = [], []
X_validation, y_validation = [], []

training_file = 'training.csv'
validation_file = 'validation.csv'

# Read test data
with open(training_file, mode='r') as f:
    reader = csv.reader(f)
    for row in reader:
        X_train.append(row[:4])
        y_train.append(row[4])

# Read validation data
with open(validation_file, mode='r') as f:
    reader = csv.reader(f)
    for row in reader:
        X_validation.append(row[:4])
        y_validation.append(row[4])

"""
EXPLANATION OF DATA
====================

Each data point in X_train is a list containing:
* X[0]: raw text of tweet
* X[1]: number of characters in tweet
* X[2]: percentage of uppercase characters in tweet
* X[3]: number of punctuation characters in tweet

Each data point in y_train is the author of its corresponding tweet in X_train.
"""