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

X_and_y_train = []
X_train, y_train = [], []
X_per_user = {'thetalkingjed': [],
    'isnascarasport': [],
    'wildenian_thot': []}

# Read all data from csv into dict
with open('tweet_data.csv', mode='r') as f:
    reader = csv.reader(f, dialect='excel')
    for row in reader:
        user = row[4].lower()
        X_per_user[user].append(row)

# Randomly sample 200 from each user
for key in X_per_user.keys():
    data = X_per_user[key]
    sample_size = 250
    X_and_y_train.extend(random.sample(data, sample_size))

# Shuffle and split into X_train and y_train
random.shuffle(X_and_y_train)
for entry in X_and_y_train:
    X_train.append(X_and_y_train[:4])
    y_train.append(entry[4])

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