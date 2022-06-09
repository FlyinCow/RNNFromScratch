from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNCell(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(RNNCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size

    self.wh = nn.Linear(hidden_size, hidden_size)
    self.wx = nn.Linear(input_size, hidden_size)

  def forward(self, x, h=None):
    if h is None:
      h = torch.zeros([1, self.hidden_size])
    x = self.wx(x)
    h = self.wh(h)
    h = F.tanh(x+h)
    return h


class RNN(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(RNN, self).__init__()
    self.cell = RNNCell(input_size, hidden_size)
    self.input_size = input_size
    self.hidden_size = hidden_size

  def forward(self, x, h0=None):
    # x: [time step * batch size * feature size]
    # h0: [batch size * hidden size]
    h = []
    for t in x:
      # t: [batch size * feature size]
      h0 = self.cell(t, h0)
      h.append(h0)

    return h0, h
