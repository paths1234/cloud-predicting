'''''''
Referred from achmfirmansyah / Dissertation-Nowcasting.py
''''''''

import os
import torch
from torch import nn
from torch.nn import functional as F
import models.config as cfg
     

class ConvLSTM(nn.Module):
  def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride=1, padding=1):
    """
    Initiate the ConvLSTM parameters 
    '''
    Parameters
    ----------
    input_channels: int 
      the size of input channel
    output_channels: int
      the size of output channel
    kernel_size: int
      kernel size for convolution operation
    b_h_w: tuple of int
      consist of batchsize, height, and width, of frame
    """
    super().__init__()
        
    # Initiate the Convolution layer at first
    self._conv = nn.Conv2d(in_channels=input_channel + output_channel,
                               out_channels=output_channel*4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
  
    self._batch_size, self._state_height, self._state_width = b_h_w
    # Initiate the Weight using zeros weighting scheme      
    self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width))
    self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width))
    self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width))
        
    self._input_channel = input_channel
    self._output_channel = output_channel

  
  # inputs and states should not be all none
  # inputs: S*B*C*H*W
  def forward(self, inputs=None, states=None, seq_len=cfg.sequence_len['IN']):
    # Initiate the Previous Cell Activation and Hidden States for first iterations
    # Since in the first iterations no prior information provided, initiate with zeros 
    if states is None:
      c = torch.zeros((inputs.size(1), self._output_channel, self._state_height,
                                  self._state_width), dtype=torch.float)
      h = torch.zeros((inputs.size(1), self._output_channel, self._state_height,
                             self._state_width), dtype=torch.float)
    else:
      h, c = states

    outputs = []

    # Iterates for each element in sequences
    for index in range(seq_len):
      # initial inputs
      if inputs is None:
        x = torch.zeros((h.size(0), self._input_channel, self._state_height,
                                      self._state_width), dtype=torch.float)
      else:
        x = inputs[index, ...]
      
      # Concantenate the Input and Hidden Layer; Hence the size is summation of input channel size with hidden layersize
      cat_x = torch.cat([x, h], dim=1)
      # Convolution Operation
      conv_x = self._conv(cat_x)
      
      # Check the equation underlies the ConvLSTM process!!

      # Initiate the Input Gate; Forget Gate; Cell Activation; and Output Channel
      i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

      # Updating the gate value based on the equations

      i = torch.sigmoid(i+self.Wci*c)
      f = torch.sigmoid(f+self.Wcf*c)
      c = f*c + i*torch.tanh(tmp_c)
      o = torch.sigmoid(o+self.Wco*c)
      h = o*torch.tanh(c)
      outputs.append(h)
    return torch.stack(outputs), (h, c)
