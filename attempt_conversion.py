import torch
import onnx
from onnx2torch import convert
from torch import nn
from torch import Tensor
import numpy as np

class LearnedController(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4,4,bias=False),
            nn.Linear(4,20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,4),
            nn.Linear(4,2,bias=False),
        )
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
