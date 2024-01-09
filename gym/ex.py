import gymnasium as gym
import matplotlib.pyplot as plt

import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple

class DecisionTransformer(nn.Module):
    def __init__(self, R, s, a, t):
        super().__init__()
        

