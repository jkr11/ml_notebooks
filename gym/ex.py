import gymnasium as gym
import matplotlib.pyplot as plt

import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple

class ActorCritic(nn.Module):
    def __init__(self, dimin, dimout, dimhidden = 32):
        self.l1 = nn.Linear(dimin, dimhidden)
        self.l2 = nn.Linear(dimhidden, dimout)

        self.c1 = nn.Linear(dimin, dimhidden)
        self.c2 = nn.Linear(dimhidden, 1)

    def forward(self, obs:Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.tanh(self.l1(obs))
        act = torch.log_softmax(self.l2(x))
        x = torch.relu(self.c1(obs))
        return act, self.c2(x)
    

if __name__ == "__main__":
    env = gym.make("Cartpole")
    
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    n_act = env.action_space.n
    state, info = env.reset()
    n_obs = len(state)

    model = ActorCritic(env.observation_space.shape[0], int(env.action_space.n))
    opt = torch.optim.Adam(model.get_parameter(), lr=1e-2)

    def train_step(x:Tensor, saction:Tensor, reward:Tensor, ologdist:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        logdist, val = model(x)
        advantage = reward.reshape(-1,1) - val
        ratio = torch.exp(logdist - ologdist)

    def get_action(obs:Tensor) -> Tensor:
        ret = torch.exp(model(obs)[0])
        ret = torch.multinomial(ret)
        return ret



      

