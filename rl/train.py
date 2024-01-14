import torch
import gym
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

class Agent(nn.Module):
    def __init__(self, obs_dim, inner, act_dim):
        self.actor = nn.Sequential(
        nn.Linear(obs_dim,inner),
        nn.Tanh(),
        nn.Linear(inner, inner),
        nn.LogSoftmax(-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim,inner),
            nn.ReLU()
            nn.Linear(inner, 1),
        )

    def forward(self, x):
        return self.act(x), self.critic(x) # log prob + obs
    
    
def eval(model, env):
    (obs, _), done, ddone = env.reset(), False, False
    rr = 0.
    while not done and not ddone:
        act = model(torch.Tensor(obs))[0].argmax().item()
        obs, rew, done, ddone, _ = env.step(act)
        rr += float(rew)
    return rr



if __name__ == "__main__":
    render=False
    hidden_shape = 32
    lr=1e-2
    epochs=64
    bs=5000
    REPLAY_BUFFER_SIZE = 2000

    env = gym.make("CartPole-v1")

    net = Agent(env.observation_space.shape[0], hidden_shape, env.action_space.n)

    opt = Adam(net.parameters(), lr = lr)

    def train_step():
        pass

    def get_action(obs):
        with torch.no_grad:
            ret = torch.exp(net(obs)[0])
            ret = torch.multinomial(ret)
        return ret
    
    SB, AB, RB = [], [], []
    from tqdm import trange
    for ep in (t:=trange(epochs)):
        obs = env.reset()[0]
        rr, done, ddone = [], False, False
        while not done and not ddone:
            act = get_action(torch.Tensor(obs)).item()
            SB.append(np.copy(obs))
            AB.append(act)

            obs, rew, done, ddone, _ = env.step(act)
            rr.append(float(rew))
        RB += [np.sum(rr[i:]) for i in range(len(rr))] # Rev cumsum also np.cumsum(r[::-1])[::-1]

        SB, AB, RB = SB[-REPLAY_BUFFER_SIZE:], AB[-REPLAY_BUFFER_SIZE:], RB[-REPLAY_BUFFER_SIZE:]
        S, A, R = torch.Tensor(SB), torch.Tensor(AB), torch.Tensor(RB)

        for i in bs:
            

        
