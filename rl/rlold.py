import torch
import gym
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

def Agent(obs_dim, inner, act_dim):
    return nn.Sequential(
        nn.Linear(obs_dim,inner),
        nn.Tanh(),
        nn.Linear(inner, inner),
        nn.Tanh(),
        nn.Linear(inner, act_dim)
    )

def train(render=False, hidden_shape = 32, lr=1e-2, epochs=64, bs=5000):
    env = gym.make("CartPole-v1", render_mode = "human")

    obs_dim = env.observation_space.shape[0]
    nacts = env.action_space.n

    net = Agent(obs_dim, hidden_shape, nacts)
    from torch.distributions.categorical import Categorical
    def get_policy(obs) -> Categorical:
        logits = net(obs)
        return Categorical(logits=logits)
    
    def get_action(obs):
        return get_policy(obs).sample().item()
    
    def loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()
    
    opt = Adam(net.parameters(), lr = lr)

    def train_one():
        BO = []
        BA = []
        BW = []
        BR = []
        BL = []
        

        obs, _ = env.reset() 
        done = False
        R = []
        ddone = False
        
        while True:

            if not ddone:
                env.render()
            BO.append(obs)

            act = get_action(torch.as_tensor(np.array(obs), dtype=torch.float32))
            obs, rew, done, _, _ = env.step(act)

            BA.append(act)
            R.append(rew)
            
            if done or len(BO) > bs:
                ep_ret, ep_len = sum(R), len(R)
                BR.append(ep_ret)
                BL.append(ep_len)

                BW += [ep_ret] * ep_len

                (obs, _), done, R = env.reset(), False, []
                ddone = True
                if len(BO) > bs:
                    break
        

        opt.zero_grad()
        
        bloss = loss(obs=torch.as_tensor(BO, dtype=torch.float32), act=torch.as_tensor(BA,dtype=torch.float32), weights=torch.as_tensor(BW,dtype=torch.float32))
        bloss.backward()
        opt.step()
        return bloss, BR, BL
    
    BBR = []

    for i in range(epochs):
        bl, BR, BL = train_one()
        BBR.extend(BR)
        plt.plot(BBR)
        plt.pause(0.05)
        
        #plt.show()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, bl, np.mean(BR), np.mean(BL)))
        
if __name__ == "__main__":
    train(render=True,lr=1e-2)