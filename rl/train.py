import torch
import gym
from gym.spaces import Discrete, Box 
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt



class Agent(nn.Module):
    def __init__(self, obs_dim, inner,act_dim):
        super().__init__()
        net = nn.Sequential(
            nn.Linear(obs_dim,inner),
            nn.Tanh(),
            nn.Linear(inner, inner),
            nn.Tanh(),
            nn.Linear(inner, act_dim)
        )

    def forward(self,x):
        return self.net(x)
    
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
    
def train(render=False, hidden_shape = [32], lr=1e-2, epochs=64, bs=5000):
    env = gym.make("CartPole-v1")
    #env = gym.make('LunarLander-v2', render_mode="rgb_array")

    obs_dim = env.observation_space.shape[0]
    nacts = env.action_space.n

    # DO NET HERE
    net = mlp(sizes=[obs_dim]+hidden_shape+[nacts])
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
            
            if done:
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