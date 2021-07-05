#! /usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

class Actor(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit):
        super(Actor, self).__init__()
        self.act_limit = act_limit
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh()
        )        

    def forward(self, obs):
        return self.act_limit * self.layers(obs)

class QFunc(nn.Module):

    def __init__(self, obs_dim, act_dim):
        super(QFunc, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, act):
        q = self.layers(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)

class ActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit):
        super(ActorCritic, self).__init__()
        self.pi = Actor(obs_dim, act_dim, act_limit)
        self.q = QFunc(obs_dim, act_dim)

    def get_action(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()